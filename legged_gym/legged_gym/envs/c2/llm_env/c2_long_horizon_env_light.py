import os
import random
import numpy as np
from collections import OrderedDict, defaultdict

from legged_gym import LEGGED_GYM_ROOT_DIR
from isaacgym.torch_utils import (
    torch_rand_float,
    get_euler_xyz,
    quat_from_euler_xyz,
    tf_apply,
)
from isaacgym import gymtorch, gymapi, gymutil
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from isaacgym.torch_utils import *
from legged_gym.envs.c2.walk.c2_walk_config import CyberWalkCfg
from legged_gym.envs.c2.llm_env.c2_long_horizon_env import (
    CyberLongHorizonEnv,
    POLICY_ID,
    POLICY_NAME,
)
from legged_gym.utils.helpers import class_to_dict

from legged_gym.envs.c2.llm_env.c2_long_horizon_config_light import (
    CyberLongHorizonLightCfg,
)
from legged_gym.envs.c2.llm_code.llm_code_long_horizon_light import *

# from legged_gym.envs.c2.llm_code.llm_code_robotool_light import *
from legged_gym.legged_gym.envs.c2.llm_code.llm_code_multi_robotool_light import *
import math
import json


class CyberLongHorizonLightEnv(CyberLongHorizonEnv):
    ### Create and Init

    def __init__(
        self,
        cfg: CyberLongHorizonLightCfg,
        sim_params,
        physics_engine,
        sim_device,
        headless,
    ):
        self.exp_scene_id = cfg.task.exp_scene_id
        self.score_mode = cfg.task.score_mode
        self.use_down_stair_box = cfg.task.use_down_stair_box
        self.fix_box_actor = cfg.env.fix_box_actor
        if cfg.task.multi_robotool:
            with open(
                r"./resources/data/random_data_light.json", "r", encoding="utf-8"
            ) as f:
                self.random_data = json.load(f)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        self.num_train_envs = cfg.env.num_envs
        self.load_RL_policy()

        print("load policy & init ok!")

    def _create_task_props(self):
        """
        Create the task-related assets
        """
        ## Record the object info
        self.object_all_size = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.object_size = torch.zeros(
            self.num_envs, 5, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.object_info = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.object_density = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # 0 mass ground-truth
        # 1 Volume ground-truth
        # 2 friction ground-truth linear damping
        # 3 Drag coefficient [0 TODO]
        # 4 object type [0 TODO]
        # 5 blank [0 TODO]
        self.objectbox_handle = []
        objectbox_pose = gymapi.Transform()
        object_base_init_state_list = (
            [0, 0, 0.125] + [0, 0, 0, 1] + [0, 0, 0] + [0, 0, 0]
        )
        self.object_base_init_state = to_torch(
            object_base_init_state_list, device=self.device, requires_grad=False
        )

        ## Prepare for wall and button
        self.wall_handles = []
        self.button_handles = []
        wall_asset_options = gymapi.AssetOptions()
        wall_asset_options.fix_base_link = True
        wall_asset_options.density = 10
        self.wall_asset = self.gym.create_box(self.sim, 0.1, 10, 4, wall_asset_options)
        box_rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(
            self.wall_asset
        )
        box_rigid_shape_props_asset[0].friction = 0.2
        self.gym.set_asset_rigid_shape_properties(
            self.wall_asset, box_rigid_shape_props_asset
        )
        init_wall_state_list = [0.0, 0.0, 0.125] + [0, 0, 0, 1] + [0, 0, 0] + [0, 0, 0]
        self.init_wall_state = to_torch(
            init_wall_state_list, device=self.device, requires_grad=False
        )
        self.wall_pose = gymapi.Transform()
        self.wall_pose.p = gymapi.Vec3(*self.init_wall_state[:3])

        button_asset_options = gymapi.AssetOptions()
        button_asset_options.fix_base_link = True
        self.button_asset = self.gym.create_box(
            self.sim, 0.01, 0.1, 0.1, button_asset_options
        )
        init_button_state_list = [0.0, 0.0, 10] + [0, 0, 0, 1] + [0, 0, 0] + [0, 0, 0]
        self.init_button_state = to_torch(
            init_button_state_list, device=self.device, requires_grad=False
        )
        self.button_pose = gymapi.Transform()
        self.button_pose.p = gymapi.Vec3(*self.init_button_state[:3])

        return (self.wall_pose, self.wall_asset)

    def _create_task_env(self, i, props=None):
        """
        Crate the task-related environment setups
        """
        self._create_box_env(i)
        self._create_wall_and_attachment(i)

    def _create_box_env(self, env_id):
        box_w = self.cfg.task.box_w
        box_h = self.cfg.task.box_h
        box_d = random.uniform(
            self.cfg.task.box_height_range[0], self.cfg.task.box_height_range[1]
        )

        for _i in range(self.num_object_actor):
            box_p = random.uniform(
                self.cfg.task.box_density_range[0], self.cfg.task.box_density_range[1]
            )

            linear_damping = random.uniform(
                self.cfg.task.box_linear_damping_range[0],
                self.cfg.task.box_linear_damping_range[1],
            )

            box_assert_options = gymapi.AssetOptions()
            box_assert_options.density = box_p
            box_assert_options.fix_base_link = self.fix_box_actor
            if _i > 0:
                box_assert_options.fix_base_link = True  
                box_assert_options.density = 200  
            box_assert_options.linear_damping = (
                linear_damping  # random.uniform(0., 0.5)
            )
            if _i == 0:
                box_assert_options.fix_base_link = False
                self.object_all_size[env_id, 0] = box_w
                self.object_all_size[env_id, 1] = box_h
                self.object_all_size[env_id, 2] = box_d

                self.object_info[env_id, 1] = box_w * box_h * box_d
                self.object_info[env_id, 0] = self.object_info[env_id, 1] * box_p
                self.object_info[env_id, 2] = box_assert_options.linear_damping
                self.object_density[env_id] = box_p

            _objectbox_pos = self.env_origins[env_id].clone()
            self.object_size[env_id, _i] = box_d

            objectbox_asset = self.gym.create_box(
                self.sim, box_w, box_h, box_d, box_assert_options
            )
            box_rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(
                objectbox_asset
            )
            box_rigid_shape_props_asset[0].friction = random.uniform(
                self.cfg.task.box_friction_range[0], self.cfg.task.box_friction_range[1]
            )
            if _i == 2:
                box_rigid_shape_props_asset[0].friction = 0.8
            self.gym.set_asset_rigid_shape_properties(
                objectbox_asset, box_rigid_shape_props_asset
            )
            if not self.use_down_stair_box:
                if (self.exp_scene_id == 1) or _i:
                    if _i == 0:
                        box_branch = random.uniform(
                            self.cfg.task.stair_rand[0], self.cfg.task.stair_rand[1]
                        )
                        if self.cfg.task.multi_robotool:
                            box_branch = (
                                self.random_data["low stair size"][env_id] - 0.3
                            )
                        if box_branch > 0:
                            box_d += random.uniform(self.cfg.task.box_height_range[0], self.cfg.task.box_height_range[1])
                            _objectbox_pos[2] += 0.12 * _i
                    else:
                        box_d += random.uniform(self.cfg.task.box_height_range[0], self.cfg.task.box_height_range[1])
                        _objectbox_pos[2] += 0.12 * _i
            else:
                box_d -= random.uniform(0.23, 0.28)
                _objectbox_pos[2] -= 0.14

            objectbox_offset = 1.0
            _objectbox_pos[0] += objectbox_offset + 0.81 * _i
            if _i == 0:
                _objectbox_pos[0] += objectbox_offset - 0.3
                _objectbox_pos[0] += 0.5

            objectbox_pose = gymapi.Transform()
            objectbox_pose.p = gymapi.Vec3(*_objectbox_pos)

            ahandle = self.gym.create_actor(
                self.envs[env_id],
                objectbox_asset,
                objectbox_pose,
                "Object",
                env_id,
                0,
                _i + 1,
            )
            self.objectbox_handle.append(ahandle)
            self.gym.set_rigid_body_color(
                self.envs[env_id],
                ahandle,
                0,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(0.75, 0, 0),
            )

    def _create_wall_and_attachment(self, i, props=None):
        objectbox_offset = 1.0
        pos_wall = self.env_origins[i].clone()
        pos_wall[0] += objectbox_offset + 0.81 * 3 - 0.35
        pos_wall[2] += 0.12 * 3
        self.wall_pose.p = gymapi.Vec3(*pos_wall)

        wall_handle = self.gym.create_actor(
            self.envs[i], self.wall_asset, self.wall_pose, "Object", i, 0, 4
        )
        self.wall_handles.append(wall_handle)
        self.gym.set_rigid_body_color(
            self.envs[i],
            wall_handle,
            0,
            gymapi.MESH_VISUAL_AND_COLLISION,
            gymapi.Vec3(0.8, 0.8, 0.8),
        )

        pos_button = self.env_origins[i].clone()
        pos_wall[1] += 0.5
        self.button_pose.p = gymapi.Vec3(*pos_button)
        button_handle = self.gym.create_actor(
            self.envs[i], self.button_asset, self.button_pose, "button", i, 0, 5
        )
        color = gymapi.Vec3(1, 0, 0)
        self.gym.set_rigid_body_color(
            self.envs[i], button_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color
        )
        self.button_handles.append(button_handle)

    def _init_buffers(self):
        super()._init_buffers()

        self._init_climb_box_buffer()
        self._init_manipulate_buffer()
        self._init_button_buffer()

        # self.generate_llm_plan()
        self.object_path_slope = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        self.record_randomization = True
        self.record_flag = False


    def _init_utils_buffer(self):
        super()._init_utils_buffer()
        ## for test

        self.test_task_id = self.cfg.task.test_task_id
        # 0 climb
        # 1 manipulate
        # 2 button
        # 3 manipulate + climb
        # 4 climb_floor2_to_button

        self.min_dis_toe2button = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.min_pos_toe2button = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.min_dis_toe2button[:] = 10
        self.dis_finish_progress = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        self.init_left_toe_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        torch.set_printoptions(precision=10)

    def _init_button_buffer(self):
        super()._init_button_buffer()

        self.button_ref_plat_height = (
            self.object_root_states[:, 2, 2] + self.object_size[:, 2] / 2
        )

        button_start_pos = self.env_origins[
            :, :3
        ].clone()  
        button_start_pos[:, 0] += 2.48 
        button_start_pos[:, 2] = self.button_ref_plat_height + 0.25
        self.observe_cam_coordinate = button_start_pos

    def _init_for_RL_policy(self):
        super()._init_for_RL_policy()

        self.task_id[:] = 5
        self.rl_policy_id[:] = POLICY_ID["manipulate"]

        if self.single_test_work:
            if (self.test_task_id == 0) or (self.test_task_id == 4):
                self.task_id[:] = 1
                self.rl_policy_id[:] = POLICY_ID["climb"]
                if self.test_task_id == 4:
                    self.task_id[:] = 3
            elif (self.test_task_id == 1) or (self.test_task_id == 3):
                self.task_id[:] = 0
                self.rl_policy_id[:] = POLICY_ID["manipulate"]
            elif self.test_task_id == 2:
                self.task_id[:] = 3
                self.rl_policy_id[:] = POLICY_ID["stand"]

    ### Policy

    def _call_llm_generate_plan(self, i):
        if self.cfg.task.multi_robotool:
            self.llm_plan_generator = globals().get(f"generate_robotool_{str(i)}")
            self.llm_plan_generator(self, i)
        elif self.cfg.task.test_llm_branch:
            self.llm_plan_generator_1 = globals().get(
                self.cfg.task.test_llm_code_branch_1
            )
            self.llm_plan_generator_2 = globals().get(
                self.cfg.task.test_llm_code_branch_2
            )
            if self.object_size[i, 1] >= 0.24:
                self.llm_plan_generator_1(self, i)
            else:
                self.llm_plan_generator_2(self, i)
        else:
            self.llm_plan_generator = globals().get(self.cfg.task.test_llm_code)
            self.llm_plan_generator(self, i)

    def hand_write_multi_planner(self):
        # self.RL_policy_name = "stand"
        self.cond_has_first_box = (
            torch.norm(
                self.object_root_states[:, 0, :2] - self.object_root_states[:, 1, :2],
                dim=1,
            )
            < 1.2
        )
        control_args = torch.zeros(
            self.num_envs, 10, dtype=torch.float, device=self.device
        )

        self.rl_policy_id[
            torch.logical_and((self.task_finish_buf == 1), (self.task_id == 0))
        ] = POLICY_ID["move_to_pos"]
        self.rl_policy_id[
            torch.logical_and((self.task_finish_buf == 1), (self.task_id == 1))
        ] = POLICY_ID["climb"]
        self.rl_policy_id[
            torch.logical_and((self.task_finish_buf == 1), (self.task_id == 2))
        ] = POLICY_ID["move_to_pos"]
        control_args[self.task_id[:, 0] == 3, 0] = 2.65
        self.rl_policy_id[
            torch.logical_and((self.task_finish_buf == 1), (self.task_id == 3))
        ] = POLICY_ID["stand"]
        self.rl_policy_id[
            torch.logical_and((self.task_finish_buf == 1), (self.task_id == 4))
        ] = POLICY_ID["button"]
        self.rl_policy_id[
            torch.logical_and((self.task_finish_buf == 1), (self.task_id == 5))
        ] = POLICY_ID["sit_down"]

        self.task_id[self.task_finish_buf == True] += 1
        self.task_finish_buf[self.task_finish_buf == True] = False

        return self.RL_policy_name, control_args

    def execute_llm_plan(self):
        self.cond_has_first_box = (
            torch.norm(
                self.object_root_states[:, 0, :2] - self.object_root_states[:, 1, :2],
                dim=1,
            )
            < 1.2
        )
        super().execute_llm_plan()

        self.button_ref_plat_height = (
            self.object_root_states[:, 2, 2] + self.object_size[:, 2] / 2
        )

        if self.num_envs == 1:
            print(self.cur_step, POLICY_NAME[int(self.rl_policy_id[:, 0])])
            # print(self.episode_length_buf[:], self.target2object_diff[:])
            # if self.rl_policy_id[0, 0] == POLICY_ID["climb"]:

            if self.rl_policy_id[0, 0] == POLICY_ID["manipulate"]:
                print(self.target2object_diff)
            if self.rl_policy_id[0, 0] == POLICY_ID["stand"]:
                print(self.base_pos[:, 2], self.button_ref_plat_height + 0.3)

        diff_button2hand = (
            self.target_hand_pos_world[:, :3] - self.world_hand_positions[:, :3]
        )
        dis_button2hand = torch.norm(diff_button2hand[:, :3], dim=1)
        more_close_cond = dis_button2hand < self.min_dis_toe2button
        self.min_dis_toe2button[more_close_cond] = dis_button2hand[more_close_cond]
        self.min_pos_toe2button[more_close_cond] = (
            self.world_hand_positions[more_close_cond, :3]
            - self.env_origins[more_close_cond, :3]
        )
        self.dis_finish_progress = self.min_dis_toe2button / torch.norm(
            self.init_left_toe_pos
            - self.target_hand_pos_world[:, :3]
            + self.env_origins[:, :3],
            dim=1,
        )
        touched = torch.logical_and(
            (torch.abs(diff_button2hand[:, 1]) < self.cfg.task.finish_button[1]),
            (torch.abs(diff_button2hand[:, 2]) < self.cfg.task.finish_button[2]),
        )
        touched = torch.logical_and(
            touched,
            (torch.abs(diff_button2hand[:, 0]) < self.cfg.task.finish_button[0]),
        )
        self.all_task_finished[touched] = True

    def check_task_finish(self):
        super().check_task_finish()
        diff_button2hand = (
            self.target_hand_pos_world[:, :3] - self.world_hand_positions[:, :3]
        )

        dis_button2hand = torch.norm(diff_button2hand[:, :3], dim=1)
        more_close_cond = dis_button2hand < self.min_dis_toe2button
        self.min_dis_toe2button[more_close_cond] = dis_button2hand[more_close_cond]
        self.min_pos_toe2button[more_close_cond] = (
            self.world_hand_positions[more_close_cond, :3]
            - self.target_hand_pos_world[more_close_cond, :3]
        )
        self.dis_finish_progress = self.min_dis_toe2button / torch.norm(
            self.init_left_toe_pos
            - self.target_hand_pos_world[:, :3]
            + self.env_origins[:, :3],
            dim=1,
        )

    ### Step

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        super().post_physics_step()

        self.init_left_toe_pos = (
            self.init_feet_positions[:, 0] - self.env_origins[:, :3]
        )

    ###  Reset   ##

    def check_termination(self):
        super().check_termination()

        if self.single_test_work:
            if self.num_envs > 1:
                self.compute_diff_physics_arg(
                    self.finish_basic.nonzero(as_tuple=False).flatten()
                )
            if (self.test_task_id == 0) or ((self.test_task_id == 3)):
                self.check_climb_finish()
            if self.test_task_id == 1:
                self.check_manipulate_finish()
            elif (self.test_task_id == 2) or (self.test_task_id == 4):
                self.check_button_finish()

            print("Success: ", self.finish_once.sum(), "/", (self.reset_once).sum())

        # return

    def check_climb_finish(self):
        # self.reset_buf[self.task_id[:, 0] == 4] = True
        self.finish_once[(self.task_id[:, 0] == 3) & (self.reset_once == 0)] = 1
        self.finish_time[
            (self.finish_once == 1) | (self.reset_once == 0)
        ] = self.episode_length_buf[(self.finish_once == 1) | (self.reset_once == 0)]

        self.finish_stairs_count[
            (self.base_pos[:, 0] - self.env_origins[:, 0] > 2.6)
            & (self.base_pos[:, 2] > 0.7),
            2,
        ] = 1
        self.finish_stairs_count[
            (self.base_pos[:, 0] - self.env_origins[:, 0] > 1.8)
            & (self.base_pos[:, 2] > 0.5),
            1,
        ] = 1
        self.finish_stairs_count[
            (self.base_pos[:, 0] - self.env_origins[:, 0] > 1.0)
            & (self.base_pos[:, 2] > 0.3),
            0,
        ] = 1

        if self.num_envs > 1:
            print("Climb 2 : ")
            self.compute_diff_physics_arg(
                self.finish_stairs_count[:, 1].nonzero(as_tuple=False).flatten()
            )

        self.obs_history_climb_full[
            self.last_finish_stairs_count[:, 0] != self.finish_stairs_count[:, 0], :
        ] = 0
        self.obs_history_climb_full[
            self.last_finish_stairs_count[:, 1] != self.finish_stairs_count[:, 1], :
        ] = 0
        self.obs_history_climb_full[
            self.last_finish_stairs_count[:, 2] != self.finish_stairs_count[:, 2], :
        ] = 0
        self.last_finish_stairs_count = self.finish_stairs_count

        if self.test_task_id == 3:
            self.finish_basic[(self.task_id[:, 0] == 1)] = True
            print("Manipulate finish", self.finish_basic.sum())

        print(
            "Task finish: ",
            self.finish_stairs_count[:, 0].sum(),
            self.finish_stairs_count[:, 1].sum(),
            self.finish_stairs_count[:, 2].sum(),
        )

    def check_manipulate_finish(self):
        self.finish_once[
            (self.task_finish_buf[:, 0] == 1) & (self.task_id[:, 0] == 0)
        ] = 1
        self.finish_time[
            (self.finish_once == 1) | (self.reset_once == 1)
        ] = self.episode_length_buf[(self.finish_once == 1) | (self.reset_once == 1)]
        self.reset_once[(self.task_id[:, 0] == 1) & (self.reset_once < 1)] = 1

    def check_button_finish(self):
        # self.reset_buf[self.task_id[:, 0] == 6] = True
        self.finish_once[(self.task_id[:, 0] == 6)] = 1
        self.finish_time[
            (self.finish_once == 1) | (self.reset_once == 1)
        ] = self.episode_length_buf[(self.finish_once == 1) | (self.reset_once == 1)]

        if self.test_task_id == 4:
            self.finish_basic[(self.task_id[:, 0] == 5)] = True
            print("button finish", self.finish_basic.sum())

        self.reset_once[(self.task_id[:, 0] == 6) & (self.reset_once < 1)] += 1
        # print()

    def _reset_root_states(self, env_ids):
        """
        Reset all actor root states
        robot - root_states         [0]
        box - object_root_states    [1:4]
        button - button_root_states [4]
        wall - wall_root_states     [5]

        """
        ### reset robot
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 0] -= 1
        self.root_states[env_ids, 1] -= 1.4
        if self.exp_scene_id == 2:
            self.root_states[env_ids, 0] += 1

        # add some noise: #ljh: new added!!!!
        if self.single_test_work:
            self.root_states[env_ids, 1] += 1.4
            self.root_states[env_ids, 0] += 1
            if (self.test_task_id == 1) or (self.test_task_id == 3):
                self.root_states[env_ids, 0] -= 1.0
            if self.test_task_id == 2:
                # self.root_states[env_ids, 1] += 1
                self.root_states[env_ids, 0] += 2.65
                self.root_states[env_ids, 2] += self.button_ref_plat_height
                # self.root_states[env_ids, 0:2] += torch_rand_float(-0.03, 0.03, (len(env_ids), 2), device=self.device)
                self.root_states[env_ids, 0:2] += torch_rand_float(
                    -0.04, 0.04, (len(env_ids), 2), device=self.device
                )
                # self.root_states[env_ids, 1] += torch_rand_float(-0.1, 0.1, (len(env_ids), 2), device=self.device)[:, 1]
            if self.test_task_id == 4:
                self.root_states[env_ids, 0] += 1.8
                self.root_states[env_ids, 2] += (
                    self.object_size[env_ids, 1] * 0.5 + 0.02
                )

        # base velocities
        if getattr(self.cfg.domain_rand, "init_base_vel_range", None) is None:
            base_vel_range = (-0.5, 0.5)
        else:
            base_vel_range = self.cfg.domain_rand.init_base_vel_range
        self.root_states[env_ids, 7:13] = torch_rand_float(
            *base_vel_range,
            (len(env_ids), 6),
            device=self.device,
        )  # [7:10]: lin vel, [10:13]: ang vel

        self._reset_box_object_root_states(env_ids)
        self._reset_button_root_states(env_ids)

        actor_ids_int32 = (
            torch.arange(
                0,
                self.num_envs * self.num_actors,
                dtype=torch.int32,
                device=self.device,
            )
            .reshape((self.num_envs, -1))[env_ids]
            .reshape(-1)
        )

        if self.cfg.task.multi_robotool:
            self._load_stored_randomization(env_ids)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.all_root_states),
            gymtorch.unwrap_tensor(actor_ids_int32),
            len(actor_ids_int32),
        )

        if self.record_randomization:
            if not self.record_flag:
                self._record_randomization_data()
                self.record_flag = True

    def _record_randomization_data(self):
        record_box_pos = self.object_root_states[:, 0, :3].tolist()
        record_low_stair_pos = self.object_root_states[:, 1, :3].tolist()
        record_high_stair_pos = self.object_root_states[:, 2, :3].tolist()
        record_button_pos = self.button_root_states[:, :3].tolist()

        record_box_size = self.object_all_size.tolist()
        record_low_stair_size = self.object_size[:, 1].tolist()
        record_high_stair_size = self.object_size[:, 2].tolist()

        record_box_pos_related = (
            self.object_root_states[:, 0, :3] - self.env_origins
        ).tolist()
        record_low_stair_pos_related = (
            self.object_root_states[:, 1, :3] - self.env_origins
        ).tolist()
        record_high_stair_pos_related = (
            self.object_root_states[:, 2, :3] - self.env_origins
        ).tolist()
        record_button_pos_related = (
            self.button_root_states[:, :3] - self.env_origins
        ).tolist()

        write_data = {
            "env origins": self.env_origins.tolist(),
            "box pos": record_box_pos,
            "button pos": record_button_pos,
            "low stair pos": record_low_stair_pos,
            "high stair pos": record_high_stair_pos,
            "box size": record_box_size,
            "low stair size": record_low_stair_size,
            "high stair size": record_high_stair_size,
            "box pos related": record_box_pos_related,
            "button pos related": record_button_pos_related,
            "low stair pos related": record_low_stair_pos_related,
            "high stair pos related": record_high_stair_pos_related,
        }
        save_file_path = "random_data.json"
        with open(save_file_path, "w") as json_file:
            json.dump(write_data, json_file, indent=4)

    def _load_stored_randomization(self, env_ids):
        with open(
            r"./resources/data/random_data_light.json", "r", encoding="utf-8"
        ) as f:
            random_data = json.load(f)

        self.object_root_states[env_ids, 0, :3] = (
            self.env_origins[env_ids]
            + torch.tensor(random_data["box pos related"]).to(self.device)[env_ids]
        )
        self.object_root_states[env_ids, 1, :3] = (
            self.env_origins[env_ids]
            + torch.tensor(random_data["low stair pos related"]).to(self.device)[
                env_ids
            ]
        )
        self.object_root_states[env_ids, 2, :3] = (
            self.env_origins[env_ids]
            + torch.tensor(random_data["high stair pos related"]).to(self.device)[
                env_ids
            ]
        )
        self.button_root_states[env_ids, :3] = (
            self.env_origins[env_ids]
            + torch.tensor(random_data["button pos related"]).to(self.device)[env_ids]
        )

    def _reset_box_object_root_states(self, env_ids):
        ## Reset Box Object
        for i in range(self.num_object_actor):
            self.object_init_states[env_ids, i, :] = self.object_base_init_state
            self.object_init_states[env_ids, i, :3] += self.env_origins[env_ids]
            self.object_init_states[env_ids, i, 0] += 1.0 + 0.81 * i
            if self.exp_scene_id == 1:
                self.object_init_states[env_ids, i, 2] += (
                    self.object_size[env_ids, i] * 0.5 + 0.01 - 0.125
                )
            if self.exp_scene_id == 2:
                self.object_init_states[env_ids, i, 2] += 0.126 * ((2 - abs(i - 2)) - 1)

            if i == 0:
                if not ((self.test_task_id == 0) and self.single_test_work):
                    self.object_init_states[
                        env_ids, i, 0
                    ] -= 0.8  # torch_rand_float(0.6, 0.8, (len(env_ids), 1), device=self.device)[:, 0]    # 0.6
                    self.object_init_states[
                        env_ids, i, 1
                    ] += 0.15  # torch_rand_float(-0.3, 0.3, (len(env_ids), 1), device=self.device)[:, 0]  # 0.15
                    self.object_init_states[env_ids, i, 0] += torch_rand_float(
                        -0.1, 0.1, (len(env_ids), 1), device=self.device
                    )[:, 0]
                if self.exp_scene_id == 2:
                    self.object_init_states[env_ids, i, 1] += 2.0
                    self.object_init_states[env_ids, i, 2] += 0.126 * (2 - abs(i - 2))

            self.object_root_states[env_ids] = self.object_init_states[env_ids]

    def _reset_button_root_states(self, env_ids):
        self.fix_targets[env_ids] = (
            torch.Tensor(self.cfg.env.fix_rel_target)
            .to(self.device)
            .unsqueeze(0)
            .repeat(len(env_ids), 1)
        )
        self.fix_targets[env_ids, :2] += self.env_origins[env_ids, :2]
        self.fix_targets[env_ids, 3:5] += self.env_origins[env_ids, :2]

        self.target_hand_pos_world[env_ids] = self.fix_targets[env_ids]
        self.button_root_states[env_ids] = self.init_button_state
        self.button_root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        self.button_root_states[env_ids, 0] += (
            0.81 * 3 - 0.35 + 1 - 0.062
        )  # 2.95 # 0.4 + 2.62
        self.button_root_states[env_ids, 2] = torch_rand_float(
            0.55, 0.65, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        if self.cfg.task.demo_flag:
            self.button_root_states[env_ids, 2] = 0.65
        self.button_root_states[env_ids, 2] += self.object_size[env_ids, 2]
        self.target_hand_pos_world[env_ids, :3] = self.button_root_states[
            env_ids, :3
        ].clone()

        self.wall_root_states[env_ids] = self.init_wall_state
        self.wall_root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        self.wall_root_states[env_ids, 2] += 0.12 * 3
        self.wall_root_states[env_ids, 0] += 0.81 * 3 - 0.35 + 1

    ### Observation

    def compute_observations_manipulate(self, control_args=None):
        if (control_args != None) and (not self.single_test_work):
            # print("control args=", control_args, self.cur_step)
            control_args = control_args.squeeze(1)
            self.target_pos[:, 0] = (
                control_args[:, 0] + self.cfg.task.push_to_wall_offset
            )  
            self.target_pos[:, 1] = (
                control_args[:, 1] - self.cfg.task.push_to_wall_offset
            )  
        else:
            self.target_pos[:, 0] = self.env_origins[:, 0] + 1.2
            self.target_pos[:, 1] = self.env_origins[:, 1] + 0.0
        self.compute_target_info()
        self.target_rot[:, 3] = 1
        obs_buf = torch.cat(
            (
                self.projected_gravity,  # 3
                self.base_target2robot_diff,  # 2
                torch.norm(self.target2robot_diff, dim=1).unsqueeze(-1),  # 1
                self.target_rot,  # 4
                self.base_object2robot_diff,  # 2
                torch.norm(self.object2robot_diff, dim=1).unsqueeze(-1),  # 1
                self.object_rot,  # 4
                self.target2object_diff,  # 2
                torch.norm(self.target2object_diff, dim=1).unsqueeze(-1),  # 1
                # self._get_object_observation_noise(),
                self._get_base_pose_obs(),  # 6
                self.target_pos[:, :2] - self.env_origins[:, :2],  # 2
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
                # self.dof_vel * self.obs_scales.dof_vel,
                self.command_interface,  # 15
                self.object_info,  # 6
                self.object_all_size,  # 3
            ),
            dim=-1,
        )

        self.obs_history_manipulate = torch.cat(
            (self.obs_history_manipulate[:, self.num_manipulate_obs :], obs_buf[:, :]),
            dim=-1,
        )

    def get_box_observation(self):
        self.n_box_passed[:] = 0
        self.n_box_passed[self.cond_has_first_box == 0] = 1
        if self.exp_scene_id == 2:
            self.n_box_passed[:] = 1
        self.n_box_passed[
            self.root_states[:, 0] > (self.object_root_states[:, 0, 0] - 0.2)
        ] = 1
        self.n_box_passed[
            self.root_states[:, 0] > (self.object_root_states[:, 1, 0] - 0.2)
        ] = 2

        if self.use_down_stair_box:
            self.n_box_passed[
                self.root_states[:, 0] > (self.object_root_states[:, 2, 0] - 0.2)
            ] = 3
            self.n_box_passed[
                self.root_states[:, 0] > (self.object_root_states[:, 3, 0] - 0.2)
            ] = 4
            self.n_box_passed[
                self.root_states[:, 0] > (self.object_root_states[:, 4, 0] - 0.2)
            ] = 5

        self.box_info_buf[:, 0] = (
            1.0
            + 0.8 * self.n_box_passed
            + self.env_origins[:, 0]
            - self.root_states[:, 0]
            - 0.4
        )

        for i in range(self.num_envs):
            if self.n_box_passed[i] == 0:
                self.box_info_buf[i, 7] = self.object_size[i, self.n_box_passed[i]]
            elif self.n_box_passed[i] == 5:
                self.box_info_buf[i, 7] = (
                    -1 * self.object_size[i, self.n_box_passed[i] - 1]
                )
            else:
                self.box_info_buf[i, 7] = (
                    self.object_size[i, self.n_box_passed[i]]
                    - self.object_size[i, self.n_box_passed[i] - 1]
                )

            if self.n_box_passed[i] < 5:
                self.box_info_buf[i, 0] = (
                    self.object_root_states[i, self.n_box_passed[i], 0]
                    - self.root_states[i, 0]
                    - 0.4
                )
            else:
                self.box_info_buf[i, 0] = 1


        return self.box_info_buf

    def _get_box_offset_obs(self, privileged=False):
        return_ = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)

        box_forward = quat_apply(self.object_root_states[:, 0, 3:7], self.forward_vec)
        box_rot = torch.atan2(box_forward[:, 1], box_forward[:, 0])
        box_y_offset = self.object_root_states[:, 0, 1] - self.env_origins[:, 1]

        return_[self.cond_has_first_box, 0] = box_rot[self.cond_has_first_box]
        return_[self.cond_has_first_box, 1] = box_y_offset[self.cond_has_first_box]

        return return_
