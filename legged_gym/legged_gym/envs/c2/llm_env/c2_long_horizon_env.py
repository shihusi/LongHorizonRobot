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
from legged_gym.envs.c2.llm_env.c2_long_horizon_config import (
    CyberLongHorizonCfg,
)
from legged_gym.utils.helpers import class_to_dict

from legged_gym.envs.base.legged_robot_noisy import LeggedRobotNoisy


import math

from varname import nameof

POLICY_NAME = {
    0: "walk",
    1: "climb",
    2: "stand",
    3: "button",
    4: "manipulate",
    5: "move_to_pos",
    6: "sit_down",
    7: "recover",
}
""" 0: "walk",
    1: "climb",
    2: "stand",
    3: "button",
    4: "manipulate",
    5: "move_to_pos",
    6: "sit_down",
    7: "recover",
    """

POLICY_ID = {
    "walk": 0,
    "climb": 1,
    "stand": 2,
    "button": 3,
    "manipulate": 4,
    "move_to_pos": 5,
    "sit_down": 6,
    "recover": 7,
}
""" "walk": 0,
    "climb": 1,
    "stand": 2,
    "button": 3,
    "manipulate": 4,
    "move_to_pos": 5,
    "sit_down": 6,
    "recover": 7, 
    """


class CyberLongHorizonEnv(LeggedRobotNoisy):
    ### Create and Init

    def __init__(
        self,
        cfg: CyberLongHorizonCfg,
        sim_params,
        physics_engine,
        sim_device,
        headless,
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        self.num_train_envs = cfg.env.num_envs
        self.load_RL_policy()
        print("load policy & init ok!")

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength = torch_rand_float(
                self.cfg.domain_rand.leg_motor_strength_range[0],
                self.cfg.domain_rand.leg_motor_strength_range[1],
                (self.num_envs, 12),
                device=self.device,
            )

        self.num_object_actor = self.cfg.task.num_object_actor
        self.num_actors = self.cfg.task.num_actors
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = False  # self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []

        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.npc_handles = (
            []
        )  # surrounding actors or objects or oppoents in each environment.
        self.sensor_handles = []
        self.actor_handles = []
        self.envs = []

        task_props_data = self._create_task_props()
        self.record_friction = []
        self.record_damping = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
                1
            )
            start_pose.p = gymapi.Vec3(*pos)
            self.start_pose = start_pose

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.asset.name, i, 0, 0
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            sensor_handle_dict = self._create_sensors(env_handle, actor_handle)
            npc_handle_dict = self._create_npc(env_handle, i)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.sensor_handles.append(sensor_handle_dict)
            self.npc_handles.append(npc_handle_dict)

            self._create_task_env(i, task_props_data)

        self.gym.set_light_parameters(
            self.sim,
            0,
            gymapi.Vec3(0.5, 0.5, 0.5),
            gymapi.Vec3(1, 1, 1),
            gymapi.Vec3(0, 0, 0),
        )

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def _create_task_props(self):
        """
        Prepare the task related asset and placement information
        """

        print(
            """
        !!!!!!!!!!!!!!!! Notice !!!!!!!!!!!!!!!!!
              _create_task_props Neet to be rewrite
        """
        )

        raise Exception("Task specified error")

    def _create_task_env(self, i):
        """
        Create the task related environment
        """

        print(
            """
        !!!!!!!!!!!!!!!! Notice !!!!!!!!!!!!!!!!!
              No task environment here
        """
        )

        raise Exception("Task specified error")

    def _process_dof_props(self, props, env_id):
        """
        Process the dof props
        """

        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = (
                    m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )
                self.dof_pos_limits[i, 1] = (
                    m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )
        sum_fric = 0.0
        sum_damping = 0.0

        if hasattr(self.cfg.domain_rand, "joint_friction_range"):
            for i in range(12):
                props["friction"][i] = random.uniform(
                    self.cfg.task.joint_friction_range[0],
                    self.cfg.task.joint_friction_range[1],
                )
                sum_fric += props["friction"][i]
        if hasattr(self.cfg.domain_rand, "joint_damping_range"):
            for i in range(12):
                props["damping"][i] = random.uniform(
                    self.cfg.task.joint_damping_range[0],
                    self.cfg.task.joint_damping_range[1],
                )
                sum_damping += props["damping"][i]

        self.record_friction.append(sum_fric / 12)
        self.record_damping.append(sum_damping / 12)
        return props

    def _randomize_dof_props(self, env_ids):
        """Randomize the properites of motor(dof)

        Args:
            env_ids (torch.Tensor)
        """
        if self.cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = (
                torch.rand(
                    len(env_ids),
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                ).unsqueeze(1)
                * (max_strength - min_strength)
                + min_strength
            )
        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = (
                torch.rand(
                    len(env_ids),
                    self.num_dof,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                * (max_offset - min_offset)
                + min_offset
            )
        if self.cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = self.cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = (
                torch.rand(
                    len(env_ids),
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                ).unsqueeze(1)
                * (max_Kp_factor - min_Kp_factor)
                + min_Kp_factor
            )
        if self.cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = self.cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = (
                torch.rand(
                    len(env_ids),
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                ).unsqueeze(1)
                * (max_Kd_factor - min_Kd_factor)
                + min_Kd_factor
            )

    def _get_climb_noise(self):
        """
        Compute the observation noise of climb
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        if self.num_core_obs == 74:
            noise_scale_engaging = torch.zeros(8, dtype=torch.float, device=self.device)
            noise_scale_engaging[0] = 0.05
            noise_scale_engaging[6:8] = 0.05
        elif self.num_core_obs == 69:
            noise_scale_engaging = torch.zeros(3, dtype=torch.float, device=self.device)
            noise_scale_engaging[:] = 0.05

        noise_vec[0:3] = noise_scales.gravity * noise_level
        noise_vec[3:18] = 0.0  # commands
        noise_vec[18:30] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[30:58] = 0.0  # actions, previous actions and clock input
        noise_vec[58:64] = (
            noise_scales.base_pose * noise_level
        )  # * self.obs_scales.base_pose
        if self.num_core_obs == 74:
            noise_vec[64:72] = (
                noise_scale_engaging * noise_level
            )  # * self.obs_scales.engaging_block
            noise_vec[72:74] = 0
        elif self.num_core_obs == 69:
            noise_vec[64:67] = (
                noise_scale_engaging * noise_level
            )  # * self.obs_scales.engaging_block
            noise_vec[67:69] = 0

        return noise_vec

    def _init_buffers(self):
        """
        Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.action_history_num = 6
        self.all_history_num = 30  
        self.num_core_obs = 80 - 5
        self.cfg.env.use_vel_obs = False
        if not self.cfg.env.use_vel_obs:
            self.num_core_obs -= 6
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_core_obs, dtype=torch.float, device=self.device
        )
        self.action_history = torch.zeros(
            self.num_envs,
            self.action_history_num,
            12,
            dtype=torch.float,
            device=self.device,
        )
        self.obs_history = torch.zeros(
            self.num_envs,
            self.all_history_num * self.num_core_obs,
            dtype=torch.float,
            device=self.device,
        )
        self.dof_pos_history = torch.zeros(
            self.num_envs,
            self.action_history_num,
            12,
            dtype=torch.float,
            device=self.device,
        )
        self.clock_inputs = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.desired_contact_states = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.gait_indices = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.test_reset_id = torch.zeros(
            1,
            dtype=torch.int32,
            device=self.device,
            requires_grad=False,
        )
        self.obs_padding = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.measured_heights = 0
        self.proprioception_refreshed = False
        self.forward_depth_refreshed = False

        # update obs_scales components incase there will be one-by-one scaling
        for k in self.all_obs_components:
            if isinstance(getattr(self.obs_scales, k, None), (tuple, list)):
                setattr(
                    self.obs_scales,
                    k,
                    torch.tensor(
                        getattr(self.obs_scales, k, 1.0),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )

        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.all_root_states.view(self.num_envs, -1, 13)[
            :, 0, :
        ]  # (num_envs, 13)
        self.all_dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state = self.all_dof_states.view(self.num_envs, -1, 2)[
            :, : self.num_dof, :
        ]  # (num_envs, 2)
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., : self.num_dof, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., : self.num_dof, 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.step_counter = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device, requires_grad=False
        )
        self.extras = {}
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        command = torch.zeros(
            self.num_envs,
            15,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_target2robot_diff = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_targetpos2robot_diff = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_object2robot_diff = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.substep_torques = torch.zeros(
            self.num_envs,
            self.cfg.control.decimation,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.substep_dof_vel = torch.zeros(
            self.num_envs,
            self.cfg.control.decimation,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.substep_exceed_dof_pos_limits = torch.zeros(
            self.num_envs,
            self.cfg.control.decimation,
            self.num_dof,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.all_rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.robot_rigid_body_states = self.all_rigid_body_states.view(
            self.num_envs, -1, 13
        )[:, :17, :]
        # add sensor dict, which will be filled during create sensor

        self.max_torques = torch.zeros_like(self.torques[..., 0])
        self.torque_exceed_count_substep = torch.zeros_like(
            self.torques[..., 0], dtype=torch.int32
        )  # The number of substeps that the torque exceeds the limit
        self.torque_exceed_count_envstep = torch.zeros_like(
            self.torques[..., 0], dtype=torch.int32
        )  # The number of envsteps that the torque exceeds the limit
        self.out_of_dof_pos_limit_count_substep = torch.zeros_like(
            self.torques[..., 0], dtype=torch.int32
        )  # The number of substeps that the dof pos exceeds the limit

        # self.num_obs = 28

        self.motor_strengths = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.motor_offsets = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.Kp_factors = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.Kd_factors = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.noise_scale_vec = self._get_climb_noise()
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.body_height_cmd,
                self.obs_scales.gait_freq_cmd,
                self.obs_scales.gait_phase_cmd,
                self.obs_scales.gait_phase_cmd,
                self.obs_scales.gait_phase_cmd,
                self.obs_scales.gait_phase_cmd,
                self.obs_scales.footswing_height_cmd,
                self.obs_scales.body_pitch_cmd,
                self.obs_scales.body_roll_cmd,
                self.obs_scales.stance_width_cmd,
                self.obs_scales.stance_length_cmd,
                self.obs_scales.aux_reward_cmd,
            ],
            device=self.device,
            requires_grad=False,
        )[: self.cfg.commands.num_commands]
        self.rew_buf_pos = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.rew_buf_neg = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )

        self.last_dof_pos = torch.zeros_like(self.dof_pos)

        self.num_actuated_dof = self.num_actions
        self.joint_pos_target = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_joint_pos_target = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_last_joint_pos_target = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]
        self.lag_buffer = [
            torch.zeros_like(self.dof_pos)
            for i in range(self.cfg.domain_rand.lag_timesteps + 1)
        ]

        # for play record
        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []

        self._init_utils_buffer()
        self._init_for_RL_policy()

        self._init_walk_buffer()
        self.button_ref_plat_height = 0
        self.task_finish = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.all_task_finished = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def _init_utils_buffer(self):
        '''
        Initialize the utils buffer for the environment
        '''
        ## for test
        self.single_test_work = self.cfg.task.single_test_work
        self.test_task_id = self.cfg.task.test_task_id
        # 0 climb
        # 1 manipulate
        # 2 button
        # 3 manipulate + climb
        # 4 climb_floor2_to_button

        self.succ_sum = 0
        self.reset_sum = 0
        self.finish_time = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reset_once = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.finish_once = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.finish_stairs_count = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.last_finish_stairs_count = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        self.finish_basic = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.dead_buf = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        ## for collecting data
        self.collect_data_work = self.cfg.task.collect_data_work
        self.collect_target = self.cfg.task.collect_target
        self.save_data_file_path = self.cfg.task.save_data_file_path

        self.box_rot = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.box_offset = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )

    def _init_button_buffer(self):
        '''
        Initialize the button buffer for the button task
        '''
        self.num_stand_obs = 42 + 3 + 2
        self.num_sit_down_obs = 42 + 3 + 2
        self.num_button_obs = 42 + 3 + 2 + 6 + 7 - 3
        self.max_episode_length = 1000

        self.obs_history_button = torch.zeros(
            self.num_envs,
            self.RL_policy_obs_dict["button"],
            dtype=torch.float,
            device=self.device,
        )
        self.obs_history_stand = torch.zeros(
            self.num_envs,
            self.RL_policy_obs_dict["stand"],
            dtype=torch.float,
            device=self.device,
        )
        self.obs_history_sit_down = torch.zeros(
            self.num_envs,
            self.RL_policy_obs_dict["sit_down"],
            dtype=torch.float,
            device=self.device,
        )
        self.wall_root_states = self.all_root_states.view(self.num_envs, -1, 13)[
            :, self.num_object_actor + 1, :
        ]
        self.button_root_states = self.all_root_states.view(self.num_envs, -1, 13)[
            :, self.num_object_actor + 2, :
        ]  

        self.last_xy = torch.zeros(
            (self.num_envs, 2), dtype=torch.float, device=self.device
        )
        self.last_heading = torch.zeros(
            (self.num_envs,), dtype=torch.float, device=self.device
        )
        self.init_feet_positions = torch.zeros(
            (self.num_envs, 4, 3), dtype=torch.float, device=self.device
        )
        # cyberdog
        self.nominal_rear_pos = (
            torch.from_numpy(np.array([0.3, -2.0, 1.5, -0.3, -2.0, 1.5]))
            .float()
            .to(self.device)
        )
        self.nominal_rear_pos_sit = (
            torch.from_numpy(np.array([0.0, -2.3, 2.1, 0.0, -2.3, 2.1]))
            .float()
            .to(self.device)
        )

        self.fix_targets = torch.zeros(
            (self.num_envs, 6), dtype=torch.float, device=self.device
        )
        self.fix_targets[:] = (
            torch.Tensor(self.cfg.env.fix_rel_target)
            .to(self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.fix_targets[:, :2] += self.env_origins[:, :2]
        self.fix_targets[:, 3:5] += self.env_origins[:, :2]
        self.target_hand_pos_world = self.fix_targets[:].clone()
        self.projected_forward_vec = quat_rotate_inverse(
            self.base_quat, self.forward_vec
        )

        self.hand_positions = torch.zeros(
            (self.num_envs, 6), dtype=torch.float, device=self.device
        )
        self.world_hand_positions = torch.zeros(
            (self.num_envs, 6), dtype=torch.float, device=self.device
        )
        self.obtain_pos_from_random_actions(
            self.dof_pos, torch.arange(self.num_envs, device=self.device)
        )
        self.world_hand_positions[:, :3] = self._base_to_world(
            self.hand_positions[:, :3]
        )
        self.world_hand_positions[:, 3:] = self._base_to_world(
            self.hand_positions[:, 3:]
        )
        self.target_hand_pos = torch.zeros(
            (self.num_envs, 6), dtype=torch.float, device=self.device
        )
        self.init_target_hand_pos = torch.zeros(
            (self.num_envs, 6), dtype=torch.float, device=self.device
        )
        self._obtain_target_from_world_to_base()
        self.start_pos = self.env_origins + torch.tensor([0.2, 0.0, 0.1]).to(
            self.device
        ).unsqueeze(0)

        self.see_button = torch.zeros(self.num_envs, device=self.device)
        self.observed_base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.continue_touch = torch.zeros(self.num_envs, device=self.device)
        self.rear_foot_shift = torch.zeros(self.num_envs, device=self.device)

        self.motor_target = torch.zeros(
            self.num_envs, self.num_dofs, dtype=torch.float, device=self.device
        )

        self.base_pos_x_noise = torch.zeros(self.num_envs, device=self.device)
        self.button_y_offset = 0.000

        self.contact_button_time = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )

        self.foot_positions = self.robot_rigid_body_states.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]

        self.button_ref_plat_height = 0

    def _init_walk_buffer(self):
        """
        Init the walk buffer
        """
        self.num_walk_obs = 58
        self.walk_commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.body_height_cmd,
                self.obs_scales.gait_freq_cmd,
                self.obs_scales.gait_phase_cmd,
                self.obs_scales.gait_phase_cmd,
                self.obs_scales.gait_phase_cmd,
                self.obs_scales.gait_phase_cmd,
                self.obs_scales.footswing_height_cmd,
                self.obs_scales.body_pitch_cmd,
                self.obs_scales.body_roll_cmd,
                self.obs_scales.stance_width_cmd,
                self.obs_scales.stance_length_cmd,
                self.obs_scales.aux_reward_cmd,
            ],
            device=self.device,
            requires_grad=False,
        )

        self.obs_history_walk = torch.zeros(
            self.num_envs, self.num_walk_obs * 30, dtype=torch.float, device=self.device
        )
        self.diff_robot2target_pos = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )

    def _init_climb_box_buffer(self):
        self.num_climb_obs = 74
        # Setup for object
        self.obs_history_climb = torch.zeros(
            self.num_envs,
            self.RL_policy_obs_dict["climb"],
            dtype=torch.float,
            device=self.device,
        )
        self.object_root_states = self.all_root_states.view(
            self.num_envs, self.num_actors, 13
        )[:, 1 : 1 + self.num_object_actor, :]
        self.object_init_states = torch.zeros_like(
            self.object_root_states, dtype=torch.float, device=self.device
        )
        self.object_pos = torch.zeros(
            (self.num_envs, self.num_object_actor, 3),
            device=self.device,
            dtype=torch.float,
        )
        self.object_rot = torch.zeros(
            (self.num_envs, self.num_object_actor, 4),
            device=self.device,
            dtype=torch.float,
        )

        self.box_info_buf = torch.zeros(
            self.num_envs, 8, dtype=torch.float, device=self.device
        )
        self.box_info_buf[:, 4] = 1
        self.box_info_buf[:, 6] = 0.08

        self.box_size = torch.zeros(
            self.num_envs, 5, dtype=torch.float, device=self.device
        )

        self.n_box_passed = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )

    def _init_climb_box_buffer(self):
        '''
        Initialize the climb box buffer for the climb task
        '''
        self.num_climb_obs = 74
        # Setup for object
        self.obs_history_climb = torch.zeros(
            self.num_envs,
            self.RL_policy_obs_dict["climb"],
            dtype=torch.float,
            device=self.device,
        )
        # print(self.all_root_states.size())
        self.object_root_states = self.all_root_states.view(
            self.num_envs, self.num_actors, 13
        )[:, 1 : 1 + self.num_object_actor, :]
        self.object_init_states = torch.zeros_like(
            self.object_root_states, dtype=torch.float, device=self.device
        )
        self.object_pos = torch.zeros(
            (self.num_envs, self.num_object_actor, 3),
            device=self.device,
            dtype=torch.float,
        )
        self.object_rot = torch.zeros(
            (self.num_envs, self.num_object_actor, 4),
            device=self.device,
            dtype=torch.float,
        )

        self.box_info_buf = torch.zeros(
            self.num_envs, 8, dtype=torch.float, device=self.device
        )
        self.box_info_buf[:, 4] = 1
        self.box_info_buf[:, 6] = 0.08

        self.box_size = torch.zeros(
            self.num_envs, 5, dtype=torch.float, device=self.device
        )

        self.n_box_passed = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )

    def _init_manipulate_buffer(self):
        '''
        Initialize the manipulate buffer for the manipulate task
        '''
        self.num_manipulate_obs = 64
        self.obs_history_manipulate = torch.zeros(
            self.num_envs,
            self.RL_policy_obs_dict["manipulate"],
            dtype=torch.float,
            device=self.device,
        )
        self.command_interface = torch.zeros(
            self.num_envs,
            15,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.high_level_command = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        self.robot_base_state = self.robot_rigid_body_states[:, 0, :]

        self.object_pos = self.object_root_states[:, 0, :3]
        self.object_rot = self.object_root_states[:, 0, 3:7]
        self.target_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.target_rot = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float
        )
        self.heading2object_diff = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.heading2target_diff = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.target2object_diff = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float
        )
        self.target2object_rotation_diff = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float
        )
        self.object2robot_diff = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float
        )
        self.target2robot_diff = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float
        )
        self.target2robot_dis = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.object2robot_dis = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )

    def _init_move2pos_buffer(self):
        '''
        Initialize the move to position buffer for the move to position task
        '''
        self.projected_forward_vec = quat_rotate_inverse(
            self.base_quat, self.forward_vec
        )
        self.num_obs_move2pos = 66 + 2 + 2 - 28 - 2 - 2
        self.move_command_actions = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.target_pos2move = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.obs_history_move2pos = torch.zeros(
            self.num_envs,
            self.num_obs_move2pos * 30,
            dtype=torch.float,
            device=self.device,
        )
        self.command_heading = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.arrive_buf = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.last_command_action = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.robot_pos_noise = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )

    def _init_for_RL_policy(self):
        """Init for decision dict for RL policy"""
        self.RL_policy_obs_dict = {}
        self.RL_policy_obs_dict["walk"] = 58 * 30
        self.RL_policy_obs_dict["climb"] = 74 * 10
        self.RL_policy_obs_dict["stand"] = (42 + 3 + 2) * 3
        self.RL_policy_obs_dict["button"] = (42 + 3 + 2 + 6 + 7 - 3) * 3
        self.RL_policy_obs_dict["manipulate"] = 64 * 30
        self.RL_policy_obs_dict["move_to_pos"] = 40  ## Not real
        self.RL_policy_obs_dict["sit_down"] = (42 + 3 + 2) * 3
        self.RL_policy_name = "manipulate"

        self.RL_policy_name_dict = [
            "walk",
            "climb",
            "stand",
            "button",
            "manipulate",
            "move_to_pos",
            "sit_down",
        ]

        self.task_finish_buf = torch.zeros(
            self.num_envs, 1, dtype=torch.int, device=self.device
        )
        self.task_id = torch.zeros(
            self.num_envs, 1, dtype=torch.int, device=self.device
        )
        self.rl_policy_id = torch.zeros(
            self.num_envs, 1, dtype=torch.int, device=self.device
        )

        self.last_rl_policy_id = torch.ones(
            self.num_envs, 1, dtype=torch.int, device=self.device
        )
        self.last_rl_policy_id *= -1

        self.command_heading = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        torch.set_printoptions(profile="full")


    ### Policy

    def load_RL_policy(self):
        """Load RL policy to Walk, Manipulate, Climb, Stand up, Button, Sit down."""
        print("begin to load policy")
        # self._init_for_RL_policy()

        ## Load Walk

        policy_zoo_dir = self.cfg.task.policy_path_dict["policy_zoo"]

        self.walk_policy_body = torch.jit.load(
            policy_zoo_dir + self.cfg.task.policy_path_dict["walk_body"]
        )
        self.walk_policy_adapt = torch.jit.load(
            policy_zoo_dir + self.cfg.task.policy_path_dict["walk_adapt"]
        )
        self.walk_policy_body.to(self.device)
        self.walk_policy_adapt.to(self.device)


        ## Load Manipulate
        manipulate_policy_logdir = (
            policy_zoo_dir + self.cfg.task.policy_path_dict["manipulate"]
        )
        self.manipulate_policy = torch.jit.load(manipulate_policy_logdir).to(
            self.device
        )

        ## Load Climb
        climb_policy_logdir = policy_zoo_dir + self.cfg.task.policy_path_dict["climb"]
        self.climb_policy = torch.jit.load(climb_policy_logdir).to(self.device)

        ## Load Stand and sit
        stand_policy_logdir = policy_zoo_dir + self.cfg.task.policy_path_dict["stand"]
        self.stand_policy = torch.jit.load(stand_policy_logdir).to(self.device)
        sit_down_ligdir = policy_zoo_dir + self.cfg.task.policy_path_dict["sit_down"]
        self.sit_down_policy = torch.jit.load(sit_down_ligdir).to(self.device)

        ## Load Button
        button_policy_logdir = policy_zoo_dir + self.cfg.task.policy_path_dict["button"]
        self.button_policy = torch.jit.load(button_policy_logdir).to(self.device)

    def RL_multi_controller(self, rl_policy_id, RL_policy_Args=None):
        policy_change_buf = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )
        policy_change_buf[self.last_rl_policy_id[:, 0] != self.rl_policy_id[:, 0]] = 1
        policy_change_env_ids = policy_change_buf.nonzero(as_tuple=False).flatten()

        # Clear obs history (for those with policy change)
        self._reset_buffers(policy_change_env_ids)
        self.task_finish_buf[policy_change_env_ids] = False
        for (
            RL_policy_name
        ) in self.RL_policy_name_dict:  # compute observations for all policies
            if RL_policy_name != "move_to_pos":
                getattr(self, "obs_history_" + RL_policy_name)[
                    policy_change_env_ids, :
                ] = 0  # clear all history buffers!

            if (RL_policy_name == "move_to_pos") or (RL_policy_name == "manipulate"):
                self.obs_history_walk[policy_change_env_ids, :] = 0

            if (
                RL_policy_name != "walk"
            ):  # compute observations for all policies except walk
                getattr(self, "compute_observations_" + RL_policy_name)(RL_policy_Args)

        return_actions = torch.zeros(
            self.num_envs, 12, dtype=torch.float, device=self.device
        )

        if "walk" in self.RL_policy_name_dict:
            self.compute_observations_walk(
                self.command_interface, self.rl_policy_id[:, 0] == 0
            )
            latent_val = self.walk_policy_adapt.forward(self.obs_history_walk)
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["walk"]
            ] = self.walk_policy_body.forward(
                torch.cat((self.obs_history_walk, latent_val), dim=-1)
            )[
                self.rl_policy_id[:, 0] == POLICY_ID["walk"]
            ]

        if "climb" in self.RL_policy_name_dict:
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["climb"]
            ] = self.climb_policy(self.obs_history_climb_full)[
                self.rl_policy_id[:, 0] == POLICY_ID["climb"]
            ]

        if "stand" in self.RL_policy_name_dict:
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["stand"]
            ] = self.stand_policy(self.obs_history_stand)[
                self.rl_policy_id[:, 0] == POLICY_ID["stand"]
            ]

        if "button" in self.RL_policy_name_dict:
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["button"]
            ] = self.button_policy(self.obs_history_button)[
                self.rl_policy_id[:, 0] == POLICY_ID["button"]
            ]

        if "manipulate" in self.RL_policy_name_dict:
            if (self.episode_length_buf[0] % 5 == 0) or (
                self.last_rl_policy_id != rl_policy_id
            ):
                self.high_level_command[
                    self.rl_policy_id[:, 0] == POLICY_ID["manipulate"]
                ] = self.manipulate_policy(self.obs_history_manipulate)[
                    self.rl_policy_id[:, 0] == POLICY_ID["manipulate"]
                ]
            self.command_interface[
                self.rl_policy_id[:, 0] == POLICY_ID["manipulate"]
            ] = self.clip_walk_command(self.high_level_command)[
                self.rl_policy_id[:, 0] == POLICY_ID["manipulate"]
            ]
            self.compute_observations_walk(
                self.command_interface,
                self.rl_policy_id[:, 0] == POLICY_ID["manipulate"],
            )
            latent_val = self.walk_policy_adapt.forward(self.obs_history_walk)
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["manipulate"]
            ] = self.walk_policy_body.forward(
                torch.cat((self.obs_history_walk, latent_val), dim=-1)
            )[
                self.rl_policy_id[:, 0] == POLICY_ID["manipulate"]
            ]

        if "move_to_pos" in self.RL_policy_name_dict:
            RL_policy_Args[:, :2] += self.env_origins[:, :2]
            self.move_to_pos(RL_policy_Args[:, :3])
            self.compute_observations_walk(
                self.command_interface,
                self.rl_policy_id[:, 0] == POLICY_ID["move_to_pos"],
            )
            latent_val = self.walk_policy_adapt.forward(self.obs_history_walk)
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["move_to_pos"]
            ] = self.walk_policy_body.forward(
                torch.cat((self.obs_history_walk, latent_val), dim=-1)
            )[
                self.rl_policy_id[:, 0] == POLICY_ID["move_to_pos"]
            ]

        if "sit_down" in self.RL_policy_name_dict:
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["sit_down"]
            ] = self.sit_down_policy(self.obs_history_sit_down)[
                self.rl_policy_id[:, 0] == POLICY_ID["sit_down"]
            ]

        self.last_rl_policy_id = self.rl_policy_id

        return return_actions

    def generate_all_llm_plan(self):
        '''
        Generate all LLM plans for the environment
        '''
        self.llm_plan = torch.zeros(
            self.num_envs, 20, 4, dtype=torch.float32, device=self.device
        )
        self.cur_step = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device
        )
        self.real_target_pos = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        for i in range(self.num_envs):
            self._call_llm_generate_plan(i)

        if self.cfg.task.multi_robotool:
            print(
                self.llm_plan[:, 1:4].size(),
                self.env_origins.unsqueeze(1).expand(-1, 20, -1).size(),
            )
            self.llm_plan[:, :, 1:4] += self.env_origins.unsqueeze(1).expand(-1, 20, -1)

    def _call_llm_generate_plan(self, i):
        self.llm_plan_generator = globals().get(self.cfg.task.test_llm_code)
        self.llm_plan_generator(self, i)

    def hand_write_multi_planner(self):
        '''
        You can write your plan here for debug
        '''
        print("!!!!!!!!!!!!!!!! Notice Need to be rewrite")

        raise Exception("Task_Specify_Error")

    def execute_llm_plan(self):
        '''
        Execute the LLM plan for the environment
        '''
        ## Judge whether the box was pushed to the target_pos

        mask0 = self.cur_step[:, 0] > -1
        mask = self.cur_step[:, 0] == -1

        self.cur_step[mask, 0] += 1
        self.rl_policy_id[:, 0] = self.llm_plan[
            torch.arange(self.num_envs, device=self.device), self.cur_step[:, 0], 0
        ]

        # check is finished
        self.task_finish[:] = False

        policy_args = (
            self.llm_plan[
                torch.arange(self.num_envs, device=self.device), self.cur_step[:, 0], 1:
            ]
            .squeeze(1)
            .to(self.device)
        )
        # task_finished = False

        ## climb
        if "climb" in self.RL_policy_name_dict:
            mask = torch.logical_and(
                self.rl_policy_id[:, 0] == POLICY_ID["climb"], mask0
            )
            self.task_finish[mask] = torch.logical_and(
                torch.abs(self.base_pos[:, 2] - policy_args[:, 2]) < 0.2,
                self.base_pos[:, 0] - (policy_args[:, 0] - 0.00) > 0.0
                # self.base_pos[:, 0] - self.env_origins[:, 0] > 2.6
            )[mask]

        ## stand
        if "stand" in self.RL_policy_name_dict:
            mask = torch.logical_and(
                self.rl_policy_id[:, 0] == POLICY_ID["stand"], mask0
            )
            self.task_finish[mask] = torch.logical_and(
                self.base_pos[:, 2] > self.button_ref_plat_height + 0.3,
                torch.abs(self.target_hand_pos_world[:, 0] - self.root_states[:, 0])
                < 0.60,
            )[mask]

        ## button
        if "button" in self.RL_policy_name_dict:
            mask = torch.logical_and(
                self.rl_policy_id[:, 0] == POLICY_ID["button"], mask0
            )
            if torch.sum(mask) > 0:
                pass
                # print("button", self.target_hand_pos_world[1, :3] - self.observe_cam_coordinate[1, :3], self.target_hand_pos_world[1, :3] - self.world_hand_positions[1, :3])
            diff_button2hand = (
                self.target_hand_pos_world[:, :3] - self.world_hand_positions[:, :3]
            )
            dis_button2hand = torch.norm(diff_button2hand[:, :3], dim=1)

            touched = torch.logical_and(
                (torch.abs(diff_button2hand[:, 1]) < self.cfg.task.finish_button[1]),
                (torch.abs(diff_button2hand[:, 2]) < self.cfg.task.finish_button[2]),
            )
            touched = torch.logical_and(
                touched,
                (torch.abs(diff_button2hand[:, 0]) < self.cfg.task.finish_button[0]),
            )
            self.contact_button_time[touched] += 1
            self.contact_button_time[self.contact_button_time > 0] += 1
            self.task_finish[mask] = (
                self.contact_button_time[:, 0] > self.cfg.task.contact_button_time
            )[mask]

        ## manipulate
        if "manipulate" in self.RL_policy_name_dict:
            mask = torch.logical_and(
                self.rl_policy_id[:, 0] == POLICY_ID["manipulate"], mask0
            )
            condition_achieve = torch.logical_and(
                (
                    torch.abs(self.target2object_diff[:, 0])
                    < self.cfg.task.finish_manipulate[0]
                ),  # real is 0.08 [0.28 - 0.2]
                (
                    torch.abs(
                        self.target2object_diff[:, 1]
                        + self.cfg.task.finish_manipulate[2]
                    )
                    < self.cfg.task.finish_manipulate[1]
                ),
            )

            condition_out_time = (
                (torch.abs(self.target2object_diff[:, 0]) < 0.3)
                & (torch.abs(self.target2object_diff[:, 1] + 0.2) < 0.3)
                & (
                    (self.episode_length_buf[:] > 2500)
                    | (
                        torch.norm(
                            (self.base_pos - self.object_root_states[:, 0, :3]), dim=-1
                        )
                        > 3
                    )
                )
            )
            self.task_finish[mask] = torch.logical_or(
                condition_achieve, condition_out_time
            )[mask]
        ## move2pos
        if "move_to_pos" in self.RL_policy_name_dict:
            mask = torch.logical_and(
                self.rl_policy_id[:, 0] == POLICY_ID["move_to_pos"], mask0
            )
            self.task_finish[mask] = torch.logical_and(
                torch.logical_and(
                    (
                        torch.abs(self.diff_robot2target_pos[:, 0])
                        < self.cfg.task.finish_move_to_pos[0]
                    ),
                    (
                        torch.abs(self.diff_robot2target_pos[:, 1])
                        < self.cfg.task.finish_move_to_pos[1]
                    ),
                ),
                torch.logical_and(
                    (
                        torch.abs(self.command_heading)
                        < self.cfg.task.finish_move_to_pos[2]
                    ),
                    (self.rl_policy_id[:, 0] == POLICY_ID["move_to_pos"]),
                ),
            )[mask]

        ## sit down
        if "sit_down" in self.RL_policy_name_dict:
            mask_sit_down = self.base_pos[:, 2] < 0.25 + self.button_ref_plat_height
            self.task_finish[
                mask_sit_down & (self.rl_policy_id[:, 0] == POLICY_ID["sit_down"])
            ] = True
            self.contact_button_time[mask_sit_down] = 0

        policy_change = self.task_finish == True
        self.policy_change = policy_change  
        self.cur_step[policy_change] += 1
        self.rl_policy_id[:, 0] = self.llm_plan[
            torch.arange(self.num_envs, device=self.device), self.cur_step[:, 0], 0
        ]
        self.task_finish[:] = False

        if "button" in self.RL_policy_name_dict:
            mask = torch.logical_and(
                policy_change, self.rl_policy_id[:, 0] == POLICY_ID["button"]
            )
            button_start_pos = self.env_origins[
                :, :3
            ].clone()  # + torch.tensor([0.2, 0.0, 0.1]).to(self.device).unsqueeze(0)
            button_start_pos[:, 2] = self.button_ref_plat_height + 0.05
            button_start_pos[:, 0] += self.cfg.task.button_start_pos

            dynamic_start_pos = False
            if dynamic_start_pos:
                button_start_pos = self.root_states[
                    :, :3
                ].clone()  # + torch.tensor([0.2, 0.0, 0.1]).to(self.device).unsqueeze(0)
                button_start_pos[:, 2] = self.button_ref_plat_height + 0.1
                button_start_pos[:, 0] += 0.2 + 0.02

            self.observe_cam_coordinate[mask[:] == True] = button_start_pos[
                mask[:] == True
            ]

        if self.num_envs == 1:
            print(
                f"\n>> Task ID: {int(self.cur_step)}, {POLICY_NAME[int(self.rl_policy_id)]}, {mask0}\npolicy_args_r: {policy_args - self.env_origins}"
            )
            if self.rl_policy_id[0] == POLICY_ID["move_to_pos"]:
                print("Pos error ", self.diff_robot2target_pos, self.command_heading)
                print("heanding", self.command_heading)
            if self.rl_policy_id[0] == POLICY_ID["manipulate"]:
                print(
                    f"Taget to object diff: {self.target2object_diff}, {condition_achieve}"
                )
            if self.rl_policy_id[0] == POLICY_ID["button"]:
                print(f"Diff button to hand: {diff_button2hand}")
            if self.rl_policy_id[0] == POLICY_ID["sit_down"]:
                print(
                    f"height/target: {self.base_pos[:, 2]}{self.button_ref_plat_height}+{self.cfg.task.finish_sit_sown[0]}"
                )
            if self.rl_policy_id[0] == POLICY_ID["stand"]:
                print(
                    "stand",
                    self.base_pos[:, 2] > self.button_ref_plat_height + 0.3,
                    torch.abs(self.target_hand_pos_world[:, 0] - self.root_states[:, 0])
                    < 0.60,
                )

    def llm_RL_multi_controller(self):
        '''
        This function is used to control the robot with multiple RL policies based on the LLM plan.
        '''
        policy_change_buf = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )
        policy_change_buf[self.last_rl_policy_id[:, 0] != self.rl_policy_id[:, 0]] = 1
        policy_change_env_ids = policy_change_buf.nonzero(as_tuple=False).flatten()

        # Clear obs history (for those with policy change)
        self._reset_buffers(policy_change_env_ids)
        self.task_finish_buf[policy_change_env_ids] = False
        for (
            RL_policy_name
        ) in self.RL_policy_name_dict:  # compute observations for all policies
            if RL_policy_name != "move_to_pos":
                getattr(self, "obs_history_" + RL_policy_name)[
                    policy_change_env_ids, :
                ] = 0  # clear all history buffers!

            if (RL_policy_name == "move_to_pos") or (RL_policy_name == "manipulate"):
                self.obs_history_walk[policy_change_env_ids, :] = 0

        return_actions = torch.zeros(
            self.num_envs, 12, dtype=torch.float, device=self.device
        )

        # 0: walk
        # 1: climb
        # 2: stand
        # 3: button
        # 4: manipulate
        # 5: move_to_pos
        # 6: sit_down

        if "walk" in self.RL_policy_name_dict:
            self.compute_observations_walk(
                self.command_interface, self.rl_policy_id[:, 0] == POLICY_ID["walk"]
            )
            latent_val = self.walk_policy_adapt.forward(self.obs_history_walk)
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["walk"]
            ] = self.walk_policy_body.forward(
                torch.cat((self.obs_history_walk, latent_val), dim=-1)
            )[
                self.rl_policy_id[:, 0] == POLICY_ID["walk"]
            ]

        if "climb" in self.RL_policy_name_dict:
            self.compute_observations_climb()
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["climb"]
            ] = self.climb_policy(self.obs_history_climb_full)[
                self.rl_policy_id[:, 0] == POLICY_ID["climb"]
            ]

        if "stand" in self.RL_policy_name_dict:
            self.compute_observations_stand()
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["stand"]
            ] = self.stand_policy(self.obs_history_stand)[
                self.rl_policy_id[:, 0] == POLICY_ID["stand"]
            ]

        if "button" in self.RL_policy_name_dict:
            self.compute_observations_button(
                self.llm_plan[
                    torch.arange(self.num_envs, device=self.device),
                    self.cur_step[:, 0],
                    1:,
                ]
            )
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["button"]
            ] = self.button_policy(self.obs_history_button)[
                self.rl_policy_id[:, 0] == POLICY_ID["button"]
            ]

        if "manipulate" in self.RL_policy_name_dict:
            self.compute_observations_manipulate(
                self.llm_plan[
                    torch.arange(self.num_envs, device=self.device),
                    self.cur_step[:, 0],
                    1:,
                ]
            )
            self.high_level_command[
                torch.logical_and(
                    (self.rl_policy_id[:, 0] == POLICY_ID["manipulate"]),
                    torch.logical_or(
                        (self.episode_length_buf[:] % 5 == 0),
                        (self.last_rl_policy_id[:, 0] != self.rl_policy_id[:, 0]),
                    ),
                )
            ] = self.manipulate_policy(self.obs_history_manipulate)[
                (self.rl_policy_id[:, 0] == POLICY_ID["manipulate"])
                & (
                    (self.episode_length_buf[:] % 5 == 0)
                    | (self.last_rl_policy_id[:, 0] != self.rl_policy_id[:, 0])
                )
            ]
            self.command_interface[
                self.rl_policy_id[:, 0] == POLICY_ID["manipulate"]
            ] = self.clip_walk_command(self.high_level_command)[
                self.rl_policy_id[:, 0] == POLICY_ID["manipulate"]
            ]
            self.compute_observations_walk(
                self.command_interface,
                self.rl_policy_id[:, 0] == POLICY_ID["manipulate"],
            )
            latent_val = self.walk_policy_adapt.forward(self.obs_history_walk)
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["manipulate"]
            ] = self.walk_policy_body.forward(
                torch.cat((self.obs_history_walk, latent_val), dim=-1)
            )[
                self.rl_policy_id[:, 0] == POLICY_ID["manipulate"]
            ]

        if "move_to_pos" in self.RL_policy_name_dict:
            self.move_to_pos(
                self.llm_plan[
                    torch.arange(self.num_envs, device=self.device),
                    self.cur_step[:, 0],
                    1:,
                ]
            )
            self.compute_observations_walk(
                self.command_interface,
                self.rl_policy_id[:, 0] == POLICY_ID["move_to_pos"],
            )
            latent_val = self.walk_policy_adapt.forward(self.obs_history_walk)
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["move_to_pos"]
            ] = self.walk_policy_body.forward(
                torch.cat((self.obs_history_walk, latent_val), dim=-1)
            )[
                self.rl_policy_id[:, 0] == POLICY_ID["move_to_pos"]
            ]

        if "sit_down" in self.RL_policy_name_dict:
            self.compute_observations_sit_down()
            return_actions[
                self.rl_policy_id[:, 0] == POLICY_ID["sit_down"]
            ] = self.sit_down_policy(self.obs_history_sit_down)[
                self.rl_policy_id[:, 0] == POLICY_ID["sit_down"]
            ]

        self.last_rl_policy_id = self.rl_policy_id

        return return_actions

    def move_to_pos(self, control_args, mode="direct"):
        '''
        Move the robot to a specific position based on the control arguments.
        '''
        target_pos = control_args
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        self.diff_robot2target_pos = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.diff_robot2target_pos = (
            target_pos[:, :2] - self.root_states[:, :2]
        ) 
        dis_robot2target_pos = torch.norm(self.diff_robot2target_pos, dim=1)
        desire_robot2target_heading = torch.atan2(
            self.diff_robot2target_pos[:, 1], self.diff_robot2target_pos[:, 0]
        )
        heading2target_diff = desire_robot2target_heading - heading

        self.base_targetpos2robot_diff[:, 0] = dis_robot2target_pos * torch.cos(
            heading2target_diff
        )
        self.base_targetpos2robot_diff[:, 1] = dis_robot2target_pos * torch.sin(
            heading2target_diff
        )

        command_action = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        command_action[:, :2] = self.base_targetpos2robot_diff * 4

        desire_command_heading = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        command_heading = desire_command_heading - heading
        command_heading[torch.abs(command_heading) < 0.05] = 0
        command_action[:, 2] = command_heading * 4
        self.command_heading = command_heading

        # rl control
        if mode == "rl":
            self.compute_observations_rl_move_to_pos(control_args[:, :3])
            rl_command_action = self.move_to_pos_policy(self.obs_history_move2pos)
            clip_rl_command_action = self.clip_walk_command(rl_command_action)
            self.move_command_actions = clip_rl_command_action[:, :3]

            self.command_interface[self.rl_policy_id[:, 0] == 5] = clip_rl_command_action[
                self.rl_policy_id[:, 0] == 5
            ]

        self.command_interface[self.rl_policy_id[:, 0] == 5] = self.clip_walk_command(
            command_action
        )[self.rl_policy_id[:, 0] == 5]

    def check_task_finish(self):
        '''
        Check if the task is finished based on the current state of the environment.
        Only used for hand-written plan test
        '''
        if "manipulate" in self.RL_policy_name_dict:
            self.task_finish_buf[
                (
                    torch.abs(self.target2object_diff[:, 0])
                    < self.cfg.task.finish_manipulate[0]
                )
                & (
                    torch.abs(self.target2object_diff[:, 1])
                    < self.cfg.task.finish_manipulate[1]
                )
                & (self.rl_policy_id[:, 0] == POLICY_ID["manipulate"])
            ] = True

        if "move_to_pos" in self.RL_policy_name_dict:
            self.task_finish_buf[
                (
                    torch.abs(self.diff_robot2target_pos[:, 0])
                    < self.cfg.task.finish_move_to_pos[0]
                )
                & (
                    torch.abs(self.diff_robot2target_pos[:, 1])
                    < self.cfg.task.finish_move_to_pos[1]
                )
                & (
                    torch.abs(self.command_heading)
                    < self.cfg.task.finish_move_to_pos[2]
                )
                & (self.rl_policy_id[:, 0] == POLICY_ID["move_to_pos"])
            ] = True

        if "stand" in self.RL_policy_name_dict:
            mask_close = (
                torch.abs(self.target_hand_pos_world[:, 0] - self.base_pos[:, 0])
                < self.cfg.task.finish_stand[1]
            ) & (self.rl_policy_id[:, 0] == POLICY_ID["stand"])
            mask_stand = self.base_pos[0, 2] > (
                self.button_ref_plat_height + self.cfg.task.finish_stand[0]
            )
            mask_on_the_stand_policy = self.rl_policy_id[:, 0] == POLICY_ID["stand"]
            mask = torch.logical_and(
                torch.logical_and(mask_close, mask_stand), mask_on_the_stand_policy
            )

            self.task_finish_buf[mask[:] == True] = True

            button_start_pos = self.env_origins[
                :, :3
            ].clone()  # + torch.tensor([0.2, 0.0, 0.1]).to(self.device).unsqueeze(0)
            button_start_pos[:, 2] = self.button_ref_plat_height + 0.05
            button_start_pos[:, 0] += self.cfg.task.button_start_pos

            self.observe_cam_coordinate[mask[:] == True] = button_start_pos[
                mask[:] == True
            ]

        if "climb" in self.RL_policy_name_dict:
            self.task_finish_buf[
                (
                    self.base_pos[:, 0] - self.env_origins[:, 0]
                    > self.cfg.task.finish_climb[0]
                )
                & (self.rl_policy_id[:, 0] == POLICY_ID["climb"])
            ] = True

        if "button" in self.RL_policy_name_dict:
            diff_button2hand = (
                self.target_hand_pos_world[:, :3] - self.world_hand_positions[:, :3]
            )
            touched = torch.logical_and(
                (torch.abs(diff_button2hand[:, 1]) < self.cfg.task.finish_button[1]),
                (torch.abs(diff_button2hand[:, 2]) < self.cfg.task.finish_button[2]),
            )
            touched_plus = torch.logical_and(
                touched, diff_button2hand[:, 0] < self.cfg.task.finish_button[0]
            )
            self.contact_button_time[touched_plus] += 1

            self.task_finish_buf[
                (self.contact_button_time[:, 0] > 0)
                & (self.rl_policy_id[:, 0] == POLICY_ID["button"])
            ] = True

        if "sit_down" in self.RL_policy_name_dict:
            mask_sit_down = (
                self.base_pos[:, 2]
                < self.button_ref_plat_height + self.cfg.task.finish_sit_sown[0]
            )
            self.task_finish_buf[
                (mask_sit_down[:] == True)
                & (self.rl_policy_id[:, 0] == POLICY_ID["sit_down"])
            ] = True

        if self.num_envs == 1:
            print(
                f">> Task ID: {int(self.task_id)}, {POLICY_NAME[int(self.rl_policy_id)]}"
            )
            if self.rl_policy_id[0] == POLICY_ID["move_to_pos"]:
                print("Pos error ", self.diff_robot2target_pos, self.command_heading)
                print("heanding", self.command_heading)
            if self.rl_policy_id[0] == POLICY_ID["manipulate"]:
                print(f"Taget to object diff: {self.target2object_diff}")
            if self.rl_policy_id[0] == POLICY_ID["button"]:
                print(f"Diff button to hand: {diff_button2hand}")
            if self.rl_policy_id[0] == POLICY_ID["sit_down"]:
                print(
                    f"height/target: {self.base_pos[:, 2]}{self.button_ref_plat_height}+{self.cfg.task.finish_sit_sown[0]}"
                )
            if self.rl_policy_id[0] == POLICY_ID["stand"]:
                print(
                    "stand gg",
                    self.target_hand_pos_world[:, 0],
                    torch.abs(self.target_hand_pos_world[:, 0] - self.base_pos[:, 0]),
                    self.base_pos[0, 2],
                    self.button_ref_plat_height + self.cfg.task.finish_stand[0],
                )

    ### Step

    def step(self, high_actions=None, use_high_level_RL=False):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        # handwrite for test
        if self.single_test_work or self.collect_data_work:
            policy_id, control_args = self.hand_write_multi_planner()
            actions_ = self.RL_multi_controller(policy_id, control_args)
            self.check_task_finish()
        else:
            # llm control:
            self.execute_llm_plan()
            actions_ = self.llm_RL_multi_controller()

        self.pre_physics_step(actions_)
        # step physics and render each frame
        self.render()
        for dec_i in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.post_decimation_step(dec_i)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def _compute_torques(self, actions):
        """The input actions will not be used, instead the scaled clipped actions will be used.
        Please check the computation logic whenever you change anything.
        """

        self.pre_physics_step(actions)
        self.actions_scaled_torque_clipped = (
            self.actions * self.cfg.control.action_scale
        )
        actions_scaled_torque_clipped = self.actions_scaled_torque_clipped

        control_type = "P"
        control_type_tensor = torch.zeros(
            self.num_envs, 1, dtype=torch.int, device=self.device
        )
        control_type_tensor[self.rl_policy_id == 0] = 1

        self.actions_scaled_torque_clipped[self.rl_policy_id[:, 0] == 1] = (
            self.actions[self.rl_policy_id[:, 0] == 1] * 0.5
        )

        actions_scaled_torque_clipped = self.actions_scaled_torque_clipped

        torques = (
            self.p_gains
            * self.Kp_factors
            * (
                actions_scaled_torque_clipped
                + self.default_dof_pos
                - self.dof_pos
                + self.motor_offsets
            )
            - self.d_gains * self.Kd_factors * self.dof_vel
        )
        # print(actions_scaled_torque_clipped, self.Kp_factors, self.Kd_factors, self.motor_strengths, self.motor_offsets)
        # else:
        #     raise NotImplementedError
        if self.cfg.control.motor_clip_torque:
            torques = torch.clip(
                torques,
                -self.torque_limits * self.cfg.control.motor_clip_torque,
                self.torque_limits * self.cfg.control.motor_clip_torque,
            )
        # print("torques", torques)
        # torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.projected_forward_vec[:] = quat_rotate_inverse(
            self.base_quat, self.forward_vec
        )
        self.foot_positions = self.robot_rigid_body_states.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_last_actions[:] = self.last_actions[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]

        for i in range(0, self.action_history_num - 1):
            self.action_history[:, i] = self.action_history[:, i + 1]
            self.dof_pos_history[:, i] = self.dof_pos_history[:, i + 1]

        self.action_history[:, self.action_history_num - 1] = self.last_actions
        self.dof_pos_history[:, self.action_history_num - 1] = self.last_dof_pos

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        if "button" in self.RL_policy_name_dict:
            self.last_xy[:] = self.root_states[:, :2]
            self.last_heading[:] = self._get_cur_heading()
            self.init_feet_positions[
                self.episode_length_buf == 1
            ] = self.foot_positions[self.episode_length_buf == 1]

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._step_contact_targets()
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        with torch.no_grad():
            if "max_pos_x" not in self.extras["episode"]:
                self.extras["episode"]["max_pos_x"] = 0.0
                self.extras["episode"]["min_pos_x"] = 0.0
                self.extras["episode"]["max_pos_y"] = 0.0
                self.extras["episode"]["min_pos_y"] = 0.0
            pos_x = self.root_states[:, 0] - self.env_origins[:, 0]
            pos_y = self.root_states[:, 1] - self.env_origins[:, 1]
            self.extras["episode"]["max_pos_x"] = max(
                self.extras["episode"]["max_pos_x"], torch.max(pos_x).cpu()
            )
            self.extras["episode"]["min_pos_x"] = min(
                self.extras["episode"]["min_pos_x"], torch.min(pos_x).cpu()
            )
            self.extras["episode"]["max_pos_y"] = max(
                self.extras["episode"]["max_pos_y"], torch.max(pos_y).cpu()
            )
            self.extras["episode"]["min_pos_y"] = min(
                self.extras["episode"]["min_pos_y"], torch.min(pos_y).cpu()
            )
            if self.check_BarrierTrack_terrain():
                self.extras["episode"]["n_obstacle_passed"] = torch.mean(
                    torch.clip(
                        torch.div(
                            pos_x, self.terrain.env_block_length, rounding_mode="floor"
                        )
                        - 1,
                        min=0.0,
                    )
                ).cpu()

        if hasattr(self, "proprioception_buffer"):
            resampling_time = getattr(
                self.cfg.sensor.proprioception, "latency_resampling_time", self.dt
            )
            resample_env_ids = (
                (self.episode_length_buf % int(resampling_time / self.dt) == 0)
                .nonzero(as_tuple=False)
                .flatten()
            )
            if len(resample_env_ids) > 0:
                self._resample_proprioception_latency(resample_env_ids)

        if hasattr(self, "forward_depth_buffer"):
            resampling_time = getattr(
                self.cfg.sensor.forward_camera, "latency_resampling_time", self.dt
            )
            resample_env_ids = (
                (self.episode_length_buf % int(resampling_time / self.dt) == 0)
                .nonzero(as_tuple=False)
                .flatten()
            )
            if len(resample_env_ids) > 0:
                self._resample_forward_camera_latency(resample_env_ids)

        self.torque_exceed_count_envstep[
            (torch.abs(self.substep_torques) > self.torque_limits).any(dim=1).any(dim=1)
        ] += 1

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)

    ###  Reset   ##

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 100.0,
            dim=1,
        )
        self.dead_buf[self.reset_buf == True] = 1

        run_too_far = torch.norm((self.base_pos - self.env_origins), dim=-1) > 4.0
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= run_too_far

        if self.collect_data_work:
            if torch.sum(self.reset_buf > 0):
                self._collect_data()
        else:
            self.reset_buf[:] = False  # ljh TODO: change back for high level RL

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.gait_indices[env_ids] = 0
        self.reset_button(env_ids)

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

        if 0 in env_ids:
            if not self.single_test_work:
                self.generate_all_llm_plan()

    def reset_button(self, env_ids):
        self.last_xy[env_ids] = torch.clone(self.root_states[env_ids, :2])
        heading = self._get_cur_heading()
        self.last_heading[env_ids] = heading[env_ids]
        self.continue_touch[env_ids] = 0
        self.base_pos_x_noise[env_ids] *= 0

    def _reset_buffers(self, env_ids):
        """
        Reset torch buffer

        """
        # super()._reset_buffers(env_ids)

        self.last_actions[env_ids] *= 0.0
        self.last_dof_vel[env_ids] *= 0.0
        self.feet_air_time[env_ids] *= 0.0
        # self.episode_length_buf[env_ids] *= 0
        self.reset_buf[env_ids] = 1
        self.step_counter[env_ids] *= 0

        self.obs_history[env_ids] *= 0
        self.last_last_actions[env_ids] *= 0
        self.last_dof_pos[env_ids] *= 0
        self.last_dof_vel[env_ids] *= 0
        self.last_root_vel[env_ids] *= 0
        self.last_last_joint_pos_target[env_ids] *= 0
        self.last_joint_pos_target[env_ids] *= 0

    def _reset_root_states(self, env_ids):
        print("!!!!!!!!!!!!!!!!!!! Notic reset_root_states need to be rewrite")

        raise Exception("Rewrite Error")

    ### Observation

    def compute_observations_climb(self, control_args=None):
        # climb control args
        self.commands[:, :15] = torch.tensor([0.6, 0.0, 0.0, 0.0, 3.0, 0.5, 0.0, 0.0, 0.5, 0.25, 0.0, 0.0, 0.28, 0.33, 0.0]).to(self.device).unsqueeze(0).repeat(self.num_envs, 1)
        for key in self.sensor_handles[0].keys():
            print("key--", key)
            if "camera" in key:
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                break

        obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                # self.commands[:, :3] * self.commands_scale,
                self.commands[:, :3] * self.commands_scale[:3],
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                # self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                # self.last_actions,
                # self.clock_inputs,
            ),
            dim=-1,
        )
        if not self.cfg.env.use_vel_obs and self.commands.size()[1] == 15:
            # self.obs_history = torch.cat((self.obs_history[:, self.num_core_obs:], self.obs_buf[:, 6:]), dim=-1)
            obs_buf = torch.cat(
                (  # self.base_lin_vel * self.obs_scales.lin_vel,
                    # self.base_ang_vel  * self.obs_scales.ang_vel,
                    self.projected_gravity,
                    # self.commands[:, :3] * self.commands_scale,
                    self.commands * self.commands_scale,
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                    # self.dof_vel * self.obs_scales.dof_vel,
                    self.actions,
                    self.last_actions,
                    self.clock_inputs,
                    self._get_base_pose_obs(),  # 6
                    self.get_box_observation(),  # 8
                    self._get_box_offset_obs(),  # 2
                ),
                dim=-1,
            )
            self.obs_history_climb = torch.cat(
                (self.obs_history_climb[:, self.num_climb_obs :], obs_buf[:, :]), dim=-1
            )
            self.obs_history_climb_full = torch.cat(
                (self.obs_history_climb, self._get_robot_config_obs()), dim=-1
            )
        else:
            self.obs_history = torch.cat(
                (self.obs_history[:, self.num_core_obs :], obs_buf), dim=-1
            )

        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            obs_buf = torch.cat((obs_buf, heights), dim=-1)

        if not self.num_privileged_obs is None:
            min_shape = min(obs_buf.shape[1], self.privileged_obs_buf.shape[1])
            self.privileged_obs_buf[:, :min_shape] = obs_buf[
                :, :min_shape
            ]  # copy content

        self.obs_super_impl = obs_buf

        # critic obs
        if not self.num_privileged_obs is None:
            self.privileged_obs_buf[:] = self._get_obs_from_components(
                self.cfg.env.privileged_obs_components,
                privileged=getattr(
                    self.cfg.env, "privileged_obs_gets_privilege", False
                ),
            )
        # fixing linear velocity in proprioception observation
        if "proprioception" in getattr(
            self.cfg.env, "privileged_obs_components", []
        ) and getattr(self.cfg.env, "privileged_use_lin_vel", False):
            self.privileged_obs_buf[:, :3] = self.base_lin_vel * self.obs_scales.lin_vel

        for key in self.sensor_handles[0].keys():
            if "camera" in key:
                self.gym.end_access_image_tensors(self.sim)
                break

        return obs_buf

    def compute_observations_walk(self, control_args, compute_buf):
        """Compute observations for the walking policy"""
        self.commands[:, 0] = 0.0
        self.commands[:, 1] = 0.6
        self.commands[:, 2] = 0.0
        self.commands[:, 3] = -0.12
        self.commands[:, 4] = 3.0
        self.commands[:, 5:8] = 0.0
        self.commands[:, 5] = 0.5
        self.commands[:, 8] = 0.5
        self.commands[:, 9] = 0.1
        self.commands[:, 10] = 0.0
        self.commands[:, 11] = 0.0
        self.commands[:, 12] = 0.32
        self.commands[:, 13] = 0.33
        self.commands[:, 14] = 0.0

        self.commands[:] = control_args[:]

        low_obs_buf = torch.cat(
            (
                self.projected_gravity,  # 3
                self.commands * self.walk_commands_scale,  # 15
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
                self.actions,  # 12
                self.last_actions,  # 12
                self.clock_inputs,
            ),
            dim=-1,
        )

        self.obs_history_walk[compute_buf] = torch.cat(
            (self.obs_history_walk[:, self.num_walk_obs :], low_obs_buf[:, :]), dim=-1
        )[compute_buf]


    def compute_observations_stand(self, control_args=None):
        '''Compute observations for the standing policy'''
        obs_commands = self.commands[:, :3]
        obs_commands[:, 0] = 0.2
        obs_commands[:, 1] = 0
        obs_commands[:, 2] = 0
        common_obs_buf = torch.cat(
            (  # self.base_lin_vel * self.obs_scales.lin_vel,
                # self.base_ang_vel  * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.projected_forward_vec,
                obs_commands * 2,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * 0,
                self.actions,
                self.clock_inputs[:, -2:],
            ),
            dim=-1,
        )

        self.obs_history_stand = torch.cat(
            (self.obs_history_stand[:, self.num_stand_obs :], common_obs_buf[:, :]),
            dim=-1,
        )

        return common_obs_buf

    def compute_observations_button(self, control_args=None):
        '''Compute observations for the button policy'''
        obs_commands = self.commands[:, :3]
        obs_commands[:, 0] = 0.0
        obs_commands[:, 1] = 0
        obs_commands[:, 2] = 0
        common_obs_buf = torch.cat(
            (  # self.base_lin_vel * self.obs_scales.lin_vel,
                # self.base_ang_vel  * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.projected_forward_vec,
                obs_commands * self.commands_scale[:3],
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.clock_inputs[:, -2:],
            ),
            dim=-1,
        )

        self._obtain_target_from_world_to_base()
        offset = self.observe_cam_coordinate
        control_args = control_args.squeeze(1)
        obs_1 = self.target_hand_pos_world[:, :3] - offset
        obs_1[:, 1] -= self.button_y_offset
        # obs_1[:, 2] -= 0.1
        if self.num_envs == 1 and self.rl_policy_id[0] == POLICY_ID["button"]:
            print(f"obs: {obs_1}\noffset: {offset}\nbase pos:{self.base_pos - offset}")

        common_obs_buf = torch.cat([common_obs_buf, obs_1], dim=-1)
        common_obs_buf = torch.cat(
            [common_obs_buf, self.base_pos - offset, self.base_quat], dim=-1
        )
        self.world_hand_positions = self._obtain_hand_pos_in_world_frame()
        if self.num_envs == 1:
            print(
                self.target_hand_pos_world[:, :3] - self.button_root_states[:, :3],
                self.target_hand_pos_world,
                obs_1,
                self.world_hand_positions,
            )

        self.obs_history_button = torch.cat(
            (self.obs_history_button[:, self.num_button_obs :], common_obs_buf[:, :]),
            dim=-1,
        )

    def compute_observations_manipulate(self, control_args=None):
        if (control_args != None) and (not self.single_test_work):
            control_args = control_args.squeeze(1)
            self.target_pos[:, 0] = (
                control_args[:, 0]
            )  
            self.target_pos[:, 1] = (
                control_args[:, 1]
            )  
        else:   # for test
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

    def compute_observations_sit_down(self, control_args=None):
        '''Compute observations for the sit down policy'''
        obs_commands = self.commands[:, :3]
        common_obs_buf = torch.cat(
            (
                self.projected_gravity,
                self.projected_forward_vec,
                obs_commands * self.commands_scale[:3] * 0.0,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.clock_inputs[:, -2:],
            ),
            dim=-1,
        )
        cur_obs_buf = common_obs_buf
        self.obs_history_sit_down = torch.cat(
            (self.obs_history_sit_down[:, self.num_sit_down_obs :], cur_obs_buf[:, :]),
            dim=-1,
        )

    def compute_observations_rl_move_to_pos(self, control_args):
        '''Compute observations for the RL move to position policy'''
        target_pos = control_args
        target_pos[:, :2] -= self.env_origins[:, :2]

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        dis_robot2target_pos = torch.norm(self.diff_robot2target_pos, dim=1)
        desire_robot2target_heading = torch.atan2(
            self.diff_robot2target_pos[:, 1], self.diff_robot2target_pos[:, 0]
        )
        heading2target_diff = desire_robot2target_heading - heading

        self.base_targetpos2robot_diff[:, 0] = dis_robot2target_pos * torch.cos(
            heading2target_diff
        )
        self.base_targetpos2robot_diff[:, 1] = dis_robot2target_pos * torch.sin(
            heading2target_diff
        )

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        ex_command_heading = control_args[:, 2] - heading

        common_obs_buf = torch.cat(
            (
                self.projected_gravity,  # 3
                self.command_interface * self.commands_scale[:15],
                target_pos,  # 3
                dis_robot2target_pos.unsqueeze(-1),  # 1
                self._get_base_pose_obs(),  # 6
                ex_command_heading.unsqueeze(-1),  # 1
                self._get_box_obstacle_observation(),
                self.move_command_actions,
            ),
            dim=-1,
        )

        self.obs_history_move2pos = torch.cat(
            (
                self.obs_history_move2pos[:, self.num_obs_move2pos :],
                common_obs_buf[:, :],
            ),
            dim=-1,
        )

        return self.obs_history_move2pos

    def compute_observations_move_to_pos(self, control_args=None):
        return self.obs_buf
        # pass

    def compute_observations(self):
        pass

    def _get_clock_input_obs(self, privileged=False):
        # print(self.clock_inputs)
        return self.clock_inputs

    def _get_action_history_obs(self, privileged=False):
        # return self.action_history.flatten(start_dim=1)s
        return self.last_actions

    def _get_dof_pos_history_obs(self, privileged=False):
        return self.dof_pos_history.flatten(start_dim=1)

    def _get_all_command_obs_obs(self, privileged=False):
        # print("get all command obs", self.obs_super_impl[:, :72])
        if self.cfg.env.use_vel_obs:
            return self.obs_super_impl[:, 6:76]
        else:
            return self.obs_super_impl[:, :76]

    def _get_all_history_obs(self, privileged=False):
        # print("get all history", self.obs_history)
        return self.obs_history

    def _write_action_history_noise(self, noise_vec):
        pass

    def _write_clock_input_noise(self, noise_vec):
        pass

    def _write_dof_pos_history_noise(self, noise_vec):
        pass

    def _write_all_command_obs_noise(self, noise_vec):
        pass

    def _write_all_history_noise(self, noise_vec):
        pass

    def _get_obs_from_components(self, components: list, privileged=False):
        obs_segments = self.get_obs_segment_from_components(components)
        obs = []
        # print("++++++++observation_segments")
        for k, v in obs_segments.items():
            # print(k)
            if k == "proprioception":
                obs.append(self._get_proprioception_obs(privileged))
            elif k == "all_command_obs":
                obs.append(self._get_all_command_obs_obs(privileged))
            elif k == "height_measurements":
                obs.append(self._get_height_measurements_obs(privileged))
            else:
                # get the observation from specific component name
                # such as "_get_forward_depth_obs"
                obs.append(
                    getattr(self, "_get_" + k + "_obs")(privileged)
                    * getattr(self.obs_scales, k, 1.0)
                )
        obs = torch.cat(obs, dim=1)

        return obs


    def get_box_observation(self):
        '''Get the box observation for the climbing task'''
        self.n_box_passed[:] = 0
        # self.n_box_passed[self.cond_has_first_box == 0] = 1
        if self.exp_scene_id == 2:
            self.n_box_passed[:] = 1
        self.n_box_passed[
            self.root_states[:, 0] > (self.object_root_states[:, 0, 0] - 0.2)
        ] = 1
        if self.num_object_actor > 1:
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

        for i in range(self.num_envs):
            # print(self.n_box_passed[i], self.object_size)
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

            if self.n_box_passed[i] < self.num_object_actor:
                self.box_info_buf[i, 0] = (
                    self.object_root_states[i, self.n_box_passed[i], 0]
                    - self.root_states[i, 0]
                    - 0.4
                )
            else:
                self.box_info_buf[i, 0] = 1

        return self.box_info_buf

    def _get_box_offset_obs(self, privileged=False):
        ''' Get the box offset observation for the climbing task'''
        return_ = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)

        box_forward = quat_apply(self.object_root_states[:, 0, 3:7], self.forward_vec)
        box_rot = torch.atan2(box_forward[:, 1], box_forward[:, 0])
        box_y_offset = self.object_root_states[:, 0, 1] - self.env_origins[:, 1]

        return_[self.cond_has_first_box, 0] = box_rot[self.cond_has_first_box]
        return_[self.cond_has_first_box, 1] = box_y_offset[self.cond_has_first_box]

        return return_

    def obtain_pos_from_random_actions(
        self, actions, env_ids
    ):  # new added! 从leggedrobot的step复制来前一半
        dofs = actions.view(-1, 4, 3)
        x1 = 0.164
        y1 = 0.042
        z1 = 0
        x2 = 0
        y2 = 0.094
        z2 = 0
        x3 = 0
        y3 = 0
        z3 = -0.12
        x4 = -0.0055
        y4 = 0
        z4 = -0.17391
        thigh_to_flfoot = torch.stack(
            [
                torch.cos(dofs[:, 0, 2]) * x4 + torch.sin(dofs[:, 0, 2]) * z4 + x3,
                (y4 + y3) * torch.ones_like(dofs[:, 0, 2]),
                -torch.sin(dofs[:, 0, 2]) * x4 + torch.cos(dofs[:, 0, 2]) * z4 + z3,
            ],
            dim=-1,
        )
        hip_to_flfoot = torch.stack(
            [
                -torch.cos(dofs[:, 0, 1]) * thigh_to_flfoot[:, 0]
                - torch.sin(dofs[:, 0, 1]) * thigh_to_flfoot[:, 2]
                + x2,
                -thigh_to_flfoot[:, 1] + y2,
                -torch.sin(dofs[:, 0, 1]) * thigh_to_flfoot[:, 0]
                + torch.cos(dofs[:, 0, 1]) * thigh_to_flfoot[:, 2]
                + z2,
            ],
            dim=-1,
        )
        trunk_to_flfoot = torch.stack(
            [
                hip_to_flfoot[:, 0] + x1,
                torch.cos(dofs[:, 0, 0]) * hip_to_flfoot[:, 1]
                - torch.sin(dofs[:, 0, 0]) * hip_to_flfoot[:, 2]
                + y1,
                torch.sin(dofs[:, 0, 0]) * hip_to_flfoot[:, 1]
                + torch.cos(dofs[:, 0, 0]) * hip_to_flfoot[:, 2]
                + z1,
            ],
            dim=-1,
        )
        # fr
        x1 = 0.164
        y1 = -0.042
        z1 = 0
        x2 = 0
        y2 = -0.094
        z2 = 0
        x3 = 0
        y3 = 0
        z3 = -0.12
        x4 = -0.0055
        y4 = 0
        z4 = -0.17391
        thigh_to_frfoot = torch.stack(
            [
                torch.cos(dofs[:, 1, 2]) * x4 + torch.sin(dofs[:, 1, 2]) * z4 + x3,
                (y4 + y3) * torch.ones_like(dofs[:, 1, 2]),
                -torch.sin(dofs[:, 1, 2]) * x4 + torch.cos(dofs[:, 1, 2]) * z4 + z3,
            ],
            dim=-1,
        )
        hip_to_frfoot = torch.stack(
            [
                -torch.cos(dofs[:, 1, 1]) * thigh_to_frfoot[:, 0]
                - torch.sin(dofs[:, 1, 1]) * thigh_to_frfoot[:, 2]
                + x2,
                -thigh_to_frfoot[:, 1] + y2,
                -torch.sin(dofs[:, 1, 1]) * thigh_to_frfoot[:, 0]
                + torch.cos(dofs[:, 1, 1]) * thigh_to_frfoot[:, 2]
                + z2,
            ],
            dim=-1,
        )
        trunk_to_frfoot = torch.stack(
            [
                hip_to_frfoot[:, 0] + x1,
                torch.cos(dofs[:, 1, 0]) * hip_to_frfoot[:, 1]
                - torch.sin(dofs[:, 1, 0]) * hip_to_frfoot[:, 2]
                + y1,
                torch.sin(dofs[:, 1, 0]) * hip_to_frfoot[:, 1]
                + torch.cos(dofs[:, 1, 0]) * hip_to_frfoot[:, 2]
                + z1,
            ],
            dim=-1,
        )

        """
        # TODO: afterwards, delete self.hand_positions
        """
        self.hand_positions[env_ids] = torch.cat(
            [trunk_to_flfoot, trunk_to_frfoot], dim=-1
        )
        return self.hand_positions

    def _obtain_target_from_world_to_base(self):
        """
        To get current target cmd in world coordinate
        """
        base_target = self.target_hand_pos_world.clone()
        base_target[:, :3] = self._world_to_base(base_target[:, :3])
        base_target[:, 3:] *= 0
        base_target[
            self.episode_length_buf - 1 < self.cfg.rewards.before_handtrack_steps
        ] = 0.0
        self.target_hand_pos = base_target

        just_start = self.episode_length_buf == 1
        self.init_target_hand_pos = self.init_target_hand_pos * (
            1 - just_start.float().unsqueeze(1)
        ) + self.target_hand_pos.clone() * just_start.float().unsqueeze(1)

    def _get_base_pose_obs(self, privileged=False):
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        roll[roll > np.pi] -= np.pi * 2  # to range (-pi, pi)
        pitch[pitch > np.pi] -= np.pi * 2  # to range (-pi, pi)
        yaw[yaw > np.pi] -= np.pi * 2  # to range (-pi, pi)
        return torch.cat(
            [
                self.root_states[:, :3] - self.env_origins,
                torch.stack([roll, pitch, yaw], dim=-1),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device),
                torch.stack([roll, pitch, yaw], dim=-1),
            ],
            dim=-1,
        )
        return torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device)

    def _get_robot_config_obs(self, privileged=False):
        # print("robot config obs", self.robot_config_buffer)
        return self.robot_config_buffer

    def _get_sidewall_distance_obs(self, privileged=False):
        if not self.check_BarrierTrack_terrain():
            return torch.zeros((self.num_envs, 2), device=self.sim_device)
        base_positions = self.root_states[:, 0:3]  # (n_envs, 3)

        return_ = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        for i in range(self.num_envs):
            if self.n_box_passed[i] > 0 and self.n_box_passed[i] < 5:
                return_[i, 0] = (
                    self.object_size[i, self.n_box_passed[i]]
                    - self.object_size[i, self.n_box_passed[i] - 1]
                )
            elif self.n_box_passed[i] == 0:
                return_[i, 0] = self.object_size[i, self.n_box_passed[i]]
            elif self.n_box_passed[i] == 5:
                return_[i, 0] = -1 * self.object_size[i, self.n_box_passed[i]]

        return return_

    def compute_base_diff(self):
        self.target2robot_dis = torch.norm(self.target2robot_diff, dim=1)
        self.object2robot_dis = torch.norm(self.object2robot_diff, dim=1)
        self.target2object_dis = torch.norm(self.target2object_diff, dim=1)

        self.base_target2robot_diff[:, 0] = self.target2robot_dis * torch.cos(
            self.heading2target_diff
        )
        self.base_target2robot_diff[:, 1] = self.target2robot_dis * torch.sin(
            self.heading2target_diff
        )
        self.base_object2robot_diff[:, 0] = self.object2robot_dis * torch.cos(
            self.heading2object_diff
        )
        self.base_object2robot_diff[:, 1] = self.object2robot_dis * torch.sin(
            self.heading2object_diff
        )

    def compute_target_info(self, object_idx=0):
        self.object_pos = self.object_root_states[:, object_idx, :3]
        self.object_rot = self.object_root_states[:, object_idx, 3:7]

        self.target2object_diff = self.target_pos[:, :2] - self.object_pos[:, :2]
        self.object2robot_diff = self.object_pos[:, :2] - self.robot_base_state[:, :2]
        self.target2robot_diff = self.target_pos[:, :2] - self.robot_base_state[:, :2]
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        desire_obj2robot_heading = torch.atan2(
            self.object2robot_diff[:, 1], self.object2robot_diff[:, 0]
        )  # atan2(y, x) arctan(y/x) for x>0
        desire_target2robot_heading = torch.atan2(
            self.target2robot_diff[:, 1], self.target2robot_diff[:, 0]
        )
        self.desire_target2obj = torch.atan2(
            self.target2object_diff[:, 1], self.target2object_diff[:, 0]
        )
        self.heading2object_diff = desire_obj2robot_heading - heading
        self.heading2target_diff = desire_target2robot_heading - heading

        self.compute_base_diff()

    ### Utils

    def _collect_data(self):
        '''Collect data for the smooth transition between tasks'''

        valuable_env_ids = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        valuable_data_list = []
        valuable_data = []
        if self.collect_target == "box_init_state_mani2climb":
            self.reset_buf[self.cur_step[:, 0] > 0] = True
            condition_close = self.target2object_dis < 0.3
            condition_reset = self.reset_buf == True
            valuable_env_ids[condition_close & condition_reset] = True

            valuable_tensor = self.object_root_states[valuable_env_ids == True, 0, :]

        valuable_data = valuable_tensor.cpu().numpy().tolist()
        print("collect_data ", torch.sum(valuable_env_ids))
        with open(self.save_data_file_path, "a", encoding="utf-8") as file_object:
            file_object.writelines(valuable_data)

    def compute_diff_physics_arg(self, env_ids_suc):
        ''' Compute the difference in physics arguments for successful and failed environments'''
        # print(env_ids_suc)
        fail_buf = torch.ones(self.num_envs, dtype=torch.int, device=self.device)
        if len(env_ids_suc):
            fail_buf[env_ids_suc] = 0
        env_ids_fail = fail_buf.nonzero(as_tuple=False).flatten()

        average_dof_fric = 0.0
        average_dof_damp = 0.0

        dof_friction = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        dof_damping = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # print(self.record_friction)
        for i in range(self.num_envs):
            dof_friction[i] = self.record_friction[i]
            dof_damping[i] = self.record_damping[i]
            # average_dof_fric = (average_dof_fric + self.record_friction[i]) / 2
            # average_dof_damp = (average_dof_damp + self.record_damping[i]) / 2

        average_suc_mass = torch.mean(self.object_info[env_ids_suc, 0])
        average_suc_vol = torch.mean(self.object_info[env_ids_suc, 1])
        average_suc_den = torch.mean(self.object_density[env_ids_suc])
        average_suc_box_init_x = torch.mean(self.object_init_states[env_ids_suc, 0])
        average_suc_box_init_y = torch.mean(self.object_init_states[env_ids_suc, 1])
        # average_suc_box_init_slope =  torch.mean(self.slope[env_ids_suc])

        average_fail_mass = torch.mean(self.object_info[env_ids_fail, 0])
        average_fail_vol = torch.mean(self.object_info[env_ids_fail, 1])
        average_fail_den = torch.mean(self.object_density[env_ids_fail])
        average_fail_box_init_x = torch.mean(self.object_init_states[env_ids_fail, 0])
        average_fail_box_init_y = torch.mean(self.object_init_states[env_ids_fail, 1])
        # average_fail_box_init_slope =  torch.mean(self.slope[env_ids_suc])

        # print("Mass succ/fail ", average_suc_mass, average_fail_mass)
        # print("Vol succ/fail ", average_suc_vol, average_fail_vol)
        # print("Den succ/fail ", average_suc_den, average_fail_den)
        # print("init x succ/fail ", average_suc_box_init_x, average_fail_box_init_x)
        # print("init y succ/fail ", average_suc_box_init_y, average_fail_box_init_y)
        # print("fric, ", dof_friction[env_ids_suc].mean(), dof_friction[env_ids_fail].mean())
        # print("fric, ", dof_damping[env_ids_suc].mean(), dof_damping[env_ids_fail].mean())
        # print("slope succ/fail ", average_suc_box_init_slope, average_fail_box_init_slope)

    def clip_walk_command(self, actions):
        ''' Clip the walking command actions to ensure they are within valid ranges.'''
        command_interface = torch.zeros(
            self.num_envs, 15, dtype=torch.float, device=self.device
        )
        # all scale
        actions = actions * 0.5  # !! 0.25

        num_high_level_command = 3

        actions[:, :3] = torch.clip(actions[:, :3], -1, 1)
        actions[:, 1:3] = torch.clip(actions[:, 1:3], -0.6, 0.6)

        command_interface[:, :num_high_level_command] = actions

        if num_high_level_command > 3:
            # scale 1
            actions[:, 3] = actions[:, 3] * 0.5
            actions[:, 9:14] = actions[:, 9:14] * 0.5
            # scale 2
            actions[:, 13] = actions[:, 13] * 0.5
            actions[:, 3] = torch.clip(actions[:, 3], -0.15, 0.15)
            actions[:, 4] += 3.0
            actions[:, 4] = torch.clip(actions[:, 4], 2.0, 4.0)

            ## gait choose the max
            mask = (actions[:, 5:8] == actions[:, 5:8].max(dim=1, keepdim=True)[0]).to(
                dtype=torch.int32
            )
            actions[:, 5:8] = mask * 0.5
            actions[:, 8] = 0.5
            actions[:, 9] += 0.14
            actions[:, 9] = torch.clip(actions[:, 9], 0.03, 0.25)
            actions[:, 10] = torch.clip(actions[:, 10], -0.2, 0.2)
            actions[:, 11] = 0
            actions[:, 12] += 0.25
            actions[:, 13] += 0.32
            actions[:, 12] = torch.clip(actions[:, 12], 0.15, 0.35)
            actions[:, 13] = torch.clip(actions[:, 13], 0.28, 0.36)
            actions[:, 14] = 0

            command_interface = actions

        else:
            command_interface[:, 3] = -0.12
            command_interface[:, 4] = 3.0
            command_interface[:, 5:8] = 0.0
            command_interface[:, 5] = 0.5
            command_interface[:, 8] = 0.5
            command_interface[:, 9] = 0.1
            command_interface[:, 10] = 0.0
            command_interface[:, 11] = 0.0
            command_interface[:, 12] = 0.32
            command_interface[:, 13] = 0.33
            command_interface[:, 14] = 0.0

        return command_interface

    def apply_target_dropout(self):
        cond0 = self.episode_length_buf > self.cfg.rewards.before_handtrack_steps + 1
        # random_drop = torch.zeros(self.num_envs, dtype=torch.float, device = self.device)
        random_drop = (
            torch.rand(self.num_envs, dtype=torch.float, device=self.device) < 0.98
        )
        cam_view_drop = torch.zeros(self.num_envs, device=self.device)
        lr = (
            torch.atan(
                torch.abs(
                    (self.target_hand_pos[:, 1] - 25e-3)
                    / (-self.target_hand_pos[:, 2] + 114.912e-3)
                )
            )
            / np.pi
            * 180
        )
        ud = (
            torch.atan(
                torch.abs(
                    (self.target_hand_pos[:, 0] - 271.994e-3)
                    / (-self.target_hand_pos[:, 2] + 114.912e-3)
                )
            )
            / np.pi
            * 180
        )
        cam_view_drop = torch.logical_or(lr > (85 / 2.0), ud > (50 / 2.0))
        self.see_button = torch.logical_not(
            torch.logical_and(cond0, torch.logical_or(random_drop, cam_view_drop))
        )
        self.observed_base_pos = self.base_pos * self.see_button.unsqueeze(
            1
        ).float() + self.observed_base_pos * (1 - self.see_button.unsqueeze(1).float())

    def _base_to_world(self, v):
        """
        Transform coordinate from base frame to world frame
        shape of v: [num_envs, 3]
        return: [num_envs, 3]
        """
        return quat_apply(self.base_quat, v) + self.base_pos

    def _world_to_base(self, v):
        """
        Transform coordinate from world frame to base frame
        shape of v: [num_envs, 3]
        return: [num_envs, 3]
        """
        return quat_apply(quat_conjugate(self.base_quat), v - self.base_pos)

    def _obtain_hand_pos_in_world_frame(self):
        """
        Get current hand position in world frame
        """
        hand_positions = self.foot_positions[:, :2, :]
        self.world_hand_positions = torch.reshape(hand_positions, (self.num_envs, 6))
        return self.world_hand_positions

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        # if cfg.domain_rand.push_robots:
        env_ids = (self.episode_length_buf % 10 == 0).nonzero(as_tuple=False).flatten()

        max_vel = 0.2
        self.root_states[env_ids, 7:9] = torch_rand_float(
            -max_vel, max_vel, (len(env_ids), 2), device=self.device
        )  # lin vel x/y
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
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.all_root_states),
            gymtorch.unwrap_tensor(actor_ids_int32),
            len(actor_ids_int32),
        )

    def _apply_external_base_force(self):
        """Apply external forces to the base of the robot."""
        mask = self.projected_gravity[:, 2] > -0.5
        force = random.uniform(0, 1) * (torch.abs(self.base_lin_vel[:, 2])).unsqueeze(
            dim=-1
        )
        direction = -self.base_lin_vel[:, :2] / torch.clamp(
            torch.norm(self.base_lin_vel[:, :2], dim=-1), min=1e-5
        ).unsqueeze(dim=-1)
        external_forces = torch.zeros(
            (self.num_envs, self.num_bodies + self.num_object_actor + 2, 3),
            dtype=torch.float32,
            device=self.device,
        )
        external_forces[mask, 0, :2] = (force * direction)[mask]
        torque = 0.0 * self.contact_forces[:, 0, 2].unsqueeze(dim=-1)
        direction = -self.base_ang_vel / torch.clamp(
            torch.norm(self.base_ang_vel, dim=-1), min=1e-5
        ).unsqueeze(dim=-1)
        external_torques = torch.zeros(
            (self.num_envs, self.num_bodies + self.num_object_actor + 2, 3),
            dtype=torch.float32,
            device=self.device,
        )
        external_torques[:, 0] = torque * direction
        direction = quat_apply(
            self.base_quat,
            to_torch([0.0, -1.0, 0.0], device=self.device).repeat((self.num_envs, 1)),
        )
        pitch_vel = quat_rotate_inverse(self.base_quat, self.base_ang_vel)[:, 1]
        pitch_vel[pitch_vel > 0] = 0.0
        torque = torch.clamp(
            0
            * -pitch_vel.unsqueeze(dim=1)
            * torch_rand_float(0.8, 1.2, (self.num_envs, 1), device=self.device),
            min=-50,
            max=50,
        )
        external_torques[mask, 0] = (torque * direction)[mask]
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(external_forces),
            gymtorch.unwrap_tensor(external_torques),
            gymapi.ENV_SPACE,
        )

    def _get_cur_heading(self):
        heading_vec = quat_apply_yaw(self.base_quat, self.forward_vec)
        heading = torch.atan2(heading_vec[:, 1], heading_vec[:, 0])
        return heading

    ### Reward

    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies = self.commands[:, 4]
            phases = self.commands[:, 5]
            offsets = self.commands[:, 6]
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]
            self.gait_indices = torch.remainder(
                self.gait_indices + self.dt * frequencies, 1.0
            )

            if self.cfg.commands.pacing_offset:
                foot_indices = [
                    self.gait_indices + phases + offsets + bounds,
                    self.gait_indices + bounds,
                    self.gait_indices + offsets,
                    self.gait_indices + phases,
                ]
            else:  # activate
                foot_indices = [
                    self.gait_indices + phases + offsets + bounds,
                    self.gait_indices + offsets,
                    self.gait_indices + bounds,
                    self.gait_indices + phases,
                ]

            self.foot_indices = torch.remainder(
                torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0
            )

            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (
                    0.5 / durations[stance_idxs]
                )
                idxs[swing_idxs] = 0.5 + (
                    torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]
                ) * (0.5 / (1 - durations[swing_idxs]))

            # if self.cfg.commands.durations_warp_clock_inputs:

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(
                0, kappa
            ).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = smoothing_cdf_start(
                torch.remainder(foot_indices[0], 1.0)
            ) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)
            ) + smoothing_cdf_start(
                torch.remainder(foot_indices[0], 1.0) - 1
            ) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)
            )
            smoothing_multiplier_FR = smoothing_cdf_start(
                torch.remainder(foot_indices[1], 1.0)
            ) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)
            ) + smoothing_cdf_start(
                torch.remainder(foot_indices[1], 1.0) - 1
            ) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)
            )
            smoothing_multiplier_RL = smoothing_cdf_start(
                torch.remainder(foot_indices[2], 1.0)
            ) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)
            ) + smoothing_cdf_start(
                torch.remainder(foot_indices[2], 1.0) - 1
            ) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)
            )
            smoothing_multiplier_RR = smoothing_cdf_start(
                torch.remainder(foot_indices[3], 1.0)
            ) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)
            ) + smoothing_cdf_start(
                torch.remainder(foot_indices[3], 1.0) - 1
            ) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)
            )

            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]

    def _get_heights_at_points(self, points):
        """Get vertical projected terrain heights at points
        points: a tensor of size (num_envs, num_points, 2) in world frame
        """
        points = points.clone()
        num_points = points.shape[1]
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs, num_points, device=self.device, requires_grad=False
            )
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_box_obstacle_observation(self, obstacle_idx=0):
        if hasattr(self, "obstacle_idx"):
            obstacle_idx = self.obstacle_idx

        self.object_pos = self.object_root_states[:, obstacle_idx, :3]
        self.object_rot = self.object_root_states[:, obstacle_idx, 3:7]

        # robot_pos_noise = self.root_states[:, :2].clone() + torch_rand_float(-0.1, 0.1, (self.num_envs, 2), device=self.device)
        self.object2robot_diff = self.object_pos[:, :2]
        object2robot_diff_noise = (
            self.object_pos[:, :2] - self.root_states[:, :2].clone()
        )

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        desire_obj2robot_heading = torch.atan2(
            self.object2robot_diff[:, 1], self.object2robot_diff[:, 0]
        )  # atan2(y, x) arctan(y/x) for x>0
        desire_target2robot_heading = torch.atan2(
            self.target2robot_diff[:, 1], self.target2robot_diff[:, 0]
        )
        self.desire_target2obj = torch.atan2(
            self.target2object_diff[:, 1], self.target2object_diff[:, 0]
        )
        self.heading2object_diff = desire_obj2robot_heading - heading
        self.heading2target_diff = desire_target2robot_heading - heading

        self.object2robot_dis = torch.norm(self.object2robot_diff, dim=1)
        object2robot_dis_noise = torch.norm(object2robot_diff_noise, dim=1)

        self.base_object2robot_diff[:, 0] = self.object2robot_dis * torch.cos(
            self.heading2object_diff
        )
        self.base_object2robot_diff[:, 1] = self.object2robot_dis * torch.sin(
            self.heading2object_diff
        )

        return_tensor = torch.zeros(
            self.num_envs, 8, dtype=torch.float, device=self.device
        )
        return_tensor[:, :2] = self.object_pos[:, :2] - self.env_origins[:, :2]
        return_tensor[:, 2] = 1.0  # self.object_size[:, 0]
        return_tensor[:, 3] = self.object2robot_dis
        return_tensor[:, 3] = object2robot_dis_noise

        ## give diff
        return_tensor[:, 4:6] = self.object2robot_diff
        return_tensor[:, 6:8] = self.base_object2robot_diff
        return_tensor[:, 4:6] = object2robot_diff_noise

        # print("obstacle info", return_tensor[:, :6])
        return return_tensor[:, :6]
