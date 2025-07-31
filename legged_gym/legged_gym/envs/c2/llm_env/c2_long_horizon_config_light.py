from legged_gym.envs.c2.llm_env.c2_long_horizon_config import (
    CyberLongHorizonCfg,
    CyberLongHorizonCfgPPO,
)
from isaacgym import gymtorch, gymapi
import numpy as np


class CyberLongHorizonLightCfg(CyberLongHorizonCfg):
    class task(CyberLongHorizonCfg.task):
        ## mode
        def __init__(self):
            super().__init__()
            self.exp_scene_id = 1
            self.use_down_stair_box = False
            self.single_test_work = False
            self.test_task_id = 2
            # 0 climb
            # 1 manipulate
            # 2 button
            # 3 manipulate + climb
            # 4 climb_floor2_to_button

            self.multi_robotool = False

            self.test_llm_branch = False  # benchmark the robotool neet to be set True
            # self.test_llm_code = "generate_multi_robotool"
            self.test_llm_code = "generate_llm_plan_main1"
            self.test_llm_code_branch_1 = "generate_robotool_high_3"
            self.test_llm_code_branch_2 = "generate_robotool_low_3"

            if self.multi_robotool:
                self.test_llm_code = "generate_multi_robotool_3"

            self.num_object_actor = 3
            self.num_actors = 1 + self.num_object_actor + 2
            self.demo_flag = False

            ## box arg
            self.box_w = 0.8
            self.box_h = 1.0
            self.box_d = 0.2  
            self.box_height_range = [0.2, 0.2]
            self.box_density_range = [10, 12]  # [7, 12]
            self.box_linear_damping_range = [1.0, 1.0]  # [0, 0.8]
            self.box_friction_range = [0.8, 0.8]  # [0.1, 2]

            self.stair_branch = "low_stair"  # "high_stair"    "both"
            if self.stair_branch == "high_stair":
                self.stair_rand = [1, 1]
            elif self.stair_branch == "low_stair":
                self.stair_rand = [-1, -1]
            elif self.stair_branch == "both":
                self.stair_rand = [-1, 1]

            # task finish condition
            self.finish_manipulate = [0.26, 0.2, 0.2]  # x, y
            self.finish_move_to_pos = [0.06, 0.04, 0.1]  # x, y, heading
            self.finish_stand = [0.35, 0.6]  # height, close
            self.finish_button = [0.03, 0.06, 0.06]
            self.button_start_pos = 2.4 + 0.08 + 0.18
            self.finish_sit_sown = [0.2]  # height
            self.finish_climb = [2.6]  # x
            self.contact_button_time = 1


    class render_config:
        camera_offset = gymapi.Vec3(-0.5, -0.5, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(45)
        )

        fix_camera = False
        camera_attach_actor_id = 0

        if fix_camera:
            camera_attach_actor_id = 2
            camera_offset = gymapi.Vec3(-2.4, -2.0, 0.5)
            camera_rotation = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(-0.0, 0.0, 1), np.deg2rad(70)
            )

            camera_offset = gymapi.Vec3(0.8, -1.4, 0.8)
            camera_rotation = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(-0.0, 0.0, 1), np.deg2rad(110)
            )


class CyberLongHorizonLightCfgPPO(CyberLongHorizonCfgPPO):
    def __init__(self):
        super().__init__()

    class runner(CyberLongHorizonCfgPPO.runner):
        experiment_name = "llm_light"
