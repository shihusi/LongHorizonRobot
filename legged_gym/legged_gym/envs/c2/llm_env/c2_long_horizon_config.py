from os import path as osp
import numpy as np

# from legged_gym.envs.a1.a1_field_config import CyberWalkCfg, CyberWalkCfgPPO
from legged_gym.envs.c2.walk.c2_walk_config import CyberWalkCfg, CyberWalkCfgPPO
from legged_gym.utils.helpers import merge_dict
from isaacgym import gymtorch, gymapi

init_pose = "upright"       # "sit" or "upright"
tracking_frame = "world"    #'world' or 'base'


class CyberLongHorizonCfg(CyberWalkCfg):
    class env(CyberWalkCfg.env):
        num_envs = 256 
        use_down_stair_box = True
        fix_box_actor = True

        fix_rel_target = [0.7500, 0.20, 0.65, 0, 0, 0]  # position in world frame

    class sensor:
        class forward_camera:
            resolution = [int(240 / 4), int(424 / 4)]
            position = dict(
                mean=[0.271, 25e-3, 114.912e-3],
                std=[0.01, 0.0025, 0.0005],
            )  # position in base_link 
            rotation = dict(
                lower=[0, 0, 0],
                upper=[0, 5 * np.pi / 180, 0],
            )  # rotation in base_link 
            resized_resolution = [48, 64]
            output_resolution = [48, 64]
            horizontal_fov = [85, 88]

            # adding randomized latency
            latency_range = [
                0.2,
                0.26,
            ]  # for [16, 32, 32] -> 128 -> 128 visual model in (240, 424 option)
            latency_resample_time = 5.0  # [s]
            refresh_duration = (
                1 / 10
            )  # [s] for (240, 424 option with onboard script fixed to no more than 20Hz)

            # config to simulate stero RGBD camera
            crop_top_bottom = [0, 0]
            crop_left_right = [int(60 / 4), int(46 / 4)]
            depth_range = [0.0, 1.5]  # [m]

        class proprioception:
            delay_action_obs = True
            latency_range = [0.04 - 0.0025, 0.04 + 0.0075]  # [min, max] in seconds
            latency_resample_time = 2.0  # [s]

    class terrain(CyberWalkCfg.terrain):
        static_friction = 0.5
        dynamic_friction = 0.5
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.0
        curriculum = True
        selected = "BarrierTrack"
        track_block_length = 2.0
        measure_heights = True


    class render_config:
        camera_offset = gymapi.Vec3(-1.0, -1.0, 1.0)
        camera_rotation = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(45)
        )
        fix_camera = False
        camera_attach_actor_id = 0

    class control(CyberWalkCfg.control):
        action_scale = 0.25

        stiffness = {"joint": 30.0}
        damping = {"joint": 3.0}
        decimation = 4
        kp_factor_range = [0.8, 1.2]
        kd_factor_range = [0.8, 1.2]

    class noise(CyberWalkCfg.noise):
        class noise_scales(CyberWalkCfg.noise.noise_scales):
            forward_depth = 0.1
            base_pose = 0.05
            engaging_block = [1, 0, 0, 0, 0, 0, 1, 1]

    class commands(CyberWalkCfg.commands):
        class ranges(CyberWalkCfg.commands.ranges):
            lin_vel_x = [0.1, 0.6]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class termination(CyberWalkCfg.termination):
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]
        z_low_kwargs = merge_dict(
            CyberWalkCfg.termination.z_low_kwargs,
            dict(
                threshold=-1.0,
            ),
        )

    class domain_rand(CyberWalkCfg.domain_rand):
        init_base_pos_range = dict(
            x=[0.2, 0.6],
            y=[-0.25, 0.25],
        )
        randomize_Kp_factor = True
        randomize_Kd_factor = True
        Kp_factor_range = [0.8, 1.2]
        Kd_factor_range = [0.8, 1.2]
        max_push_vel_xy = 0.2

        randomize_motor_offset = True
        randomize_motor_strength = True
        motor_strength_range = [1.0, 1.0]   
        motor_offset_range = [0, 0]         

    class rewards(CyberWalkCfg.rewards):
        ## button
        liftup_target = 0.42
        lift_up_threshold = [0.15, 0.42]
        scale_factor_low = 0.25
        scale_factor_high = 0.35
        foot_target = 0.05
        if init_pose == "upright":
            allow_contact_steps = 0
        elif init_pose == "sit":
            allow_contact_steps = 30
        before_handtrack_steps = 0 if init_pose == "upright" else 50
        upright_vec = [0.2, 0.0, 1.0]
        ang_rew_mode = "heading"
        frame = tracking_frame
        reach_criteria = 0.05
        height_cond = 0.36
        upright_cond = 0.5

        class scales:
            tracking_ang_vel = 0.5
            tracking_lin_vel = 4.0

    class curriculum(CyberWalkCfg.curriculum):
        penetrate_volume_threshold_harder = 8000
        penetrate_volume_threshold_easier = 12000
        penetrate_depth_threshold_harder = 1000
        penetrate_depth_threshold_easier = 1600

    class task:
        def __init__(self):
            self.exp_scene_id = 1
            self.score_mode = 0
            self.use_down_stair_box = False
            self.single_test_work = False
            self.test_task_id = 2
            self.multi_robotool = False

            ## collect data
            self.collect_data_work = False
            self.collect_target = "box_init_state_mani2climb"
            self.save_data_file_path = "./box_climb_init_state.txt"

            # robot arg
            self.joint_friction_range = [0.05, 0.06]
            self.joint_damping_range = [0.03, 0.04]
            self.walk_control_type = "P"
            self.push_to_wall_offset = 0.2

            self.policy_path_dict = {
                "policy_zoo": "./policy_zoo/rl_skills",       # policy zoo root path
                "walk_body": "/walk_policy_body.jit",
                "walk_adapt": "/walk_policy_adapter.jit",
                "manipulate": "/manipulate_policy.pt",
                "climb": "/climb_policy.pt",
                "stand": "/stand_policy.pt",
                "sit_down": "/sit_down_policy.pt",
                "button": "/button_policy.pt",
                "move_2_pos": "/move_policy.pt",
            }


class CyberLongHorizonCfgPPO(CyberWalkCfgPPO):

    class runner(CyberWalkCfgPPO.runner):
        policy_class_name = "ActorCritic"
        experiment_name = "long_horizon"
        run_name = "Vaild Test"
        resume = True
        load_run = "{Your traind walking model directory}"
        load_run = "{Your virtually trained climb model directory}"
        max_iterations = 20000
        save_interval = 500
