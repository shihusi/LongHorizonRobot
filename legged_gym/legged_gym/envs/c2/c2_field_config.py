import numpy as np
# from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.envs.c2.c2_common_config import CyberCommonCfg, CyberCommonCfgPPO

factor_o = 4.0

class CyberFieldCfg( CyberCommonCfg ):
    class env( CyberCommonCfg.env ):
        num_envs = 2048 # 8192
        obs_components = [
            "proprioception", # 48
            # "height_measurements", # 187
            "base_pose",
            "robot_config",
            "engaging_block",
            "sidewall_distance",
            "action_history",
            "dof_pos_history",
            "clock_input"
        ]
        # privileged_use_lin_vel = True # for the possible of setting "proprioception" in obs and privileged obs different

        ######## configs for training a walk policy ############
        # obs_components = [
        #     "proprioception", # 48
        #     # "height_measurements", # 187
        #     # "forward_depth",
        #     # "base_pose",
        #     # "robot_config",
        #     # "engaging_block",
        #     # "sidewall_distance",
        # ]
        # privileged_obs_components = [
        #     "proprioception",
        #     # "height_measurements",
        #     # "forward_depth",
        #     "robot_config",
        # ]
        ######## End configs for training a walk policy ############

    class sensor:
        class forward_camera:
            resolution = [16, 16]
            position = [0.26, 0., 0.03] # position in base_link
            rotation = [0., 0., 0.] # ZYX Euler angle in base_link
    
        class proprioception:
            delay_action_obs = False
            latency_range = [0.0, 0.0]
            latency_resample_time = 2.0 # [s]
    
    class terrain( CyberCommonCfg.terrain ):
        mesh_type = "trimesh" # Don't change
        num_rows = 20
        num_cols = 50
        selected = "BarrierTrack" # "BarrierTrack" or "TerrainPerlin", "TerrainPerlin" can be used for training a walk policy.
        max_init_terrain_level = 0
        border_size = 5
        slope_treshold = 20.

        curriculum = False # for walk
        horizontal_scale = 0.025 # [m]
        # vertical_scale = 1. # [m] does not change the value in hightfield
        pad_unavailable_info = True
        
        BarrierTrack_kwargs = dict(
            options= [
                # "climb",
                # "crawl",
                # "tilt",
                # "leap",
            ], # each race track will permute all the options
            track_width= 1.6,
            track_block_length= 2., # !!!! the x-axis distance from the env origin point
            wall_thickness= (0.04, 0.2), # [m]
            wall_height= 0,
            climb= dict(
                height= (0.0, 0.0),
                depth= (0.1, 0.8), # size along the forward axis
                fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
                climb_down_prob= 0.0,
            ),
            crawl= dict(
                height= (0., 0.),
                depth= (0.1, 0.6), # size along the forward axis
                wall_height= 0,
                no_perlin_at_obstacle= False,
            ),
            tilt= dict(
                width= (0.24, 0.32),
                depth= (0.4, 1.), # size along the forward axis
                opening_angle= 0.0, # [rad] an opening that make the robot easier to get into the obstacle
                wall_height= 0.,
            ),
            leap= dict(
                length= (0.2, 1.0),
                depth= (0.4, 0.8),
                height= 0.0,
            ),
            add_perlin_noise= False,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 1.2,
            curriculum_perlin= False,
            no_perlin_threshold= 0.0,
        )

        TerrainPerlin_kwargs = dict(
            zScale= 0,
            # zScale= 0.1, # Use a constant zScale for training a walk policy
            frequency= 10,
        )
    
    class commands( CyberCommonCfg.commands ):
        heading_command = False
        resampling_time = 20 # [s]
        discretize = False
        default_gait_freq = 0.8 # 2.5
        curriculum = True
        max_curriculum = 1.2
        class ranges( CyberCommonCfg.commands.ranges ):
            lin_vel_x = [-0.4, 0.4]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.4, 0.4]
            ######## configs for training a walk policy #########
            # lin_vel_y = [-1., 1.]
            # ang_vel_yaw = [-1., 1.]

    class control( CyberCommonCfg.control ):
        stiffness = {'joint': 50.}
        damping = {'joint': 1.}
        action_scale = 0.5
        torque_limits = 25 # override the urdf
        computer_clip_torque = True
        motor_clip_torque = False

    class asset( CyberCommonCfg.asset ):
        penalize_contacts_on = ["base", "thigh", "calf"]
        terminate_after_contacts_on = ["base"]

    class termination:
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            # "out_of_track",
        ]

        roll_kwargs = dict(
            threshold= 0.8, # [rad]
            tilt_threshold= 1.5,
        )
        pitch_kwargs = dict(
            threshold= 1.6, # [rad] # for leap, climb
            climb_threshold= 1.6,
            leap_threshold= 1.5,
        )
        z_low_kwargs = dict(
            threshold= 0.06, # [m]
        )
        z_high_kwargs = dict(
            threshold= 1.5, # [m]
        )
        out_of_track_kwargs = dict(
            threshold= 1., # [m]
        )

        check_obstacle_conditioned_threshold = True
        timeout_at_border = False

    class domain_rand( CyberCommonCfg.domain_rand ):
        randomize_com = True
        randomize_lag_timesteps = False
        class com_range:
            x = [-0.01, 0.01]
            y = [-0., 0.]
            z = [-0.01, 0.01]

        randomize_motor = True
        leg_motor_strength_range = [0.9, 1.1]

        randomize_base_mass = True
        added_mass_range = [-0.5, 0.5]

        randomize_friction = True
        friction_range = [1.0, 2.]

        init_base_pos_range = dict(
            x= [0.2, 0.6],
            y= [-0.25, 0.25],
        )

        push_robots = True 

    class rewards( CyberCommonCfg.rewards ):
        class scales:
            legs_energy_substeps =  0   # -2e-5

            ## original
            # base_height = -2.0 * factor_o
            # # world_vel_l2norm = -1.
            # legs_energy = -0.
            # alive = 2.
            # # penalty for hardware safety
            # exceed_dof_pos_limits = -1e-1
            # exceed_torque_limits_i = -2e-1
            # orientation = -32. * factor_o
            # collision = - 4. * factor_o
            # lin_vel_z = -16.0 * factor_o
            # ang_vel_xyz = -0.5 * factor_o
            # feet_slip = -1.0 * factor_o
            # feet_clearance_cmd_linear_4 = -3000 * factor_o
            
            # feet_contact_forces = -2 * factor_o
            # tracking_ang_vel = 2.0 * factor_o
            # tracking_lin_vel =  8.0 * factor_o
            # legs_energy_substeps =  0   # -2e-5
            # off_ground = -4. * factor_o
            # action_rate = -0.04 * factor_o
            # # off_ground = -4 # -2. * factor_o

            # tracking_ang_vel = 2.0  # * factor_o
            # tracking_lin_vel =  6.0 # 8.0 * factor_o
            

            # ## v1
            # tracking_ang_vel = 1.0  # * factor_o
            # tracking_lin_vel =  20.0 # 8.0 * factor_o
            # feet_slip = -0.04
            # action_smoothness_1 = -0.1
            # action_smoothness_2 = -0.1
            # dof_vel = -1e-4
            # dof_pos = -0.0
            # raibert_heuristic = -10.0
            # feet_clearance_cmd_linear_4 = -200.0
            # # orientation_control = -5.0
            # orientation = -40.0
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.001
            # # tracking_contacts_shaped_force = 4.0
            # tracking_contacts_shaped_vel = 4.0
            # collision = -5.0
            # action_rate = -0.01

            ## v2
            tracking_ang_vel = 0.5  # * factor_o
            tracking_lin_vel =  1.0 # 8.0 * factor_o
            feet_slip = -0.04
            action_smoothness_1 = -0.1
            action_smoothness_2 = -0.1
            dof_vel = -1e-4
            dof_pos = -0.0
            raibert_heuristic = -10.0
            feet_clearance_cmd_linear_4 = -30.0
            # orientation_control = -5.0
            orientation = -5.0
            lin_vel_z = -0.02
            ang_vel_xy = -0.001
            # tracking_contacts_shaped_force = 4.0
            tracking_contacts_shaped_vel = 4.0
            collision = -5.0
            action_rate = -0.01

        only_positive_rewards = True
        only_positive_rewards_ji22_style = True
        sigma_rew_neg = 0.02
        kappa_gait_probs = 0.07
        soft_dof_pos_limit = 0.01
        foot_target = 0.03
        # allow_contact_steps = 30
        base_height_target = 0.26   # 0.3
        gait_vel_sigma = 0.2
        tracking_sigma = 0.2

    class normalization( CyberCommonCfg.normalization ):
        class obs_scales( CyberCommonCfg.normalization.obs_scales ):
            forward_depth = 1.
            base_pose = [0., 0., 1., 1., 1., 1.]
            engaging_block = 1.

    class noise( CyberCommonCfg.noise ):
        class noise_scales( CyberCommonCfg.noise.noise_scales ):
            forward_depth = 0.1
            base_pose = 1.0

    class viewer( CyberCommonCfg.viewer ):
        pos = [0, 0, 5]  # [m]
        lookat = [5., 5., 2.]  # [m]

        draw_volume_sample_points = False

    class sim( CyberCommonCfg.sim ):
        body_measure_points = { # transform are related to body frame
            "base": dict(
                x= [i for i in np.arange(-0.2, 0.31, 0.03)],
                y= [-0.08, -0.04, 0.0, 0.04, 0.08],
                z= [i for i in np.arange(-0.061, 0.061, 0.03)],
                transform= [0., 0., 0.005, 0., 0., 0.],
            ),
            "thigh": dict(
                x= [
                    -0.16, -0.158, -0.156, -0.154, -0.152,
                    -0.15, -0.145, -0.14, -0.135, -0.13, -0.125, -0.12, -0.115, -0.11, -0.105, -0.1, -0.095, -0.09, -0.085, -0.08, -0.075, -0.07, -0.065, -0.05,
                    0.0, 0.05, 0.1,
                ],
                y= [-0.015, -0.01, 0.0, -0.01, 0.015],
                z= [-0.03, -0.015, 0.0, 0.015],
                transform= [0., 0., -0.1,   0., 1.57079632679, 0.],
            ),
            "calf": dict(
                x= [i for i in np.arange(-0.13, 0.111, 0.03)],
                y= [-0.015, 0.0, 0.015],
                z= [-0.015, 0.0, 0.015],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
        }

    class curriculum:
        no_moveup_when_fall = False
        # chosen heuristically, please refer to `LeggedRobotField._get_terrain_curriculum_move` with fixed body_measure_points

class CyberFieldCfgPPO( CyberCommonCfgPPO ):
    class algorithm( CyberCommonCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 1e-12

    class policy( CyberCommonCfgPPO.policy ):
        rnn_type = 'gru'
        mu_activation = "tanh"
    
    class runner( CyberCommonCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        experiment_name = "field_c2"
        run_name = "".join(["WalkingBase",
        ("_pEnergySubsteps" + np.format_float_scientific(-CyberFieldCfg.rewards.scales.legs_energy_substeps, precision=1, exp_digits=1, trim="-") if CyberFieldCfg.rewards.scales.legs_energy_substeps != 0 else ""),
        ("_propDelay{:.2f}-{:.2f}".format(
                CyberFieldCfg.sensor.proprioception.latency_range[0],
                CyberFieldCfg.sensor.proprioception.latency_range[1],
            ) if CyberFieldCfg.sensor.proprioception.delay_action_obs else ""
        ),
        ("_aScale{:d}{:d}{:d}".format(
                int(CyberFieldCfg.control.action_scale[0] * 10),
                int(CyberFieldCfg.control.action_scale[1] * 10),
                int(CyberFieldCfg.control.action_scale[2] * 10),
            ) if isinstance(CyberFieldCfg.control.action_scale, (tuple, list)) \
            else "_aScale{:.1f}".format(CyberFieldCfg.control.action_scale)
        ),
        ])
        resume = False
        max_iterations = 10000
        save_interval = 500
    
