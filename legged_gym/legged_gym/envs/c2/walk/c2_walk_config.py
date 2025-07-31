import numpy as np
# from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.envs.c2.c2_common_config import CyberCommonCfg, CyberCommonCfgPPO

factor_o = 4.0

class CyberWalkCfg( CyberCommonCfg ):
    class env( CyberCommonCfg.env ):
        num_envs = 1024 # 8192
        num_observation_history = 15
        observe_gait_commands = True
        use_vel_obs = False
        record_video = True
        obs_components = [
            # "proprioception", # 48
            # "all_command_obs",
            "all_history",
            # "height_measurements", # 187
            # "base_pose",
            "robot_config",
            # "dof_pos_history",
            # "clock_input",    # write into core obs
            # "engaging_block",
            # "sidewall_distance",
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
        #     "all_history",
        #     # "height_measurements", # 187
        #     "base_pose",
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
    

    class curriculum_thresholds:
        tracking_lin_vel = 0.8  # closer to 1 is tighter
        tracking_ang_vel = 0.5
        tracking_contacts_shaped_force = 0.8  # closer to 1 is tighter
        tracking_contacts_shaped_vel = 0.8

    class terrain( CyberCommonCfg.terrain ):
        mesh_type = "trimesh" # Don't change
        num_rows = 20
        num_cols = 50
        selected = "BarrierTrack" # "BarrierTrack" or "TerrainPerlin", "TerrainPerlin" can be used for training a walk policy.
        max_init_terrain_level = 0
        border_size = 5
        slope_treshold = 20.
        measure_heights = False

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
            n_obstacles_per_track = 3,
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

        static_friction = 0.2
        dynamic_friction = 0.2
    
    class commands( CyberCommonCfg.commands ):
        resampling_time = 20 # [s]
        discretize = False
        default_gait_freq = 0.8 # 2.5
        curriculum = True
        max_curriculum = 1.2

        command_curriculum = False
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        exclusive_command_sampling = False
        num_commands = 15
        resampling_time = 10.  # time before command are changed[s]
        subsample_gait = False
        gait_interval_s = 10.  # time between resampling gait params
        vel_interval_s = 10.
        jump_interval_s = 20.  # time between jumps
        jump_duration_s = 0.1  # duration of jump
        # jump_height = 0.3
        heading_command = False  # if true: compute ang vel command from heading error
        global_reference = False
        observe_accel = False
        distributional_commands = False
        curriculum_type = "RewardThresholdCurriculum"
        lipschitz_threshold = 0.9

        num_lin_vel_bins = 20
        lin_vel_step = 0.3
        num_ang_vel_bins = 20
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100


        # Command
        lin_vel_x = [-0.6, 0.6]  # min max [m/s]
        lin_vel_y = [-0.4, 0.4]  # min max [m/s]
        ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]

        body_height_cmd = [-0.2, 0.1]
        gait_frequency_cmd_range = [2.0, 4.0]
        # Gait phase
        gait_phase_cmd_range = [0.0, 1.0]
        gait_offset_cmd_range = [0.0, 1.0]
        gait_bound_cmd_range = [0.0, 1.0]
        gait_duration_cmd_range = [0.5, 0.5]

        footswing_height_range = [0.05, 0.2]

        body_pitch_range = [-0.2, 0.2]
        body_roll_range = [-0.0, 0.0]
        
        stance_width_range = [0.2, 0.35]
        stance_length_range = [0.28, 0.36]
        
        aux_reward_coef_range = [0.0, 0.01]


        limit_vel_x = [-1.0, 1.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-1.0, 1.0]

        limit_body_height = [-0.2, 0.1]
        limit_gait_frequency = [2.0, 4.0]

        limit_gait_phase = [0, 1]
        limit_gait_offset = [0, 1]
        limit_gait_bound = [0, 1]
        limit_gait_duration = [0.5, 0.5]

        limit_footswing_height = [0.05, 0.2]

        limit_body_pitch = [-0.2, 0.2]
        limit_body_roll = [-0.0, 0.0]
        limit_aux_reward_coef = [0.0, 0.01]
        # limit_compliance = [0.0, 0.01]
        limit_stance_width = [0.2, 0.33]
        limit_stance_length = [0.28, 0.36]

        num_bins_vel_x = 15
        num_bins_vel_y = 3
        num_bins_vel_yaw = 15
        num_bins_body_height = 3
        num_bins_gait_frequency = 3
        num_bins_gait_phase = 1
        num_bins_gait_offset = 1
        num_bins_gait_bound = 1
        num_bins_gait_duration = 1
        num_bins_footswing_height = 3
        num_bins_body_pitch = 1
        num_bins_body_roll = 1
        num_bins_aux_reward_coef = 1
        num_bins_compliance = 1
        num_bins_compliance = 1
        num_bins_stance_width = 3
        num_bins_stance_length = 1

        heading = [-3.14, 3.14]

        exclusive_phase_offset = False
        binary_phases = False
        pacing_offset = False
        balance_gait_distribution = True
        gaitwise_curricula = True
        

        class ranges( CyberCommonCfg.commands.ranges ):
            lin_vel_x = [-0.6, 0.6]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.4, 0.4]
            ######## configs for training a walk policy #########
            # lin_vel_y = [-1., 1.]
            # ang_vel_yaw = [-1., 1.]

    class control( CyberCommonCfg.control ):
        stiffness = {'joint': 30.}
        damping = {'joint': 3.}
        action_scale = 0.5
        # torque_limits = 25 # override the urdf
        computer_clip_torque = True
        motor_clip_torque = False

    class asset( CyberCommonCfg.asset ):
        penalize_contacts_on = ["base", "thigh", "calf"]
        terminate_after_contacts_on = ["base"]

    class termination:
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            # "roll",
            # "pitch",
            # "z_low",
            # "z_high",
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
        num_commands = 15
        class com_range:
            x = [-0.01, 0.01]
            y = [-0., 0.]
            z = [-0.01, 0.01]

        randomize_motor = True
        leg_motor_strength_range = [0.9, 1.1]

        randomize_base_mass = True
        added_mass_range = [-0.5, 0.5]

        randomize_friction = True
        friction_range = [0.5, 1.5]

        init_base_pos_range = dict(
            x= [0.2, 0.6],
            y= [-0.25, 0.25],
        )

        lag_timesteps = 6

        push_robots = True 

    class rewards( CyberCommonCfg.rewards ):
        class scales:
            legs_energy_substeps =  0   # -2e-5

            ## v2
            tracking_ang_vel = 0.5  # * factor_o
            tracking_lin_vel =  2.0 # 8.0 * factor_o
            feet_slip = -0.04
            action_smoothness_1 = -0.1
            action_smoothness_2 = -0.1
            dof_vel = -1e-4
            dof_pos = -0.0
            dof_acc = -2.5e-7
            raibert_heuristic = -40.0
            feet_clearance_cmd_linear_4 = -100.0
            orientation_control = -5.0
            # orientation = -5.0
            lin_vel_z = -0.2
            ang_vel_xy = -0.001
            tracking_contacts_shaped_force = 4.0
            tracking_contacts_shaped_vel = 4.0
            collision = -5.0
            action_rate = -0.01
            jump = 10.0
            # torques = -0.0001
            # dof_pos_limits = -10.0


        only_positive_rewards_ji22_style = True
        sigma_rew_neg = 0.04    # 0.02 0.1
        kappa_gait_probs = 0.07
        soft_dof_pos_limit = 0.95
        # foot_target = 0.03
        # allow_contact_steps = 30
        base_height_target = 0.26   # 0.3
        gait_vel_sigma = 0.2 # 10.0
        tracking_sigma = 0.2   #0.25
        max_contact_force = 50

    class normalization( CyberCommonCfg.normalization ):
        # clip_actions = 5.
        class obs_scales( CyberCommonCfg.normalization.obs_scales ):
            forward_depth = 1.
            base_pose = [0., 0., 1., 1., 1., 1.]
            engaging_block = 1.
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            imu = 0.1
            height_measurements = 5.0
            friction_measurements = 1.0
            body_height_cmd = 2.0
            gait_phase_cmd = 1.0
            gait_freq_cmd = 1.0
            footswing_height_cmd = 0.15
            body_pitch_cmd = 0.3
            body_roll_cmd = 0.3
            aux_reward_cmd = 1.0
            compliance_cmd = 1.0
            stance_width_cmd = 1.0
            stance_length_cmd = 1.0
            segmentation_image = 1.0
            rgb_image = 1.0
            depth_image = 1.0

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

class CyberWalkCfgPPO( CyberCommonCfgPPO ):
    use_wandb = True
    class algorithm( CyberCommonCfgPPO.algorithm ):
        entropy_coef = 0.01
        # clip_min_std = 1e-12

    # class policy( CyberCommonCfgPPO.policy ):
    #     rnn_type = 'gru'
    #     mu_activation = "tanh"    # "tanh"   "elu"
    
    class runner( CyberCommonCfgPPO.runner ):
        policy_class_name = "ActorCritic"
        experiment_name = "field_c2"
        run_name = "".join(["WalkingAllCommand",
        ("_pEnergySubsteps" + np.format_float_scientific(-CyberWalkCfg.rewards.scales.legs_energy_substeps, precision=1, exp_digits=1, trim="-") if CyberWalkCfg.rewards.scales.legs_energy_substeps != 0 else ""),
        ("_propDelay{:.2f}-{:.2f}".format(
                CyberWalkCfg.sensor.proprioception.latency_range[0],
                CyberWalkCfg.sensor.proprioception.latency_range[1],
            ) if CyberWalkCfg.sensor.proprioception.delay_action_obs else ""
        ),
        ("_aScale{:d}{:d}{:d}".format(
                int(CyberWalkCfg.control.action_scale[0] * 10),
                int(CyberWalkCfg.control.action_scale[1] * 10),
                int(CyberWalkCfg.control.action_scale[2] * 10),
            ) if isinstance(CyberWalkCfg.control.action_scale, (tuple, list)) \
            else "_aScale{:.1f}".format(CyberWalkCfg.control.action_scale)
        ),
        ])
        resume = False
        max_iterations = 100000
        save_interval = 500
    
