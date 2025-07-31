from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class CyberCommonCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        obs_base_vel = False
        obs_base_vela = False
        obs_height = False
        binarize_base_vela = False
        single_base_vel = False
        single_base_vela = False
        single_height = False
    
    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': -45 / 57.3,     # [rad]
            'RL_thigh_joint': -45 / 57.3,   # [rad]
            'FR_thigh_joint': -45 / 57.3,     # [rad]
            'RR_thigh_joint': -45 / 57.3,   # [rad]

            'FL_calf_joint': 70 / 57.3,   # [rad]
            'RL_calf_joint': 70 / 57.3,    # [rad]
            'FR_calf_joint': 70 / 57.3,  # [rad]
            'RR_calf_joint': 70 / 57.3,    # [rad]
        }
        # default_joint_angles = {
        #     # 'FL_hip_joint':0.28981366753578186, 
        #     # 'RL_hip_joint':-0.9788374304771423, 
        #     # 'FR_hip_joint':1.233295202255249, 
        #     # 'RR_hip_joint': -0.10544154793024063, 
            
        #     # 'FL_thigh_joint': -0.9346483945846558, 
        #     # 'RL_thigh_joint': 1.323702335357666, 
        #     # 'FR_thigh_joint': 0.33093586564064026, 
        #     # 'RR_thigh_joint': -1.1398987770080566, 
            
        #     # 'FL_calf_joint': 1.0004403591156006, 
        #     # 'RL_calf_joint':-0.17597976326942444, 
        #     # 'FR_calf_joint': -1.0723696947097778, 
        #     # 'RR_calf_joint': 1.0730541944503784,

        #     'FL_hip_joint':0.4757556617259979, 
        #     'RL_hip_joint':-1.2022411823272705, 
        #     'FR_hip_joint':0.7840592265129089,
        #     'RR_hip_joint':-0.5821757316589355, 
            
        #     'FL_thigh_joint': -0.3176952004432678,
        #     'RL_thigh_joint': 0.8873132467269897,
        #     'FR_thigh_joint': 0.5686444044113159, 
        #     'RR_thigh_joint': -0.19131043553352356,
            
        #     'FL_calf_joint': 1.723366379737854,
        #     'RL_calf_joint': 0.10798931121826172, 
        #     'FR_calf_joint': -1.382157325744629, 
        #     'RR_calf_joint': 0.7832280397415161,

        #     # 0.4757556617259979, -1.2022411823272705, 0.7840592265129089, -0.5821757316589355, -0.3176952004432678, 0.8873132467269897, 0.5686444044113159, -0.19131043553352356, 1.723366379737854, 0.10798931121826172, -1.382157325744629, 0.7832280397415161
        # }
        pos = [0.0, 0.0, 0.3] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        # rot = [-0.04215244576334953, -0.03334926813840866, -0.04430733993649483, 0.9975710511207581,]
        # rot = [-0.04216809943318367, -0.24881726503372192, 0.018970025703310966, 0.9674461483955383,]
        # lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        # ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # default_joint_angles = { # target angles when action = 0.0
        #     "joint_a": 0., 
        #     "joint_b": 0.}
    
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 30.0}
        damping = {'joint': 3.0}
        decimation = 4
        kp_factor_range = [0.8, 1.2]
        kd_factor_range = [0.8, 1.2]
        
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_reduction_scale = 1.0
        ratio_delay = 0.0

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cyberdog2/urdf/cyberdog2_v2_c.urdf'
        name = "cyber2"
        foot_name = "foot"
        # self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

        door_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/door.urdf'

    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [1.0, 3.0]
        randomize_restitution = True
        randomize_base_mass = True
        added_mass_range = [-0.5, 0.5]
        randomize_com_displacement = True
        com_displacement_range = [[-0.01, 0.0, -0.01], [0.01, 0.0, 0.01]]
        randomize_joint_props = True
        joint_friction_range = [0.03, 0.08]
        joint_damping_range = [0.02, 0.06]
        restitution_range = [0.0, 0.4]
        
        use_dynamic_kp_scale = False
        lag_timesteps = 6
        swing_lag_timesteps = [6, 6]
        stance_lag_timesteps = [1, 1]

        randomize_Kp_factor = True
        randomize_Kd_factor = True
        Kp_factor_range = [0.8, 1.2]
        Kd_factor_range = [0.8, 1.2]

        randomize_motor_offset = True
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [-0.05, 0.05]
    
    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards_ji22_style = False
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.
        base_height_target = 0.3   # 0.3
        class scales:
            pass
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        static_friction = 0.5
        dynamic_friction = 0.5


class CyberCommonCfgPPO(LeggedRobotCfgPPO):
    use_wandb = True
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 2.5e-4
        schedule = 'fixed'
