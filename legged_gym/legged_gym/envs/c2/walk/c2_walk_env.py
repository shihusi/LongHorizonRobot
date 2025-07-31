import random
import numpy as np
from collections import OrderedDict, defaultdict

from isaacgym.torch_utils import torch_rand_float, get_euler_xyz, quat_from_euler_xyz, tf_apply
from isaacgym import gymtorch, gymapi, gymutil
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from isaacgym.torch_utils import *
from .c2_walk_config import CyberWalkCfg
from legged_gym.utils.helpers import class_to_dict

from legged_gym.envs.base.legged_robot_noisy import LeggedRobotNoisy

class CyberWalkEnv(LeggedRobotNoisy):

    def __init__(self, cfg: CyberWalkCfg, sim_params, physics_engine, sim_device, headless):
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # print("action_before_pre: ", actions)
        self.pre_physics_step(actions)
        # print("aciton_afeter_pre", self.actions)
        # step physics and render each frame
        self.render()
        for dec_i in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.post_decimation_step(dec_i)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
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
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.foot_positions = self.robot_rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_last_actions[:] = self.last_actions[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]

        for i in range(0, self.action_history_num - 1):
            self.action_history[:, i] = self.action_history[:, i+1]
            self.dof_pos_history[:, i] = self.dof_pos_history[:, i+1]

        self.action_history[:, self.action_history_num-1] = self.last_actions
        self.dof_pos_history[:, self.action_history_num-1] = self.last_dof_pos

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()


    
    def _init_command_distribution(self, env_ids):
        # new style curriculum
        self.category_names = ['nominal']
        if self.cfg.commands.gaitwise_curricula:
            self.category_names = ['pronk', 'trot', 'pace', 'bound']

        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            from .curriculum import RewardThresholdCurriculum
            CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                               x_vel=(self.cfg.commands.limit_vel_x[0],
                                                      self.cfg.commands.limit_vel_x[1],
                                                      self.cfg.commands.num_bins_vel_x),
                                               y_vel=(self.cfg.commands.limit_vel_y[0],
                                                      self.cfg.commands.limit_vel_y[1],
                                                      self.cfg.commands.num_bins_vel_y),
                                               yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                                                        self.cfg.commands.limit_vel_yaw[1],
                                                        self.cfg.commands.num_bins_vel_yaw),
                                               body_height=(self.cfg.commands.limit_body_height[0],
                                                            self.cfg.commands.limit_body_height[1],
                                                            self.cfg.commands.num_bins_body_height),
                                               gait_frequency=(self.cfg.commands.limit_gait_frequency[0],
                                                               self.cfg.commands.limit_gait_frequency[1],
                                                               self.cfg.commands.num_bins_gait_frequency),
                                               gait_phase=(self.cfg.commands.limit_gait_phase[0],
                                                           self.cfg.commands.limit_gait_phase[1],
                                                           self.cfg.commands.num_bins_gait_phase),
                                               gait_offset=(self.cfg.commands.limit_gait_offset[0],
                                                            self.cfg.commands.limit_gait_offset[1],
                                                            self.cfg.commands.num_bins_gait_offset),
                                               gait_bounds=(self.cfg.commands.limit_gait_bound[0],
                                                            self.cfg.commands.limit_gait_bound[1],
                                                            self.cfg.commands.num_bins_gait_bound),
                                               gait_duration=(self.cfg.commands.limit_gait_duration[0],
                                                              self.cfg.commands.limit_gait_duration[1],
                                                              self.cfg.commands.num_bins_gait_duration),
                                               footswing_height=(self.cfg.commands.limit_footswing_height[0],
                                                                 self.cfg.commands.limit_footswing_height[1],
                                                                 self.cfg.commands.num_bins_footswing_height),
                                               body_pitch=(self.cfg.commands.limit_body_pitch[0],
                                                           self.cfg.commands.limit_body_pitch[1],
                                                           self.cfg.commands.num_bins_body_pitch),
                                               body_roll=(self.cfg.commands.limit_body_roll[0],
                                                          self.cfg.commands.limit_body_roll[1],
                                                          self.cfg.commands.num_bins_body_roll),
                                               stance_width=(self.cfg.commands.limit_stance_width[0],
                                                             self.cfg.commands.limit_stance_width[1],
                                                             self.cfg.commands.num_bins_stance_width),
                                               stance_length=(self.cfg.commands.limit_stance_length[0],
                                                                self.cfg.commands.limit_stance_length[1],
                                                                self.cfg.commands.num_bins_stance_length),
                                               aux_reward_coef=(self.cfg.commands.limit_aux_reward_coef[0],
                                                           self.cfg.commands.limit_aux_reward_coef[1],
                                                           self.cfg.commands.num_bins_aux_reward_coef),
                                               )]

        if self.cfg.commands.curriculum_type == "LipschitzCurriculum":
            for curriculum in self.curricula:
                curriculum.set_params(lipschitz_threshold=self.cfg.commands.lipschitz_threshold,
                                      binary_phases=self.cfg.commands.binary_phases)
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int)
        low = np.array(
            [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
             self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.body_height_cmd[0],
             self.cfg.commands.gait_frequency_cmd_range[0],
             self.cfg.commands.gait_phase_cmd_range[0], self.cfg.commands.gait_offset_cmd_range[0],
             self.cfg.commands.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
             self.cfg.commands.footswing_height_range[0], self.cfg.commands.body_pitch_range[0],
             self.cfg.commands.body_roll_range[0],self.cfg.commands.stance_width_range[0],
             self.cfg.commands.stance_length_range[0], self.cfg.commands.aux_reward_coef_range[0], ])
        high = np.array(
            [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
             self.cfg.commands.ang_vel_yaw[1], self.cfg.commands.body_height_cmd[1],
             self.cfg.commands.gait_frequency_cmd_range[1],
             self.cfg.commands.gait_phase_cmd_range[1], self.cfg.commands.gait_offset_cmd_range[1],
             self.cfg.commands.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1],
             self.cfg.commands.footswing_height_range[1], self.cfg.commands.body_pitch_range[1],
             self.cfg.commands.body_roll_range[1],self.cfg.commands.stance_width_range[1],
             self.cfg.commands.stance_length_range[1], self.cfg.commands.aux_reward_coef_range[1], ])
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def _reset_buffers(self, env_ids):
        super()._reset_buffers(env_ids)
        self.obs_history[env_ids] = 0
        self.last_last_actions[env_ids] = 0
        self.last_dof_pos[env_ids] = 0
        self.last_dof_vel[env_ids] = 0
        self.last_root_vel[env_ids] = 0
        self.last_last_joint_pos_target[env_ids] = 0
        self.last_joint_pos_target[env_ids] = 0
        self.actions[env_ids] = 0
        self.last_base_lin_vel[env_ids] = 0
        self.last_base_ang_vel[env_ids] = 0
    
    def _randomize_dof_props(self, env_ids):
        if self.cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_strength - min_strength) + min_strength
        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     max_offset - min_offset) + min_offset
        if self.cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = self.cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kp_factor - min_Kp_factor) + min_Kp_factor
        if self.cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = self.cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor

    def reset_idx(self, env_ids):
        self._randomize_dof_props(env_ids)
        super().reset_idx(env_ids)
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["train/episode"]["terrain_level"] = torch.mean(
                self.terrain_levels[:self.num_train_envs].float())
        if self.cfg.commands.command_curriculum:
            self.extras["env_bins"] = torch.Tensor(self.env_command_bins)[:self.num_train_envs]
            self.extras["train/episode"]["min_command_duration"] = torch.min(self.commands[:, 8])
            self.extras["train/episode"]["max_command_duration"] = torch.max(self.commands[:, 8])
            self.extras["train/episode"]["min_command_bound"] = torch.min(self.commands[:, 7])
            self.extras["train/episode"]["max_command_bound"] = torch.max(self.commands[:, 7])
            self.extras["train/episode"]["min_command_offset"] = torch.min(self.commands[:, 6])
            self.extras["train/episode"]["max_command_offset"] = torch.max(self.commands[:, 6])
            self.extras["train/episode"]["min_command_phase"] = torch.min(self.commands[:, 5])
            self.extras["train/episode"]["max_command_phase"] = torch.max(self.commands[:, 5])
            self.extras["train/episode"]["min_command_freq"] = torch.min(self.commands[:, 4])
            self.extras["train/episode"]["max_command_freq"] = torch.max(self.commands[:, 4])
            self.extras["train/episode"]["min_command_x_vel"] = torch.min(self.commands[:, 0])
            self.extras["train/episode"]["max_command_x_vel"] = torch.max(self.commands[:, 0])
            self.extras["train/episode"]["min_command_y_vel"] = torch.min(self.commands[:, 1])
            self.extras["train/episode"]["max_command_y_vel"] = torch.max(self.commands[:, 1])
            self.extras["train/episode"]["min_command_yaw_vel"] = torch.min(self.commands[:, 2])
            self.extras["train/episode"]["max_command_yaw_vel"] = torch.max(self.commands[:, 2])
            if self.cfg.commands.num_commands > 9:
                self.extras["train/episode"]["min_command_swing_height"] = torch.min(self.commands[:, 9])
                self.extras["train/episode"]["max_command_swing_height"] = torch.max(self.commands[:, 9])
            for curriculum, category in zip(self.curricula, self.category_names):
                self.extras["train/episode"][f"command_area_{category}"] = np.sum(curriculum.weights) / \
                                                                           curriculum.weights.shape[0]

            self.extras["train/episode"]["min_action"] = torch.min(self.actions)
            self.extras["train/episode"]["max_action"] = torch.max(self.actions)

            self.extras["curriculum/distribution"] = {}
            for curriculum, category in zip(self.curricula, self.category_names):
                self.extras[f"curriculum/distribution"][f"weights_{category}"] = curriculum.weights
                self.extras[f"curriculum/distribution"][f"grid_{category}"] = curriculum.grid

        self.gait_indices[env_ids] = 0

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0
        
    
    def compute_reward(self):
        
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
            # print("ac", self.rew_buf)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        
        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1


    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
        #     self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
        #     self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    
    def _compute_torques(self, actions):
        return_ = super()._compute_torques(actions)
        actions_scaled = actions * self.cfg.control.action_scale
        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        else:
            # print("scale + default:", actions_scaled, " and default: ", self.default_dof_pos)
            self.joint_pos_target = actions_scaled + self.default_dof_pos

        return return_





    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._step_contact_targets()
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        with torch.no_grad():
            pos_x = self.root_states[:, 0] - self.env_origins[:, 0]
            pos_y = self.root_states[:, 1] - self.env_origins[:, 1]
            self.extras["episode"]["max_pos_x"] = max(self.extras["episode"]["max_pos_x"], torch.max(pos_x).cpu())
            self.extras["episode"]["min_pos_x"] = min(self.extras["episode"]["min_pos_x"], torch.min(pos_x).cpu())
            self.extras["episode"]["max_pos_y"] = max(self.extras["episode"]["max_pos_y"], torch.max(pos_y).cpu())
            self.extras["episode"]["min_pos_y"] = min(self.extras["episode"]["min_pos_y"], torch.min(pos_y).cpu())
            if self.check_BarrierTrack_terrain():
                self.extras["episode"]["n_obstacle_passed"] = torch.mean(torch.clip(
                    torch.div(pos_x, self.terrain.env_block_length, rounding_mode= "floor") - 1,
                    min= 0.0,
                )).cpu()
        
        if hasattr(self, "proprioception_buffer"):
            resampling_time = getattr(self.cfg.sensor.proprioception, "latency_resampling_time", self.dt)
            resample_env_ids = (self.episode_length_buf % int(resampling_time / self.dt) == 0).nonzero(as_tuple= False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_proprioception_latency(resample_env_ids)
        
        if hasattr(self, "forward_depth_buffer"):
            resampling_time = getattr(self.cfg.sensor.forward_camera, "latency_resampling_time", self.dt)
            resample_env_ids = (self.episode_length_buf % int(resampling_time / self.dt) == 0).nonzero(as_tuple= False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_forward_camera_latency(resample_env_ids)

        self.torque_exceed_count_envstep[(torch.abs(self.substep_torques) > self.torque_limits).any(dim= 1).any(dim= 1)] += 1

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.curriculum_thresholds = class_to_dict(self.cfg.curriculum_thresholds)

    
    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}
    
    def _resample_commands(self, env_ids):

        if len(env_ids) == 0: return

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.max_episode_length, timesteps)

        # update curricula based on terminated environment bins and categories
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]

            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    # print(self.curriculum_thresholds, self.reward_scales)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update(old_bins, task_rewards, success_thresholds,
                                  local_range=np.array(
                                      [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0]))

        # assign resampled environments to new categories
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1. / len(self.category_names)
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                      random_env_floats < probability_per_category * (i + 1))] for i in
                            range(len(self.category_names))]

        # sample from new category curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue
            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :self.cfg.commands.num_commands]).to(
                self.device)

        if self.cfg.commands.num_commands > 5:
            if self.cfg.commands.gaitwise_curricula:    # activate
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    if category == "pronk":  # pronking
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    elif category == "trot":  # trotting
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":  # pacing
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":  # bounding
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

            elif self.cfg.commands.exclusive_phase_offset:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                trotting_envs = env_ids[random_env_floats < 0.34]
                pacing_envs = env_ids[torch.logical_and(0.34 <= random_env_floats, random_env_floats < 0.67)]
                bounding_envs = env_ids[0.67 <= random_env_floats]
                self.commands[pacing_envs, 5] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[trotting_envs, 6] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 7] = 0

            elif self.cfg.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[torch.logical_and(0.25 <= random_env_floats, random_env_floats < 0.50)]
                pacing_envs = env_ids[torch.logical_and(0.50 <= random_env_floats, random_env_floats < 0.75)]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                self.commands[pronking_envs, 5] = (self.commands[pronking_envs, 5] / 2 - 0.25) % 1
                self.commands[pronking_envs, 6] = (self.commands[pronking_envs, 6] / 2 - 0.25) % 1
                self.commands[pronking_envs, 7] = (self.commands[pronking_envs, 7] / 2 - 0.25) % 1
                self.commands[trotting_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 5] = 0
                self.commands[pacing_envs, 7] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 5] = self.commands[trotting_envs, 5] / 2 + 0.25
                self.commands[pacing_envs, 6] = self.commands[pacing_envs, 6] / 2 + 0.25
                self.commands[bounding_envs, 7] = self.commands[bounding_envs, 7] / 2 + 0.25

            if self.cfg.commands.binary_phases:
                self.commands[env_ids, 5] = (torch.round(2 * self.commands[env_ids, 5])) / 2.0 % 1
                self.commands[env_ids, 6] = (torch.round(2 * self.commands[env_ids, 6])) / 2.0 % 1
                self.commands[env_ids, 7] = (torch.round(2 * self.commands[env_ids, 7])) / 2.0 % 1

        # setting the smaller commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.


    def _get_clock_input_obs(self, privileged=False):
        # print(self.clock_inputs)
        return self.clock_inputs
    
    def _get_action_history_obs(self, privileged=False):
        # return self.action_history.flatten(start_dim=1)
        return self.last_actions
    
    def _get_dof_pos_history_obs(self, privileged=False):
        return self.dof_pos_history.flatten(start_dim=1)
    
    def _get_all_command_obs_obs(self, privileged= False):
        # print("get all command obs", self.obs_super_impl[:, :72])
        if self.cfg.env.use_vel_obs:
            return self.obs_super_impl[:, 6:76]
        else:
            return self.obs_super_impl[:, :76]
    
    def _get_all_history_obs(self, privileged= False):
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

    def _get_obs_from_components(self, components: list, privileged= False):
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
                    getattr(self, "_get_" + k + "_obs")(privileged) * \
                    getattr(self.obs_scales, k, 1.)
                )
        obs = torch.cat(obs, dim= 1)
        
        return obs
    
    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)

        if self.cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []
    
    def _get_base_pose_obs(self, privileged= False):
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        roll[roll > np.pi] -= np.pi * 2 # to range (-pi, pi)
        pitch[pitch > np.pi] -= np.pi * 2 # to range (-pi, pi)
        yaw[yaw > np.pi] -= np.pi * 2 # to range (-pi, pi)
        # print("base_obs", torch.cat([
        #     self.root_states[:, :3] - self.env_origins,
        #     torch.stack([roll, pitch, yaw], dim= -1),
        # ], dim= -1))
        return torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device)
    
    def _get_robot_config_obs(self, privileged= False):
        # print("robot config obs", self.robot_config_buffer)
        return self.robot_config_buffer

    def _get_engaging_block_obs(self, privileged= False):
        """ Compute the obstacle info for the robot """
        if not self.check_BarrierTrack_terrain():
            # This could be wrong, check BarrierTrack implementation to get the exact shape.
            return torch.zeros((self.num_envs, (1 + (4 + 1) + 2)), device= self.sim_device)
        base_positions = self.root_states[:, 0:3] # (n_envs, 3)
        self.refresh_volume_sample_points()
        # print("engaging_block_info", self.terrain.get_engaging_block_info(
        #     base_positions,
        #     self.volume_sample_points - base_positions.unsqueeze(-2), # (n_envs, n_points, 3)
        # ))
        # return self.terrain.get_engaging_block_info(
        #     base_positions,
        #     self.volume_sample_points - base_positions.unsqueeze(-2), # (n_envs, n_points, 3)
        # )
        return torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

    def _get_sidewall_distance_obs(self, privileged= False):
        if not self.check_BarrierTrack_terrain():
            return torch.zeros((self.num_envs, 2), device= self.sim_device)
        base_positions = self.root_states[:, 0:3] # (n_envs, 3)
        # print("sideall obs", self.terrain.get_sidewall_distance(base_positions))
        # return self.terrain.get_sidewall_distance(base_positions)
        return torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        
    def compute_observations(self):
        for key in self.sensor_handles[0].keys():
            print("key--", key)
            if "camera" in key:
                # NOTE: Different from the documentation and examples from isaacgym
                # gym.fetch_results() must be called before gym.start_access_image_tensors()
                # refer to https://forums.developer.nvidia.com/t/camera-example-and-headless-mode/178901/10
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                break
        add_noise = self.add_noise; self.add_noise = False
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,    
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    # self.commands[:, :3] * self.commands_scale,
                                    self.commands[:, :3] * self.commands_scale[:3],
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    # self.last_actions,
                                    # self.clock_inputs,
                                    ),dim=-1)
        if not self.cfg.env.use_vel_obs and self.commands.size()[1] == 15:
            # self.obs_history = torch.cat((self.obs_history[:, self.num_core_obs:], self.obs_buf[:, 6:]), dim=-1)
            self.obs_buf = torch.cat((  # self.base_lin_vel * self.obs_scales.lin_vel,
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
                                    self._get_engaging_block_obs(), # 8
                                    self._get_sidewall_distance_obs(),  # 2
                                    ),dim=-1)
            # print(self.obs_buf.size())
            self.obs_history = torch.cat((self.obs_history[:, self.num_core_obs:], self.obs_buf[:, :]), dim=-1)
        else:

            self.obs_history = torch.cat((self.obs_history[:, self.num_core_obs:], self.obs_buf), dim=-1)
        # print(self.obs_history[0])
        # print(self.base_lin_vel)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if not self.num_privileged_obs is None:
            min_shape = min(self.obs_buf.shape[1], self.privileged_obs_buf.shape[1])
            self.privileged_obs_buf[:, :min_shape] = self.obs_buf[:, :min_shape] # copy content
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # if not self.cfg.env.use_lin_vel:
        #     self.obs_buf[:, :3] = 0.
        self.obs_super_impl = self.obs_buf
        self.add_noise = add_noise

        

        # actor obs
        self.obs_buf = self._get_obs_from_components(
            self.cfg.env.obs_components,
            privileged= False,
        )

        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # critic obs
        if not self.num_privileged_obs is None:
            self.privileged_obs_buf[:] = self._get_obs_from_components(
                self.cfg.env.privileged_obs_components,
                privileged= getattr(self.cfg.env, "privileged_obs_gets_privilege", False),
            )
        # fixing linear velocity in proprioception observation
        if "proprioception" in getattr(self.cfg.env, "privileged_obs_components", []) \
            and getattr(self.cfg.env, "privileged_use_lin_vel", False):
            # NOTE: according to self.get_obs_segment_from_components, "proprioception" observation
            # is always the first part of this flattened observation. check super().compute_observations
            # and self.cfg.env.use_lin_vel for the reason of this if branch.
            self.privileged_obs_buf[:, :3] = self.base_lin_vel * self.obs_scales.lin_vel

        for key in self.sensor_handles[0].keys():
            if "camera" in key:
                self.gym.end_access_image_tensors(self.sim)
                break
        return self.obs_buf
    
    

    
    def _init_buffers(self):
        self.action_history_num = 6
        self.all_history_num = 30
        self.num_core_obs = 80 - 5  # -5 remove one hot of terrain info
        if not self.cfg.env.use_vel_obs:
            self.num_core_obs -= 6
        # self.num_critical_obs = 72
        self.action_history = torch.zeros(self.num_envs, self.action_history_num, 12, dtype=torch.float, device=self.device)
        self.obs_history = torch.zeros(self.num_envs, self.all_history_num * self.num_core_obs, dtype=torch.float, device=self.device)
        self.dof_pos_history = torch.zeros(self.num_envs, self.action_history_num, 12, dtype=torch.float, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # if self.cfg.terrain.measure_heights:
        self.height_points = self._init_height_points()
        self.test_reset_id = torch.zeros(1, dtype=torch.int32, device=self.device,
                                                  requires_grad=False, )
        self.obs_padding = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.measured_heights = 0
        return_ = super()._init_buffers()
        self.num_obs = self.num_core_obs * self.all_history_num + 17
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                            self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
                                            self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
                                           self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd],
                                           device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]
        self.rew_buf_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_neg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self.last_dof_pos = torch.zeros_like(self.dof_pos)

        self.num_actuated_dof = self.num_actions
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                                      device=self.device,
                                                      requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]
        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]

        self.last_base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.last_base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)



        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []
        return return_
    
    def check_termination(self):
        # print(self.episode_length_buf)
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.out_env_buf = self.root_states[:, 2] < 0
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.out_env_buf
        # feet_contact = torch.any(self.contact_forces[:, self.feet_indices, 2] > 0.001, dim=1)
        # falled_ = self.step_counter > 10
        # # print(self.step_counter)
        # # pen_off_ground = (~feet_contact & falled_)
        # # print(feet_contact, pen_off_ground)
        # # self.reset_buf |= pen_off_ground

        # return 
    
    
    


    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            if self.cfg.commands.num_commands > 4:
                frequencies = self.commands[:, 4]
                phases = self.commands[:, 5]
                offsets = self.commands[:, 6]
                bounds = self.commands[:, 7]
                durations = self.commands[:, 8]
            else:
                frequencies = 2
                phases = 0
                offsets = 0
                bounds = 0
                durations = 0.5 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # 0.5
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            if self.cfg.commands.pacing_offset:
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + phases]
            else:   # activate
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + bounds,
                                self.gait_indices + phases]

            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                            0.5 / (1 - durations[swing_idxs]))

            # if self.cfg.commands.durations_warp_clock_inputs:

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            # self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            # self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            # self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            # self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            # self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            # self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            # self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            # self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                    kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # print("Command_lin ", self.commands[:, :2])
        # print("real_vel ", self.base_lin_vel[:, :2])
        # print("lin_vel_error", lin_vel_error)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # print("Command_ang ", self.commands[:, 2])
        # print("real_ang ", self.base_ang_vel[:, 2])
        # print("reward", torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma))
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_collision(self):
        # print(torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1))
        # print(self.penalised_contact_indices)
        # print(self.root_states[:, 2])
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.001
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.robot_rigid_body_state = self.all_rigid_body_states.view(self.num_envs, -1, 13)[:, :17, :]
        self.foot_velocities = self.robot_rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                            )[:, self.feet_indices, 7:10]
        self.foot_velocities_ang = self.robot_rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                                )[:, self.feet_indices, 10:13]
        # xy lin vel
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        # yaw ang vel
        foot_ang_velocities = torch.square(torch.norm(self.foot_velocities_ang[:, :, 2:] / np.pi, dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(contact_filt * (foot_velocities + foot_ang_velocities), dim=1)
        # print(rew_slip)
        return rew_slip
    
    def _get_heights_at_points(self, points):
        """ Get vertical projected terrain heights at points 
        points: a tensor of size (num_envs, num_points, 2) in world frame
        """
        points = points.clone()
        num_points = points.shape[1]
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(self.num_envs, num_points, device=self.device, requires_grad=False)
        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    # def _reward_feet_clearance_cmd_linear(self):
    #     phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices[:, -2:] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    #     phases_4 = 1 - torch.abs(1.0 - torch.clip((self.foot_indices[:, :] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    #     foot_height = (self.foot_positions[:, -2:, 2]).view(self.num_envs, -1)# - reference_heights

    #     terrain_at_foot_height = self._get_heights_at_points(self.foot_positions[:, -2:, :2])
    #     # target_height = 0.15 * torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device) * phases + 0.02 + terrain_at_foot_height # offset for foot radius 2cm
    #     target_height = self.cfg.rewards.foot_target * phases + 0.02 + terrain_at_foot_height
    #     # print(phases_4)
    #     rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states[:, -2:])
    #     # rew_foot_clearance = torch.exp(-torch.abs(target_height - foot_height) * (1 - self.desired_contact_states[:, -2:]) / 0.01)
    #     condition = self.episode_length_buf > self.cfg.rewards.allow_contact_steps
    #     rew_foot_clearance = rew_foot_clearance * condition.unsqueeze(dim=-1).float()
    #     # print(rew_foot_clearance)
    #     return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_feet_clearance_cmd_linear_4(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices[:, :] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)# - reference_heights
        terrain_at_foot_height = self._get_heights_at_points(self.foot_positions[:, :, :2])
        # target_height = 0.15 * torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device) * phases + 0.02 + terrain_at_foot_height # offset for foot radius 2cm
        target_height = self.commands[:, 9].unsqueeze(1) * phases + 0.03 + terrain_at_foot_height
        # print(target_height, foot_height)
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states[:, :])
        # print(torch.square(target_height - foot_height), rew_foot_clearance)
        # rew_foot_clearance = torch.exp(-torch.abs(target_height - foot_height) * (1 - self.desired_contact_states[:, -2:]) / 0.01)
        # condition = self.episode_length_buf > self.cfg.rewards.allow_contact_steps
        # rew_foot_clearance = rew_foot_clearance * condition.unsqueeze(dim=-1).float()
        return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_feet_clearance_cmd_linear(self):    # already 4, convient for distill check reward term.
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices[:, :] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)# - reference_heights
        terrain_at_foot_height = self._get_heights_at_points(self.foot_positions[:, :, :2])
        if self.cfg.commands.num_commands > 4:
            target_height = self.commands[:, 9].unsqueeze(1) * phases + 0.02 + terrain_at_foot_height
        else:
            target_height = 0.15 * torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device) * phases + 0.02 + terrain_at_foot_height # offset for foot radius 2cm
        # 
        # print(target_height, foot_height)
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states[:, :])
        # print(torch.square(target_height - foot_height), rew_foot_clearance)
        # rew_foot_clearance = torch.exp(-torch.abs(target_height - foot_height) * (1 - self.desired_contact_states[:, -2:]) / 0.01)
        # condition = self.episode_length_buf > self.cfg.rewards.allow_contact_steps
        # rew_foot_clearance = rew_foot_clearance * condition.unsqueeze(dim=-1).float()
        return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_off_ground(self):
        falled_ = self.step_counter > 8
        ## ver1
        # feet_contact = torch.any(self.contact_forces[:, self.feet_indices, 2] > 0.001, dim=1)
        # # print(self.step_counter)
        # pen_off_ground = ((~feet_contact & falled_) * 1.0)
        # # print(feet_contact, pen_off_ground)
        # print(self.root_states[:, 2])
        ## ver2
        feet_contact_f = torch.any(self.contact_forces[:, self.feet_indices[:2], 2] > 0.001, dim=1)
        feet_contact_b = torch.any(self.contact_forces[:, self.feet_indices[2:], 2] > 0.001, dim=1)
        feet_contact_fb = feet_contact_f & feet_contact_b
        pen_off_ground = ((~feet_contact_fb & falled_) * 1.0)
        return pen_off_ground
    
    def _reward_rear_air(self):
        contact = self.contact_forces[:, self.feet_indices[-2:], 2] < 1.
        # init_condition = self.root_states[:, 2] < 0.3
        # init_condition = self.episode_length_buf < self.cfg.rewards.allow_contact_steps
        # reward = (torch.all(contact, dim=1) * (~init_condition) + torch.any(contact, dim=1) * init_condition).float()
        calf_contact = self.contact_forces[:, self.calf_indices[-2:], 2] < 1.
        unhealthy_condition = torch.logical_and(~calf_contact, contact)
        reward = torch.all(contact, dim=1).float() + unhealthy_condition.sum(dim=-1).float()
        return reward
    
    def _reward_stand_air(self):
        stand_air_condition = torch.logical_and(
            torch.logical_and(
                self.episode_length_buf < self.cfg.rewards.allow_contact_steps,
                quat_apply(self.base_quat, self.forward_vec)[:, 2] < 0.9
            ), torch.any(self.foot_positions[:, -2:, 2] > 0.03, dim=1)
        )
        # penalty = torch.mean((self.foot_positions[:, -2:, 2] - 0.03).clamp(min=0) / 0.01, dim=-1)
        return stand_air_condition.float()
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
                                     dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.joint_pos_target[:, :self.num_actuated_dof] - self.last_joint_pos_target[:, :self.num_actuated_dof])
        diff = diff * (self.last_actions[:, :self.num_dof] != 0)  # ignore first step
        return torch.sum(diff, dim=1)
    
    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.joint_pos_target[:, :self.num_actuated_dof] - 2 * self.last_joint_pos_target[:, :self.num_actuated_dof] + self.last_last_joint_pos_target[:, :self.num_actuated_dof])
        diff = diff * (self.last_actions[:, :self.num_dof] != 0)  # ignore first step
        diff = diff * (self.last_last_actions[:, :self.num_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)
    
    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)))
        # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        # base_in_collision = torch.norm(self.contact_forces[:, self.base_contact_indice, :], dim=-1) > 0.001
        # reward = reward * (1 - base_in_collision.float())
        return reward / 4
    
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))
        # print(foot_forces)
        return reward / 4

    
    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        roll_pitch_commands = self.commands[:, 10:12]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.gravity_vec)

        return torch.sum(torch.square(self.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)
    
    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # print(self.foot_positions)
        # nominal positions: [FR, FL, RR, RL]
        if self.cfg.commands.num_commands >= 13:
            desired_stance_width = self.commands[:, 12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        if self.cfg.commands.num_commands >= 14:
            desired_stance_length = self.commands[:, 13:14]
            desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)


        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        # frequencies = self.cfg.commands.default_gait_freq * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        frequencies = self.commands[:, 4]
        # frequencies = self.cfg.commands.default_gait_freq # 3.57#2.5
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1)) #* 0.8

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)
        

        # desired_footsteps_body_frame[:, :, 0] += 0.02
        # print("foot", desired_footsteps_body_frame, footsteps_in_body_frame)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        # base_in_collision = torch.norm(self.contact_forces[:, self.base_contact_indice, :], dim=-1) > 0.001
        # reward = reward * (1 - base_in_collision.float())
        # super()._reward_dof_pos_limits

        return reward
    
    def _render_headless(self):
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 0.5),
                                         gymapi.Vec3(bx, by, bz))
            self.video_frame = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera,
                                                         gymapi.IMAGE_COLOR)
            self.video_frame = self.video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))
            self.video_frames.append(self.video_frame)
    
    def _reward_jump(self):
        reference_heights = 0
        body_height = self.base_pos[:, 2] - reference_heights
        jump_height_target = self.commands[:, 3] + self.cfg.rewards.base_height_target
        reward = - torch.square(body_height - jump_height_target)
        # print(body_height, jump_height_target)
        return reward
    
    def start_recording(self):
        self.complete_video_frames = None
        self.record_now = True


    def pause_recording(self):
        self.complete_video_frames = []
        self.video_frames = []
        self.record_now = False
    
    def get_complete_frames(self):
        if self.complete_video_frames is None:
            return []
        return self.complete_video_frames

    