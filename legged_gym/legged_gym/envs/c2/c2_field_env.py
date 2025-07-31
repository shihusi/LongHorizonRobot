import random
import numpy as np

from isaacgym.torch_utils import torch_rand_float, get_euler_xyz, quat_from_euler_xyz, tf_apply
from isaacgym import gymtorch, gymapi, gymutil
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from isaacgym.torch_utils import *

from legged_gym.envs.base.legged_robot_noisy import LeggedRobotNoisy

class CyberFieldEnv(LeggedRobotNoisy):


    def post_physics_step(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.step_counter += 1
        # print(self.torques.abs().sum())

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

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


        # return super().post_physics_step()
    

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
        

    def _reset_buffers(self, env_ids):
        super()._reset_buffers(env_ids)
        self.action_history[env_ids] = 0
        self.dof_pos_history[env_ids] = 0
    
    def _compute_torques(self, actions):
        return_ = super()._compute_torques(actions)
        actions_scaled = actions * self.cfg.control.action_scale
        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
            # print("flag 1")
        else:
            self.joint_pos_target = actions_scaled + self.default_dof_pos
            # print("flag2")

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

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        if self.cfg.commands.discretize:
            conti_velx_cmd = self.commands[env_ids, 0:1]
            # conti_heading_cmd = self.commands[env_ids, 3:4]
            self.commands[env_ids, 0:1] = torch.sign(conti_velx_cmd) * torch.round(torch.abs(conti_velx_cmd) / 0.1) * 0.1
            # self.commands[env_ids, 3:4] = torch.sign(conti_heading_cmd) * torch.round(torch.abs(conti_heading_cmd) / (np.pi / 12)) * (np.pi / 12)


    def _get_clock_input_obs(self, privileged=False):
        print("clock", self.clock_inputs.abs().sum())
        return self.clock_inputs
    
    def _get_action_history_obs(self, privileged=False):
        print("action his", self.action_history.flatten(start_dim=1).abs().sum())
        return self.action_history.flatten(start_dim=1)
        # return self.last_actions
    
    def _get_dof_pos_history_obs(self, privileged=False):
        print("dof obs", self.dof_pos_history.flatten(start_dim=1).abs().sum())
        return self.dof_pos_history.flatten(start_dim=1)
    
    def _write_action_history_noise(self, noise_vec):
        pass

    def _write_clock_input_noise(self, noise_vec):
        pass

    def _write_dof_pos_history_noise(self, noise_vec):
        pass
        
    def _________compute_observation(self):
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
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.last_actions,
                                    self.clock_inputs
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if True:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if not self.num_privileged_obs is None:
            min_shape = min(self.obs_buf.shape[1], self.privileged_obs_buf.shape[1])
            self.privileged_obs_buf[:, :min_shape] = self.obs_buf[:, :min_shape] # copy content
        if self.num_obs == 48:
            self.obs_buf = self.obs_buf[:, :48]
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        if not self.cfg.env.use_lin_vel:
            self.obs_buf[:, :3] = 0.
        self.obs_super_impl = self.obs_buf
        self.add_noise = add_noise


        # actor obs
        self.obs_buf = self._get_obs_from_components(
            self.cfg.env.obs_components,
            privileged= False,
        )
        # print(self.obs_buf.size())
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
        self.action_history = torch.zeros(self.num_envs, self.action_history_num, 12, dtype=torch.float, device=self.device)
        self.dof_pos_history = torch.zeros(self.num_envs, self.action_history_num, 12, dtype=torch.float, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # if self.cfg.terrain.measure_heights:
        self.height_points = self._init_height_points()
        self.measured_heights = 0
        return_ = super()._init_buffers()
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

        return return_
    
    def check_termination(self):
        return_ = super().check_termination()
        feet_contact = torch.any(self.contact_forces[:, self.feet_indices, 2] > 0.001, dim=1)
        falled_ = self.step_counter > 10
        # print(self.step_counter)
        # pen_off_ground = (~feet_contact & falled_)
        # print(feet_contact, pen_off_ground)
        # self.reset_buf |= pen_off_ground

        return return_

    def _step_contact_targets(self):
        # TODO: fill in reasonable numbers
        # frequencies = 3 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_positions = self.robot_rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        
        if self.cfg.commands.num_commands >= 10: #false here
            frequencies = self.commands[:, 6]
            phases = self.commands[:, 7]
            offsets = self.commands[:, 8]
            bounds = self.commands[:, 9]
        else:
            frequencies = self.cfg.commands.default_gait_freq # 3.57#2.5
            phases = 0.5
            offsets = 0
            bounds = 0
        
        # if mode == "trotting":
        #     phases = 0.5 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # [0, 1]
        #     offsets = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # [0, 1]
        #     bounds = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # [0, 1]
        # elif mode == "bounding":
        #     phases = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     offsets = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     bounds = 0.5 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # elif mode == "pacing":
        #     phases = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     offsets = 0.5 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     bounds = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # elif mode == "pronking":
        #     phases = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     offsets = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     bounds = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # else:
        #     raise NotImplementedError
        durations = 0.5 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # 0.5
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        # TODO: 
        # if self.cfg.commands.pacing_offset:
        #     foot_indices = [self.gait_indices + phases + offsets + bounds,
        #                     self.gait_indices + bounds,
        #                     self.gait_indices + offsets,
        #                     self.gait_indices + phases]
        # else:
        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]
        
        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
        #print("self.foot_indices=",self.foot_indices[0][:2])
        #print("gait_indices=",self.gait_indices[0],"dt,freq=",self.dt,frequencies,self.dt*frequencies)

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

        # print("-----------clock", self.clock_inputs)

        # self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
        # self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
        # self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
        # self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

        # self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
        # self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
        # self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
        # self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

        # von mises distribution
        if hasattr(self.cfg.rewards, "kappa_gait_probs"):
            #print("kappa aaaaaaaaaaaaa")
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
            # print("desired contact state=",self.desired_contact_states[0, :])

        # if self.cfg.commands.num_commands > 9:
        #     self.desired_footswing_height = self.commands[:, 9]
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # print("Command_lin ", self.commands[:, :2])
        # print("real_vel ", self.base_lin_vel[:, :2])
        # print("lin_vel_error", lin_vel_error)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_world_tracking_lin_vel(self):
        lin_vel_error = torch.sum((self.commands[:, :2] - self.root_states[:, 7:9]), dim= 1)
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
    
    def _reward_pen_yaw(self):
        lin_vel_y2 = torch.square(self.base_lin_vel[:, 1])
        ang_vel_xy = 1e-3 * torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        # print(lin_vel_y2, ang_vel_xy)
        # print(0.05 * torch.sum(lin_vel_y2 + ang_vel_xy))
        # # print(self.root_states[:, 2].unsqueeze(1), self.measured_heights)
        

        return 0.05 * torch.sum(lin_vel_y2 + ang_vel_xy)   
    
    def _reward_ang_vel_xyz(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :3]), dim=1)
    
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
    
    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices[:, -2:] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        phases_4 = 1 - torch.abs(1.0 - torch.clip((self.foot_indices[:, :] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_positions[:, -2:, 2]).view(self.num_envs, -1)# - reference_heights

        terrain_at_foot_height = self._get_heights_at_points(self.foot_positions[:, -2:, :2])
        # target_height = 0.15 * torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device) * phases + 0.02 + terrain_at_foot_height # offset for foot radius 2cm
        target_height = self.cfg.rewards.foot_target * phases + 0.02 + terrain_at_foot_height
        # print(phases_4)
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states[:, -2:])
        # rew_foot_clearance = torch.exp(-torch.abs(target_height - foot_height) * (1 - self.desired_contact_states[:, -2:]) / 0.01)
        condition = self.episode_length_buf > self.cfg.rewards.allow_contact_steps
        rew_foot_clearance = rew_foot_clearance * condition.unsqueeze(dim=-1).float()
        # print(rew_foot_clearance)
        return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_feet_clearance_cmd_linear_4(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices[:, :] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)# - reference_heights
        terrain_at_foot_height = self._get_heights_at_points(self.foot_positions[:, :, :2])
        # target_height = 0.15 * torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device) * phases + 0.02 + terrain_at_foot_height # offset for foot radius 2cm
        target_height = self.cfg.rewards.foot_target * phases + 0.03 + terrain_at_foot_height
        # print(target_height)
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
        # print(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1))
        rew = torch.sum(1000 * (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  0.002).clip(0, 10), dim=1)
        # rew *= (torch.norm(self.base_lin_vel[:, :2], dim=1) > 0.1)
        # rew *= (self.root_states[:, 2] > 0.4)
        return rew 
    
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
        base_in_collision = torch.norm(self.contact_forces[:, self.base_contact_indice, :], dim=-1) > 0.001
        reward = reward * (1 - base_in_collision.float())
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
        if self.cfg.commands.num_commands >= 12:
            desired_stance_width = self.commands[:, 10]
            desired_stance_length = self.commands[:, 11]
            desired_ys_nom = torch.stack([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=-1)
            desired_xs_nom = torch.stack([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=-1)
        else:
            desired_stance_width = 0.25 # 0.28
            desired_stance_length = 0.33
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], 
                                        device=self.device).unsqueeze(0)
            desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], 
                                        device=self.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.cfg.commands.default_gait_freq * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # frequencies = self.commands[:, 6]
        # frequencies = self.cfg.commands.default_gait_freq # 3.57#2.5
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        base_in_collision = torch.norm(self.contact_forces[:, self.base_contact_indice, :], dim=-1) > 0.001
        reward = reward * (1 - base_in_collision.float())
        # super()._reward_dof_pos_limits

        return reward
    
    