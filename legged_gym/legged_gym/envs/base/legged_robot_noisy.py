import random

from isaacgym.torch_utils import torch_rand_float, get_euler_xyz, quat_from_euler_xyz, tf_apply
from isaacgym import gymtorch, gymapi, gymutil
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from legged_gym.envs.base.legged_robot_field import LeggedRobotField

class LeggedRobotNoisy(LeggedRobotField):
    """ This class should be independent from the terrain, but depend on the sensors of the parent
    class.
    """

    def clip_position_action_by_torque_limit(self, actions_scaled):
        """ For position control, scaled actions should be in the coordinate of robot default dof pos
        """
        if hasattr(self, "proprioception_output"):
            dof_vel = self.proprioception_output[:, -24:-12] / self.obs_scales.dof_vel
            dof_pos_ = self.proprioception_output[:, -36:-24] / self.obs_scales.dof_pos
        else:
            dof_vel = self.dof_vel
            dof_pos_ = self.dof_pos - self.default_dof_pos
        p_limits_low = (-self.torque_limits) + self.d_gains*dof_vel
        p_limits_high = (self.torque_limits) + self.d_gains*dof_vel
        actions_low = (p_limits_low/self.p_gains) + dof_pos_
        actions_high = (p_limits_high/self.p_gains) + dof_pos_
        actions_scaled_torque_clipped = torch.clip(actions_scaled, actions_low, actions_high)
        return actions_scaled_torque_clipped

    def pre_physics_step(self, actions):
        self.forward_depth_refreshed = False # incase _get_forward_depth_obs is called multiple times
        self.proprioception_refreshed = False
        return_ = super().pre_physics_step(actions)

        if isinstance(self.cfg.control.action_scale, (tuple, list)):
            self.cfg.control.action_scale = torch.tensor(self.cfg.control.action_scale, device= self.sim_device)
        # if getattr(self.cfg.control, "computer_clip_torque", False):
        #     self.actions_scaled = self.actions * self.cfg.control.action_scale
        #     control_type = self.cfg.control.control_type
        #     if control_type == "P":
        #         self.actions_scaled_torque_clipped = self.clip_position_action_by_torque_limit(self.actions_scaled)
        #     else:
        #         raise NotImplementedError
        # else:
        self.actions_scaled_torque_clipped = self.actions * self.cfg.control.action_scale
        
        return return_
    
    def _compute_torques(self, actions):
        """ The input actions will not be used, instead the scaled clipped actions will be used.
        Please check the computation logic whenever you change anything.
        """
        # if not hasattr(self.cfg.control, "motor_clip_torque"):
        #     print("falg has attr")
        #     return super()._compute_torques(actions)
        # else:
        self.pre_physics_step(actions)
        self.actions_scaled_torque_clipped = self.actions * self.cfg.control.action_scale
        actions_scaled_torque_clipped = self.actions_scaled_torque_clipped
        # print("clip_action", self.actions_scaled_torque_clipped)
        # if hasattr(self, "motor_strength"):
        #     actions_scaled_torque_clipped = self.motor_strength * self.actions_scaled_torque_clipped
        #     # print("flag action_scaled", actions_scaled_torque_clipped)
        # else:
        #     actions_scaled_torque_clipped = self.actions_scaled_torque_clipped
        #     print("flag 5")
        control_type = self.cfg.control.control_type
        if control_type == "P":
            # torques = self.p_gains * (actions_scaled_torque_clipped + self.default_dof_pos - self.dof_pos) \
            #     - self.d_gains * self.dof_vel
            torques = self.p_gains * self.Kp_factors * (
                actions_scaled_torque_clipped + self.default_dof_pos - self.dof_pos  + self.motor_offsets) - self.d_gains * self.Kd_factors * self.dof_vel
            # print(self.Kp_factors, self.Kd_factors, self.motor_strengths, self.motor_offsets)
        else:
            raise NotImplementedError
        if self.cfg.control.motor_clip_torque:
            torques = torch.clip(
                torques,
                -self.torque_limits * self.cfg.control.motor_clip_torque,
                self.torque_limits * self.cfg.control.motor_clip_torque,
            )
        # print("torques", torques)
        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
    def post_decimation_step(self, dec_i):
        return_ = super().post_decimation_step(dec_i)
        self.max_torques = torch.maximum(
            torch.max(torch.abs(self.torques), dim= -1)[0],
            self.max_torques,
        )
        self.torque_exceed_count_substep[(torch.abs(self.torques) > self.torque_limits).any(dim= -1)] += 1
        
        ### count how many times in the episode the robot is out of dof pos limit (summing all dofs)
        self.out_of_dof_pos_limit_count_substep += self._reward_dof_pos_limits().int()
        ### or using a1_const.h value to check whether the robot is out of dof pos limit
        # joint_pos_limit_high = torch.tensor([0.802, 4.19, -0.916] * 4, device= self.device) - 0.001
        # joint_pos_limit_low = torch.tensor([-0.802, -1.05, -2.7] * 4, device= self.device) + 0.001
        # self.out_of_dof_pos_limit_count_substep += (self.dof_pos > joint_pos_limit_high.unsqueeze(0)).sum(-1).int()
        # self.out_of_dof_pos_limit_count_substep += (self.dof_pos < joint_pos_limit_low.unsqueeze(0)).sum(-1).int()
        
        return return_
    
    def _fill_extras(self, env_ids):
        return_ = super()._fill_extras(env_ids)
        
        self.extras["episode"]["max_torques"] = self.max_torques[env_ids]
        self.max_torques[env_ids] = 0.
        self.extras["episode"]["torque_exceed_count_substeps_per_envstep"] = self.torque_exceed_count_substep[env_ids] / self.episode_length_buf[env_ids]
        self.torque_exceed_count_substep[env_ids] = 0
        self.extras["episode"]["torque_exceed_count_envstep"] = self.torque_exceed_count_envstep[env_ids]
        self.torque_exceed_count_envstep[env_ids] = 0
        self.extras["episode"]["out_of_dof_pos_limit_count_substep"] = self.out_of_dof_pos_limit_count_substep[env_ids] / self.episode_length_buf[env_ids]
        self.out_of_dof_pos_limit_count_substep[env_ids] = 0

        return return_
    
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

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
        
    def _resample_proprioception_latency(self, env_ids):
        self.current_proprioception_latency[env_ids] = torch_rand_float(
            self.cfg.sensor.proprioception.latency_range[0],
            self.cfg.sensor.proprioception.latency_range[1],
            (len(env_ids), 1),
            device= self.device,
        ).flatten()

    def _resample_forward_camera_latency(self, env_ids):
        self.current_forward_camera_latency[env_ids] = torch_rand_float(
            self.cfg.sensor.forward_camera.latency_range[0],
            self.cfg.sensor.forward_camera.latency_range[1],
            (len(env_ids), 1),
            device= self.device,
        ).flatten()

    def _init_buffers(self):
        return_ = super()._init_buffers()
        all_obs_components = self.all_obs_components

        if "proprioception" in all_obs_components and hasattr(self.cfg.sensor, "proprioception"):
            """ Adding proprioception delay buffer """
            self.cfg.sensor.proprioception.buffer_length = int((self.cfg.sensor.proprioception.latency_range[1] + self.dt) / self.dt)
            self.proprioception_buffer = torch.zeros(
                (
                    self.cfg.sensor.proprioception.buffer_length,
                    self.num_envs,
                    self.get_num_obs_from_components(["proprioception"]),
                ),
                dtype= torch.float32,
                device= self.device,
            )
            self.proprioception_delayed_frames = torch.ones((self.num_envs,), device= self.device, dtype= int) * self.cfg.sensor.proprioception.buffer_length
            self.current_proprioception_latency = torch_rand_float(
                self.cfg.sensor.proprioception.latency_range[0],
                self.cfg.sensor.proprioception.latency_range[1],
                (self.num_envs, 1),
                device= self.device,
            ).flatten()

        if "forward_depth" in all_obs_components and hasattr(self.cfg.sensor, "forward_camera"):
            output_resolution = getattr(self.cfg.sensor.forward_camera, "output_resolution", self.cfg.sensor.forward_camera.resolution)
            self.cfg.sensor.forward_camera.buffer_length = int((self.cfg.sensor.forward_camera.latency_range[1] + self.cfg.sensor.forward_camera.refresh_duration) / self.dt)
            self.forward_depth_buffer = torch.zeros(
                (
                    self.cfg.sensor.forward_camera.buffer_length,
                    self.num_envs, 
                    1,
                    output_resolution[0],
                    output_resolution[1],
                ),
                dtype= torch.float32,
                device= self.device,
            )
            self.forward_depth_delayed_frames = torch.ones((self.num_envs,), device= self.device, dtype= int) * self.cfg.sensor.forward_camera.buffer_length
            self.current_forward_camera_latency = torch_rand_float(
                self.cfg.sensor.forward_camera.latency_range[0],
                self.cfg.sensor.forward_camera.latency_range[1],
                (self.num_envs, 1),
                device= self.device,
            ).flatten()
            if hasattr(self.cfg.sensor.forward_camera, "resized_resolution"):
                self.forward_depth_resize_transform = T.Resize(
                    self.cfg.sensor.forward_camera.resized_resolution,
                    interpolation= T.InterpolationMode.BICUBIC,
                )
        self.contour_detection_kernel = torch.zeros(
            (8, 1, 3, 3),
            dtype= torch.float32,
            device= self.device,
        )
        # emperical values to be more sensitive to vertical edges
        self.contour_detection_kernel[0, :, 1, 1] = 0.5
        self.contour_detection_kernel[0, :, 0, 0] = -0.5
        self.contour_detection_kernel[1, :, 1, 1] = 0.1
        self.contour_detection_kernel[1, :, 0, 1] = -0.1
        self.contour_detection_kernel[2, :, 1, 1] = 0.5
        self.contour_detection_kernel[2, :, 0, 2] = -0.5
        self.contour_detection_kernel[3, :, 1, 1] = 1.2
        self.contour_detection_kernel[3, :, 1, 0] = -1.2
        self.contour_detection_kernel[4, :, 1, 1] = 1.2
        self.contour_detection_kernel[4, :, 1, 2] = -1.2
        self.contour_detection_kernel[5, :, 1, 1] = 0.5
        self.contour_detection_kernel[5, :, 2, 0] = -0.5
        self.contour_detection_kernel[6, :, 1, 1] = 0.1
        self.contour_detection_kernel[6, :, 2, 1] = -0.1
        self.contour_detection_kernel[7, :, 1, 1] = 0.5
        self.contour_detection_kernel[7, :, 2, 2] = -0.5

        self.max_torques = torch.zeros_like(self.torques[..., 0])
        self.torque_exceed_count_substep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of substeps that the torque exceeds the limit
        self.torque_exceed_count_envstep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of envsteps that the torque exceeds the limit
        self.out_of_dof_pos_limit_count_substep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of substeps that the dof pos exceeds the limit
        
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)

        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)

        return return_

    def _reset_buffers(self, env_ids):
        return_ = super()._reset_buffers(env_ids)
        if hasattr(self, "forward_depth_buffer"):
            self.forward_depth_buffer[:, env_ids] = 0.
            self.forward_depth_delayed_frames[env_ids] = self.cfg.sensor.forward_camera.buffer_length
        if hasattr(self, "proprioception_buffer"):
            self.proprioception_buffer[:, env_ids] = 0.
            self.proprioception_delayed_frames[env_ids] = self.cfg.sensor.proprioception.buffer_length
        return return_

    def _draw_debug_vis(self):
        return_ = super()._draw_debug_vis()
        # ljh: changed!
        # if hasattr(self, "forward_depth_output"):
        #     if self.num_envs == 1:
        #         import matplotlib.pyplot as plt
        #         forward_depth_np = self.forward_depth_output[0, 0].detach().cpu().numpy() # (H, W)
        #         plt.imshow(forward_depth_np, cmap= "gray", vmin= 0, vmax= 1)
        #         plt.pause(0.001)
        #     else:
        #         print("LeggedRobotNoisy: More than one robot, stop showing camera image")
        return return_

    """ Steps to simulate stereo camera depth image """
    def _add_depth_contour(self, depth_images):
        mask =  F.max_pool2d(
            torch.abs(F.conv2d(depth_images, self.contour_detection_kernel, padding= 1)).max(dim= -3, keepdim= True)[0],
            kernel_size= self.cfg.noise.forward_depth.contour_detection_kernel_size,
            stride= 1,
            padding= int(self.cfg.noise.forward_depth.contour_detection_kernel_size / 2),
        ) > self.cfg.noise.forward_depth.contour_threshold
        depth_images[mask] = 0.
        return depth_images

    @torch.no_grad()
    def form_artifacts(self,
            H, W, # image resolution
            tops, bottoms, # artifacts positions (in pixel) shape (n_,)
            lefts, rights,
        ):
        """ Paste an artifact to the depth image.
        NOTE: Using the paradigm of spatial transformer network to build the artifacts of the
        entire depth image.
        """
        batch_size = tops.shape[0]
        tops, bottoms = tops[:, None, None], bottoms[:, None, None]
        lefts, rights = lefts[:, None, None], rights[:, None, None]

        # build the source patch
        source_patch = torch.zeros((batch_size, 1, 25, 25), device= self.device)
        source_patch[:, :, 1:24, 1:24] = 1.

        # build the grid
        grid = torch.zeros((batch_size, H, W, 2), device= self.device)
        grid[..., 0] = torch.linspace(-1, 1, W, device= self.device).view(1, 1, W)
        grid[..., 1] = torch.linspace(-1, 1, H, device= self.device).view(1, H, 1)
        grid[..., 0] = (grid[..., 0] * W + W - rights - lefts) / (rights - lefts)
        grid[..., 1] = (grid[..., 1] * H + H - bottoms - tops) / (bottoms - tops)

        # sample using the grid and form the artifacts for the entire depth image
        artifacts = torch.clip(
            F.grid_sample(
                source_patch,
                grid,
                mode= "bilinear",
                padding_mode= "zeros",
                align_corners= False,
            ).sum(dim= 0).view(H, W),
            0, 1,
        )

        return artifacts

    def _add_depth_artifacts(self, depth_images,
            artifacts_prob,
            artifacts_height_mean_std,
            artifacts_width_mean_std,
        ):
        """ Simulate artifacts from stereo depth camera. In the final artifacts_mask, where there
        should be an artifacts, the mask is 1.
        """
        N, _, H, W = depth_images.shape
        def _clip(x, dim):
            return torch.clip(x, 0., (H, W)[dim])

        # random patched artifacts
        artifacts_mask = torch_rand_float(
            0., 1.,
            (N, H * W),
            device= self.device,
        ).view(N, H, W) < artifacts_prob
        artifacts_mask = artifacts_mask & (depth_images[:, 0] > 0.)
        artifacts_coord = torch.nonzero(artifacts_mask).to(torch.float32) # (n_, 3) n_ <= N * H * W
        artifcats_size = (
            torch.clip(
                artifacts_height_mean_std[0] + torch.randn(
                    (artifacts_coord.shape[0],),
                    device= self.device,
                ) * artifacts_height_mean_std[1],
                0., H,
            ),
            torch.clip(
                artifacts_width_mean_std[0] + torch.randn(
                    (artifacts_coord.shape[0],),
                    device= self.device,
                ) * artifacts_width_mean_std[1],
                0., W,
            ),
        ) # (n_,), (n_,)
        artifacts_top_left = (
            _clip(artifacts_coord[:, 1] - artifcats_size[0] / 2, 0),
            _clip(artifacts_coord[:, 2] - artifcats_size[1] / 2, 1),
        )
        artifacts_bottom_right = (
            _clip(artifacts_coord[:, 1] + artifcats_size[0] / 2, 0),
            _clip(artifacts_coord[:, 2] + artifcats_size[1] / 2, 1),
        )
        for i in range(N):
            # NOTE: make sure the artifacts points are as few as possible
            artifacts_mask = self.form_artifacts(
                H, W,
                artifacts_top_left[0][artifacts_coord[:, 0] == i],
                artifacts_bottom_right[0][artifacts_coord[:, 0] == i],
                artifacts_top_left[1][artifacts_coord[:, 0] == i],
                artifacts_bottom_right[1][artifacts_coord[:, 0] == i],
            )
            depth_images[i] *= (1 - artifacts_mask)

        return depth_images
    
    def _recognize_top_down_too_close(self, too_close_mask):
        """ Based on real D435i image pattern, there are two situations when pixels are too close
        Whether there is too-close pixels all the way across the image vertically.
        """
        # vertical_all_too_close = too_close_mask.all(dim= 2, keepdim= True)
        vertical_too_close = too_close_mask.sum(dim= -2, keepdim= True) > (too_close_mask.shape[-2] * 0.6)
        return vertical_too_close
    
    def _add_depth_stereo(self, depth_images):
        """ Simulate the noise from the depth limit of the stereo camera. """
        N, _, H, W = depth_images.shape
        far_mask = depth_images > self.cfg.noise.forward_depth.stereo_far_distance
        too_close_mask = depth_images < self.cfg.noise.forward_depth.stereo_min_distance
        near_mask = (~far_mask) & (~too_close_mask)

        # add noise to the far points
        far_noise = torch_rand_float(
            0., self.cfg.noise.forward_depth.stereo_far_noise_std,
            (N, H * W),
            device= self.device,
        ).view(N, 1, H, W)
        far_noise = far_noise * far_mask
        depth_images += far_noise

        # add noise to the near points
        near_noise = torch_rand_float(
            0., self.cfg.noise.forward_depth.stereo_near_noise_std,
            (N, H * W),
            device= self.device,
        ).view(N, 1, H, W)
        near_noise = near_noise * near_mask
        depth_images += near_noise

        # add artifacts to the too close points
        vertical_block_mask = self._recognize_top_down_too_close(too_close_mask)
        full_block_mask = vertical_block_mask & too_close_mask
        half_block_mask = (~vertical_block_mask) & too_close_mask
        # add artifacts where vertical pixels are all too close
        for pixel_value in random.sample(
                self.cfg.noise.forward_depth.stereo_full_block_values,
                len(self.cfg.noise.forward_depth.stereo_full_block_values),
            ):
            artifacts_buffer = torch.ones_like(depth_images)
            artifacts_buffer = self._add_depth_artifacts(artifacts_buffer,
                self.cfg.noise.forward_depth.stereo_full_block_artifacts_prob,
                self.cfg.noise.forward_depth.stereo_full_block_height_mean_std,
                self.cfg.noise.forward_depth.stereo_full_block_width_mean_std,
            )
            depth_images[full_block_mask] = ((1 - artifacts_buffer) * pixel_value)[full_block_mask]
        # add artifacts where not all the same vertical pixels are too close
        half_block_spark = torch_rand_float(
            0., 1.,
            (N, H * W),
            device= self.device,
        ).view(N, 1, H, W) < self.cfg.noise.forward_depth.stereo_half_block_spark_prob
        depth_images[half_block_mask] = (half_block_spark.to(torch.float32) * self.cfg.noise.forward_depth.stereo_half_block_value)[half_block_mask]

        return depth_images
    
    def _recognize_top_down_seeing_sky(self, too_far_mask):
        N, _, H, W = too_far_mask.shape
        # whether there is too-far pixels with all pixels above it too-far
        num_too_far_above = too_far_mask.cumsum(dim= -2)
        all_too_far_above_threshold = torch.arange(H, device= self.device).view(1, 1, H, 1)
        all_too_far_above = num_too_far_above > all_too_far_above_threshold # (N, 1, H, W) mask
        return all_too_far_above
    
    def _add_sky_artifacts(self, depth_images):
        """ Incase something like ceiling pattern or stereo failure happens. """
        N, _, H, W = depth_images.shape
        
        possible_to_sky_mask = depth_images > self.cfg.noise.forward_depth.sky_artifacts_far_distance
        to_sky_mask = self._recognize_top_down_seeing_sky(possible_to_sky_mask)
        isinf_mask = depth_images.isinf()
        
        # add artifacts to the regions where they are seemingly pointing to sky
        for pixel_value in random.sample(
                self.cfg.noise.forward_depth.sky_artifacts_values,
                len(self.cfg.noise.forward_depth.sky_artifacts_values),
            ):
            artifacts_buffer = torch.ones_like(depth_images)
            artifacts_buffer = self._add_depth_artifacts(artifacts_buffer,
                self.cfg.noise.forward_depth.sky_artifacts_prob,
                self.cfg.noise.forward_depth.sky_artifacts_height_mean_std,
                self.cfg.noise.forward_depth.sky_artifacts_width_mean_std,
            )
            depth_images[to_sky_mask & (~isinf_mask)] *= artifacts_buffer[to_sky_mask & (~isinf_mask)]
            depth_images[to_sky_mask & isinf_mask & (artifacts_buffer < 1)] = 0.
            depth_images[to_sky_mask] += ((1 - artifacts_buffer) * pixel_value)[to_sky_mask]
            pass
        
        return depth_images

    def _crop_depth_images(self, depth_images):
        H, W = depth_images.shape[-2:]
        return depth_images[...,
            self.cfg.sensor.forward_camera.crop_top_bottom[0]: H - self.cfg.sensor.forward_camera.crop_top_bottom[1],
            self.cfg.sensor.forward_camera.crop_left_right[0]: W - self.cfg.sensor.forward_camera.crop_left_right[1],
        ]

    def _normalize_depth_images(self, depth_images):
        depth_images = torch.clip(
            depth_images,
            self.cfg.sensor.forward_camera.depth_range[0],
            self.cfg.sensor.forward_camera.depth_range[1],
        )
        # normalize depth image to (0, 1)
        depth_images = (depth_images - self.cfg.sensor.forward_camera.depth_range[0]) / (
            self.cfg.sensor.forward_camera.depth_range[1] - self.cfg.sensor.forward_camera.depth_range[0]
        )
        return depth_images
    
    @torch.no_grad()
    def _process_depth_image(self, depth_images):
        # depth_images length N list with shape (H, W)
        # reverse the negative depth (according to the document)
        depth_images_ = torch.stack(depth_images).unsqueeze(1).contiguous().detach().clone() * -1
        if hasattr(self.cfg.noise, "forward_depth"):
            if getattr(self.cfg.noise.forward_depth, "countour_threshold", 0.) > 0.:
                depth_images_ = self._add_depth_contour(depth_images_)
            if getattr(self.cfg.noise.forward_depth, "artifacts_prob", 0.) > 0.:
                depth_images_ = self._add_depth_artifacts(depth_images_,
                    self.cfg.noise.forward_depth.artifacts_prob,
                    self.cfg.noise.forward_depth.artifacts_height_mean_std,
                    self.cfg.noise.forward_depth.artifacts_width_mean_std,
                )
            if getattr(self.cfg.noise.forward_depth, "stereo_min_distance", 0.) > 0.:
                depth_images_ = self._add_depth_stereo(depth_images_)
            if getattr(self.cfg.noise.forward_depth, "sky_artifacts_prob", 0.) > 0.:
                depth_images_ = self._add_sky_artifacts(depth_images_)
        # if self.num_envs == 1:
        #     import matplotlib.pyplot as plt
        #     plt.cla()
        #     __depth_images = depth_images_[0, 0].detach().cpu().numpy() # (H, W)
        #     plt.imshow(__depth_images, cmap= "gray",
        #         vmin= self.cfg.sensor.forward_camera.depth_range[0],
        #         vmax= self.cfg.sensor.forward_camera.depth_range[1],
        #     )
        #     plt.draw()
        #     plt.pause(0.001)
        depth_images_ = self._normalize_depth_images(depth_images_)
        depth_images_ = self._crop_depth_images(depth_images_)
        if hasattr(self, "forward_depth_resize_transform"):
            depth_images_ = self.forward_depth_resize_transform(depth_images_)
        return depth_images_.unsqueeze(0) # (1, N, 1, H, W)

    def _get_forward_depth_obs(self, privileged= False):
        if not self.forward_depth_refreshed and hasattr(self.cfg.sensor, "forward_camera") and (not privileged):
            self.forward_depth_buffer = torch.cat([
                self.forward_depth_buffer[1:],
                self._process_depth_image(self.sensor_tensor_dict["forward_depth"]),
            ], dim= 0)
            delay_refresh_mask = (self.episode_length_buf % int(self.cfg.sensor.forward_camera.refresh_duration / self.dt)) == 0
            # NOTE: if the delayed frames is greater than the last frame, the last image should be used.
            frame_select = (self.current_forward_camera_latency / self.dt).to(int)
            self.forward_depth_delayed_frames = torch.where(
                delay_refresh_mask,
                torch.minimum(
                    frame_select,
                    self.forward_depth_delayed_frames + 1,
                ),
                self.forward_depth_delayed_frames + 1,
            )
            self.forward_depth_delayed_frames = torch.clip(
                self.forward_depth_delayed_frames,
                0,
                self.cfg.sensor.forward_camera.buffer_length,
            )
            self.forward_depth_output = self.forward_depth_buffer[
                -self.forward_depth_delayed_frames,
                torch.arange(self.num_envs, device= self.device),
            ].clone()
            self.forward_depth_refreshed = True
        if not hasattr(self.cfg.sensor, "forward_camera") or privileged:
            return super()._get_forward_depth_obs(privileged).reshape(self.num_envs, -1)

        return self.forward_depth_output.flatten(start_dim= 1)

    def _get_proprioception_obs(self, privileged= False):
        if not self.proprioception_refreshed and hasattr(self.cfg.sensor, "proprioception") and (not privileged):
            self.proprioception_buffer = torch.cat([
                self.proprioception_buffer[1:],
                super()._get_proprioception_obs().unsqueeze(0),
            ], dim= 0)
            # NOTE: if the delayed frames is greater than the last frame, the last image should be used. [0.04-0.0075, 0.04+0.0025]
            self.proprioception_delayed_frames = ((self.current_proprioception_latency / self.dt) + 1).to(int)
            self.proprioception_output = self.proprioception_buffer[
                -self.proprioception_delayed_frames,
                torch.arange(self.num_envs, device= self.device),
            ].clone()
            # The last-action is not delayed.
            if getattr(self.cfg.sensor.proprioception, "delay_action_obs", False):
                not_delayed_mask = torch.randint(0, 1, size= (self.num_envs,), device= self.device).bool()
                self.proprioception_output[not_delayed_mask, -12:] = self.proprioception_buffer[-1, not_delayed_mask, -12:]
            else:
                self.proprioception_output[:, -12:] = self.proprioception_buffer[-1, :, -12:]
            self.proprioception_refreshed = True
        if not hasattr(self.cfg.sensor, "proprioception") or privileged:
            return super()._get_proprioception_obs(privileged)

        return self.proprioception_output.flatten(start_dim= 1)

    def get_obs_segment_from_components(self, obs_components):
        obs_segments = super().get_obs_segment_from_components(obs_components)
        if "forward_depth" in obs_components:
            obs_segments["forward_depth"] = (1, *getattr(
                self.cfg.sensor.forward_camera,
                "output_resolution",
                self.cfg.sensor.forward_camera.resolution,
            ))
        return obs_segments
    
    def _reward_exceed_torque_limits_i(self):
        """ Indicator function """
        max_torques = torch.abs(self.substep_torques).max(dim= 1)[0]
        exceed_torque_each_dof = max_torques > self.torque_limits
        exceed_torque = exceed_torque_each_dof.any(dim= 1)
        return exceed_torque.to(torch.float32)
    
    def _reward_exceed_torque_limits_square(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - self.torque_limits
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.square(exceeded_torques).sum(dim= 1).sum(dim= 1)
    
    def _reward_exceed_dof_pos_limits(self):
        return self.substep_exceed_dof_pos_limits.to(torch.float32).sum(dim= -1).mean(dim= -1)













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
        # target_height = 0.15 * torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device) * phases + 0.02 + terrain_at_foot_height # offset for foot radius 2cm
        target_height = self.commands[:, 9].unsqueeze(1) * phases + 0.03 + terrain_at_foot_height
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
        # base_in_collision = torch.norm(self.contact_forces[:, self.base_contact_indice, :], dim=-1) > 0.001
        # reward = reward * (1 - base_in_collision.float())
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
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1)) * 0.8

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)
        

        desired_footsteps_body_frame[:, :, 0] += 0.02
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