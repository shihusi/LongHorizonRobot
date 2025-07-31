# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
from collections import OrderedDict
import os
import json
import time
import numpy as np
np.float = np.float32
import isaacgym
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import update_class_from_dict
from legged_gym.utils.observation import get_obs_slice
from legged_gym.debugger import break_into_debugger
from legged_gym.utils.record_utils import save_object_to_json

import numpy as np
import torch

def create_recording_camera(gym, env_handle,
        resolution= (2560, 1440),
        h_fov= 86,
        actor_to_attach= None,
        transform= None, # related to actor_to_attach
    ):
    camera_props = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.width = resolution[0]
    camera_props.height = resolution[1]
    camera_props.horizontal_fov = h_fov
    camera_handle = gym.create_camera_sensor(env_handle, camera_props)
    if actor_to_attach is not None:
        gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_to_attach,
            transform,
            gymapi.FOLLOW_POSITION,
        )
    elif transform is not None:
        gym.set_camera_transform(
            camera_handle,
            env_handle,
            transform,
        )
    return camera_handle

@torch.no_grad()
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    if env_cfg.terrain.selected == "BarrierTrack":
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
        env_cfg.env.episode_length_s = 20
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
    else:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
        env_cfg.env.episode_length_s = 60
        env_cfg.terrain.terrain_length = 8
        env_cfg.terrain.terrain_width = 8
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.init_base_pos_range = dict(
        x= [0.6, 0.6],
        y= [-0.05, 0.05],
    )
    env_cfg.termination.termination_terms = []
    env_cfg.termination.timeout_at_border = False
    env_cfg.termination.timeout_at_finished = False
    train_cfg.runner.resume = True
    train_cfg.runner_class_name = "OnPolicyRunner"

    
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    obs = env.get_observations()
    critic_obs = env.get_privileged_observations()

    if RECORD_FRAMES:


        camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        camera_vel = np.array([1., 1., 0.])
        camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
        img_idx = 0

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 640
        camera_properties.height = 480
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(-0.4, -1.2, 1.0)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0, 0, 1),
                                                    np.deg2rad(110))
        camera_offset = gymapi.Vec3(-0.5, -0.5, 0.3)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(45))
        
        camera_offset = env_cfg.render_config.camera_offset
        camera_rotation = env_cfg.render_config.camera_rotation
        
        actor_handle = env.gym.get_actor_handle(env.envs[0], env.cfg.render_config.camera_attach_actor_id)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)
        # env.gym.set_camera_location(h1, env.envs[0], gymapi.Vec3(env.env_origins[0, 0] + 2.6 , env.env_origins[0, 1] - 1.6 +0.3, env.env_origins[0, 2] + 1.4), gymapi.Vec3(env.env_origins[0, 0] + 2.6, env.env_origins[0, 1] + 0.3, 0))

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 4 # which joint is used for logging
    stop_state_log = 512 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([0.6, 0., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    camera_follow_id = 0 # only effective when CAMERA_FOLLOW
    img_idx = 0

    if hasattr(env, "motor_strength"):
        print("motor_strength:", env.motor_strength[robot_index].cpu().numpy().tolist())
    print("torque_limits:", env.torque_limits)
    start_time = time.time_ns()
    vertices = np.zeros((2, 3))
    vertices[0, 2] = 1
    color = np.zeros(3)
    color[2] = 1
    # env.gym.add_lines(h1, env, 1, vertices, color)
    export_root_name =  os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    os.makedirs(export_root_name, exist_ok=True)
    for i in range(10*int(env.max_episode_length)):
        if "obs_slice" in locals().keys():
            obs_component = obs[:, obs_slice[0]].reshape(-1, *obs_slice[1])
            print(obs_component[robot_index])
        obs, critic_obs, rews, dones, infos = env.step()

        if RECORD_FRAMES:
            img_idx += 1
            if True: #img_idx % 2 == 0:
                name = str(img_idx).zfill(4)
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', name + ".png")
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                # print("new", h1)
                env.gym.write_camera_image_to_file(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR, filename)
                print(filename)
            # img_idx += 1 
        if MOVE_CAMERA:
            if CAMERA_FOLLOW:
                camera_position[:] = env.root_states[camera_follow_id, :3].cpu().numpy() - camera_direction
            else:
                camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)


        if i < stop_state_log:
            if torch.is_tensor(env.cfg.control.action_scale):
                action_scale = env.cfg.control.action_scale.detach().cpu().numpy()[joint_index]
            else:
                action_scale = env.cfg.control.action_scale
            base_roll = get_euler_xyz(env.base_quat)[0][robot_index].item()
            base_pitch = get_euler_xyz(env.base_quat)[1][robot_index].item()
            if base_pitch > torch.pi: base_pitch -= torch.pi * 2

        elif i==stop_state_log:
            logger.plot_states()
            env._get_terrain_curriculum_move(torch.tensor([0], device= env.device))
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        
        if i % 100 == 0:
            print("frame_rate:" , 100/(time.time_ns() - start_time) * 1e9, 
                  "command_x:", env.commands[robot_index, 0],
            )
            start_time = time.time_ns()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = True
    MOVE_CAMERA = True
    CAMERA_FOLLOW = True
    args = get_args()
    play(args)
