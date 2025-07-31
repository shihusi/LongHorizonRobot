import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

import rsl_rl.algorithms as algorithms
import rsl_rl.modules as modules
from rsl_rl.env import VecEnv

import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import json
import shutil

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class EasyEnv(VecEnv):
    def __init__(self, cfg, device="cuda:0"):
        self.num_envs = cfg.env.num_envs
        self.num_obs =  229 # cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.device = device

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}
    
    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        pass
        # raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        return self.obs_buf, self.privileged_obs_buf
        # pass
        # self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        # return obs, privileged_obs

    def step(self, actions):
        pass

    def render(self, sync_frame_time=True):
        pass
    

class EasyTaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        # print("+++++++++++++env_near", env_cfg.env.num_observations)
        env = EasyEnv(cfg=env_cfg)
        # print("++++++++", env.num_obs, env.num_privileged_obs)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default", env_cfg=None, save_cfg= True) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.
            env_cfg (Dict, optional): for logging

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

        log_cfg_dict = dict()
        train_cfg_dict = class_to_dict(train_cfg)
        log_cfg_dict.update(train_cfg_dict)
        if not env_cfg is None:
            env_cfg_dict = class_to_dict(env_cfg)
            log_cfg_dict.update(env_cfg_dict)

        runner = build_runner(
            getattr(train_cfg, "runner_class_name", "OnPolicyRunner"),
            env,
            train_cfg_dict,
            log_dir,
            device=args.rl_device,
        )
        
        if save_cfg:
            os.makedirs(log_dir, exist_ok= True)
            with open(os.path.join(log_dir, "config.json"), "w") as f:
                json.dump(log_cfg_dict, f, indent= 4)
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            train_cfg.runner.load_run = "Nov17_09-50-15_WalkingBase_aScale0.5"
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            if save_cfg:
                shutil.copyfile(resume_path, os.path.join(log_dir, os.path.basename(resume_path)))
            runner.load(resume_path)
        return runner, train_cfg

# make global task registry
easy_task_registry = EasyTaskRegistry()





        




class Easyrunner():

    def __init__(self):
        actor_critic = modules.build_actor_critic(
            self.env,
            self.cfg["policy_class_name"],
            self.policy_cfg,
        ).to(self.device)
        alg_class = getattr(algorithms, self.cfg["algorithm_class_name"]) # PPO
        self.alg: algorithms.PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)


    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference