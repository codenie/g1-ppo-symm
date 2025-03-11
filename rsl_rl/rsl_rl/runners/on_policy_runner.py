import time
import os
import numpy as np
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticSymmetry
from rsl_rl.env import VecEnv
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from rsl_rl.utils.torch_utils import preserve_training_state

actor_critic_cls_dict = {
    'ActorCritic': ActorCritic,
    'ActorCriticSymmetry': ActorCriticSymmetry,
}

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):


        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        self.num_vae_encoder_output = None # 19 # 该参数不应该在PPO中使用.
        self.num_obs_step = env.num_obs_step
        self.num_critic_obs = env.num_critic_obs
        self.num_history = env.num_obs_history
        
        assert self.cfg["policy_class_name"] in actor_critic_cls_dict, "ActorCritic类型不在可用列表中"
        
        actor_critic: ActorCritic = actor_critic_cls_dict[self.cfg["policy_class_name"]](
                                                num_vae=self.num_vae_encoder_output,
                                                num_obs_step=self.num_obs_step,
                                                num_critic_obs=self.num_critic_obs,
                                                num_history= self.num_history,
                                                num_actions=self.env.num_actions,
                                                actor_hidden_dims=self.policy_cfg["actor_hidden_dims"],
                                                critic_hidden_dims=self.policy_cfg["critic_hidden_dims"],
                                                activation='elu',
                                                init_noise_std=self.policy_cfg["init_noise_std"]
                                                ).to(self.device)

        # 指定使用PPO
        # print(f"[INFO] 指定使用PPO algorithm.")
        assert self.cfg["algorithm_class_name"] == 'PPO', "指定使用PPO, 而当前设置没有使用该算法."
        self.alg: PPO = PPO(actor_critic, device=self.device, **self.alg_cfg)
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, \
                        self.num_obs_step, self.num_critic_obs, self.env.num_actions)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
        _, _= self.env.reset() 
        
        # log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.cfg['experiment_name'])
        # resume_path = get_load_path(log_root, load_run=self.cfg['retrain_load_run'], checkpoint= self.cfg['retrain_checkpoint'])
        # print(f"Loading model from: {resume_path}")
        # self.load(resume_path)
        # self.current_learning_iteration = 0
        

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        ### NOTE very useful
        self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs, critic_obs = self.env.get_observations()
        obs_history, base_vel = self.env.get_extra_info()

        # obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        
        
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            # self.alg.actor_critic.train() # switch to train mode (for dropout for example)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    actions = self.alg.act(obs, critic_obs, obs_history, base_vel)
                    obs, critic_obs, rewards, dones, infos = self.env.step(actions)  

                    obs_history, base_vel = self.env.get_extra_info()
                    # obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        
                    self.alg.process_env_step(rewards, dones, infos, obs)

                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

                
            mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_symmetry_loss,\
                mean_recons_loss, mean_vel_loss, mean_kld_loss = self.alg.update()

                
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))




    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]: ### name
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']: ### num_steps_per_env
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor) ### ping jun yi bu de  reward
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()

        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/entropy', locs['mean_entropy_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_symmetry_loss', locs['mean_symmetry_loss'], locs['it'])

        self.writer.add_scalar('Loss/recons', locs['mean_recons_loss'], locs['it'])
        self.writer.add_scalar('Loss/vel', locs['mean_vel_loss'], locs['it'])
        self.writer.add_scalar('Loss/kld', locs['mean_kld_loss'], locs['it'])

        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])


        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])



        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path:str, infos=None):
        """保存模型的权重。 如果使用了ESCNN模型，注意保存&load时都要切换为 training 状态.
            Ref. https://quva-lab.github.io/escnn/api/escnn.nn.html?highlight=state_dict    

        Args:
            path (_type_): 保存位置
            infos (_type_, optional): 可选的信息，一般为None. Defaults to None.
        """
        with preserve_training_state(self.alg.actor_critic):
            self.alg.actor_critic.eval() # 注意ECSNN网络一定要在eval模式下进行保存
            torch.save({
                'model_state_dict': self.alg.actor_critic.state_dict(),  
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'iter': self.current_learning_iteration,
                'infos': infos,
                }, path)
        # 
        
    def load(self, path:str, load_optimizer:bool=True):
        """加载模型权重。同save函数要求要在eval模式下进行。ESCNN文档

        Args:
            path (str): 文件路径
            load_optimizer (bool, optional): 是否加载optimizer. Defaults to True.

        Returns:
            _type_: _description_
        """
        with preserve_training_state(self.alg.actor_critic):
            # 注意一定要在eval模式下加载这个网络！ 这是ESCNN的要求
            self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
            
            loaded_dict = torch.load(path)
            self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict']) # TODO: 加载错误
            if load_optimizer:
                self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.current_learning_iteration = loaded_dict['iter']

            # with torch.no_grad():
            #     self.alg.actor_critic.std.copy_(torch.ones_like(self.alg.actor_critic.std)*0.2)

        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference






