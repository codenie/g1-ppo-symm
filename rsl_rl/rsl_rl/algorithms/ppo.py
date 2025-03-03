import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
import numpy as np


class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 obs_symmetry = None,
                 act_symmetry = None,
                 sym_coef = 1.0,
                 ):


        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
       

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam([
                {'params':self.actor_critic.actor.parameters(), 'lr':learning_rate},
                {'params':self.actor_critic.critic.parameters(), 'lr':learning_rate},
                {'params':self.actor_critic.std, 'lr':learning_rate},])
   

        self.num_vae_substeps = 1
        ###  vae_optimizer       
        # self.vae_optimizer = torch.optim.Adam(self.actor_critic.vae.parameters(), lr=1e-3)


        self.transition = RolloutStorage.Transition()

#### ------------ symmetry -------------------------
        # self.sym_coef = sym_coef
        # self.act_sym_mat = torch.zeros((len(act_symmetry), len(act_symmetry)), device=self.device)
        # for i, perm in enumerate(act_symmetry):
        #     self.act_sym_mat[int(abs(perm))][i] = np.sign(perm) 
        
        # self.obs_sym_mat = torch.zeros((len(obs_symmetry), len(obs_symmetry)), device=self.device)
        # for i, perm in enumerate(obs_symmetry):
        #     self.obs_sym_mat[int(abs(perm))][i] = np.sign(perm)  

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, obs_shape, \
                                        critic_obs_shape, action_shape, \
                                        self.device)

    def act(self, obs, critic_obs, obs_history, base_vel):

        # Compute the actions and values
        self.transition.actions= self.actor_critic.act(obs, obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = \
                                self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # need to record obs and critic_obs before env.step()  s_t 
        self.transition.observations = obs.detach()
        self.transition.critic_observations = critic_obs.detach()

###- ------------  vae  ------------------
        self.transition.observation_histories = obs_history.detach()
        self.transition.base_vel = base_vel.detach()

        return self.transition.actions 
    

    def process_env_step(self, rewards, dones, infos, next_obs):

        self.transition.rewards = rewards.detach()
        self.transition.dones = dones.detach()
        self.transition.next_observations = next_obs.detach()
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)  ###  copy_
        self.transition.clear()


    def compute_returns(self, last_critic_obs):

        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)



    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_symmetry_loss = 0

        mean_recons_loss = 0
        mean_vel_loss = 0
        mean_kld_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, \
                advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, \
                dones_batch, obs_history_batch, base_vel_batch, next_obs_batch in generator:


###--------------------   locomotion  ----------------------------         
                self.actor_critic.act(obs_batch, obs_history_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                ### -------------------- symmetry -------------------------
                # mirror_obs_1 = torch.matmul(obs_history_batch[:,92*0:92*1],self.obs_sym_mat)
                # mirror_obs_2 = torch.matmul(obs_history_batch[:,92*1:92*2],self.obs_sym_mat)
                # mirror_obs_3 = torch.matmul(obs_history_batch[:,92*2:92*3],self.obs_sym_mat)
                # mirror_obs_4 = torch.matmul(obs_history_batch[:,92*3:92*4],self.obs_sym_mat)
                # mirror_obs_5 = torch.matmul(obs_history_batch[:,92*4:92*5],self.obs_sym_mat)

                # mirror_obs_history_batch = torch.cat((mirror_obs_1,
                #                                       mirror_obs_2,
                #                                       mirror_obs_3,
                #                                       mirror_obs_4,
                #                                       mirror_obs_5,
                #                             ),dim=-1)
                # mirror_vel_est, mirror_latent = self.actor_critic.vae.sample(mirror_obs_history_batch)
                # mirror_actor_obs = torch.cat((mirror_vel_est, mirror_latent, mirror_obs_1), dim = -1)
                # mirror_act = self.actor_critic.actor(mirror_actor_obs)
                # mirror_mirror_act = torch.matmul(mirror_act,self.act_sym_mat)
                # sym_loss = (mu_batch-mirror_mirror_act).pow(2).mean()

                # KL  更新 学习率
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            # param_group['lr'] = self.learning_rate  
                            if param_group['params'] == self.actor_critic.actor.parameters():
                                param_group['lr'] = self.learning_rate
                            elif param_group['params'] == self.actor_critic.critic.parameters():
                                param_group['lr'] = self.learning_rate
                            elif param_group['params'] == self.actor_critic.std:
                                param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() \
                #                             + self.sym_coef * sym_loss

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() 

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy_loss +=entropy_batch.mean().item()
                # mean_symmetry_loss += sym_loss.item()


                # for epoch in range(self.num_vae_substeps):
                #     #! 需要 next obs
                #     #! 通过 dones 来隔断训练 
                #     vae_loss_dict = self.actor_critic.vae.loss_fn(obs_history_batch, next_obs_batch, base_vel_batch, 1.0)
                #     valid = (dones_batch == 0).squeeze()
                #     vae_loss = torch.mean(vae_loss_dict['loss'][valid])
                #     self.vae_optimizer.zero_grad()
                #     vae_loss.backward()
                #     nn.utils.clip_grad_norm_(self.actor_critic.vae.parameters(), self.max_grad_norm)
                #     self.vae_optimizer.step()
                #     with torch.no_grad():
                #         recons_loss = torch.mean(vae_loss_dict['recons_loss'][valid])
                #         kld_loss = torch.mean(vae_loss_dict['kld_loss'][valid])
                #         vel_loss = torch.mean(vae_loss_dict['vel_loss'][valid])
                #     mean_recons_loss += recons_loss.item()
                #     mean_vel_loss += vel_loss.item()
                #     mean_kld_loss += kld_loss.item()


        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_symmetry_loss /= num_updates


        mean_recons_loss /= (num_updates * self.num_vae_substeps)
        mean_vel_loss /= (num_updates * self.num_vae_substeps)
        mean_kld_loss /= (num_updates * self.num_vae_substeps)

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_symmetry_loss,\
                mean_recons_loss, mean_vel_loss, mean_kld_loss, \



