# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:16:21 2020

@author: lizijian
"""

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from constants import action_scale, f_minportion, explore_rate

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
        
    def __call__(self):
        x = self.x_prev + self.theta *(self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else \
                      np.zeros_like(self.mu)
        
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.slot_num_memory = np.zeros(self.mem_size)
        
    def store_transition(self, state, action, reward, state_, done, slot_num):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done
        self.slot_num_memory[index] = slot_num
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        slot_num = self.slot_num_memory[batch]
        
        return states, actions, rewards, new_states, terminal, slot_num
    
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:1' if T.cuda.is_available() else 'cpu')
        # change this when run in the device you have 
        self.to(self.device)
        
        
    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)  # weather this should appedned 
        # with a relu is an open debate
        
        # action_value = F.relu(self.action_value(action))
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        
        return state_action_value
    
    def save_checkpoint(self, path):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), os.path.join(path, self.checkpoint_file) if \
               path else self.checkpoint_file)
        
    def load_checkpoint(self, path):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(os.path.join(path, self.checkpoint_file) \
                                    if path else self.checkpoint_file))
        
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:1' if T.cuda.is_available() else 'cpu')
        # change this when run in the device you have
        self.to(self.device)
        
    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))
        
        return x
        
    def save_checkpoint(self, path):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), os.path.join(path, self.checkpoint_file) if \
               path else self.checkpoint_file)
        
    def load_checkpoint(self, path):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(os.path.join(path, self.checkpoint_file) \
                                    if path else self.checkpoint_file))

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, n_actions, env,
                 gamma=0.999, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64, path=None):
        # old gamma = 0.99
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.path = path
        
        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                  n_actions=n_actions, name='Actor')
        
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions,
                                         name='TargetActor')
        
        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                    n_actions=n_actions, name='Critic')
        
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size,
                                           layer2_size, n_actions=n_actions,
                                           name='TargetCritic')
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        
        self.update_network_parameters(tau=1)
        
    def choose_action(self, observation, with_noise = True):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(\
                              self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        if with_noise:
            mu_prime = mu + explor_rate*T.tensor(self.noise(),
                                     dtype=T.float).to(self.actor.device)
            # when adding noise, the bound operation should be implemented
            self.actor.train()
        return self.preprocess(mu_prime.cpu().detach().numpy())
    
            
    def preprocess(self, action):
        action = np.clip(action, -.9999999, .9999999)
        action = np.multiply((action.reshape((-1,4))+1)/2, action_scale)
        action[:,:-1] = action[:,:-1].astype(int)
        action[:,-1] = np.maximum(f_minportion, action[:,-1])
        return action
        
    def remember(self, state, action, reward, new_state, done, slot_num):
        self.memory.store_transition(state, action, reward, new_state, done,
                                     slot_num)
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done,  slot_num = \
                            self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)
        slot_num = T.tensor(slot_num, dtype=T.float).to(self.critic.device)
        
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)
        
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + pow(self.gamma, slot_num[j])*critic_value_[j]*done[j])  # #######################
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)
        
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)
        
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()
                                      
        self.target_critic.load_state_dict(critic_state_dict)
        
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                     (1-tau)*target_actor_dict[name].clone()
                                     
        self.target_actor.load_state_dict(actor_state_dict)
        
    def save_models(self):
        self.actor.save_checkpoint(self.path)
        self.critic.save_checkpoint(self.path)
        self.target_actor.save_checkpoint(self.path)
        self.target_critic.save_checkpoint(self.path)
        
    def load_modules(self):
        self.actor.load_checkpoint(self.path)
        self.critic.load_checkpoint(self.path)
        self.target_actor.load_checkpoint(self.path)
        self.target_critic.load_checkpoint(self.path)

        