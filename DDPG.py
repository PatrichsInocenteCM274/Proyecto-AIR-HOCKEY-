import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Important announcement:
# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Extracted from the official repositories of https://github.com/sfujim/TD3, 
# it is not an own version and it is only used for comparison purposes, 
# I give full credit to its creators cited in the paper https://arxiv.org /abs/1509.02971.


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400 + action_dim, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		return self.l3(q)


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0005)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.discount = discount
		self.tau = tau
		self.perdida_critico,self.perdida_actor = [],[]


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=64, done_bool=False):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		if done_bool:
			self.perdida_critico.append(critic_loss)
			self.perdida_actor.append(actor_loss)
			print("almacena perdidas")

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic.pth")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pth")
		
		torch.save(self.actor.state_dict(), filename + "_actor.pth")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pth")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pth"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pth"))
		self.actor_target = copy.deepcopy(self.actor)
		