import numpy as np
import torch
import gym
import pickle
import argparse
import os
import simple_air_hockey
import utils
import DDPG
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from IPython.display import HTML
from IPython.display import display
import cv2
import warnings
warnings.filterwarnings('ignore')

# Important announcement:
# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Extracted from the official repositories of https://github.com/sfujim/TD3, 
# it is not an own version and it is only used for comparison purposes, 
# I give full credit to its creators cited in the paper https://arxiv.org /abs/1509.02971.


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	print("--------------Evaluacion---------------")
	avg_reward = 0.
	for num_episode in range(eval_episodes):
		frames_episode = []
		last_episode_reward= 0.
		state, done = env.reset(scara=args.scara), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = env.step([action,args.scara])
			avg_reward += reward
			last_episode_reward += reward

		print(f'Disco coords: {env.coords()} Reward: {last_episode_reward:.2f}')
	avg_reward /= eval_episodes

	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")

	return avg_reward

avg_reward_list = []
ep_reward_list = []
avg_reward = 0
cont = 0


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="DDPG")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="SimpleAirHockey-v0")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=5e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=2.5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.9999)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.1)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.3)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=1, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", default="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--scara", default="left")					# "right" seleccionara entrenamiento de scara derecha
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model:
		if not os.path.exists("./models_DDPG"):
			os.makedirs("./models_DDPG")
		if not os.path.exists("./replaybuffer"):
			os.makedirs("./replaybuffer")
		
	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	
	policy = DDPG.DDPG(**kwargs)
    
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    
	if args.load_model != "":
		args.start_timesteps = 0
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models_DDPG/{policy_file}")
		replay_buffer.load()
		np.random.set_state(pickle.load(open("./replaybuffer/random_state.pkl", "rb")))
		env.random_set_state(pickle.load(open("./replaybuffer/random_action.pkl", "rb")))
        
	episode_num = 0
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]
	#evaluations = [np.load("./results/%s.npy" % (file_name))]
    
	state, done = env.reset(scara=args.scara), False
	episode_reward = 0
	episode_timesteps = 0
	

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step([action,args.scara])
		
		done_bool = float(done) if episode_timesteps < env.max_steps() else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size, done_bool)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			if t > args.start_timesteps and cont >= 2:
				ep_reward_list.append(episode_reward)
				avg_reward = np.mean(ep_reward_list[-100:])
				avg_reward_list.append(avg_reward)
			
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Disco coords: {env.coords()} Reward: {episode_reward:.3f} Avg_reward: {avg_reward:.3f}")
			# Reset environment
			state, done = env.reset(scara=args.scara), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			cont = cont +1
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: 
				policy.save(f"./models_DDPG/{file_name}")
				pickle.dump(avg_reward_list, file = open("./results/avg_reward_ddpg_"+str(args.scara)+".pkl", "wb"))
				pickle.dump(ep_reward_list, file = open("./results/ep_reward_ddpg_"+str(args.scara)+".pkl", "wb"))
				pickle.dump(policy.perdida_actor, file = open("./results/actor_loss_ddpg_"+str(args.scara)+".pkl", "wb"))
				pickle.dump(policy.perdida_critico, file = open("./results/critico_loss_ddpg_"+str(args.scara)+".pkl", "wb"))
				
				
