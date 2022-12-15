from IPython.display import display
import os
import simple_air_hockey
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from IPython.display import HTML
from IPython.display import display
import cv2
import argparse
import warnings
warnings.filterwarnings("ignore")

class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x)) 
    return x

class Critic(nn.Module):

  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Definimos el primero de los Críticos como red neuronal profunda
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)
    # Definimos el segundo de los Críticos como red neuronal profunda
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u):
    xu = torch.cat([x, u], 1)
    # Propagación hacia adelante del primero de los Críticos
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    # Propagación hacia adelante del segundo de los Críticos
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2
  
  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1

# Selección del dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construir todo el proceso de entrenamiento en una clase
class TD3(object):

  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state):
    state = torch.Tensor(state.reshape(1, -1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()

  # Método para cargar el modelo entrenado
  def load(self, filename, directory,scara):
    print("Cargando modelo guardado")
    self.actor.load_state_dict(torch.load("%s/%s_actor_%s.pth" % (directory, filename,scara)))
    self.critic.load_state_dict(torch.load("%s/%s_critic_%s.pth" % (directory, filename,scara)))
    
def display_video(frames, framerate=60, episode_reward=0,episode_num=0):
  """Generates video from `frames`.

  Args:
    frames (ndarray): Array of shape (n_frames, height, width, 3).
    framerate (int): Frame rate in units of Hz.

  Returns:
    Display object.
  """
  height, width, _ = frames[0].shape
  print("Grabando Video")
  dpi = 70
  orig_backend = matplotlib.get_backend()
  matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
  fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
  matplotlib.use(orig_backend)  # Switch back to the original backend.
  ax.set_axis_off()
  ax.set_aspect('equal')
  ax.set_position([0, 0, 1, 1])
  im = ax.imshow(frames[0])
  title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
  def update(frame):
    im.set_data(frame)
    title.set_text(u"Episode Num: {} Reward: {}".format(episode_num,episode_reward))
    return im,title

  interval = 1000/framerate
  anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                  interval=interval, blit=True, repeat=False)
                                  
  writervideo = animation.FFMpegWriter(fps=framerate) 
  anim.save("video_inferencia_"+str(args.scara)+str(episode_num)+".mp4", writer=writervideo) 

"""## Hacemos una función que evalúa la política calculando su recompensa promedio durante 10 episodios"""

def evaluate_policy(policy_right,policy_left,eval_episodes=7):

  if policy_right and policy_left:
    print("Comienza Demostración")
    while not env.winner:
      obs_reset,_ = env.reset(scara="all")
      obs_right = obs_reset[0]
      obs_left = obs_reset[1]
      done_right = False
      done_left = False
      while not done_right and not done_left:
        action_right = policy_right.select_action(np.array(obs_right))
        action_left = policy_left.select_action(np.array(obs_left))
        ob, reward, done, _ = env.step([action_right,action_left,"all"])
        obs_right, reward_right, done_right = ob[0], reward[0], done[0]
        obs_left, reward_left, done_left = ob[1], reward[1], done[1] 
  
  else:
    avg_reward=0
    for num_episode in range(eval_episodes):
      frames_episode = []
      last_episode_reward= 0.
      obs = env.reset(scara=args.scara)
      done = False
      while not done:
        if args.scara == "right":
          action = policy_right.select_action(np.array(obs))
        if args.scara == "left":
          action = policy_left.select_action(np.array(obs))
        obs, reward, done, _ = env.step([action,args.scara])
        last_episode_reward += reward
        avg_reward += reward
        font = cv2.FONT_HERSHEY_DUPLEX 
        frames_episode.append(cv2.putText(env.render(),"Current reward: {:.2f} Accumulated reward: {:.2f}".format(reward,last_episode_reward),(12,25), font, 0.40,0.7))
      display_video(frames_episode, framerate=60,episode_reward=last_episode_reward,episode_num=num_episode)
    avg_reward /= eval_episodes
    print("Recompensa Promedio de episodios: ",avg_reward)
    


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", default="") # "" por defecto no usara la GUI, "game" usara GUI    
  parser.add_argument("--scara", default="right")  # "right" mostrará solo scara derecha, "left" mostrará solo scara izquierda
                                                   # "all" mostrará ambas scaras por GUI
  args = parser.parse_args()
  mode = args.mode
  scara = args.scara
  env_name = "SimpleAirHockey-v0"
  seed = 0

  file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
  print ("---------------------------------------")
  print ("Configuración: %s" % (file_name))
  print ("---------------------------------------")

  eval_episodes = 7
  save_env_vid = False

  env = gym.make(env_name)
  if mode == "game" or scara == "all":
    env.set_mode(True)

  max_episode_steps = env.max_steps()
  env.seed(seed)

  torch.manual_seed(seed)
  np.random.seed(seed)
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]
  max_action = float(env.action_space.high[0])
  policy_right = None
  policy_left = None
  if scara == "right" or scara == "all":
    policy_right = TD3(state_dim, action_dim, max_action)
    policy_right.load(file_name, './pytorch_models',scara="right")
  if scara == "left" or scara == "all":
    policy_left = TD3(state_dim, action_dim, max_action)
    policy_left.load(file_name, './pytorch_models',scara="left")
  evaluate_policy(policy_right,policy_left, eval_episodes=eval_episodes)
