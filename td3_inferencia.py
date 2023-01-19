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
import DDPG
import recording_util
import td3_
warnings.filterwarnings("ignore")

def evaluate_policy(policy_right,policy_left,models=1,eval_episodes=0,init=False):

  if policy_right and policy_left:
    print("Comienza Demostración")
    while not env.winner:
      obs_reset,_ = env.reset(scara="all")
      obs_right = obs_reset[0]
      obs_left = obs_reset[1]
      done_right = False
      done_left = False
      while not done_right and not done_left:
        action_right = policy_right.accion(np.array(obs_right))
        if models == 1 : action_left = policy_left.accion(np.array(obs_left))
        else : action_left = policy_left.select_action(np.array(obs_left))
        ob, reward, done, _ = env.step([action_right,action_left,"all"])
        obs_right, reward_right, done_right = ob[0], reward[0], done[0]
        obs_left, reward_left, done_left = ob[1], reward[1], done[1] 
  
  else:
    avg_reward=0
    frames_episode = []
    last_episode_reward= 0.
    obs = env.reset(scara=args.scara)
    done = False
    while not done:
      if args.scara == "right":
        accion = policy_right.accion(np.array(obs))
      if args.scara == "left":
        if models == 1 : accion = policy_left.accion(np.array(obs))
        else : accion = policy_left.select_action(np.array(obs))
      obs, reward, done, _ = env.step([accion,args.scara])
      last_episode_reward += reward
      avg_reward += reward
      font = cv2.FONT_HERSHEY_DUPLEX 
      frames_episode.append(cv2.putText(env.render(),"Current reward: {:.2f} Accumulated reward: {:.2f}".format(reward,last_episode_reward),(12,25), font, 0.40,0.7))
    
    recording_util.recording_video(frames_episode, framerate=60,episode_reward=last_episode_reward,scara=args.scara,episode_num=eval_episodes)
    print("Recompensa de Episodio: ",avg_reward)
    


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", default="") # "" por defecto no usara la GUI, "game" usara GUI    
  parser.add_argument("--scara", default="right")  # "right" mostrará solo scara derecha, "left" mostrará solo scara izquierda
                                                   # "all" mostrará ambas scaras por GUI
  parser.add_argument("--models", default=1, type=int)    # "1" solo modelo td3 para ambas scaras
                                                  # "2" scara left con modelo ddpg y scara right con modelo td3 
                                                   
  args = parser.parse_args()
  mode = args.mode
  scara = args.scara
  models = args.models
  env_name = "SimpleAirHockey-v0"
  seed = 0

  file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
  print ("---------------------------------------")
  print ("Configuración: %s" % (file_name))
  print ("---------------------------------------")

  eval_episodes = 7
  save_env_vid = False

  env = gym.make(env_name)
  if models == 2:
    env.set_models(2)
  if mode == "game" or scara == "all":
    env.set_mode(True)

  max_episode_steps = env.max_steps()
  env.seed(seed)

  torch.manual_seed(seed)
  np.random.seed(seed)
  dimension_estados = env.observation_space.shape[0]
  dimension_acciones = env.action_space.shape[0]
  policy_right = None
  policy_left = None
  if scara == "right" or scara == "all":
    policy_right = td3_.TD3(scara = "right", dimension_estados = dimension_estados , dimension_acciones=dimension_acciones)
    policy_right.cargar_pesos_redes(scara="right")
  if scara == "left" or scara == "all":
    if models == 2:
      print ("---------------------------------------")
      print("Enfrentamiento de Modelo DDPG con TD3")
      print ("---------------------------------------")
      policy_left = DDPG.DDPG(state_dim = dimension_estados, action_dim = dimension_acciones, max_action = 1.0)
      policy_left.load(f"./models_DDPG/DDPG_SimpleAirHockey-v0_0")
    else:
      print ("---------------------------------------")
      print("Enfrentamiento de Modelo TD3 con TD3")
      print ("---------------------------------------")
      policy_left = td3_.TD3(scara = "left", dimension_estados = dimension_estados , dimension_acciones=dimension_acciones)
      policy_left.cargar_pesos_redes(scara="left")
  
  for i in range(eval_episodes):
    evaluate_policy(policy_right,policy_left, models, eval_episodes=i)

    
