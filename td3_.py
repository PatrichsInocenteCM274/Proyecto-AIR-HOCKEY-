# -*- coding: utf-8 -*-

import os
import sys
import argparse
import simple_air_hockey
from PIL import ImageDraw
import time
from datetime import datetime
import pytz
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
import pickle
import warnings
import time
warnings.filterwarnings("ignore")
America = pytz.timezone("America/New_York")

"""## Configuramos los parámetros"""
print("Iniciando TD3__10:.py")
env_name = "SimpleAirHockey-v0" # Nombre del entorno (puedes indicar cualquier entorno continuo que quieras probar aquí)
seed = 0 # Valor de la semilla aleatoria
start_timesteps = 5e3 # Número de of iteraciones/timesteps durante las cuales el modelo elige una acción al azar, y después de las cuales comienza a usar la red de políticas
eval_freq = 2.5e3 # Con qué frecuencia se realiza el paso de evaluación (después de cuántos pasos timesteps)
max_timesteps = 10e6 # Número total de iteraciones/timesteps
save_models = True # Check Boolean para saber si guardar o no el modelo pre-entrenado
expl_noise = 0.1 # Ruido de exploración: desviación estándar del ruido de exploración gaussiano
batch_size = 100 # Tamaño del bloque
discount = 0.999 # Factor de descuento gamma, utilizado en el cáclulo de la recompensa de descuento total
tau = 0.005 # Ratio de actualización de la red de objetivos
policy_noise = 0.2 #0.2 # Desviación estándar del ruido gaussiano añadido a las acciones para fines de exploración
noise_clip = 0.5 # Valor máximo de ruido gaussiano añadido a las acciones (política)
policy_freq = 2 # Número de iteraciones a esperar antes de actualizar la red de políticas (actor modelo)
learning_rate = 0.0005 # Factor de Aprendizaje para el optimizador
load = False
#scara = 'left'
#scara = 'right'

"""## Inicializamos las variables"""

total_timesteps = 0
timesteps_since_eval = 0
cont = 0
episode_num = 0
done = True
t0 = time.time()



"""## Paso 1: Inicializamos la memoria de la repetición de experiencias"""

class ReplayBuffer(object):
  def __init__(self, max_size=2e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage)== self.max_size: 
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)
      
  def save(self):
    pickle.dump(self.ptr, file = open("./replaybuffer/ptr_"+str(args.scara)+".pkl", "wb"))
    np.save("./replaybuffer/storage_"+str(args.scara)+".npy", self.storage)
    print("Replay Buffer Guardado: Indice de Replay Buffer: ",self.ptr,"Tamaño de Replay Buffer: ",len(self.storage))
    now = datetime.now(America)
    print("Fecha: ",now.date(),"Hora: ",now.time())

  def load(self):
    self.ptr  = pickle.load(open("./replaybuffer/ptr_"+str(args.scara)+".pkl", "rb"))
    self.storage = np.load("./replaybuffer/storage_"+str(args.scara)+".npy",allow_pickle=True).tolist()
    print("Replay Buffer Cargado: Indice de Replay Buffer: ",self.ptr,"Tamaño de Replay Buffer: ",len(self.storage))

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size = batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind:
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy = False))
      batch_next_states.append(np.array(next_state, copy = False))
      batch_actions.append(np.array(action, copy = False))
      batch_rewards.append(np.array(reward, copy = False))
      batch_dones.append(np.array(done, copy = False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

"""## Paso 2: Construimos una red neuronal para el **actor del modelo** y una red neuronal para el **actor del objetivo**"""

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
    x = self.max_action*torch.tanh(self.layer_3(x))
    #x = x.squeeze()
    #x = torch.stack([self.max_action[0] * (x[0]+1)/2,self.max_action[1] * x[1]])
    #x = torch.unsqueeze(x, 0)
    return x

"""
## Paso 3: Construimos dos redes neuronales para los dos **críticos del modelo** y dos redes neuronales para los dos **críticos del objetivo**"""

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

"""## Pasos 4 a 15: Proceso de Entrenamiento"""

# Selección del dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construir todo el proceso de entrenamiento en una clase
class TD3(object):

  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=learning_rate)
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=learning_rate)
    self.max_action = max_action

  def select_action(self, state):
    state = torch.Tensor(state.reshape(1, -1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clipping=0.5, policy_freq=2):
    for it in range(iterations):
      
      # Paso 4: Tomamos una muestra de transiciones (s, s’, a, r) de la memoria.
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Paso 5: A partir del estado siguiente s', el Actor del Target ejecuta la siguiente acción a'.
      next_action = self.actor_target(next_state)

      # Paso 6: Añadimos ruido gaussiano a la siguiente acción a' y lo cortamos para tenerlo en el rango de valores aceptado por el entorno.
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device) 
      noise = noise.clamp(-noise_clipping, noise_clipping)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

      # Paso 7: Los dos Críticos del Target toman un par (s’, a’) como entrada y devuelven dos Q-values Qt1(s’,a’) y Qt2(s’,a’) como salida.
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)

      # Paso 8: Nos quedamos con el mínimo de los dos Q-values: min(Qt1, Qt2). Representa el valor aproximado del estado siguiente.
      target_Q = torch.min(target_Q1, target_Q2)

      # Paso 9: Obtenemos el target final de los dos Crítico del Modelo, que es: Qt = r + γ * min(Qt1, Qt2), donde γ es el factor de descuento.
      target_Q = reward + ((1-done) * discount * target_Q).detach()

      # Paso 10: Los dos Críticos del Modelo toman un par (s, a) como entrada y devuelven dos Q-values Q1(s,a) y Q2(s,a) como salida.
      current_Q1, current_Q2 = self.critic(state, action)

      # Paso 11: Calculamos la pérdida procedente de los Crítico del Modelo: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

      # Paso 12: Propagamos hacia atrás la pérdida del crítico y actualizamos los parámetros de los dos Crítico del Modelo con un SGD.
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

      # Paso 13: Cada dos iteraciones, actualizamos nuestro modelo de Actor ejecutando el gradiente ascendente en la salida del primer modelo crítico.
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()##OJO ME DEJÉ EL LOSS
        self.actor_optimizer.step()

        # Paso 14: Todavía cada dos iteraciones, actualizamos los pesos del Actor del Target usando el promedio Polyak.
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

        # Paso 15: Todavía cada dos iteraciones, actualizamos los pesos del target del Crítico usando el promedio Polyak.
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

  # Método para guardar el modelo entrenado
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), "%s/%s_actor_%s.pth" % (directory, filename, args.scara))
    torch.save(self.critic.state_dict(), "%s/%s_critic_%s.pth" % (directory, filename, args.scara))

  # Método para cargar el modelo entrenado
  def load(self, filename, directory):
    print("Cargando modelo guardado")
    self.actor.load_state_dict(torch.load("%s/%s_actor_%s.pth" % (directory, filename, args.scara)))
    self.critic.load_state_dict(torch.load("%s/%s_critic_%s.pth" % (directory, filename, args.scara)))

"""# Visualizador de Acciones"""

def display_video(frames, framerate=60, episode_reward=0.):
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
  plt.rcParams['font.family'] = 'serif'
  title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center",size=10,fontdict=None)
  
  def update(frame):
    im.set_data(frame)
    title.set_text(u"Total Timesteps: {} Episode Num: {} Reward: {:.2f}".format(total_timesteps,episode_num,episode_reward))
    return im,title

  interval = 1000/framerate
  anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                  interval=interval, blit=True, repeat=False)
                                  
  writervideo = animation.FFMpegWriter(fps=60) 
  anim.save("video"+str(total_timesteps)+".mp4", writer=writervideo) 

"""## Hacemos una función que evalúa la política calculando su recompensa promedio durante 10 episodios"""
import cv2

def evaluate_policy(policy, eval_episodes=10):
  print ("------------------------------------------------")  
  #env.seed(0)
  avg_reward = 0.
  
  print("Comienza evaluacion")
  for num_episode in range(eval_episodes):
    frames_episode = []
    last_episode_reward= 0.
    obs = env.reset(scara=args.scara)
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step([action,args.scara])
      last_episode_reward += reward
      avg_reward += reward
    print(f'Disco coords: {env.coords()} Reward: {last_episode_reward:.2f}')
  avg_reward /= eval_episodes
  env.reset_steps_linea()
  env.reset_step_anotacion()
  
  print ("Recompensa promedio en el paso de Evaluación: %f" % (avg_reward))
  print ("------------------------------------------------")
  return avg_reward

"""## Creamos un nombre de archivo para los dos modelos guardados: los modelos Actor y Crítico."""

file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Configuración: %s" % (file_name))
print ("---------------------------------------")

"""## Creamos una carpeta dentro de la cual se guardarán los modelos entrenados"""

if not os.path.exists("./results"):
  os.makedirs("./results")
if not os.path.exists("./replaybuffer"):
			os.makedirs("./replaybuffer")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")

"""## Creamos un entorno de `PyBullet`"""

env = gym.make(env_name)
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
"""## Creamos la red neronal de la política (el actor del modelo)"""

policy = TD3(state_dim, action_dim, max_action)

"""## Creamos la memoria de la repetición de experiencias"""

replay_buffer = ReplayBuffer()
replay_buffer_episode = []
avg_reward_list = []
ep_reward_list = []
avg_reward = 0




"""## Definimos una lista donde se guardaran los resultados de evaluación de los 10 episodios"""



"""## Entrenamiento"""


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--load", default="") # "" por defecto no cargara modelo, "yes" cargara modelo de pytorch models    
  parser.add_argument("--scara", default="right")  # "right" por defecto, "left" para elegir scara izquierda"
  args = parser.parse_args()


  """# cargamos la politica, replay_buffer y metricas ----------------------------"""
  if args.load == "yes":
      start_timesteps = 0
      policy.load(file_name, "./pytorch_models",scara=args.scara)
      replay_buffer.load()
      ep_reward_list = pickle.load(open("./results/ep_reward_"+str(args.scara)+".pkl", "rb"))
      total_timesteps = pickle.load(open("./results/total_timesteps_"+str(args.scara)+".pkl", "rb"))
      episode_num = pickle.load(open("./results/episode_num_"+str(args.scara)+".pkl", "rb"))
      print("Punto de Carga: Timesteps Totales: ",total_timesteps,"Numero de Episodio: ",episode_num)
  """# cargamos la politica, replay_buffer y metricas ----------------------------"""

  evaluations = [evaluate_policy(policy)]
  init = True

  # Iniciamos el bucle principal 
  while total_timesteps < max_timesteps:
    
    # Si el episodio ha terminado
    if done:
      if env.step_golpe_disco != 0:
          replay_buffer_episode = replay_buffer_episode[:env.step_golpe_disco]  
      for element in replay_buffer_episode:
          replay_buffer.add(element)
      replay_buffer_episode = []
      # Si no estamos en la primera de las iteraciones, arrancamos el proceso de entrenar el modelo
      if not init:
        if total_timesteps > start_timesteps and cont >= 2:
          ep_reward_list.append(episode_reward)
          avg_reward = np.mean(ep_reward_list[-100:])
          avg_reward_list.append(avg_reward)
        print("Total Timesteps: {} Episode Timesteps: {} Episode Num: {} Disco coords: {} Reward: {:.3f} Avg.Reward: {:.3f}".format(total_timesteps,episode_timesteps, episode_num,env.coords(), episode_reward,avg_reward ))
        policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

      # Evaluamos el episodio y guardamos la política si han pasado las iteraciones necesarias
      if timesteps_since_eval >= eval_freq:
        cont = cont +1
        timesteps_since_eval %= eval_freq
        evaluations.append(evaluate_policy(policy))
        policy.save(file_name, directory="./pytorch_models")
        pickle.dump(avg_reward_list, file = open("./results/avg_reward_"+str(args.scara)+".pkl", "wb"))
        pickle.dump(ep_reward_list, file = open("./results/ep_reward_"+str(args.scara)+".pkl", "wb"))
        pickle.dump(total_timesteps, file = open("./results/total_timesteps_"+str(args.scara)+".pkl", "wb"))
        pickle.dump(episode_num, file = open("./results/episode_num_"+str(args.scara)+".pkl", "wb"))
        if cont % 5 == 0:
            replay_buffer.save()
        
      # Cuando el entrenamiento de un episodio finaliza, reseteamos el entorno
      obs = env.reset(scara=args.scara)
      
      # Configuramos el valor de done a False
      done = False
      
      # Configuramos la recompensa y el timestep del episodio a cero
      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1
    
    # Antes de los 10000 timesteps, ejectuamos acciones aleatorias
    if total_timesteps < start_timesteps:
      action = env.action_space.sample()
      
    else: # Después de los 10000 timesteps, cambiamos al modelo
      action = policy.select_action(np.array(obs))
      # Si el valor de explore_noise no es 0, añadimos ruido a la acción y lo recortamos en el rango adecuado
      if expl_noise != 0:
        action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
    
    # El agente ejecuta una acción en el entorno y alcanza el siguiente estado y una recompensa
    new_obs, reward, done, _ = env.step([action,args.scara])
    
    # Comprobamos si el episodio ha terminado
    done_bool = 0 if episode_timesteps + 1 == env.max_steps() else float(done)
    
    # Incrementamos la recompensa total
    episode_reward += reward
    
    # Almacenamos la nueva transición en la memoria de repetición de experiencias (ReplayBuffer)
    # Descomentar las impresiones por si desea observar el trabajo interno de modificacion de la recompensa de transiciones
    if env.step_colision_linea_enemiga != 0:
      #print("IMPACTO EN ENEMIGO-------------------------------------")
      #print("TRANSICION ACTUAL:")
      #print(obs, new_obs, action, reward, done_bool)
      #print("TRANSICION DE GOLPE:")
      #print(replay_buffer_episode[env.step_golpe_disco-1])
      transicion = list(replay_buffer_episode[env.step_golpe_disco-1])
      transicion[3] = transicion[3] + reward
      replay_buffer_episode[env.step_golpe_disco-1] = tuple(transicion)
      #print("TRANSICION DE GOLPE ACTUALIZADA:")
      #print(replay_buffer_episode[env.step_golpe_disco-1])
      #print("-------------------------------------------------------")
      env.reset_steps_linea()
      reward = 0
      
    if env.step_anotacion != 0:
      #print("ANOTACION----------------------------------")
      #print("TRANSICION ACTUAL:")
      #print(obs, new_obs, action, reward, done_bool)
      #print("TRANSICION DE GOLPE:")
      #print(replay_buffer_episode[env.step_golpe_disco-1])
      transicion = list(replay_buffer_episode[env.step_golpe_disco-1])
      transicion[3] = transicion[3] + reward
      replay_buffer_episode[env.step_golpe_disco-1] = tuple(transicion)
      #print("TRANSICION DE GOLPE ACTUALIZADA:")
      #print(replay_buffer_episode[env.step_golpe_disco-1])
      #print("-------------------------------------------------------")
      env.reset_step_anotacion()
      reward = 0
      
    '''    
    if env.step_autoanotacion != 0:
      #print("AUTO ANOTACION----------------------------------")
      #print("TRANSICION ACTUAL:")
      #print(obs, new_obs, action, reward, done_bool)
      #print("TRANSICION DE GOLPE:")
      #print(replay_buffer_episode[env.step_golpe_disco-1])
      transicion = list(replay_buffer_episode[env.step_golpe_disco-1])
      transicion[3] = transicion[3] + reward
      replay_buffer_episode[env.step_golpe_disco-1] = tuple(transicion)
      #print("TRANSICION DE GOLPE ACTUALIZADA:")
      #print(replay_buffer_episode[env.step_golpe_disco-1])
      #print("-------------------------------------------------------")
      env.reset_step_autoanotacion()
      reward = 0
    '''
    replay_buffer_episode.append((obs, new_obs, action, reward, done_bool))
    

    # Actualizamos el estado, el timestep del número de episodio, el total de timesteps y el número de pasos desde la última evaluación de la política
    obs = new_obs
    episode_timesteps += 1
    total_timesteps += 1
    init = False
    timesteps_since_eval += 1

  # Añadimos la última actualización de la política a la lista de evaluaciones previa y guardamos nuestro modelo
  evaluations.append(evaluate_policy(policy))
  if save_models: 
      policy.save("%s" % (file_name), directory="./pytorch_models")
      replay_buffer.save()


