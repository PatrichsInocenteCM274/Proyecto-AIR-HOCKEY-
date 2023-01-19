import numpy as np

def evaluate(politica, env, scara, eval_episodes=10):
  print ("------------------------------------------------")  
  #env.seed(0)
  avg_reward = 0.
  
  print("Comienza evaluacion")
  for num_episode in range(eval_episodes):
    
    frames_episode = []
    last_episode_reward= 0.
    obs = env.reset(scara=scara)
    done = False
    while not done:
      accion = politica.accion(np.array(obs))
      obs, reward, done, _ = env.step([accion,scara])
      last_episode_reward += reward
      avg_reward += reward
    print(f'Disco coords: {env.coords()} Reward: {last_episode_reward:.2f}')
  avg_reward /= eval_episodes
  env.reset_steps_linea()
  env.reset_step_anotacion()
  
  print ("Recompensa promedio en el paso de Evaluación: %f" % (avg_reward))
  print ("------------------------------------------------")
  return avg_reward