import matplotlib.pyplot as plt
import pickle
import argparse

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--scara", default="right")       # "left" para mostrar desempeño de scara Izquierda.
  parser.add_argument("--models", default="1",type=int)  # "2" Se hará la comparativa de los metodos TD3 y DDPG, donde DDPG
                                                         # se ha entrenado en la scara izquierda.
  args = parser.parse_args()
  fig, ax = plt.subplots()
  
  if args.models <= 2: 
     avg_reward_list1 = pickle.load(open("./results/avg_reward_"+args.scara+".pkl", "rb"))
     ax.plot(avg_reward_list1,'-b',label='TD3 en scara '+args.scara)
    
  if args.models == 2: 
     avg_reward_list2 = pickle.load(open("./results/avg_reward_ddpg_left.pkl", "rb"))
     ax.plot(avg_reward_list2,'-r',label='DDPG en scara left')   
  
  #plt.style.use('classic')
  
  leg = ax.legend()
  plt.xlabel("Episode")
  plt.ylabel("Avg. Epsiodic Reward en scara ")  
  plt.show() 
