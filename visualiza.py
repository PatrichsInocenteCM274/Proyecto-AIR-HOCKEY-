import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--scara", default="right")       # "left" para mostrar desempeño de scara Izquierda.
  parser.add_argument("--models", default="1",type=int)  # "2" Se hará la comparativa de los metodos TD3 y DDPG, donde DDPG
                                                         # se ha entrenado en la scara izquierda.
  args = parser.parse_args()
  fig, ax = plt.subplots()
  
  if args.models <= 2 : 
     
     avg_reward_list1 = pickle.load(open("./results/avg_reward_"+args.scara+".pkl", "rb"))
     avg_reward_list1 = avg_reward_list1
     x = np.arange(1, len(avg_reward_list1)+1)
     ep_reward_list1 = pickle.load(open("./results/ep_reward_"+args.scara+".pkl", "rb"))
     ep_reward_list1 = ep_reward_list1
     std_reward_list1 = []
     for i in range(1,len(ep_reward_list1)+1):
         std_reward_list1.append(np.std(ep_reward_list1[:i][-100:])/2)
     #plt.errorbar(x,avg_reward_list1, std_reward_list1, linestyle='solid', marker='^')
     ax.errorbar(x, avg_reward_list1, yerr=std_reward_list1,ecolor="deepskyblue",color="royalblue", fmt='-',alpha=0.5,label='TD3 en scara '+args.scara)
     #ax.plot(avg_reward_list1,'-b',label='TD3 en scara '+args.scara)
    
  if args.models == 2: 
     avg_reward_list2 = pickle.load(open("./results/avg_reward_ddpg_"+args.scara+".pkl", "rb"))
     avg_reward_list2 = avg_reward_list2[:770]
     y = np.arange(1, len(avg_reward_list2)+1)
     ep_reward_list2 = pickle.load(open("./results/ep_reward_ddpg_"+args.scara+".pkl", "rb"))
     ep_reward_list2 = ep_reward_list2[:770]
     std_reward_list2 = []
     for i in range(1,len(ep_reward_list2)+1):
         std_reward_list2.append(np.std(ep_reward_list2[:i][-100:])/2)
     #plt.errorbar(x,avg_reward_list1, std_reward_list1, linestyle='solid', marker='^')
     ax.errorbar(y, avg_reward_list2, yerr=std_reward_list2,ecolor="palegreen",color="darkgreen", fmt='-',alpha=0.3,label='DDPG en scara '+args.scara)
  
  #plt.style.use('classic')
  
  leg = ax.legend()
  plt.xlabel("Episode")
  plt.ylabel("Avg. Epsiodic Reward en scara Left")  
  plt.legend(loc='center right')
  plt.show() 
