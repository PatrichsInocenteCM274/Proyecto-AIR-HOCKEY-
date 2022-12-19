import matplotlib.pyplot as plt
import pickle
import argparse

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--scara", default="right")
  parser.add_argument("--modelo", default="td3")
  args = parser.parse_args()
  
  #if args.modelo == "ddpg": 
  avg_reward_list1 = pickle.load(open("./results/avg_reward_ddpg_left.pkl", "rb"))
  #else:
  avg_reward_list2 = pickle.load(open("./results/avg_reward_"+args.scara+".pkl", "rb"))

  plt.plot(avg_reward_list2)
  #plt.plot(avg_reward_list1)

  plt.xlabel("Episode")
  plt.ylabel("Avg. Epsiodic Reward en scara "+args.scara)  
  plt.show()