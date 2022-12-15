import matplotlib.pyplot as plt
import pickle
import argparse

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--scara", default="right")
  args = parser.parse_args()
  avg_reward_list = pickle.load(open("./results/avg_reward_"+args.scara+".pkl", "rb"))
  plt.plot(avg_reward_list)
  plt.xlabel("Episode")
  plt.ylabel("Avg. Epsiodic Reward")  
  plt.show()