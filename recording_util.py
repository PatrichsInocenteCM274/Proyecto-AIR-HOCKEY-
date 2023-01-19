import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from IPython.display import HTML
from IPython.display import display
import warnings


def recording_video(frames, framerate=60, episode_reward=0,scara="right",episode_num=0):
  matplotlib.use('Agg')
  height, width, _ = frames[0].shape
  print("Grabando Video")
  fig, ax = plt.subplots(1, 1)
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
  FFwriter = animation.FFMpegWriter(fps=framerate, extra_args=['-vcodec', 'libx264'])                                
  anim.save("video_inferencia_"+str(scara)+str(episode_num)+".mp4",writer = FFwriter) 