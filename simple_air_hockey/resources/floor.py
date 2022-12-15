import pybullet as p
import os


class Floor:
    def __init__(self, client):
        f_name = os.path.join(os.path.dirname(__file__), 'floor.urdf')
        p.loadURDF(fileName=f_name,
                   basePosition=[0, 0, 0.05],
                   useFixedBase=True,
                   physicsClientId=client)

