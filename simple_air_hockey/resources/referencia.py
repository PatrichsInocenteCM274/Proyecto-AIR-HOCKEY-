import pybullet as p
import os

class Referencia:
    def __init__(self, client, start_coord = (0, 3.15)):

        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'referencia.urdf')
        self.disco = p.loadURDF(fileName=f_name,
                        basePosition= [start_coord[0],start_coord[1],2.30],
                        useFixedBase=True,
                        physicsClientId=client)
                        

    def get_observation(self):

        pos, ang = p.getBasePositionAndOrientation(self.disco, self.client)
        return pos
