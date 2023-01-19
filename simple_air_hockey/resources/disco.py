import pybullet as p
import os

class Disco:
    def __init__(self, client, start_coord = (-1.6, 4.0)):

        self.client = client
        self.start_coord = start_coord
        self.f_name = os.path.join(os.path.dirname(__file__), 'disco.urdf')
        self.disco = p.loadURDF(fileName=self.f_name,
                        basePosition= [self.start_coord[0],self.start_coord[1],2.80],
                        physicsClientId=client)
        # Hemos realizado unas restricciones prismaticas en X,Y unidas al marco mundial
        # Con el objetivo de poder restringir movimientos en eje Z no deseados causados meramente por el simulador
        self.joint_indexes = [j for j in range(p.getNumJoints(self.disco))]
        self.joint_forces_zero = [0.0 for j in range(p.getNumJoints(self.disco))]
        p.setJointMotorControlArray(self.disco, self.joint_indexes, p.VELOCITY_CONTROL, forces=self.joint_forces_zero)

    def get_observation(self):
        pos = (p.getJointState(self.disco,0)[0]+self.start_coord[0],p.getJointState(self.disco,1)[0]+self.start_coord[1])
        return pos

        

