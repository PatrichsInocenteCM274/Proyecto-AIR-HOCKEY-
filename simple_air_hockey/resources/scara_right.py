import pybullet as p
import os
import math
import numpy as np
from numpy import interp
import simple_air_hockey.resources.helpers.helper_id_joint as helper_id

class ScaraR:
    
    def __init__(self, client, max_velocity = 7.0):
        
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'scara-right.urdf')
        self.scara_right = p.loadURDF(fileName=f_name,
                              basePosition=[0.0, 8.2, 2.78],
                              useFixedBase=True,
                              physicsClientId=client)
        # Indices de Joints que representan los servos del scara
        self.max_velocity = max_velocity
        self.servos_joints = helper_id.get_id_joints_revolution(self.scara_right)
        self.angle_servo = -0.5
        self.angle_efector = 0.8
        self.velocity_servo = 1.0
        self.velocity_efector = 1.0
        self.apply_action(action=[-0.4,0.8,self.velocity_servo,self.velocity_efector])

    def get_ids(self):
        return self.scara_right, self.client


    def apply_action(self, action):
        # Accion de dos dimensiones
        self.angle_servo,self.angle_efector,self.velocity_servo,self.velocity_efector = action

        self.angle_servo = max(min(self.angle_servo, 1), -1)
        self.angle_servo = interp(self.angle_servo,[-1,1],[-2.9,2.9])

        self.angle_efector = max(min(self.angle_efector, 1), -1)
        self.angle_efector = interp(self.angle_efector,[-1,1],[-2.9,2.9])

        self.velocity_servo = max(min(self.velocity_servo, 1), -1)
        self.velocity_servo = interp(self.velocity_servo,[-1,1],[0,10.0])

        self.velocity_efector = max(min(self.velocity_efector, 1), -1)
        self.velocity_efector = interp(self.velocity_efector,[-1,1],[0,10.0])
        
        p.setJointMotorControl2(self.scara_right,self.servos_joints[0],
                                p.POSITION_CONTROL,
                                targetPosition=self.angle_servo,
                                maxVelocity= self.velocity_servo ,
                                physicsClientId=self.client)   

        p.setJointMotorControl2(self.scara_right,self.servos_joints[1],
                                p.POSITION_CONTROL,
                                targetPosition=self.angle_efector,
                                maxVelocity= self.velocity_efector,
                                physicsClientId=self.client)  
                                   

    # Cinematica Directa de Efector, metodo Geométrico
    def get_observation(self):

        pos, ang = p.getBasePositionAndOrientation(self.scara_right, self.client)
        pos = pos[:2]
        current_angle_servo = p.getJointState(self.scara_right, self.servos_joints[0])[0]
        current_angle_efector = p.getJointState(self.scara_right, self.servos_joints[1])[0]
        observation = (pos[0]+math.sin(current_angle_servo+current_angle_efector)*2.15 + math.sin(current_angle_servo)*2.9,
                        pos[1]-math.cos(current_angle_servo+current_angle_efector)*2.15 - math.cos(current_angle_servo)*2.9,
                        current_angle_servo,
                        current_angle_efector)
        
        return observation

    def cinematica_inversa(self,action):
        px,py,self.velocity_servo,self.velocity_efector = action

        #px = max(min(px, 1), -1)
        #px = interp(px,[-1,1],[-4.70,4.70])

        #py = max(min(py, 1), -1)
        #py = interp(py,[-1,1],[3.15,7.40])

        self.velocity_servo = max(min(self.velocity_servo, 1), -1)
        self.velocity_servo = interp(self.velocity_servo,[-1,1],[0,5.0])

        self.velocity_efector = max(min(self.velocity_efector, 1), -1)
        self.velocity_efector = interp(self.velocity_efector,[-1,1],[0,5.0])

        pos, ang = p.getBasePositionAndOrientation(self.scara_right, self.client)
        pos = pos[:2]
        px = (px-pos[0])
        py = -(py-pos[1])
        r_limit = 2.15+2.899
        r_current = math.sqrt(px**2+py**2)
        
        if r_current > r_limit: #truncamos al limite del rango de movimiento 
            px= px*(r_limit/r_current)
            py= py*(r_limit/r_current)
            
        angle_efector = math.acos((px**2+py**2-2.15**2-2.9**2)/(2*2.15*2.9))
        angle_servo = np.arctan2(px,py) - np.arctan2((2.15*math.sin(angle_efector)),(2.9+2.15*math.cos(angle_efector))) 

        p.setJointMotorControl2(self.scara_right,self.servos_joints[0],
                                p.POSITION_CONTROL,
                                targetPosition=angle_servo,
                                maxVelocity= self.velocity_servo ,
                                physicsClientId=self.client)   

        p.setJointMotorControl2(self.scara_right,self.servos_joints[1],
                                p.POSITION_CONTROL,
                                targetPosition=angle_efector,
                                maxVelocity= self.velocity_efector,
                                physicsClientId=self.client) 
