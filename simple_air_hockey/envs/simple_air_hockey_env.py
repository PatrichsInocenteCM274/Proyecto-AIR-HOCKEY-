from pickle import NONE
import pybullet as p
from time import sleep
import numpy as np
from numpy import interp
import gym
import math
from simple_air_hockey.resources.disco import Disco
from simple_air_hockey.resources.mesa import Mesa
from simple_air_hockey.resources.scara_right import ScaraR
from simple_air_hockey.resources.scara_left import ScaraL
from simple_air_hockey.resources.floor import Floor
from simple_air_hockey.resources.human_mallet import Human_mallet
from simple_air_hockey.resources.referencia import Referencia
from simple_air_hockey.resources.score import Score


class SimpleAirHockeyEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.action_space = gym.spaces.box.Box(  # type: ignore
            low=np.array([-1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box( # type: ignore
            low=np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], dtype=np.float32), # Limite inferior de coordenada x,y disco y coordenada x,y mazo
            high=np.array([1,1,1,1,1,1,1,1,1,1], dtype=np.float32))
        print("Iniciando Entorno AIR HOCKEY, Bienvenido!")
        self.np_random, _ = gym.utils.seeding.np_random() # type: ignore
        self.client = None
        self.mode_GAME = False
        self._max_episode_steps = 500
        self.scara_right_angle_servo_base = None
        self.scara_right_angle_servo_efector = None
        self.scara_left_angle_servo_base = None
        self.scara_left_angle_servo_efector = None
        self.scara_right = None
        self.scara_left = None
        self.disco = None
        self.done = False
        self.init = True
        self.steps = 0
        self.prev_dist_to_disco = None
        self.prev_disc_obs = None
        self.choque=False
        self.choque_pared_enemiga=False
        self.choque_pared_aliada = False
        self.timestep = 1/60
        self.x = 0
        self.y = 0
        self.step_golpe_disco = None
        self.step_colision_linea_enemiga = 0
        self.step_anotacion = 0
        self.anotacion = False
        self.score = [0,0]
        self.score_right = 0
        self.score_left = 0
        self.gx = 0
        self.observacion_right = None
        self.observacion_left = None
        self.winner = False
        self.models = 1
        self.pared_evalua=False
        
    def set_mode(self, mode_GAME):
        self.mode_GAME = mode_GAME

    def set_timestep(self, timestep):
        self.timestep = timestep

    def step(self,parameters):

        if len(parameters) == 2:
            action,scara = parameters
            if scara == 'right': self.scara_right.apply_action(action)
            if scara == 'left': self.scara_left.apply_action(action)
        if len(parameters) == 3:
            action_right,action_left,scara = parameters
            self.scara_right.apply_action(action_right)
            self.scara_left.apply_action(action_left)
        
        if not self.mode_GAME:
            p.stepSimulation()
        disco_ob = self.disco.get_observation() 
        reward = 0

        if scara == 'right' or scara == 'all': 
            mazo_right_ob = self.scara_right.get_observation()
            mazo_to_disco = math.sqrt(((mazo_right_ob[0] - disco_ob[0]) ** 2 +
                                  (mazo_right_ob[1] - disco_ob[1]) ** 2)) 
            mazo_to_disco_angle = math.atan2(mazo_right_ob[1] - disco_ob[1], mazo_right_ob[0] - disco_ob[0])
            mazo_to_goal_angle = math.atan2(mazo_right_ob[1] - (-7.48), mazo_right_ob[0] - 0.0)
            disco_to_goal_angle = math.atan2(disco_ob[1] - (-7.48), disco_ob[0] - 0.0)
            if self.steps > self._max_episode_steps and not self.mode_GAME:
                self.done = True
                self.steps = 0

            if self.steps <= 1:
                p.setGravity(self.gx, 500, -10) 
            else:
                p.setGravity(0, 0.0, -10)     
                
            if mazo_to_disco < 0.40+0.47 and not self.choque:
                self.step_golpe_disco = self.steps
                reward += 50
                self.choque=True
                #self.done = True 
                #self.steps = 0 
                
            if mazo_to_disco > 0.40+0.47 + 0.04:
                self.choque=False
                
            #if (disco_ob[0]<-5.0 or disco_ob[0]>5.0) and self.pared_evalua and not self.mode_GAME:
            #    reward -= 5
            #    self.done = True
            #    self.steps = 0    

            if disco_ob[1]<-8.0 and not self.anotacion:
                self.score_right += 1
                if self.score_right==8: self.winner = True
                self.step_anotacion = self.steps
                reward += 100
                self.done = True
                self.steps = 0
                
            if mazo_right_ob[1] <= disco_ob[1]+0.40:
                reward -= 0.02
                
            if disco_ob[1]<=-7.47 and not self.choque_pared_enemiga:
                reward += (507-abs(disco_ob[0])*100)
                self.step_colision_linea_enemiga = self.steps
                self.choque_pared_enemiga=True
                
            if disco_ob[1]>-7.0 and self.choque_pared_enemiga and not self.mode_GAME:
                self.choque_pared_enemiga=False
                self.done = True
                self.steps = 0
                
            if disco_ob[1]>=7.47 and not self.choque_pared_aliada:
                self.choque_pared_aliada=True
                
            if disco_ob[1]<7.0 and self.choque_pared_aliada and not self.mode_GAME:
                self.choque_pared_aliada=False
                self.done = True
                self.steps = 0
                
            if disco_ob[1]>8.0 :
                reward -= 100
                self.done = True
                self.steps = 0
                
            if mazo_right_ob[1] >= 7.43 and not self.mode_GAME:
                reward -= 1.0    

            self.steps = self.steps+1
            self.prev_dist_to_disco = mazo_to_disco
            self.prev_disc_obs = disco_ob[1]

                # Interpolacion lineal -------------
            disco_ob_x = interp(disco_ob[0],[-5.07,5.07],[-1,1])
            disco_ob_y = interp(disco_ob[1],[-7.49,7.49],[-1,1])
            mazo_right_ob_x = interp(mazo_right_ob[0],[-4.99,4.99],[-1,1])
            mazo_right_ob_y = interp(mazo_right_ob[1],[3.15,7.42],[-1,1])
            angle_servo = interp(mazo_right_ob[2],[-2.0,2.0],[-1,1])
            angle_efector = interp(mazo_right_ob[3],[-2.9,2.9],[-1,1])
            mazo_to_disco = interp(mazo_to_disco,[0.85,18.0],[-1,1])
            mazo_to_disco_angle = interp(mazo_to_disco_angle,[-3.1415,3.1415],[-1,1])
            mazo_to_goal_angle = interp(mazo_to_goal_angle,[1.25,1.89],[-1,1])
            disco_to_goal_angle = interp(disco_to_goal_angle,[0,3.14],[-1,1])
            # -----------------------

            ob = np.array(tuple([disco_ob_x])+tuple([disco_ob_y])+tuple([mazo_right_ob_x])+tuple([mazo_right_ob_y])+
                tuple([angle_servo])+tuple([angle_efector])+tuple([mazo_to_disco])+tuple([mazo_to_disco_angle])
                +tuple([mazo_to_goal_angle])+tuple([disco_to_goal_angle]), dtype=np.float32)
            if not scara == 'all':
                return ob, reward, self.done, dict()
            else:
                ob_right = ob
                reward_right = reward
                done_right = self.done

        if scara == 'left' or scara == 'all': 
            mazo_left_ob = self.scara_left.get_observation()
            mazo_to_disco = math.sqrt(((mazo_left_ob[0] - disco_ob[0]) ** 2 +
                                  (mazo_left_ob[1] - disco_ob[1]) ** 2)) 
            mazo_to_disco_angle = math.atan2(mazo_left_ob[1] - disco_ob[1], mazo_left_ob[0] - disco_ob[0])
            mazo_to_goal_angle = math.atan2(mazo_left_ob[1] - (7.48), mazo_left_ob[0] - 0.0) 
            disco_to_goal_angle = math.atan2(disco_ob[1] - (7.48), disco_ob[0] - 0.0) 

            if self.steps > self._max_episode_steps and not self.mode_GAME:
                self.done = True
                self.steps = 0

            if self.steps <= 1:
                p.setGravity(self.gx, -500.0, -10) 
            else:
                p.setGravity(0, 0.0, -10)
                
            if mazo_to_disco < 0.40+0.47 and not self.choque:
                self.step_golpe_disco = self.steps
                reward += 50
                self.choque=True
                #self.done = True 
                #self.steps = 0 
                
                 
            if mazo_to_disco > 0.40+0.47 + 0.04:
                self.choque=False
                
            if disco_ob[1]>8.0 and not self.anotacion:
                self.score_left += 1
                if self.score_left==8: self.winner = True
                self.step_anotacion = self.steps
                reward += 100
                self.done = True
                self.steps = 0
                
            if mazo_left_ob[1] >= disco_ob[1]-0.40:
                reward -= 0.02
                
            if disco_ob[1]>=7.47 and not self.choque_pared_enemiga:
                reward += (507-abs(disco_ob[0])*100)
                self.step_colision_linea_enemiga = self.steps
                self.choque_pared_enemiga=True
                
            if disco_ob[1]<7.0 and self.choque_pared_enemiga and not self.mode_GAME:
                self.choque_pared_enemiga=False
                self.done = True
                self.steps = 0
                
            if disco_ob[1]<=-7.47 and not self.choque_pared_aliada:
                self.choque_pared_aliada=True
                
            if disco_ob[1]>-7.0 and self.choque_pared_aliada and not self.mode_GAME:
                self.choque_pared_aliada=False
                self.done = True
                self.steps = 0
                
            if disco_ob[1]<-8.0 :
                reward -= 100
                self.done = True
                self.steps = 0
                
            if mazo_left_ob[1] <= -7.43 and not self.mode_GAME:
                reward -= 1.0    

            self.steps=self.steps+1
            self.prev_dist_to_disco = mazo_to_disco
            self.prev_disc_obs = disco_ob[1]    

                # Interpolacion lineal -------------
            disco_ob_x = interp(disco_ob[0],[-5.07,5.07],[-1,1])
            disco_ob_y = interp(disco_ob[1],[-7.49,7.49],[-1,1])
            mazo_left_ob_x = interp(mazo_left_ob[0],[-4.99,4.99],[-1,1])
            mazo_left_ob_y = interp(mazo_left_ob[1],[-3.15,-7.42],[-1,1])
            angle_servo = interp(mazo_left_ob[2],[-2.0,2.0],[-1,1])
            angle_efector = interp(mazo_left_ob[3],[-2.9,2.9],[-1,1])
            mazo_to_disco = interp(mazo_to_disco,[0.85,18.0],[-1,1])
            mazo_to_disco_angle = interp(mazo_to_disco_angle,[-3.1415,3.1415],[-1,1])
            mazo_to_goal_angle = interp(mazo_to_goal_angle,[1.25,1.89],[-1,1])
            disco_to_goal_angle = interp(disco_to_goal_angle,[-3.14,0],[-1,1])
            # -----------------------
            
            ob = np.array(tuple([disco_ob_x])+tuple([disco_ob_y])+tuple([mazo_left_ob_x])+tuple([mazo_left_ob_y])+
                tuple([angle_servo])+tuple([angle_efector])+tuple([mazo_to_disco])+tuple([mazo_to_disco_angle])
                +tuple([mazo_to_goal_angle])+tuple([disco_to_goal_angle]), dtype=np.float32)
            if not scara == 'all':
                return ob, reward, self.done, dict()
            else:
                ob_left = ob
                reward_left = reward
                done_left = self.done
            
        if scara == 'all':
            return (ob_right,ob_left) ,(reward_right,reward_left), (done_right,done_left), dict()  

    def max_steps(self):
        return self._max_episode_steps

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed) # type: ignore
        return [seed]

    def reset_steps_linea(self):
        self.step_colision_linea_enemiga = 0
        
    def reset_step_anotacion(self):
        self.step_anotacion = 0
     
    def get_step_golpe_disco(self):    
        return self.step_golpe_disco 

    def get_step_colision_linea(self):    
        return self.step_colision_linea_enemiga

    def random_set_state(self,state=None):
        self.np_random.set_state(state)
    
    def coords(self):
        return round(self.x, 2),round(self.y, 2)
        
    def random_get_state(self):
        return self.np_random.get_state()

    def change_score(self):
        self.score = [0,0]

    def set_models(self,nro_models):
        self.models = nro_models

    def reset(self, scara = 'right'):
        if not self.init:
            p.resetSimulation(self.client)
        if self.init:
            if self.mode_GAME:
                self.client = p.connect(p.GUI)
                #p.setTimeStep(1/240, self.client)  
                p.setRealTimeSimulation(1)
            else:
                self.client = p.connect(p.DIRECT)
                p.setTimeStep(self.timestep, self.client)                
            self.init = False
        p.setGravity(0, 0.0, -10)
        self.done = False
        self.choque=False
        self.choque_pared_enemiga = False
        self.choque_pared_aliada = False
        self.step_golpe_disco = 0   
        self.pared_evalua=False     
        score = Score(self.client,self.models)
        score.change_score(self.score_right,self.score_left)
        floor = Floor(self.client)
        mesa = Mesa(self.client)
        #human_mallet = Human_mallet(self.client)
        angle = 0.0
        if scara == 'right':
            self.x = self.np_random.uniform(-4.20, 4.20)
            self.y = self.np_random.uniform(-4.20, -2.20)
        if scara == 'left':
            self.x = self.np_random.uniform(-4.20, 4.20)
            self.y = self.np_random.uniform(2.20, 4.20)
        self.gx = self.np_random.uniform(-500, 500)
        self.disco = Disco(self.client,(self.x,self.y))
        disco_ob = self.disco.get_observation()

        if scara == 'right' or scara == 'all':
            self.scara_right = ScaraR(self.client)
            for _ in range(200):
                p.stepSimulation()
            mazo_right_ob = self.scara_right.get_observation() 
            mazo_to_disco = math.sqrt(((mazo_right_ob[0] - disco_ob[0]) ** 2 +
                                    (mazo_right_ob[1] - disco_ob[1]) ** 2))             
            mazo_to_disco_angle = math.atan2(mazo_right_ob[1] - disco_ob[1], mazo_right_ob[0] - disco_ob[0]) 
            mazo_to_goal_angle = math.atan2(mazo_right_ob[1] - (-7.48), mazo_right_ob[0] - 0.0)
            disco_to_goal_angle = math.atan2(disco_ob[1] - (-7.48), disco_ob[0] - 0.0) 
            self.prev_disc_obs = disco_ob[1]
            
            # Interpolacion lineal -------------
            disco_ob_x = interp(disco_ob[0],[-5.07,5.07],[-1,1])
            disco_ob_y = interp(disco_ob[1],[-7.49,7.49],[-1,1])
            mazo_right_ob_x = interp(mazo_right_ob[0],[-4.99,4.99],[-1,1])
            mazo_right_ob_y = interp(mazo_right_ob[1],[3.15,7.42],[-1,1])
            angle_servo = interp(mazo_right_ob[2],[-2.0,2.0],[-1,1])
            angle_efector = interp(mazo_right_ob[3],[-2.9,2.9],[-1,1])
            mazo_to_disco = interp(mazo_to_disco,[0.85,18.0],[-1,1])
            mazo_to_disco_angle = interp(mazo_to_disco_angle,[-3.1415,3.1415],[-1,1])
            mazo_to_goal_angle = interp(mazo_to_goal_angle,[1.25,1.89],[-1,1])
            disco_to_goal_angle = interp(disco_to_goal_angle,[0, 3.14],[-1,1])
            # -----------------------
            #ob = np.array(disco_ob+mazo_right_ob+tuple([mazo_to_disco_angle]), dtype=np.float32)
            ob = np.array(tuple([disco_ob_x])+tuple([disco_ob_y])+tuple([mazo_right_ob_x])+tuple([mazo_right_ob_y])+
            tuple([angle_servo])+tuple([angle_efector])+tuple([mazo_to_disco])+tuple([mazo_to_disco_angle])
            +tuple([mazo_to_goal_angle])+tuple([disco_to_goal_angle]), dtype=np.float32)
            if not scara == 'all':
                return ob
            else:
                ob_right = ob

        if scara == 'left' or scara == 'all':
            self.scara_left = ScaraL(self.client)
            for _ in range(20):
                p.stepSimulation()
            mazo_left_ob = self.scara_left.get_observation() 
            mazo_to_disco = math.sqrt(((mazo_left_ob[0] - disco_ob[0]) ** 2 +
                                    (mazo_left_ob[1] - disco_ob[1]) ** 2))             
            mazo_to_disco_angle = math.atan2(mazo_left_ob[1] - disco_ob[1], mazo_left_ob[0] - disco_ob[0]) 
            mazo_to_goal_angle = math.atan2(mazo_left_ob[1] - (7.48), mazo_left_ob[0] - 0.0)
            disco_to_goal_angle = math.atan2(disco_ob[1] - (7.48), disco_ob[0] - 0.0) 
            self.prev_disc_obs = disco_ob[1]
            
            # Interpolacion lineal -------------
            disco_ob_x = interp(disco_ob[0],[-5.07,5.07],[-1,1])
            disco_ob_y = interp(disco_ob[1],[-7.49,7.49],[-1,1])
            mazo_left_ob_x = interp(mazo_left_ob[0],[-4.99,4.99],[-1,1])
            mazo_left_ob_y = interp(mazo_left_ob[1],[-3.15,-7.42],[-1,1])
            angle_servo = interp(mazo_left_ob[2],[-2.0,2.0],[-1,1])
            angle_efector = interp(mazo_left_ob[3],[-2.9,2.9],[-1,1])
            mazo_to_disco = interp(mazo_to_disco,[0.85,18.0],[-1,1])
            mazo_to_disco_angle = interp(mazo_to_disco_angle,[-3.1415,3.1415],[-1,1])
            mazo_to_goal_angle = interp(mazo_to_goal_angle,[1.25,1.89],[-1,1])
            disco_to_goal_angle = interp(disco_to_goal_angle,[-3.14,0],[-1,1])
            # -----------------------
            
            ob = np.array(tuple([disco_ob_x])+tuple([disco_ob_y])+tuple([mazo_left_ob_x])+tuple([mazo_left_ob_y])+
            tuple([angle_servo])+tuple([angle_efector])+tuple([mazo_to_disco])+tuple([mazo_to_disco_angle])
            +tuple([mazo_to_goal_angle])+tuple([disco_to_goal_angle]), dtype=np.float32)
            if not scara == 'all':
                return ob
            else:
                ob_left = ob
            
        if scara == 'all':
            return (ob_right,ob_left),dict()



    def render(self, mode='human',resolution=350):
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)

        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[1.50,2.0,2.0],
                                                            distance=10.0,
                                                            yaw=150.0,
                                                            pitch=-50,
                                                            roll=0,
                                                            upAxisIndex=2)

        frame = p.getCameraImage(resolution, resolution, view_matrix, proj_matrix)[2]
        return frame

    def demo(self):
        self.client = p.connect(p.GUI)
        p.setGravity(0, 0.0, -10)
        p.setRealTimeSimulation(1)
        p.setTimeStep(1/60, self.client)  
        floor = Floor(self.client)
        mesa = Mesa(self.client)
        #referencia = Referencia(self.client)
        self.disco = Disco(self.client)
        self.scara_right = ScaraR(self.client,7.0)
        self.score = Score(self.client)
        self.score.change_score(self.score_right,self.score_left)
        self.scara_left = ScaraL(self.client,7.0)
        c_directa = p.addUserDebugParameter(" Cinematica directa",1,0,1)
        self.scara_right_angle_servo_base = p.addUserDebugParameter(' Scara Right Servo Base', -1, 1, 0)
        self.scara_right_angle_servo_efector = p.addUserDebugParameter(' Scara Right Servo Efector', -1, 1, 0)
        self.scara_left_angle_servo_base = p.addUserDebugParameter(' Scara Left Servo Base', -1, 1, 0)
        self.scara_left_angle_servo_efector = p.addUserDebugParameter(' Scara Left Servo Efector', -1, 1, 0)
        c_inversa = p.addUserDebugParameter(" Cinematica inversa",1,0,1)
        px_right = p.addUserDebugParameter(' posicion x Scara Right', -4.70,4.70, 0)
        py_right = p.addUserDebugParameter(' posicion y Scara Right', 3.15, 7.40, 3.15)
        px_left = p.addUserDebugParameter(' posicion x Scara Left', -4.70,4.70, 0)
        py_left = p.addUserDebugParameter(' posicion y Scara Left', -7.40, -3.15, -3.15)
        
        sleep(1.0)
        camera = 0
        while True:
            
            angle_servo_scara_right = p.readUserDebugParameter(self.scara_right_angle_servo_base)
            angle_efector_scara_right = p.readUserDebugParameter(self.scara_right_angle_servo_efector)
            angle_servo_scara_left = p.readUserDebugParameter(self.scara_left_angle_servo_base)
            angle_efector_scara_left = p.readUserDebugParameter(self.scara_left_angle_servo_efector)

            if p.readUserDebugParameter(c_directa) % 2 == 0:
                p.addUserDebugText(
                "Cinematica Directa Activa",
                textPosition=[0,0, 4.0],
                textColorRGB=[0, 0, 0],
                textSize=1,
                lifeTime=0.1,
                )
                self.scara_right.apply_action([angle_servo_scara_right,angle_efector_scara_right,1.0,1.0])
                self.scara_left.apply_action([angle_servo_scara_left,angle_efector_scara_left,1.0,1.0])


            pos_px_right = p.readUserDebugParameter(px_right)
            pos_py_right = p.readUserDebugParameter(py_right)
            pos_px_left = p.readUserDebugParameter(px_left)
            pos_py_left = p.readUserDebugParameter(py_left)
            if p.readUserDebugParameter(c_inversa) % 2 == 0:
                p.addUserDebugText(
                "Cinematica Inversa Activa",
                textPosition=[0,0, 4.0],
                textColorRGB=[0, 0, 0],
                textSize=1,
                lifeTime=0.1,
                )
                self.scara_right.cinematica_inversa([pos_px_right,pos_py_right,1.0,1.0])
                self.scara_left.cinematica_inversa([pos_px_left,pos_py_left,1.0,1.0])
            
            scara_left_ob = self.scara_left.get_observation()
            scara_right_ob = self.scara_right.get_observation()
            disco_ob = self.disco.get_observation()
            mazo_to_disco_angle = math.atan2(scara_right_ob[1] - disco_ob[1], scara_right_ob[0] - disco_ob[0])
            mazo_to_goal_angle = math.atan2(scara_right_ob[1] - (-7.48), scara_right_ob[0] - 0.0)   
            mazo_to_disco = math.sqrt(((scara_right_ob[0] - disco_ob[0]) ** 2 +
                                  (scara_right_ob[1] - disco_ob[1]) ** 2))
            disco_to_goal_angle = math.atan2(disco_ob[1] - (7.48), disco_ob[0] - 0.0) 
            p.addUserDebugText(
                "X = "+"{:.2f}".format(scara_left_ob[0])+" Y = "+"{:.2f}".format(scara_left_ob[1]),
                textPosition=[scara_left_ob[0], scara_left_ob[1], 2.80],
                textColorRGB=[1, 0, 0],
                textSize=1,
                lifeTime=0.1,
            )
            p.addUserDebugText(
                "X = "+"{:.2f}".format(scara_right_ob[0])+" Y = "+"{:.2f}".format(scara_right_ob[1])+ " d = "+"{:.2f}".format(mazo_to_disco),
                textPosition=[scara_right_ob[0], scara_right_ob[1], 2.80],
                textColorRGB=[1, 0, 0],
                textSize=1,
                lifeTime=0.1,
            )
            
            p.addUserDebugText(
                "X = "+"{:.2f}".format(disco_ob[0])+" Y = "+"{:.2f}".format(disco_ob[1])+" angle {:.2f}".format(mazo_to_disco_angle)
                +" angle_to_goal {:.2f}".format(mazo_to_goal_angle)+"disco_goal{:.2f}".format(disco_to_goal_angle),
                textPosition=[disco_ob[0], disco_ob[1], 2.40],
                textColorRGB=[1, 0, 0],
                textSize=1,
                lifeTime=0.1,
            )
            
            
            

    def close(self):
        p.disconnect(self.client)
