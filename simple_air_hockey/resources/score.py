import pybullet as p
import os
import math
import numpy as np
from numpy import interp

class Score:
    
    def __init__(self, client, models):
        self.client = client
        score = os.path.join(os.path.dirname(__file__), 'score.urdf')
        table_score = os.path.join(os.path.dirname(__file__), 'table_score.urdf')
        self.f_image_competitors_td3_td3 = os.path.join(os.path.dirname(__file__), 'td3_td3.png')
        self.f_image_competitors_td3_ddpg = os.path.join(os.path.dirname(__file__), 'td3_ddpg.png')
        self.models = models

        self.table_score1 = p.loadURDF(table_score,
                              [0.0, -1.1, 8.5],
                              [0,0,0,1],
                              useFixedBase=True,
                              physicsClientId=client)

        self.score1 = p.loadURDF(score,
                              [-0.5, -1.1, 8.],
                              [0,0,3.1415,0],
                              useFixedBase=True,
                              physicsClientId=client)

        self.score2 = p.loadURDF(score,
                              [0.5, -1.1, 8.],
                              [0,0,3.1415,0],
                              useFixedBase=True,
                              physicsClientId=client)

        self.table_score2 = p.loadURDF(table_score,
                              [0.0, 1.1, 8.5],
                              [0,0,3.1415,0],
                              useFixedBase=True,
                              physicsClientId=client)

        self.score3 = p.loadURDF(score,
                              [0.5, 1.1, 8.],
                              [0,0,0,1],
                              useFixedBase=True,
                              physicsClientId=client)

        self.score4 = p.loadURDF(score,
                              [-0.5, 1.1, 8.],
                              [0,0,0,1],
                              useFixedBase=True,
                              physicsClientId=client)
                     
    def change_score(self, player1_score, player2_score):
            player1_score = max(min(player1_score, 7), 0)
            player2_score = max(min(player2_score, 7), 0)
            f_number_image_player1 = os.path.join(os.path.dirname(__file__), str(player1_score)+'.png')
            f_number_image_player2 = os.path.join(os.path.dirname(__file__), str(player2_score)+'.png')
            self.number_score_player1 = p.loadTexture(textureFilename=f_number_image_player1,physicsClientId=self.client)
            self.number_score_player2 = p.loadTexture(textureFilename=f_number_image_player2,physicsClientId=self.client)
            if self.models == 1:
                self.competitors = p.loadTexture(textureFilename=self.f_image_competitors_td3_td3,physicsClientId=self.client)
            else:
                self.competitors = p.loadTexture(textureFilename=self.f_image_competitors_td3_ddpg,physicsClientId=self.client)
            p.changeVisualShape(self.score1,-1,textureUniqueId = self.number_score_player1, physicsClientId=self.client )    
            p.changeVisualShape(self.score2,-1,textureUniqueId = self.number_score_player2, physicsClientId=self.client )       
            p.changeVisualShape(self.score3,-1,textureUniqueId = self.number_score_player1, physicsClientId=self.client )    
            p.changeVisualShape(self.score4,-1,textureUniqueId = self.number_score_player2, physicsClientId=self.client )  
            p.changeVisualShape(self.table_score1,-1,textureUniqueId = self.competitors, physicsClientId=self.client )    
            p.changeVisualShape(self.table_score2,-1,textureUniqueId = self.competitors, physicsClientId=self.client )
