import pybullet as p
import os

class Mesa:
    def __init__(self, client):
        f_name = os.path.join(os.path.dirname(__file__), 'mesa.urdf')
        f_name_image = os.path.join(os.path.dirname(__file__), 'imagen_tablero.png')
        self.client = client
        self.mesa_cargada=p.loadURDF(fileName=f_name,
                        basePosition=[0, 0, 2.2],
                        useFixedBase=True,
                        physicsClientId=client)
        
        self.hockey_textura = p.loadTexture(textureFilename=f_name_image,physicsClientId=self.client)
        p.changeVisualShape(self.mesa_cargada ,-1, textureUniqueId = self.hockey_textura, physicsClientId=self.client )
