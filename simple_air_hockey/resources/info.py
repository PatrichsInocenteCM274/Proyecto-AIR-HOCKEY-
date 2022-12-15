import pybullet as p
p.connect(p.DIRECT)
car = p.loadURDF('score.urdf')
number_of_joints = p.getVisualShapeData(car)
for link in number_of_joints:
    print(link)