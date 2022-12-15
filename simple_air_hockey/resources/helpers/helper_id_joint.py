
import pybullet as p

def get_id_joints_revolution(id_urdf):
        joints = []
        number_of_joints = p.getNumJoints(id_urdf)

        for joint_number in range(number_of_joints):
            info = p.getJointInfo(id_urdf, joint_number)
            if 'base_to_arm_1' in str(info[1]) or 'arm1_to_arm_2' in str(info[1]):
                joints.append(info[0]) #Agrega el joint de revolucion
        return joints