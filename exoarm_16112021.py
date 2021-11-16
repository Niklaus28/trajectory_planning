from datetime import datetime
from random import random, randrange

from numpy.core.defchararray import join, startswith
from exoarm_state import RobotState as state

import yaml
import random
import hebi
import math
from math import pi
from time import sleep, time
import numpy as np
from matplotlib import pyplot as plt
from util import math_utils

#status = state.PENDING
interrupted = False
go_home = False

class Point3d(object):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        
def get_group():
    families = ['rightarm']
    names = ['base','J1','J2']
    #com = [0.00410,0.00074,0.06684]
    #inertia = [0.00156069598, 0.00160407352, 0.00027313650, -0.00000017350, 0.00010727634, 0.000020488]
    #mass = 0.25157
    #output = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.09491],[0,0,0,1]])
    lookup = hebi.Lookup()
    group = lookup.get_group_from_names(families, names)
    if group is None:
        return None

    # Set gains
    gains_command = hebi.GroupCommand(group.size)
    try:
        gains_command.read_gains("gains/exoarm_gains_planar_task.xml")
    except Exception as e:
        print('Failed to read gains: {0}'.format(e))
        return None
    if not group.send_command_with_acknowledgement(gains_command):
        print('Failed to receive ack from group')
        return None

    model = hebi.robot_model.import_from_hrdf("hrdf/exoarm_planar_task.hrdf")
    
    return group,model

def decay_function(point,stiffness,decay_rate,t):
    #STARTING POINT = 15
    #DECAY RATE = 0.022
    #T = 9 , T for elbow = 3
    starting_point = 15.0
    #decay_rate = 0.03
    #t=2
    stiffness = 3
    array_size = 100
    factor = 100
    offset = stiffness + 0.5
    decay_formula2 = []
    for i in range(array_size):
        decay =  starting_point*pow((1-decay_rate),t)
        decay_formula2.append(decay)
        starting_point = decay

    for i in range (array_size):
        decay_formula2[i]= offset + decay_formula2[i]
    
    plt.plot(decay_formula2[0:100])
    #plt.legend('label1','label2','label3','label4')
    index = decay_formula2[int(factor*point)]
    return index
    
def exponential_stiffness(joint_error):
    joint_base = abs(joint_error[0])
    joint_J1 = abs(joint_error[1])
    joint_J2 = abs(joint_error[2])
    
    stiffness_base = 1.5
    stiffness_J1 = 3.0
    stiffness_J2 = 2.5


    if joint_base < 0.5:
        base_stiffness = decay_function(abs(joint_base),stiffness_base,0.03,4.5)
    else:
        base_stiffness = stiffness_base

    if joint_J1 < 0.5:
        J1_stiffness = decay_function(abs(joint_J1),stiffness_J1,0.03,2)
    else:
        J1_stiffness = stiffness_J1

    if joint_J2 < 0.5:
        J2_stiffness = decay_function(abs(joint_J2),stiffness_J2,0.01,2)
    else:
        J2_stiffness = stiffness_J2
    
    return np.expand_dims(np.array([base_stiffness,J1_stiffness,J2_stiffness]),axis=-1)

def obstacle_decay(point):
    starting_point = 1.5
    decay_rate = 0.04
    t=5
    array_size = 500
    factor = 100
    #offset = stiffness + 0.5
    decay_formula2 = []
    for i in range(array_size):
        decay =  starting_point*pow((1-decay_rate),t)
        decay_formula2.append(decay)
        starting_point = decay
        
    for i in range (array_size):
        decay_formula2[i]= decay_formula2[i]
    
    plt.plot(decay_formula2[:160])
    
    index = 15*decay_formula2[int(factor*point)]
    return index

def repulsive_force(q_current):
    
    torque_repulsive = np.empty((3,1))
    q_base = q_current[0]
    q_J1 = q_current[1]
    q_J2 = q_current[2]
    
    q_base_negative_limit = -1.8
    q_base_positive_limit = 1.8
    
    q_J1_negative_limit = -3.0
    q_J1_positive_limit = 3.0
    
    q_J2_negative_limit = -2
    q_J2_positive_limit = 3*pi/4
      
    if q_base >= 0:
        dist_base = q_base_positive_limit - q_base
        torque_repulsive_base = -obstacle_decay(dist_base)
    elif q_base <0:
        dist_base = q_base - q_base_negative_limit
        torque_repulsive_base = obstacle_decay(dist_base)
        
    if q_J1 >= 0:
        dist_J1 = q_J1_positive_limit - q_J1
        torque_repulsive_J1 = -obstacle_decay(dist_J1)
    elif q_J1 < 0:
        dist_J1 = q_J1 - q_J1_negative_limit
        torque_repulsive_J1 = obstacle_decay(dist_J1)
    
    if q_J2 > 1.75:
        dist_J2 = q_J2_positive_limit - q_J2
        torque_repulsive_J2 = -obstacle_decay(dist_J2)
    elif q_J2 <= 1.75:
        dist_J2 = q_J2 -q_J2_negative_limit
        torque_repulsive_J2 = obstacle_decay(dist_J2)
   
    return np.expand_dims(np.array([torque_repulsive_base,torque_repulsive_J1,torque_repulsive_J2]),axis=-1)
    
def goal_position(xyz_targets):
    global group
    global model
        
    num_joints = group.size
    command = hebi.GroupCommand(num_joints)
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    
    # Choose an "elbow up" initial configuration for IK
    xyz_cols = xyz_targets.shape[1]
    elbow_up_angle = feedback.position
    rotation_target = math_utils.rotate_z(0)
    joint_target = np.empty((group.size, xyz_cols))
    for col in range(xyz_cols):
        ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_targets[:, col])  # define xyz position
        endeffector_so3_objective = hebi.robot_model.endeffector_so3_objective(rotation_target)
        ik_res_angles = model.solve_inverse_kinematics(elbow_up_angle, endeffector_so3_objective, ee_position_objective)
        joint_target[:, col] = ik_res_angles
    
    ##Verify Joint angle limit
    
    
    #selected_pt = np.expand_dims(loc_6,axis=-1) + offset_angle
    group.get_next_feedback(reuse_fbk=feedback)
    current_joint_angle = np.expand_dims(np.array(feedback.position),axis=-1)
    joint_error = joint_target - current_joint_angle
    
    return joint_target,joint_error

def home_position(xyz_targets):
    global group
    global model
    
    num_joints = group.size
    command = hebi.GroupCommand(num_joints)
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    
    # Choose an "elbow up" initial configuration for IK
    xyz_cols = xyz_targets.shape[1]
    elbow_up_angle = feedback.position
    rotation_target = math_utils.rotate_z(0)
    joint_target = np.empty((group.size, xyz_cols))
    for col in range(xyz_cols):
        ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_targets[:, col])  # define xyz position
        endeffector_so3_objective = hebi.robot_model.endeffector_so3_objective(rotation_target)
        ik_res_angles = model.solve_inverse_kinematics(elbow_up_angle, endeffector_so3_objective, ee_position_objective)
        joint_target[:, col] = ik_res_angles
            
    log_directory = 'dirs'
    log_filename = 'planar_motion'
    group.start_log(log_directory,log_filename, mkdirs=True)
    
    #selected_pt = np.expand_dims(loc_6,axis=-1) + offset_angle
    group.get_next_feedback(reuse_fbk=feedback)
    current_joint_angle = np.expand_dims(np.array(feedback.position),axis=-1)
    joint_error = joint_target - current_joint_angle
    
    return joint_target,joint_error

def execute_trajectory(joint_target,joint_error):
    global group
    global model
        
    num_joints = group.size
    command = hebi.GroupCommand(num_joints)
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    
    while (abs(joint_error[0]) >=0.02) or (abs(joint_error[1]) >=0.02) or (abs(joint_error[2]) >=0.02):
        group.get_next_feedback(reuse_fbk=feedback)
        current_joint_error = np.expand_dims(np.array(feedback.position),axis=-1)
        joint_error = joint_target - current_joint_error
        torque_command = exponential_stiffness(joint_error) * joint_error
        for i in range(group.size):
            if abs(torque_command[i]) > 1.5:
                torque_command[i] = 0.5 * torque_command[i]
                        
        repulsive_torque = repulsive_force(feedback.position)
        print(repulsive_torque)
        command.effort = torque_command +repulsive_torque 
        group.send_command(command)
        #print(joint_error)
        sleep(0.05)
    
    print("success")
    
def inverse_function_3_joint(xyz_target):
    global group
    global model
    
    link_1 = 0.3
    link_2 = 0.3
    link_3 = 0.3
    xyz_targets = xyz_target.copy()
    #xyz_targets[-1] = 0.19491
    target_position = xyz_targets[:2]
    
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    current_joint_angle = feedback.position
    joint_angle_2 = current_joint_angle[1]
    joint_angle_3 = current_joint_angle[2]

    while joint_angle_2 < 0 or joint_angle_3 <0:  ##verify >0 or <0
        if joint_angle_2 < 0:
            cmd_2 = 0.5
        else: 
            cmd_2 = 0
                
            
        if joint_angle_3 < 0:
            cmd_3 = 0.5
        else:
            cmd_3 = 0
            
        command = np.array([0, cmd_2, cmd_3])   ##verify format
        group.send_command(command)

        group.get_next_feedback(reuse_fbk=feedback)
        current_joint_angle = feedback.position
            
            
        joint_angle_2 = current_joint_angle[1]
        joint_angle_3 = current_joint_angle[2]
    
    x_3 = xyz_target[0]
    y_3 = xyz_target[1]
    phi_angle = math.atan2(y_3,x_3)
    x_2 = x_3 - link_3*math.cos(phi_angle)
    y_2 = y_3 - link_3*math.sin(phi_angle)


    c_sq = math.pow(x_2,2) + math.pow(y_2,2)
    cosine_beta = ((math.pow(link_1,2) + math.pow(link_2,2) - c_sq) / (2*link_1*link_2))
    beta_angle = math.acos(cosine_beta)
    cosine_alpha = ((c_sq + math.pow(link_1,2) - math.pow(link_2,2)) / (2*link_1*(math.sqrt(c_sq))))
    alpha_angle = math.acos(cosine_alpha)
    gamma_angle = math.acos(y_2/x_2)
    joint_angle_1 = gamma_angle - alpha_angle
    joint_angle_2 = pi - beta_angle
    joint_angle_3 = phi_angle - joint_angle_1 - joint_angle_2
    
    joint_target = np.expand_dims(np.array([joint_angle_1,joint_angle_2, joint_angle_3]),axis=-1)
    group.get_next_feedback(reuse_fbk=feedback)

    joint_error = joint_target - np.expand_dims(np.array(feedback.position), axis=-1)
    
    return joint_target,joint_error

def main():
    global group
    global model
    group,model = get_group()
    
    try:
        with open('config/user1.yaml') as file:
            config = yaml.safe_load(file)
            pt_home = Point3d(config['home']['x'], config['home']['y'], config['home']['z'])
    except Exception as e:
        print("failed")
        
    
    joint_target,joint_error =home_position(np.expand_dims(np.array([pt_home.x,pt_home.y,pt_home.z]),axis=-1))   
    execute_trajectory(joint_target,joint_error)
    target = input("Key in goal coordinate\n")
    x, y, z = map(float, target.split())
    joint_target,joint_error = goal_position(np.expand_dims(np.array([x,y,z]),axis=-1)) 
    execute_trajectory(joint_target,joint_error)
    
if __name__ == '__main__':
    main()