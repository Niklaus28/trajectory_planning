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
resolution = -(2*pi)/0.072

class Point3d(object):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

def get_group():
    global group
    global model
    global z_group
    families = ['rightarm']
    names = ['base','J1','J2']
    z_name = ['linear']
    
    lookup = hebi.Lookup()
    group = lookup.get_group_from_names(families, names)
    if group is None:
        return None

    z_group = lookup.get_group_from_names(families, z_name)
    if z_group is None:
        return None

    # Set gains
    gains_command = hebi.GroupCommand(group.size)
    try:
        gains_command.read_gains("gains/exoarm_gains_plannar_task.xml")
    except Exception as e:
        print('Failed to read gains: {0}'.format(e))
        return None
    if not group.send_command_with_acknowledgement(gains_command):
        print('Failed to receive ack from group')
        return None
    
    z_gains_command = hebi.GroupCommand(z_group.size)
    try:
        z_gains_command.read_gains("gains/exoarm_gains_z.xml")
    except Exception as e:
        
        print('Failed to read gains: {0}'.format(e))
        return None
    if not z_group.send_command_with_acknowledgement(z_gains_command):
        print('Failed to receive ack from group')
        return None

    model = hebi.robot_model.import_from_hrdf("hrdf/exoarm_hrdf.hrdf")
    
    return group,model,z_group

def decay_function(point,stiffness,decay_rate,t):
    #STARTING POINT = 15
    #DECAY RATE = 0.022
    #T = 9 , T for elbow = 3
    starting_point = 15.0
    #decay_rate = 0.05
    #t=2
    #stiffness = 2.5
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
    index = decay_formula2[int(factor*point)]
    return index

def exponential_stiffness(joint_error):
    joint_base = abs(joint_error[0])
    joint_J1 = abs(joint_error[1])
    joint_J2 = abs(joint_error[2])
    
    stiffness_base = 2.5
    stiffness_J1 = 2.5
    stiffness_J2 = 2.3


    if joint_base < 0.5:
        base_stiffness = decay_function(abs(joint_base),stiffness_base,0.05,2)
    else:
        base_stiffness = stiffness_base

    if joint_J1 < 0.5:
        J1_stiffness = decay_function(abs(joint_J1),stiffness_J1,0.05,2)
    else:
        J1_stiffness = stiffness_J1

    if joint_J2 < 0.5:
        J2_stiffness = decay_function(abs(joint_J2),stiffness_J2,0.05,1.5)
    else:
        J2_stiffness = stiffness_J2
    
    return np.expand_dims(np.array([base_stiffness,J1_stiffness,J2_stiffness]),axis=-1)

def obstacle_decay(point):
    starting_point = 1.5
    decay_rate = 0.04
    t=5
    array_size = 50000
    factor = 100
    decay_formula2 = []
    for i in range(array_size):
        decay =  starting_point*pow((1-decay_rate),t)
        decay_formula2.append(decay)
        starting_point = decay
        
    for i in range (array_size):
        decay_formula2[i]= decay_formula2[i]
    
    #plt.plot(decay_formula2[:160])
    
    index = 15*decay_formula2[int(factor*point)]
    return index

def repulsive_force(q_current):
    
    q_base = q_current[0]
    q_J1 = q_current[1]
    q_J2 = q_current[2]
    
    q_base_negative_limit = -pi
    q_base_positive_limit = 1.5
    
    q_J1_negative_limit = -pi
    q_J1_positive_limit = pi
    
    q_J2_negative_limit = -2*pi
    q_J2_positive_limit = 2*pi
      
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
    
def goal_position(new_xyz):
    global group
    global model
    global z_group
    global resolution
    
    command = hebi.GroupCommand(group.size)
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    z_command = hebi.GroupCommand(z_group.size)
    z_feedback = hebi.GroupFeedback(z_group.size)
    z_group.get_next_feedback(reuse_fbk=z_feedback)

    xyz_targets = new_xyz.copy()
    xyz_targets[-1] = 0.3
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
    '''
    ##Verify Joint angle limit
    result = joint_angle_verification(joint_target)
    while not all(result):
        command.effort = np.array([-0.5, 0, 0])
        group.send_command(command)
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
        
        result = joint_angle_verification(joint_target)
    '''    
    #conversion between rotation to linear movement
    
    z_target = new_xyz[-1]
    z_error = (new_xyz[-1] - z_feedback.position) *resolution
    
    group.get_next_feedback(reuse_fbk=feedback)
    current_joint_angle = np.expand_dims(np.array(feedback.position),axis=-1)
    joint_error = joint_target - current_joint_angle
    
    return joint_target,joint_error,z_target,z_error

def home_position(new_xyz):
    global group
    global model
    global z_group
    global resolution
    
    command = hebi.GroupCommand(group.size)
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    z_command = hebi.GroupCommand(z_group.size)
    z_feedback = hebi.GroupFeedback(z_group.size)
    z_group.get_next_feedback(reuse_fbk=z_feedback)

    xyz_targets = new_xyz.copy()
    xyz_targets[-1] = 0.3
    # Choose an "elbow up" initial configuration for IK
    xyz_cols = xyz_targets.shape[1]
    elbow_up_angle = feedback.position
    rotation_target = math_utils.rotate_y(0)
    joint_target = np.empty((group.size, xyz_cols))
    for col in range(xyz_cols):
        ee_position_objective = hebi.robot_model.endeffector_position_objective(new_xyz[:, col])  # define xyz position
        endeffector_so3_objective = hebi.robot_model.endeffector_so3_objective(rotation_target)
        #joint_limit = hebi.robot_model.joint_limit_constraint(minimum=(np.array([-pi,-pi,-pi/2])),maximum=(np.array([pi,pi,pi/2])), weight=1.0)
        #ik_res_angles = model.solve_inverse_kinematics(elbow_up_angle, endeffector_so3_objective,joint_limit, ee_position_objective)
        ik_res_angles = model.solve_inverse_kinematics(elbow_up_angle, endeffector_so3_objective, ee_position_objective)
        joint_target[:, col] = ik_res_angles
    '''        
    result = joint_angle_verification(joint_target)
    while not all(result):
        command.effort = np.array([-0.5, 0, 0])
        group.send_command(command)
        group.get_next_feedback(reuse_fbk=feedback)
    
        # Choose an "elbow up" initial configuration for IK
        xyz_cols = xyz_targets.shape[1]
        elbow_up_angle = feedback.position
        rotation_target = math_utils.rotate_y(0)
        joint_target = np.empty((group.size, xyz_cols))
        for col in range(xyz_cols):
            ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_targets[:, col])  # define xyz position
            endeffector_so3_objective = hebi.robot_model.endeffector_so3_objective(rotation_target)
            ik_res_angles = model.solve_inverse_kinematics(elbow_up_angle, endeffector_so3_objective,joint_limit, ee_position_objective)
            joint_target[:, col] = ik_res_angles
        
        result = joint_angle_verification(joint_target)
        print('verifying')
    #conversion between rotation to linear movement
    '''
    z_target = new_xyz[-1]
    z_current = z_feedback.position / resolution
    z_error_in_metre = (z_target - z_current)
    z_error = z_error_in_metre * resolution
    
    group.get_next_feedback(reuse_fbk=feedback)
    current_joint_angle = np.expand_dims(np.array(feedback.position),axis=-1)
    joint_error = joint_target - current_joint_angle
    
    return joint_target,joint_error,z_target,z_error

def execute_trajectory(joint_target,joint_error,z_target,z_error):
    global group
    global model
    global z_group
    global resolution
        
    command = hebi.GroupCommand(group.size)
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    z_command = hebi.GroupCommand(z_group.size)
    z_feedback = hebi.GroupFeedback(z_group.size)
    z_group.get_next_feedback(reuse_fbk=z_feedback)
    
    while (abs(joint_error[0]) >=0.02) or (abs(joint_error[1]) >=0.025) or (abs(joint_error[2]) >=0.05) or (abs(z_error))>=0.1:
        group.get_next_feedback(reuse_fbk=feedback)
        z_group.get_next_feedback(reuse_fbk=z_feedback)
        
        current_joint_error = np.expand_dims(np.array(feedback.position),axis=-1)
        joint_error = joint_target - current_joint_error
        torque_command = exponential_stiffness(joint_error) * joint_error
        for i in range(group.size):
            if abs(torque_command[i]) > 1.5:
                torque_command[i] = 0.5 * torque_command[i]
        repulsive_torque = repulsive_force(feedback.position)
        command.effort = torque_command +repulsive_torque 

        z_torque =  -z_stiffness(z_error) * z_error      
        #z_torque = z_stiffness(z_error)     
        #z_repulsive = z_limit_repulsive(z_feedback.position)
        z_current = z_feedback.position / resolution
        z_error_in_metre = (z_target - z_current)
        z_error = z_error_in_metre * resolution
        z_command.effort = z_torque
        
        group.send_command(command)
        z_group.send_command(z_command)
        print('joint_error:' +str(joint_error))
        print('z_error:' + str(z_error))
        print('repulsive_force:' +str(repulsive_torque))
        sleep(0.1)
    
    print("success")

def z_stiffness(point):
    starting_point = 4.0
    decay_rate = 0.0001
    t= 2
    stiffness = 2.0
    array_size = 10000
    factor = 100
    offset = stiffness + 0.5
    decay_formula2 = []
    for i in range(array_size):
        decay =  starting_point*pow((1+decay_rate),t)
        decay_formula2.append(decay)
        starting_point = decay

    for i in range (array_size):
        decay_formula2[i]= offset + decay_formula2[i]
    
    #plt.plot(decay_formula2[0:100])
    index = -decay_formula2[int(factor*abs(point))]
    return index

def z_limit_repulsive(position):
    
    if position >= 0:
        force = pi - position
        z_repulsive_torque = -1.5 * obstacle_decay(force)
    elif position <pi/2:
        force = position - 0
        z_repulsive_torque = 1.5 * obstacle_decay(force)
    
    return z_repulsive_torque


def joint_angle_verification(joint_target):
    base_joint_min = -3
    base_joint_max = 3
    J1_min = -3.0
    J1_max = 2.8
    J2_min = -pi
    J2_max = pi
    
    if joint_target[0] > base_joint_max or joint_target[0] < base_joint_min:
        result_base = True
    else:
        result_base = False
    
    if joint_target[1] > J1_max or joint_target[1] < J1_min:
        result_J1 = True
    else:
        result_J1 = False
        
    if joint_target[2] > J2_max or joint_target[2] < J2_min:
        result_J2 = True
    else:
        result_J2 = False
        
    return np.expand_dims(np.array([result_base,result_J1,result_J2]), axis=-1)
    
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
    global z_group
    global resolution
    group,model,z_group = get_group()
    
    try:
        with open('config/user1.yaml') as file:
            config = yaml.safe_load(file)
            pt_home = Point3d(config['home']['x'], config['home']['y'], config['home']['z'])
    except Exception as e:
        print("failed")
        
    
    joint_target,joint_error,z_target,z_error =home_position(np.expand_dims(np.array([pt_home.x,pt_home.y,pt_home.z]),axis=-1))   
    execute_trajectory(joint_target,joint_error,z_target,z_error)
    target = input("Key in goal coordinate\n")
    x, y, z = map(float, target.split())
    joint_target,joint_error,z_target,z_error = goal_position(np.expand_dims(np.array([x,y,z]),axis=-1)) 
    execute_trajectory(joint_target,joint_error,z_target,z_error)
    
if __name__ == '__main__':
    main()