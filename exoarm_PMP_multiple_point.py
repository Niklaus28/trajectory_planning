#! /usr/bin/env python

from datetime import datetime
from random import random, randrange

from numpy.core.defchararray import join, startswith
from exoarm_state import RobotState as state

import random
import hebi
from math import pi
from time import sleep, time
import numpy as np
from matplotlib import pyplot as plt
from util import math_utils

#status = state.PENDING
interrupted = False
go_home = False

def get_group():
    families = ['rightarm']
    names = ['base','J1','J2','elbow']
    lookup = hebi.Lookup()
    group = lookup.get_group_from_names(families, names)
    if group is None:
        return None

    # Set gains
    gains_command = hebi.GroupCommand(group.size)
    try:
        gains_command.read_gains("gains/exoarm_gains_torquecontrol.xml")
    except Exception as e:
        print('Failed to read gains: {0}'.format(e))
        return None
    if not group.send_command_with_acknowledgement(gains_command):
        print('Failed to receive ack from group')
        return None

    model = hebi.robot_model.import_from_hrdf("hrdf/exoarm_hrdf.hrdf")
    model.add_link("X5",0.27,0)
    model.add_end_effector("custom")
    return group,model

def decay_function(point,stiffness,decay_rate,t):
    starting_point = 15.0
    #decay_rate = 0.03
    #t=6
    #stiffness = 1.4
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
    
    #plt.plot(decay_formula2[0:100])
    #plt.legend('label1','label2','label3','label4')
    index = decay_formula2[int(factor*point)]
    return index
    
def exponential_stiffness(joint_error):
    joint_base = abs(joint_error[0])
    joint_J1 = abs(joint_error[1])
    joint_J2 = abs(joint_error[2])
    joint_J3 = abs(joint_error[3])
    stiffness_base = 1.4
    stiffness_J1 = 1.8 
    stiffness_J2 = 2.0
    stiffness_J3 = 2.5

    if joint_base < 0.03:
        base_stiffness = 10.0
    elif joint_base >=0.03 and joint_base < 0.5:
        base_stiffness = decay_function(abs(joint_base),stiffness_base,0.03,6)
    else:
        base_stiffness = stiffness_base

    if joint_J1 < 0.03:
        J1_stiffness = 15.0
    elif joint_J1 >=0.03 and joint_J1 < 0.5:
        J1_stiffness = decay_function(abs(joint_J1),stiffness_J1,0.04,1.5)
    else:
        J1_stiffness = stiffness_J1

    if joint_J2 < 0.03:
        J2_stiffness = 15.0
    elif joint_J2 >=0.03 and joint_J2 < 0.5:
        J2_stiffness = decay_function(abs(joint_J2),stiffness_J2,0.01,2)

    else:
        J2_stiffness = stiffness_J2

    if joint_J3 < 0.03:
        J3_stiffness = 15.0
    elif joint_J3 >=0.03 and joint_J3 < 0.5:
        J3_stiffness = decay_function(abs(joint_J3),stiffness_J3,0.015,3)
    else:
        J3_stiffness = stiffness_J3
    
    return np.expand_dims(np.array([base_stiffness,J1_stiffness,J2_stiffness,J3_stiffness]),axis=-1)

def obstacle_decay(point):
    starting_point = 1.5
    decay_rate = 0.025
    t=15
    array_size = 500
    factor = 100
    decay_formula2 = []
    for i in range(array_size):
        decay =  starting_point*pow((1-decay_rate),t)
        decay_formula2.append(decay)
        starting_point = decay
        
    for i in range (array_size):
        decay_formula2[i]= decay_formula2[i]
    
    #plt.plot(decay_formula2[:160])
    
    index = decay_formula2[int(factor*point)]
    return index

def obstacle_force_function(q_current):
    
    torque_repulsive = np.empty((4,1))
    q_base = q_current[0]
    q_J1 = q_current[1]
    q_J2 = q_current[2]
    q_J3 = q_current[3]
    
    q_base_negative_limit = -1.5
    q_base_positive_limit = 1.5
    
    q_J1_negative_limit = -2.7
    q_J1_positive_limit = 0
    
    q_J2_negative_limit = -1.2
    q_J2_positive_limit = 0.6
    
    q_J3_negative_limit = -pi/2
    q_J3_positive_limit = pi/4
        
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
    
    if q_J3 >= 0:
        dist_J3 = q_J3_positive_limit - q_J3
        torque_repulsive_J3 = -obstacle_decay(dist_J3)
    elif q_J3 < 0:
        dist_J3 = q_J3 - q_J3_negative_limit
        torque_repulsive_J3 = obstacle_decay(dist_J3)
    
    torque_repulsive[0] = torque_repulsive_base
    torque_repulsive[1] = torque_repulsive_J1
    torque_repulsive[2] = torque_repulsive_J2
    torque_repulsive[3] = torque_repulsive_J3
    
    return torque_repulsive

def Point_1():
    global group
    global model
    group,model = get_group()
    num_joints = group.size
    command = hebi.GroupCommand(num_joints)
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    joint_target = np.expand_dims(np.array([-0.83994198, -2.95503402, -0.92154652,  0.41823718]),axis=-1)
    
    group.get_next_feedback(reuse_fbk=feedback)
    current_joint_angle = np.expand_dims(np.array(feedback.position),axis=-1)
    joint_error = joint_target- current_joint_angle
    while (abs(joint_error[0]) >=0.03) or (abs(joint_error[1]) >=0.03) or (abs(joint_error[2]) >=0.03) or (abs(joint_error[3]) >=0.05):
        group.get_next_feedback(reuse_fbk=feedback)
        current_joint_error = np.expand_dims(np.array(feedback.position),axis=-1) 
        joint_error = joint_target - current_joint_error
        print(joint_error)
        stiff = exponential_stiffness(joint_error) 
        torque_command = stiff * joint_error
        for i in range(group.size):
            if abs(torque_command[i]) > 1.5:
                torque_command[i] = 0.5 * torque_command[i]
                
        repulsive_torque = obstacle_force_function(feedback.position) 
        command.effort = torque_command +repulsive_torque
        group.send_command(command)
        
        sleep(0.05)

def Point_2():
    global group
    global model
        
    num_joints = group.size
    command = hebi.GroupCommand(num_joints)
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    joint_target = np.expand_dims(np.array([-0.63994198, -2.35503402, 0.32154652,  0.41823718]),axis=-1)
    
    group.get_next_feedback(reuse_fbk=feedback)
    current_joint_angle = np.expand_dims(np.array(feedback.position),axis=-1)
    joint_error = joint_target- current_joint_angle
    while (abs(joint_error[0]) >=0.03) or (abs(joint_error[1]) >=0.03) or (abs(joint_error[2]) >=0.03) or (abs(joint_error[3]) >=0.05):
        group.get_next_feedback(reuse_fbk=feedback)
        current_joint_error = np.expand_dims(np.array(feedback.position),axis=-1) 
        joint_error = joint_target - current_joint_error
        print(joint_error)
        stiff = exponential_stiffness(joint_error) 
        torque_command = stiff * joint_error
        for i in range(group.size):
            if abs(torque_command[i]) > 1.5:
                torque_command[i] = 0.5 * torque_command[i]
                
        repulsive_torque = obstacle_force_function(feedback.position) 
        command.effort = torque_command +repulsive_torque
        group.send_command(command)
        sleep(0.05)
        
def Point_3():
    global group
    global model
        
    num_joints = group.size
    command = hebi.GroupCommand(num_joints)
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    joint_target = np.expand_dims(np.array([-0.43994198, -1.95503402, -0.32154652,  0.41823718]),axis=-1)
    
    group.get_next_feedback(reuse_fbk=feedback)
    current_joint_angle = np.expand_dims(np.array(feedback.position),axis=-1)
    joint_error = joint_target- current_joint_angle
    while (abs(joint_error[0]) >=0.03) or (abs(joint_error[1]) >=0.03) or (abs(joint_error[2]) >=0.03) or (abs(joint_error[3]) >=0.05):
        group.get_next_feedback(reuse_fbk=feedback)
        current_joint_error = np.expand_dims(np.array(feedback.position),axis=-1) 
        joint_error = joint_target - current_joint_error
        stiff = exponential_stiffness(joint_error) 
        torque_command = stiff * joint_error
        for i in range(group.size):
            if abs(torque_command[i]) > 1.5:
                torque_command[i] = 0.5 * torque_command[i]
                
        repulsive_torque = obstacle_force_function(feedback.position) 
        command.effort = torque_command +repulsive_torque
        group.send_command(command)
        sleep(0.05)
        
def Point_4():
    global group
    global model
        
    num_joints = group.size
    command = hebi.GroupCommand(num_joints)
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    joint_target = np.expand_dims(np.array([-0.83994198, -2.95503402, -0.92154652,  0.41823718]),axis=-1)
    
    group.get_next_feedback(reuse_fbk=feedback)
    current_joint_angle = np.expand_dims(np.array(feedback.position),axis=-1)
    joint_error = joint_target- current_joint_angle
    while (abs(joint_error[0]) >=0.03) or (abs(joint_error[1]) >=0.03) or (abs(joint_error[2]) >=0.03) or (abs(joint_error[3]) >=0.05):
        group.get_next_feedback(reuse_fbk=feedback)
        current_joint_error = np.expand_dims(np.array(feedback.position),axis=-1) 
        joint_error = joint_target - current_joint_error
        stiff = exponential_stiffness(joint_error) 
        torque_command = stiff * joint_error
        for i in range(group.size):
            if abs(torque_command[i]) > 1.5:
                torque_command[i] = 0.5 * torque_command[i]
                
        repulsive_torque = obstacle_force_function(feedback.position) 
        command.effort = torque_command +repulsive_torque
        group.send_command(command)
        sleep(0.05)

def ExoArmTerminateAction():
    global interrupted
    interrupted = True

def ExoArmGetStatus():
    global status
    return status

def ExoArmExecute(x, y, z):
    global status
    global interrupted
    global go_home

    interrupted = False
    print("\n\n")
    print("[ExoArmExecute] Received pt: ", x, y, z);
    status = state.RECEIVED

    time.sleep(1)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("[ExoArmExecute] Time Start =", current_time)
    #status = state.ACTIVE

    test_1 = np.array([0.18762295, 0.12799542, 0.10009696])
    try:
        xyz_targets = np.expand_dims(test_1,axis=-1)
        joint_PMP(xyz_targets)
        print('home_position')
            #xyz_targets = np.expand_dims(np.array([x,y,z]),axis=-1)
            #setup(xyz_targets)
            #status = state.SUCCEEDED
    except KeyboardInterrupt:
        home_position()
        status = state.ABORTED
        #sys.exit()
    
    test_2 = np.array([0.22185977, 0.39639634, 0.10009648])
    try:
        xyz_targets = np.expand_dims(test_2,axis=-1)
        joint_PMP(xyz_targets)
        print('point1')
            #xyz_targets = np.expand_dims(np.array([x,y,z]),axis=-1)
            #setup(xyz_targets)
            #status = state.SUCCEEDED
    except KeyboardInterrupt:
        home_position()
        status = state.ABORTED
        #sys.exit()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("[ExoArmExecute] Time End   =", current_time)

    return status

def main():
    home = np.array([0.0700284 , 0.0723787 , 0.10009857])
    try:
        xyz_targets = np.expand_dims(home,axis=-1)
        joint_PMP(xyz_targets)
        print('home_position')
    except KeyboardInterrupt:
        home_position()
        status = state.ABORTED
    
    test_2 = np.array([0.1329625 , 0.35062519, 0.10009726])
    try:
        xyz_targets = np.expand_dims(test_2,axis=-1)
        joint_PMP(xyz_targets)
        print('point1')

    except KeyboardInterrupt:
        home_position()
        status = state.ABORTED

if __name__ == "__main__":   
    main()