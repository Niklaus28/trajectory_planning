#! /usr/bin/env python

from datetime import datetime
from exoarm_state import RobotState as state

import hebi
from math import pi
from time import sleep, time
import numpy as np
from matplotlib import pyplot as plt

log_level = 20  # DEBUG[10] INFO[20] WARNING[30] ERROR[40] CRITICAL[50]
plot = False

status = state.PENDING
interrupted = False
go_home = False

group = None
model = None
robot_initialized = False

gain_file = "gains/exoarm_gains.xml"
hrdf_file = "hrdf/exoarm_hrdf.hrdf"
user_file = "config/user1.yaml"

class Point3d(object):
    def __init__( self, x, y, z):
        self.x, self.y, self.z = x, y, z

def get_group():

    """
    Helper function to create a group from named modules, and set specified gains on the modules in that group.
    """
    families = ['rightarm']
    names = ['base','J1','J2','elbow']

    lookup = hebi.Lookup()
    sleep(2.0)
    group = lookup.get_group_from_names(families, names)
    if group is None:
        return None, None

    # Set gains
    gains_command = hebi.GroupCommand(group.size)
    try:
        gains_command.read_gains(gain_file)
    except Exception as e:
        print('Failed to read gains: {0}'.format(e))
        return None, None
    if not group.send_command_with_acknowledgement(gains_command):
        print('Failed to receive ack from group')
        return None, None

    model = None
    try:
        model = hebi.robot_model.import_from_hrdf(hrdf_file)
    except Exception as e:
        print('Could not load HRDF: {0}'.format(e))
        return group, None

    return group, model

def decay_function(point,stiffness,decay_rate,t):

    starting_point = 15.0
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
        
    index = decay_formula2[int(factor*point)]
    return index
    
def exponential_stiffness(joint_error):
    joint_base = abs(joint_error[0])
    joint_J1 = abs(joint_error[1])
    joint_J2 = abs(joint_error[2])
    joint_J3 = abs(joint_error[3])
    
    stiffness_base = 1.1
    stiffness_J1 = 1.8 
    stiffness_J2 = 2.0
    stiffness_J3 = 2.5

    if joint_base < 0.03:
        base_stiffness = 10.0
    elif joint_base >=0.03 and joint_base < 0.5:
        base_stiffness = decay_function(abs(joint_base),stiffness_base,0.015,9)
    else:
        base_stiffness = stiffness_base

    if joint_J1 < 0.03:
        J1_stiffness = 15.0
    elif joint_J1 >=0.03 and joint_J1 < 0.5:
        J1_stiffness = decay_function(abs(joint_J1),stiffness_J1,0.05,1.5)
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
        
    index = decay_formula2[int(factor*point)]
    return index

def obstacle_force_function(q_current):
    
    torque_repulsive = np.empty((4,1))
    q_base = q_current[0]
    q_J1 = q_current[1]
    q_J2 = q_current[2]
    q_J3 = q_current[3]
    
    q_base_negative_limit = -pi
    q_base_positive_limit = pi
    
    q_J1_negative_limit = -pi
    q_J1_positive_limit = pi
    
    q_J2_negative_limit = pi/2
    q_J2_positive_limit = 3*pi/4
    
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

def execute_trajectory(selected_pt,joint_error):
    """
    Helper function to actually execute the trajectory on a group of modules
    """
    global go_home
    global status
    global interrupted
    global group
    global model

    if not robot_initialized:
        raise RuntimeError("[execute_trajectory] Failed: robot not initialized")

    num_joints = group.size
    command = hebi.GroupCommand(num_joints)
    feedback = hebi.GroupFeedback(group.size)

    status = state.ACTIVE
    while (abs(joint_error[0]) >=0.03) or (abs(joint_error[1]) >=0.03) or (abs(joint_error[2]) >=0.03) or (abs(joint_error[3]) >=0.03):
        if (interrupted):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("[ExoArmExecute] Time Cancel   =", current_time)
            status = state.PREEMPTED
            interrupted = False
            return
            
        group.get_next_feedback(reuse_fbk=feedback)
        current_joint_error = np.expand_dims(np.array(feedback.position),axis=-1)
        joint_error = selected_pt - current_joint_error
        torque_command = exponential_stiffness(joint_error) * joint_error
        for i in range(group.size):
            if abs(torque_command[i]) > 1.5:
                torque_command[i] = 0.5 * torque_command[i]
        
        repulsive_torque = obstacle_force_function(feedback.position) 
        command.effort = torque_command +repulsive_torque
        group.send_command(command)
        sleep(0.05)

    status = state.SUCCEEDED

def setup(xyz_targets):
    logger.debug("[setup]")
    logger.debug("[setup] xyz_targets: %f %f %f", xyz_targets[0], xyz_targets[1], xyz_targets[2])
    
    # Go to the XYZ positions at four corners of the box, and create a rotation matrix
    # that has the end effector point straight forward.
    xyz_cols = xyz_targets.shape[1]
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    # Choose an "elbow up" initial configuration for IK
    elbow_up_angles = feedback.position
    
    joint_targets = np.empty((group.size, xyz_cols))
    for col in range(xyz_cols):
        ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_targets[:, col]) #define xyz position
        ik_res_angles = model.solve_inverse_kinematics(elbow_up_angles, ee_position_objective)
        joint_targets[:, col] = ik_res_angles
        elbow_up_angles = ik_res_angles # reset seed after each loop, define the next initial joint angle of robot

    # Set up feedback object, and start logging
    log_directory = 'dirs'
    log_filename = 'planar_motion'
    group.start_log(log_directory,log_filename, mkdirs=True)

    group.get_next_feedback(reuse_fbk=feedback)
    joint_error = joint_targets- np.expand_dims(np.array(feedback.position),axis=-1)

    # Call helper function to execute this motion on the robot
    try:
        execute_trajectory(joint_targets,joint_error)
        logger.info("[setup] robot status: %s", status.name)
        if (status == state.SUCCEEDED):
            logger.info("[setup] Reached target")
    except Exception as e:
        logger.error("[setup] exception: %s", e)

    if (plot):
        log_file = group.stop_log()
        hebi.util.plot_logs(log_file,'position',figure_spec=101)
        hebi.util.plot_logs(log_file,'velocity',figure_spec=102)
        hebi.util.plot_logs(log_file,'effort',figure_spec=103)

def home_position():
    global go_home
    global status
    global group
    global model
    global pt_home
    go_home = True
        
    logger.info("[home_position]")
    xyz_target = np.expand_dims(np.array([pt_home.x, pt_home.y, pt_home.z]), axis=-1)
    logger.debug("[home_position] xyz_targets: %f %f %f", xyz_target[0], xyz_target[1], xyz_target[2])
    xyz_col = xyz_target.shape[1]
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    elbow_up_angle = feedback.position

    joint_target = np.empty((group.size, xyz_col))
    for col in range(xyz_col):
        ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_target[:, col]) #define xyz position
        ik_res_angles = model.solve_inverse_kinematics(elbow_up_angle, ee_position_objective)
        joint_target[:, col] = ik_res_angles
    
    # Set up feedback object, and start logging
    group.get_next_feedback(reuse_fbk=feedback)
    joint_error = joint_target- np.expand_dims(np.array(feedback.position),axis=-1)

    # Call helper function to execute this motion on the robot
    try:
        execute_trajectory(joint_target,joint_error)
        if (status == state.SUCCEEDED):
            status = state.PENDING
            logger.info("[home_position] Reached home")
    except Exception as e:
        logger.error("[home_position] exception:\n %s", e)

    go_home = False

def setUserFile(file):
    global user_file
    user_file = file

def setGainFile(file):
    global gain_file
    gain_file = file

def setHrdfFile(file):
    global hrdf_file
    hrdf_file = file

def LoadConfig():
    import yaml
    global pt_home

    logger.info("user_file: %s", user_file)
    try:
        with open(user_file) as file:
            config = yaml.safe_load(file)
            pt_home = Point3d(config['home']['x'], config['home']['y'], config['home']['z'])
    except Exception as e:
        logger.error("[LoadConfig]:\n %s", e)
        return False
    return True

def ExoArmHome():
    home_position()
    print(status.name)
    print("Ready to execute goal")

def ExoArmInit():
    global group
    global model
    global robot_initialized

    success = True

    if not LoadConfig():
        logger.error("Config file not loaded")
        success = False

    group, model = get_group()
    if group is None:
        logger.error('Group not found! Check that the family and name of a module on the network matches what is given in the source file.')
        success = False

    if model is None:
        logger.error('Model not loaded! Check hrdf file is provided correctly.')
        success = False

    robot_initialized = success
    return success
    
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
    global robot_initialized

    if (not robot_initialized):
        print("[ExoArmExecute] Robot not initialized")
        status = state.ABORTED
        return status

    interrupted = False
    print("")
    print("[ExoArmExecute] Received pt: ", x, y, z);
    status = state.RECEIVED

    sleep(1)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("[ExoArmExecute] Time Start =", current_time)

    if (int(x)==-100) and (int(y)==-100) and (int(z)==-100):
        print("[ExoArmExecute] Go home request received")
        home_position()
        if (status == state.PENDING):
            print("[ExoArmExecute] Go home request success")
            status = state.SUCCEEDED
        else:
            print("[ExoArmExecute] Go home request end in state: ", status.name)
        return status

    try:
        xyz_targets = np.expand_dims(np.array([x,y,z]),axis=-1)
        setup(xyz_targets)
    except KeyboardInterrupt:
        home_position()
        status = state.ABORTED

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("[ExoArmExecute] Time End   =", current_time)

    return status

def rosmain():
    print("ExoArm ROS main")

    global logger
    import logging
    formatter = logging.Formatter('%(message)s')
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    logger = logging.getLogger('local_logger')
    logger.setLevel(log_level)
    logger.addHandler(handler)

def main():
    print("ExoArm main")

    global logger
    import logging as logger
    logger.basicConfig(format='%(message)s', level=log_level)

    global status
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gain', type=str, default=gain_file)
    parser.add_argument('--hrdf', type=str, default=hrdf_file)
    parser.add_argument('--user', type=str, default=user_file)
    parser.add_argument('--plot', type=bool, default=False)
    args = parser.parse_args()

    setGainFile(args.gain)
    setHrdfFile(args.hrdf)
    setUserFile(args.user)

    logger.info(" - gain file: %s", gain_file)
    logger.info(" - hrdf file: %s", hrdf_file)
    logger.info(" - user file: %s", user_file)

    if not ExoArmInit():
        logger.error("ExoArm Initialization Failed")
        exit()

    ExoArmHome()
    input("Press Enter to continue...")
    result = ExoArmExecute(1.0, 1.0, 1.0)
    print("RESULT: ", result.name)

if __name__ == "__main__":   
    main()