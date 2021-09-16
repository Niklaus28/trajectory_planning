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

def execute_trajectory(group, model, trajectory, feedback):
    """
    Helper function to actually execute the trajectory on a group of modules
    """
    global go_home
    global status
    global interrupted

    if not robot_initialized:
        raise RuntimeError("[execute_trajectory] Failed: robot not initialized")

    num_joints = group.size
    command = hebi.GroupCommand(num_joints)
    duration = trajectory.duration

    start = time()
    t = time() - start

    status = state.ACTIVE
    while t < duration:
        if (interrupted):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("[ExoArmExecute] Time Cancel   =", current_time)
            status = state.PREEMPTED
            interrupted = False
            return
            
        # Get feedback and update the timer
        group.get_next_feedback(reuse_fbk=feedback)
        t = time() - start

        # Get new commands from the trajectory
        pos_cmd, vel_cmd, acc_cmd = trajectory.get_state(t)

        # Calculate commanded efforts to assist with tracking the trajectory.
        # Gravity Compensation uses knowledge of the arm's kinematics and mass to
        # compensate for the weight of the arm. Dynamic Compensation uses the
        # kinematics and mass to compensate for the commanded accelerations of the arm.
        eff_cmd = math_utils.get_grav_comp_efforts(model, feedback.position, [0, 0, 1])
        # NOTE: dynamic compensation effort computation has not yet been added to the APIs

        vel_check = any(vel > 1.5 for vel in np.abs(vel_cmd))
        eff_check = any(eff > 2.0 for eff in np.abs(eff_cmd))

        if vel_check == 1 or eff_check == 1 :
            home_position()
            status = state.FAILED
            return
        else:
            # Fill in the command and send commands to the arm
            command.position = pos_cmd
            command.velocity = vel_cmd
            command.effort = eff_cmd
            group.send_command(command)

    status = state.SUCCEEDED

def get_grav_comp_efforts(robot_model, positions, gravityVec):
    # Normalize gravity vector (to 1g, or 9.8 m/s^2)
    normed_gravity = gravityVec / np.linalg.norm(gravityVec) * 9.81

    jacobians = robot_model.get_jacobians('CoM', positions)
    # Get torque for each module
    # comp_torque = J' * wrench_vector
    # (for each frame, sum this quantity)
    comp_torque = np.zeros((robot_model.dof_count, 1))

    # Wrench vector
    wrench_vec = np.zeros(6)  # For a single frame; this is (Fx/y/z, tau x/y/z)
    num_frames = robot_model.get_frame_count('CoM')

    for i in range(num_frames):
        # Add the torques for each joint to support the mass at this frame
        wrench_vec[0:3] = normed_gravity * robot_model.masses[i]
        comp_torque += jacobians[i].transpose() * np.reshape(wrench_vec, (6, 1))

    return np.squeeze(comp_torque)

def setup(xyz_targets):
    logger.debug("[setup]")
    logger.debug("[setup] xyz_targets: %f %f %f", xyz_targets[0], xyz_targets[1], xyz_targets[2])
    
    # Go to the XYZ positions at four corners of the box, and create a rotation matrix
    # that has the end effector point straight forward.
    #home_pos = np.expand_dims(np.array([pt_home.x, pt_home.y, pt_home.z]),axis=-1)
    #xyz_targets = np.concatenate((home_pos, xyz),axis=1)
    xyz_cols = xyz_targets.shape[1]

    # Choose an "elbow up" initial configuration for IK
    elbow_up_angles = [-pi/6.0, pi/3.0, pi/6.0, 0.0]
    
    joint_targets = np.empty((group.size, xyz_cols))
    for col in range(xyz_cols):
        ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_targets[:, col]) #define xyz position
        ik_res_angles = model.solve_inverse_kinematics(elbow_up_angles, ee_position_objective)
        joint_targets[:, col] = ik_res_angles
        elbow_up_angles = ik_res_angles # reset seed after each loop, define the next initial joint angle of robot

    # Set up feedback object, and start logging
    feedback = hebi.GroupFeedback(group.size)
    log_directory = 'dirs'
    log_filename = 'planar_motion'
    group.start_log(log_directory,log_filename, mkdirs=True)

    waypoints = np.empty((group.size, 2))
    group.get_next_feedback(reuse_fbk=feedback)
    waypoints[:, 0] = feedback.position
    waypoints[:, 1] = joint_targets[:, 1]
    time_vector = [0, 5]  # Seconds for the motion - do this slowly
    trajectory = hebi.trajectory.create_trajectory(time_vector, waypoints)

    # Call helper function to execute this motion on the robot
    try:
        execute_trajectory(group, model, trajectory, feedback)
        logger.info("[setup] robot status: %s", status.name)
        if (status == state.SUCCEEDED):
            logger.info("[setup] Reached target")
    except Exception as e:
        logger.error("[setup] exception: %s", e)

    ## for more point to achieve
    # if xyz_cols < 3:
    #     pass
    # else:
    #     for col in range(xyz_cols-2):
    #         waypoints[:, 0] = feedback.position
    #         waypoints[:, 1] = joint_targets[:, col+2] 
    #         vel = np.zeros((group.size,2))
    #         acc = np.zeros((group.size,2))
    #         trajectory = hebi.trajectory.create_trajectory(time_vector, waypoints,np.transpose(vel),np.transpose(acc))
    #         execute_trajectory(group, model, trajectory, feedback)

    if (plot):
        log_file = group.stop_log()
        hebi.util.plot_logs(log_file,'position',figure_spec=101)
        hebi.util.plot_logs(log_file,'velocity',figure_spec=102)
        hebi.util.plot_logs(log_file,'effort',figure_spec=103)

def home_position():
    global go_home
    global status
    go_home = True

    logger.info("[home_position]")
    xyz_target = np.expand_dims(np.array([pt_home.x, pt_home.y, pt_home.z]), axis=-1)
    logger.debug("[home_position] xyz_targets: %f %f %f", xyz_target[0], xyz_target[1], xyz_target[2])
    xyz_col = xyz_target.shape[1]
    elbow_up_angle = [-pi/6.0, pi/3.0, pi/6.0, 0.0]

    joint_target = np.empty((group.size, xyz_col))
    for col in range(xyz_col):
        ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_target[:, col]) #define xyz position
        ik_res_angles = model.solve_inverse_kinematics(elbow_up_angle, ee_position_objective)
        joint_target[:, col] = ik_res_angles
    
    # Set up feedback object, and start logging
    feedback = hebi.GroupFeedback(group.size)
    
    waypoints = np.empty((group.size, 2))
    group.get_next_feedback(reuse_fbk=feedback)
    waypoints[:, 0] = feedback.position
    waypoints[:, 1] = joint_target[:, 0]
    time_vector = [0, 5]  # Seconds for the motion - do this slowly
    trajectory = hebi.trajectory.create_trajectory(time_vector, waypoints)

    # Call helper function to execute this motion on the robot
    try:
        execute_trajectory(group, model, trajectory, feedback)
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
    global math_utils
    from .util import math_utils as math_utils

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
    global math_utils
    from util import math_utils as math_utils

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