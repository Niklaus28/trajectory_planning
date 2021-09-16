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
    """
    Helper function to create a group from named modules, and set specified gains on the modules in that group.
    """
    families = ['rightarm']
    names = ['base','J1','J2','elbow']

    lookup = hebi.Lookup()
    #sleep(2.0)
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

    return group,model

def exponential_stiffness(joint_error):
    joint_base = abs(joint_error[0])
    joint_J1 = abs(joint_error[1])
    joint_J2 = abs(joint_error[2])
    joint_J3 = abs(joint_error[3])
    start_range = 5
    end_range = 20

    if joint_base < 0.03:
        base_stiffness = 5.0
    elif joint_base >=0.03 and joint_base < 0.5:
        #base_stiffness = random.randint(start_range,end_range)
        base_stiffness = 2.2
    else:
        base_stiffness = 1.1

    if joint_J1 < 0.03:
        J1_stiffness = 8.0
    elif joint_J1 >=0.03 and joint_J1 < 0.5:
        #J1_stiffness = random.randint(start_range,end_range)
        J1_stiffness = 2.0

    else:
        J1_stiffness = 1.1

    if joint_J2 < 0.03:
        J2_stiffness = 8.0
    elif joint_J2 >=0.03 and joint_J2 < 0.5:
        #J1_stiffness = random.randint(start_range,end_range)
        J2_stiffness = 3.5

    else:
        J2_stiffness = 1.8

    if joint_J3 < 0.03:
        J3_stiffness = 12.0
    elif joint_J3 >=0.03 and joint_J3 < 0.5:
        #J3_stiffness = random.randint(start_range,end_range)
        J3_stiffness = 5.0

    else:
        J3_stiffness = 2.0

    return np.array([base_stiffness,J1_stiffness,J2_stiffness,J3_stiffness])
    

def joint_PMP(xyz):

    duration = 10
    group,model = get_group()
    if group is None:
        print('Group not found! Check that the family and name of a module on the network')
        print('matches what is given in the source file.')
        exit(1)

    num_joints = group.size
    command = hebi.GroupCommand(num_joints)
    #xyz_target = np.expand_dims(np.array([pt_Home.x,pt_Home.y,pt_Home.z]),axis=-1)
    #xyz_target = np.expand_dims(xyz,axis=-1)
    #xyz_col = xyz_target.shape[1]
    #elbow_up_angles = [-pi/6.0, pi/3.0, pi/6.0, 0.0]

    #joint_target = np.empty((group.size, xyz_col))
    #for col in range(xyz_col):
    #    ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_target[:, col]) #define xyz position
    #    ik_res_angles = model.solve_inverse_kinematics(elbow_up_angles, ee_position_objective)
    #    joint_target[:, col] = ik_res_angles
    #desired_angle = joint_target[:,0]
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    desired_angle = np.array([0,  1.57 ,  0.03, 0.03])
    pt_3 = np.array([0.54604268,1.19073391,1.6567339,0.48])
    pt_1 = np.array([-0.84453547,  1.7589885 ,  1.81740582,  0.48])
    pt_2 = np.array([0.2961472 , 0.4095065 , 1.45216823, 0.48])
    home_angle = np.array([0,0,0,0])
    current_joint = np.array([2.88530774e-02, 0, 8.87403381e-04, 2.59414185e-02])
    selected_pt = pt_2
    
    joint_error = (selected_pt - feedback.position)
    stiffness = exponential_stiffness(joint_error)
    stiffness_array = []
    while (abs(joint_error[0]) >=0.04) or (abs(joint_error[1]) >=0.04) or (abs(joint_error[2]) >=0.04) or (abs(joint_error[3]) >=0.10):
        
        group.get_next_feedback(reuse_fbk=feedback)
        current_joint_error = (selected_pt - feedback.position)

        #for i in range(group.size):
        #    if abs(current_joint_error[i]) > abs(joint_error[i]):
        #        stiffness[i] = -stiffness[i]
        #    else:
        #        stiffness[i] = stiffness[i]

        joint_error = current_joint_error
        #if all(np.abs(joint_error)) <= pi:
        #    joint = joint_error
            #print('1')
        #elif all(selected_pt) < pi:
        #    joint = joint_error + np.array([2*pi, 2*pi, 2*pi, 2*pi])
        #    print('2')

        #else:
        #    joint = joint_error - np.array([2*pi, 2*pi, 2*pi, 2*pi])
        #    print('3')

        torque_command = exponential_stiffness(joint_error) * joint_error
        stiffness_array.append(stiffness)
        for i in range(group.size):
            if abs(torque_command[i]) > 1.5:
                torque_command[i] = 0.5
         
        command.effort = torque_command 
        group.send_command(command)
        
        print(joint_error)

def execute_trajectory(group, model, waypoints, feedback):
  """
  Helper function to actually execute the trajectory on a group of modules
  """
  global go_home
  num_joints = group.size
  command = hebi.GroupCommand(num_joints)
  duration = 1
  stiffness = 0.001
  #duration = trajectory.duration

  start = time()
  t = time() - start
  feedback = hebi.GroupFeedback(group.size)
  while t < duration:
        if (interrupted) and (not go_home):
            home_position()
            
        # Get feedback and update the timer
        group.get_next_feedback(reuse_fbk=feedback)
        t = time() - start

        ## Get new commands from the trajectory
        #pos_cmd, vel_cmd, acc_cmd = trajectory.get_state(t)

        # Calculate commanded efforts to assist with tracking the trajectory.
        # Gravity Compensation uses knowledge of the arm's kinematics and mass to
        # compensate for the weight of the arm. Dynamic Compensation uses the
        # kinematics and mass to compensate for the commanded accelerations of the arm.
        eff_grav = math_utils.get_grav_comp_efforts(model, feedback.position, [0, 0, 1])
        # NOTE: dynamic compensation effort computation has not yet been added to the APIs
        desired_angle = waypoints[:,1]
        command_angle = (feedback.position - desired_angle)
        eff_cmd = stiffness * command_angle
        print(eff_cmd)
        #vel_check = any(vel > 1.5 for vel in np.abs(vel_cmd))
        #eff_check = any(eff > 2.0 for eff in np.abs(eff_cmd))

        #if vel_check == 1 or eff_check == 1 :
        #    home_position()
        #    go_home = True
        #else:
            # Fill in the command and send commands to the arm
        #    command.position = pos_cmd
        #    command.velocity = vel_cmd
        #    command.effort = eff_cmd
        #    group.send_command(command)

        command.effort = eff_cmd + eff_grav
        group.send_command(command)

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
    
    # Go to the XYZ positions at four corners of the box, and create a rotation matrix
    # that has the end effector point straight forward.
    #home_pos = np.expand_dims(np.array([0.8552229404,0.0604135394,-0.0069096684]),axis=-1)
    #xyz_targets = np.concatenate((home_pos,xyz),axis=1)    
    xyz_cols = xyz_targets.shape[1]

    # Choose an "elbow up" initial configuration for IK
    elbow_up_angles = [-pi/6.0, pi/3.0, pi/6.0, 0.0]
    
    joint_targets = np.empty((group.size, xyz_cols+1))
    for col in range(xyz_cols):
        ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_targets[:, col]) #define xyz position
        ik_res_angles = model.solve_inverse_kinematics(elbow_up_angles, ee_position_objective)
        joint_targets[:, col] = ik_res_angles
        elbow_up_angles = ik_res_angles # reset seed after each loop, define the next initial joint angle of robot
    joint_targets[:,-1] = joint_targets[:,0]
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
    execute_trajectory(group, model, waypoints, feedback)

    ## for more point to achieve
    if xyz_cols < 3:
        pass
    else:
        for col in range(xyz_cols-2):
            waypoints[:, 0] = feedback.position
            waypoints[:, 1] = joint_targets[:, col+2] 
            trajectory = hebi.trajectory.create_trajectory(time_vector, waypoints)
            execute_trajectory(group, model, trajectory, feedback)

    log_file = group.stop_log()
    hebi.util.plot_logs(log_file,'position',figure_spec=101)
    hebi.util.plot_logs(log_file,'velocity',figure_spec=102)
    hebi.util.plot_logs(log_file,'effort',figure_spec=103)

def home_position():
    
    group,model = get_group()
    if group is None:
        print('Group not found! Check that the family and name of a module on the network')
        print('matches what is given in the source file.')
        exit(1)
    
    xyz_target = np.expand_dims(np.array([0.6629155874,0.1086072624,-0.0039570332]),axis=-1)
    xyz_col = xyz_target.shape[1]
    
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    elbow_up_angle = feedback.position
    print('reached home position')

    joint_target = np.empty((group.size, xyz_col))
    for col in range(xyz_col):
        ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_target[:, col]) #define xyz position
        ik_res_angles = model.solve_inverse_kinematics(elbow_up_angle, ee_position_objective)
        joint_target[:, col] = ik_res_angles
    
    # Set up feedback object, and start logging
  
        
    feedback = hebi.GroupFeedback(group.size)
    desired_angle = np.array([-0.01046623, 0.3000, 0.60515541, -0.72230321])
    waypoints = np.empty((group.size, 2))
    group.get_next_feedback(reuse_fbk=feedback)
    

    waypoints[:, 0] = feedback.position
    waypoints[:, 1] = desired_angle
    #time_vector = [0, 5]  # Seconds for the motion - do this slowly
    #trajectory = hebi.trajectory.create_trajectory(time_vector, waypoints)

    # Call helper function to execute this motion on the robot
    execute_trajectory(group, model, waypoints, feedback)

def PMP(desired_cartesian_coor):
    group,model = get_group()
    num_joints = group.size
    command = hebi.GroupCommand(num_joints)
    duration =  1
    k = 0.01
    start = time()
    t = time() - start
    while t < duration:
            
        feedback = hebi.GroupFeedback(group.size)
        group.get_next_feedback(reuse_fbk=feedback)

        robot_joint_angle = feedback.position
        robot_cartesian_coor = model.get_end_effector(robot_joint_angle)[0:3,3]
        desired_cartesian_coor = np.array([0.925,0, 0.31561023])
        desired_force = k * (desired_cartesian_coor - robot_cartesian_coor)
        robot_jacobian = np.transpose(model.get_jacobian_end_effector(robot_joint_angle))
        eff_cmd = robot_jacobian *desired_force
        eff_grav = math_utils.get_grav_comp_efforts(model, feedback.position, [0, 0, 1])
        
        command.effort = eff_cmd + eff_grav
        group.send_command(command)


        

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
    xyz_pt3 = np.expand_dims(np.array([0.01428743,0.5567537,0.2]),axis=-1)
    xyz_pt1 = np.expand_dims(np.array([0.29830876, 0.19300742, 0.2]),axis=-1)
    xyz_pt2 = np.expand_dims(np.array([0.54231006, 0.54188025, 0.2]),axis=-1)
    if (interrupted):
        try:
            xyz_targets = np.expand_dims(np.array([x,y,z]),axis=-1)
            setup(xyz_targets)
            status = state.SUCCEEDED
        except KeyboardInterrupt:
            home_position()
            status = state.ABORTED
            #sys.exit()
    else:
        pass

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("[ExoArmExecute] Time End   =", current_time)

    return status

def main():
    global status
    home_position()
    print("ExoArm main")

    #ExoArmExecute(1.0, 1.0, 1.0)
    xyz = np.array([[0.7835284472,0.3937107325],
                    [0.2143040746,0.6715198159],
                    [0.0002663136,0.0056665242]])
    setup(xyz)

if __name__ == "__main__":   
    main()