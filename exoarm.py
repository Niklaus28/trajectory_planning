#! /usr/bin/env python

from datetime import datetime
from exoarm_state import RobotState as state

import hebi
from math import pi
from time import sleep, time
import numpy as np
from matplotlib import pyplot as plt

status = state.PENDING
interrupted = False
go_home = False

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
        return None

    # Set gains
    gains_command = hebi.GroupCommand(group.size)
    try:
        gains_command.read_gains("gains/exoarm_gains.xml")
    except Exception as e:
        print('Failed to read gains: {0}'.format(e))
        return None
    if not group.send_command_with_acknowledgement(gains_command):
        print('Failed to receive ack from group')
        return None

    model = hebi.robot_model.import_from_hrdf("hrdf/exoarm_hrdf.hrdf")

    return group,model

def execute_trajectory(group, model, trajectory, feedback):
  """
  Helper function to actually execute the trajectory on a group of modules
  """
  global go_home
  num_joints = group.size
  command = hebi.GroupCommand(num_joints)
  duration = trajectory.duration

  start = time()
  t = time() - start

  while t < duration:
        if (interrupted) and (not go_home):
            home_position()
            
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
            exit(1)
        else:
            # Fill in the command and send commands to the arm
            command.position = pos_cmd
            command.velocity = vel_cmd
            command.effort = eff_cmd
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
    xyz_cols = xyz_targets.shape[1]

    # Choose an "elbow up" initial configuration for IK
    elbow_up_angles = [-pi/3.0, pi / 2.0, pi/6.0]

    joint_targets = np.empty((group.size, xyz_cols))
    for col in range(xyz_cols):
        #so3_objective = hebi.robot_model.endeffector_so3_objective(rotation_target)   #create a rotation matrix
        ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_targets[:, col]) #define xyz position
        #ik_res_angles = model.solve_inverse_kinematics(elbow_up_angles, so3_objective, ee_position_objective)
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
    waypoints[:, 1] = joint_targets[:, 0]
    time_vector = [0, 5]  # Seconds for the motion - do this slowly
    vel = np.zeros((2,group.size))
    acc = np.zeros((2,group.size))
    trajectory = hebi.trajectory.create_trajectory(time_vector, waypoints,np.transpose(vel),np.transpose(acc))

    # Call helper function to execute this motion on the robot
    execute_trajectory(group, model, trajectory, feedback)

    ## for more point to achieve
    if xyz_cols < 2:
        pass
    else:
        for col in range(xyz_cols-1):
            waypoints[:, 0] = feedback.position
            waypoints[:, 1] = joint_targets[:, 0] 
            vel = np.zeros((group.size,2))
            acc = np.zeros((group.size,2))
            trajectory = hebi.trajectory.create_trajectory(time_vector, waypoints,np.transpose(vel),np.transpose(acc))
            execute_trajectory(group, model, trajectory, feedback)

    log_file = group.stop_log()
    hebi.util.plot_logs(log_file,'position',figure_spec=101)
    hebi.util.plot_logs(log_file,'velocity',figure_spec=102)
    hebi.util.plot_logs(log_file,'effort',figure_spec=103)

def home_position():
    xyz_target = np.expand_dims(np.array([0.845534,0.173321,0.236279]),axis=2)
    xyz_col = xyz_target.shape[1]
    elbow_up_angle = [-pi/3.0, pi / 2.0, pi/6.0]
    print('reached home position')

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
    vel = np.zeros((2,group.size))
    acc = np.zeros((2,group.size))
    trajectory = hebi.trajectory.create_trajectory(time_vector, waypoints,np.transpose(vel),np.transpose(acc))

    # Call helper function to execute this motion on the robot
    execute_trajectory(group, model, trajectory, feedback)

    
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
    
    
    
    if (interrupted):
        try:
            xyz_targets = np.expand_dims(np.array([x,y,z]),axis=-1)
            setup(xyz_targets)
            status = state.SUCCEEDED
        except KeyboardInterrupt:
            home_position()
            status = state.ABORTED
            sys.exit()
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
    ExoArmExecute(1.0, 1.0, 1.0)

if __name__ == "__main__":   
    main()
