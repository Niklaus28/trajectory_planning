#!/usr/bin/env python3

from __future__ import print_function
from cProfile import label
from turtle import color
from hebi_code.srv import Targets,TargetsResponse

import rospy
import hebi
import math
from math import pi
from time import sleep, time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from util import math_utils
import numpy as np
import yaml
'''
# Add the root folder of the repository to the search path for modules
import os, sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path = [root_path] + sys.path
'''

gain_file = "/home/niklauslim/catkin_ws/src/hebi_code/src/hebi-python-examples/basic/gains/exoarm_gains_traj_planning.xml"
hrdf_file = "/home/niklauslim/catkin_ws/src/hebi_code/src/hebi-python-examples/basic/hrdf/exoarm_hrdf_parallelogram_version_2.hrdf"
user_file = "/home/niklauslim/catkin_ws/src/hebi_code/src/hebi-python-examples/basic/config/user2.yaml"

class Point3d(object):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

# A helper function to create a group from named modules, and set specified gains on the modules in that group.
def get_group():
  families = ['rightarm']
  names = ["base", "right_2", "left_2"]
  lookup = hebi.Lookup()
  sleep(2.0)
  group = lookup.get_group_from_names(families, names)
  if group is None:
    return None

  # Set gains
  gains_command = hebi.GroupCommand(group.size)
  try:
    gains_command.read_gains(gain_file)
  except Exception as e:
    print('Failed to read gains: {0}'.format(e))
    return None
  if not group.send_command_with_acknowledgement(gains_command):
    print('Failed to receive ack from group')
    return None
  
  model = hebi.robot_model.import_from_hrdf(hrdf_file)
  
  return group,model

def IK_calculation(targets):
    global group
    global model
    xyz_cols = targets.shape[1]
    
    # Convert these to joint angle waypoints using IK solutions for each of the xyz locations. Copy the initial waypoint
    # at the end so we close the square.
    feedback = hebi.GroupFeedback(group.size)
    group.get_next_feedback(reuse_fbk=feedback)
    # Choose an "elbow up" initial configuration for IK
    elbow_up_angles = feedback.position
    #version 1
    #    minimum_joint_limit = np.array([-2*pi/6,-1.9,-40/180*pi])
    #    maximum_joint_limit = np.array([pi/4,-10/180*pi,65/180*pi])
    #version 2 left and right change
    minimum_joint_limit = np.array([-pi/4,-0.65,-1.25])
    maximum_joint_limit = np.array([pi/4,0.6,0.55])
    joint_targets = np.empty((group.size, xyz_cols))
    
    for col in range(xyz_cols):
      ee_position_objective = hebi.robot_model.endeffector_position_objective(targets)
      joint_limit_objective = hebi.robot_model.joint_limit_constraint(minimum=minimum_joint_limit,maximum=maximum_joint_limit,weight=1.0)
      ik_res_angles = model.solve_inverse_kinematics(elbow_up_angles, ee_position_objective, joint_limit_objective)
      joint_targets[:, col] = ik_res_angles
    
    waypoints = np.empty((group.size, 2))
    group.get_next_feedback(reuse_fbk=feedback)
    waypoints[:, 0] = feedback.position
    waypoints[:, 1] = joint_targets[:, 0]
    time_vector = [0.0, 5.0]  # Seconds for the motion - do this slowly
    trajectory = hebi.trajectory.create_trajectory(time_vector, waypoints)
    
    return trajectory,joint_targets

def callback(resp1):
  global trajectory
  global group
  global model
  global joint_targets
  global t
  global start
  t = Point3d(resp1.x, resp1.y, resp1.z)
  cartesian_tartget = np.expand_dims(np.array([t.x,t.y,t.z]),axis=-1)
  trajectory,joint_targets = IK_calculation(cartesian_tartget)
  duration = trajectory.duration
  xyz = model.get_end_effector(joint_targets)[0:3,3]
  start = time()
  return TargetsResponse("Target Executed")


def main():
  global group
  global model
  global trajectory
  global t
  global feedback
  global joint_targets
  global start
  
  group,model = get_group()
  feedback = hebi.GroupFeedback(group.size)
  group.get_next_feedback(reuse_fbk=feedback)
  command = hebi.GroupCommand(group.size)

  command_array = []
  command_pos_0 = []
  command_pos_1 = []
  command_pos_2 = []
  time_array =[]
  feedback_array =[]
  error_array = []
  x_feedback = []
  y_feedback = []
  z_feedback = []
  x_command =[]
  y_command =[]
  z_command =[]
  error_cartesian =[]
  vel_feedback=[]
  command_vel_0=[]
  command_vel_1 =[]
  command_vel_2=[]
  vel_error_arr=[]

  try:
    with open(user_file) as file:
      config = yaml.safe_load(file)
      pt_home = Point3d(config['home']['x'], config['home']['y'], config['home']['z'])
      cartesian_tartget = np.expand_dims(np.array([pt_home.x,pt_home.y,pt_home.z]),axis=-1)
  except Exception as e:
      print("failed")
  
  # Set up feedback object, and start logging 
  trajectory, joint_targets = IK_calculation(cartesian_tartget)
  duration = trajectory.duration
  services = rospy.Service('target_topic',Targets, callback)
  
  start = time()

  while not rospy.is_shutdown():
    group.get_next_feedback(reuse_fbk=feedback)
    joint_error = np.array(joint_targets) - np.expand_dims(feedback.position,axis=-1)
    t = time() - start
    # Get new commands from the trajectory
    if t > duration:
      t = duration
      pos_cmd, vel_cmd, acc_cmd = trajectory.get_state(t)
    else:
      pos_cmd, vel_cmd, acc_cmd = trajectory.get_state(t)

    # Calculate commanded efforts to assist with tracking the trajectory.
    # Gravity Compensation uses knowledge of the arm's kinematics and mass to
    # compensate for the weight of the arm. Dynamic Compensation uses the
    # kinematics and mass to compensate for the commanded accelerations of the arm.
    eff_cmd = model.get_grav_comp_efforts(feedback.position, np.array([0.0, 0.0, 1.0]))
    # NOTE: dynamic compensation effort computation has not yet been added to the APIs
    
    # Fill in the command and send commands to the arm
    command.position = pos_cmd
    command.velocity = vel_cmd
    command.effort = eff_cmd
    group.send_command(command)
    group.get_next_feedback(reuse_fbk=feedback)
    current_cartesian = model.get_end_effector(feedback.position)[0:3,3]
    command_cartesian = model.get_end_effector(pos_cmd)[0:3,3]
    #print('feedback:',current_cartesian)
    #print('command:',command_cartesian)
    cartesian_error = (np.array(command_cartesian)-np.array(current_cartesian))
    joint_error = feedback.position - pos_cmd
    pose = model.get_end_effector(feedback.position)[0:3,3]
    vel_error = feedback.velocity - vel_cmd
    #print(cartesian_error)
    #print('feedback:', feedback.position)
    #print('joint target:', joint_targets)
    print('current cartersian coor :', pose)
    #print('joint error:', joint_error)

    command_pos_0.append(pos_cmd[0])
    command_pos_1.append(pos_cmd[1])
    command_pos_2.append(pos_cmd[2])
    command_vel_0.append(vel_cmd[0])
    command_vel_1.append(vel_cmd[1])
    command_vel_2.append(vel_cmd[2])
    time_array.append(t)
    feedback_array.append(feedback.position)
    vel_feedback.append(feedback.velocity)
    error_array.append(joint_error)
    vel_error_arr.append(vel_error)
    x_feedback.append(current_cartesian[0])
    y_feedback.append(current_cartesian[1])
    z_feedback.append(current_cartesian[2])
    x_command.append(command_cartesian[0])
    y_command.append(command_cartesian[1])
    z_command.append(command_cartesian[2])
    error_cartesian.append(cartesian_error)
    #command_array.append(command.position[])
    
  command_pos_0 = np.array(command_pos_0)
  command_pos_1 = np.array(command_pos_1)
  command_pos_2 = np.array(command_pos_2)
  vel_feedback = np.array(vel_feedback)
  feedback_array = np.array(feedback_array)
  error_array = np.array(error_array)
  vel_error_arr = np.array(vel_error_arr)
  error_cartesian = np.array(error_cartesian)

  '''
  plt.figure(1)
  plt.plot(time_array ,feedback_array[:,0],label='feedback',color='green', marker='o')
  plt.plot(time_array, command_pos_0,label='command',color='red',linestyle='dashed')
  plt.title("Joint 1")
  plt.xlabel('time(s)')
  plt.ylabel('rad')
  plt.legend()
  
  plt.figure(2)
  plt.plot(time_array ,feedback_array[:,1],label='feedback',color='green', marker='o')
  plt.plot(time_array, command_pos_1,label='command',color='red',linestyle='dashed')
  plt.title("Joint 2")
  plt.xlabel('time(s)')
  plt.ylabel('rad')
  plt.legend()

  plt.figure(3)
  plt.plot(time_array ,feedback_array[:,2],label='feedback',color='green', marker='o')
  plt.plot(time_array, command_pos_2,label='command',color='red',linestyle='dashed')
  plt.title("Joint 3")
  plt.xlabel('time(s)')
  plt.ylabel('rad')
  plt.legend()
  

  plt.figure(4)
  plt.plot(time_array ,vel_feedback[:,0],label='feedback',color='green', marker='o')
  plt.plot(time_array, command_vel_0,label='command',color='red',linestyle='dashed')
  plt.title("Joint 1")
  plt.xlabel('time(s)')
  plt.ylabel('rad/s')
  plt.legend()

  plt.figure(5)
  plt.plot(time_array ,vel_feedback[:,1],label='feedback',color='green', marker='o')
  plt.plot(time_array, command_vel_1,label='command',color='red',linestyle='dashed')
  plt.title("Joint 2")
  plt.xlabel('time(s)')
  plt.ylabel('rad/s')
  plt.legend()

  plt.figure(6)
  plt.plot(time_array ,vel_feedback[:,2],label='feedback',color='green', marker='o')
  plt.plot(time_array, command_vel_2,label='command',color='red',linestyle='dashed')
  plt.title("Joint 3")
  plt.xlabel('time(s)')
  plt.ylabel('rad/s')
  plt.legend()

    

   plt.figure(8)
  plt.plot(time_array,vel_error_arr[:,0],label='1',color='green')
  plt.plot(time_array,vel_error_arr[:,1],label='2',color='blue')
  plt.plot(time_array,vel_error_arr[:,2],label='3',color='red')
  plt.title("joint vel error")
  plt.xlabel('time(s)')
  plt.ylabel('rad/s')
  plt.legend()

  plt.figure(4)
  plt.plot(time_array ,x_feedback,label='feedback',color='green', marker='o')
  plt.plot(time_array,x_command,label='command',color='black',linestyle='dashed')
  plt.title("X")
  plt.xlabel('time(s)')
  plt.ylabel('m')
  plt.legend()
  
  plt.figure(5)
  plt.plot(time_array ,y_feedback,label='feedback',color='green', marker='o')
  plt.plot(time_array, y_command,label='command',color='black',linestyle='dashed')
  plt.title("Y")
  plt.xlabel('time(s)')
  plt.ylabel('m')
  plt.legend()

  plt.figure(6)
  plt.plot(time_array ,z_feedback,label='feedback',color='green', marker='o')
  plt.plot(time_array, z_command,label='command',color='black',linestyle='dashed')
  plt.title("Z")
  plt.xlabel('time(s)')
  plt.ylabel('m')
  plt.legend()

  '''
  plt.figure(7)
  plt.plot(time_array,error_array[:,0],label='1',color='green')
  plt.plot(time_array,error_array[:,1],label='2',color='blue')
  plt.plot(time_array,error_array[:,2],label='3',color='red')
  plt.title("joint error")
  plt.xlabel('time(s)')
  plt.ylabel('rad')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  rospy.init_node('exoarm')
  main()

  

