#! /usr/bin/env python

from datetime import datetime
from exoarm_state import RobotState as state

import hebi
from math import pi
from time import sleep, time
import numpy as np
from matplotlib import pyplot as plt

LOG_LEVEL = 10  # DEBUG[10] INFO[20] WARNING[30] ERROR[40] CRITICAL[50]
LOG_HEBI = False

gain_file = "gains/exoarm_gains.xml"
hrdf_file = "hrdf/exoarm_hrdf.hrdf"
user_file = "config/user1.yaml"


class Point3d(object):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class ExoArmHandle():
    def __init__(self):
        self.__status = state.PENDING
        self.__interrupted = False
        self.__go_home = False
        self.__group = None
        self.__model = None
        self.__robot_initialized = False
        self.__gain_file = gain_file
        self.__hrdf_file = hrdf_file
        self.__user_file = user_file
        self.__pt_home = None

    def set_user_file(self, file):
        self.__user_file = file

    def set_gain_file(self, file):
        self.__gain_file = file

    def set_hrdf_file(self, file):
        self.__hrdf_file = file

    def exoarm_init(self):
        import os
        success = True

        if not os.path.isfile(self.__user_file):
            logger.error("user_file %s not exist", self.__user_file)
            success = False

        if not os.path.isfile(self.__gain_file):
            logger.error("gain_file %s not exist", self.__gain_file)
            success = False

        if not os.path.isfile(self.__hrdf_file):
            logger.error("hrdf_file %s not exist", self.__hrdf_file)
            success = False

        if not success:
            return success

        if not self.__load_config():
            logger.error("Config file not loaded")
            success = False

        self.__group, self.__model = self.__get_group()
        if self.__group is None:
            logger.error('Group not found! Check that the family and name of a module on the network matches what is given in the source file.')
            success = False

        if self.__model is None:
            logger.error('Model not loaded! Check hrdf file is provided correctly.')
            success = False

        self.__robot_initialized = success
        return success

    def exoarm_get_status(self):
        return self.__status

    def exoarm_terminate_action(self):
        self.__interrupted = True

    def exoarm_execute(self, x, y, z):
        if (not self.__robot_initialized):
            print("[ExoArmExecute] Robot not initialized")
            self.__status = state.ABORTED
            return self.__status

        self.__interrupted = False
        print("")
        print("[ExoArmExecute] Received pt: ", x, y, z)
        self.__status = state.RECEIVED

        sleep(1)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("[ExoArmExecute] Time Start =", current_time)

        if (int(x) == -100) and (int(y) == -100) and (int(z) == -100):
            print("[ExoArmExecute] Go home request received")
            self.__home_position()
            if (self.__status == state.PENDING):
                print("[ExoArmExecute] Go home request success")
                self.__status = state.SUCCEEDED
            else:
                print("[ExoArmExecute] Go home request end in state: ", self.__status.name)
            return self.__status

        try:
            xyz_target = np.expand_dims(np.array([x, y, z]), axis=-1)
            self.__goal_position(xyz_target)
        except KeyboardInterrupt:
            self.__home_position()
            self.__status = state.ABORTED

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("[ExoArmExecute] Time End   =", current_time)

        return self.__status

    def exoarm_home(self):
        self.__home_position()
        print(self.__status.name)
        print("Ready to execute goal")

    def __load_config(self):
        import yaml
        try:
            with open(self.__user_file) as file:
                config = yaml.safe_load(file)
                self.__pt_home = Point3d(config['home']['x'], config['home']['y'], config['home']['z'])
        except Exception as e:
            logger.error("[LoadConfig]:\n %s", e)
            return False
        return True

    def __get_group(self):
        """
        Helper function to create a group from named modules, and set specified gains on the modules in that group.
        """

        families = ['rightarm']
        names = ['base', 'J1', 'J2', 'elbow']

        lookup = hebi.Lookup()
        sleep(2.0)
        group = lookup.get_group_from_names(families, names)
        if group is None:
            return None, None

        # Set gains
        gains_command = hebi.GroupCommand(group.size)
        try:
            gains_command.read_gains(self.__gain_file)
        except Exception as e:
            print('Failed to read gains: {0}'.format(e))
            return group, None
        if not group.send_command_with_acknowledgement(gains_command):
            print('Failed to receive ack from group')
            return group, None

        model = None
        try:
            model = hebi.robot_model.import_from_hrdf(self.__hrdf_file)
        except Exception as e:
            print('Could not load hrdf: {0}'.format(e))
            return group, None
        model.add_rigid_body([], inertia, mass, output)

        return group, model

    def __setup(self, xyz_target):
        num_joints = self.__group.size
        xyz_col = xyz_target.shape[1]
        feedback = hebi.GroupFeedback(num_joints)
        self.__group.get_next_feedback(reuse_fbk=feedback)
        elbow_up_angle = feedback.position
        joint_target = np.empty((num_joints, xyz_col))

        for col in range(xyz_col):
            ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_target[:, col])  # define xyz position
            ik_res_angles = self.__model.solve_inverse_kinematics(elbow_up_angle, ee_position_objective)
            joint_target[:, col] = ik_res_angles
            elbow_up_angle = ik_res_angles


        command = hebi.GroupCommand(num_joints)
        q_base_negative_limit = -1.5
        q_base_positive_limit = 1.5

        q_J1_negative_limit = -2.7
        q_J1_positive_limit = 2.7

        q_J2_negative_limit = pi/2
        q_J2_positive_limit = 3 * pi/4

        q_J3_negative_limit = -pi/2
        q_J3_positive_limit = pi/4

        while joint_target[0] > q_base_positive_limit or joint_target[0] < q_base_negative_limit:
            if joint_target[0] > q_base_positive_limit:
                command.effort = np.array([-0.5, 0, 0, 0])
            elif joint_target[0] < q_base_negative_limit:
                command.effort = np.array([0.5, 0, 0, 0])

            self.__group.send_command(command)
            self.__group.get_next_feedback(reuse_fbk=feedback)
            elbow_up_angles = feedback.position
            joint_target = np.empty((num_joints, xyz_col))
            logger.debug(joint_target)
            for col in range(xyz_col):
                ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_target[:, col])  # define xyz position
                ik_res_angles = self.__model.solve_inverse_kinematics(elbow_up_angles, ee_position_objective)
                joint_target[:, col] = ik_res_angles
                elbow_up_angles = ik_res_angles

        while joint_target[1] > q_J1_positive_limit or joint_target[1] < q_J1_negative_limit:
            if joint_target[1] > q_J_positive_limit:
                command.effort = np.array([0, -0.5, 0, 0])
            elif joint_target[1] < q_J1_negative_limit:
                command.effort = np.array([0, 0.5, 0, 0])

            self.__group.send_command(command)
            self.__group.get_next_feedback(reuse_fbk=feedback)
            elbow_up_angles = feedback.position
            joint_target = np.empty((num_joints, xyz_col))
            logger.debug(joint_target)
            for col in range(xyz_col):
                ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_target[:, col])  # define xyz position
                ik_res_angles = self.__model.solve_inverse_kinematics(elbow_up_angles, ee_position_objective)
                joint_target[:, col] = ik_res_angles
                elbow_up_angles = ik_res_angles

        while joint_target[2] > q_J2_positive_limit or joint_target[2] < q_J2_negative_limit:
            if joint_target[2] > q_J2_positive_limit:
                command.effort = np.array([0, 0, -0.5, 0])
            elif joint_target[2] < q_J2_negative_limit:
                command.effort = np.array([0, 0, 0.5, 0])

            self.__group.send_command(command)
            self.__group.get_next_feedback(reuse_fbk=feedback)
            elbow_up_angles = feedback.position
            joint_target = np.empty((num_joints, xyz_col))
            logger.debug(joint_target)
            for col in range(xyz_col):
                ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_target[:, col])  # define xyz position
                ik_res_angles = self.__model.solve_inverse_kinematics(elbow_up_angles, ee_position_objective)
                joint_target[:, col] = ik_res_angles
                elbow_up_angles = ik_res_angles

        while joint_target[3] > q_J3_positive_limit or joint_target[3] < q_J3_negative_limit:
            if joint_target[3] > q_J3_positive_limit:
                command.effort = np.array([0, 0, 0, -0.5])
            elif joint_target[3] < q_J3_negative_limit:
                command.effort = np.array([0, 0, 0, 0.5])    

            self.__group.send_command(command)
            self.__group.get_next_feedback(reuse_fbk=feedback)
            elbow_up_angles = feedback.position
            joint_target = np.empty((num_joints, xyz_col))
            logger.debug(joint_target)
            for col in range(xyz_col):
                ee_position_objective = hebi.robot_model.endeffector_position_objective(xyz_target[:, col])  # define xyz position
                ik_res_angles = self.__model.solve_inverse_kinematics(elbow_up_angles, ee_position_objective)
                joint_target[:, col] = ik_res_angles
                elbow_up_angles = ik_res_angles

        if LOG_HEBI:
            log_directory = 'dirs'
            log_filename = 'planar_motion'
            self.__group.start_log(log_directory, log_filename, mkdirs=True)

        self.__group.get_next_feedback(reuse_fbk=feedback)
        joint_error = joint_target - np.expand_dims(np.array(feedback.position), axis=-1)

        return joint_target, joint_error

    def __goal_position(self, xyz_target):
        logger.info("[goal_position]")
        logger.debug("[goal_position] Target: %f %f %f", xyz_target[0], xyz_target[1], xyz_target[2])

        #joint_target, joint_error = self.__setup(xyz_target)
        joint_target = np.expand_dims(np.array([-0.83994198, -2.95503402, -0.92154652,  0.41823718]),axis=-1)
        num_joints = self.__group.size
        feedback = hebi.GroupFeedback(num_joints)
        self.__group.get_next_feedback(reuse_fbk=feedback)
        joint_error = joint_target - np.expand_dims(np.array(feedback.position), axis=-1)
        try:
            self.__execute_trajectory(joint_target, joint_error)
            logger.info("[goal_position] Robot status: %s", self.__status.name)
            if (self.__status == state.SUCCEEDED):
                logger.info("[goal_position] Reached target")
        except Exception as e:
            logger.error("[goal_position] Exception: %s", e)

        if (LOG_HEBI):
            log_file = self.__group.stop_log()
            hebi.util.plot_logs(log_file, 'position', figure_spec=101)
            hebi.util.plot_logs(log_file, 'velocity', figure_spec=102)
            hebi.util.plot_logs(log_file, 'effort', figure_spec=103)

    def __home_position(self):
        self.__go_home = True
        xyz_target = np.expand_dims(np.array([self.__pt_home.x, self.__pt_home.y, self.__pt_home.z]), axis=-1)

        logger.info("[home_position]")
        logger.debug("[home_position] Target: %f %f %f", xyz_target[0], xyz_target[1], xyz_target[2])

        #joint_target, joint_error = self.__setup(xyz_target)
        joint_target = np.expand_dims(np.array([-0.83994198, -2.95503402, -0.92154652,  0.41823718]),axis=-1)
        num_joints = self.__group.size
        feedback = hebi.GroupFeedback(num_joints)
        self.__group.get_next_feedback(reuse_fbk=feedback)
        joint_error = joint_target - np.expand_dims(np.array(feedback.position), axis=-1)

        try:
            self.__execute_trajectory(joint_target, joint_error)
            if (self.__status == state.SUCCEEDED):
                self.__status = state.PENDING
                logger.info("[home_position] Reached home")
        except Exception as e:
            logger.error("[home_position] Exception:\n %s", e)

        self.__go_home = False

    def __execute_trajectory(self, selected_pt, joint_error):
        if not self.__robot_initialized:
            raise RuntimeError("[execute_trajectory] Failed: robot not initialized")

        num_joints = self.__group.size
        command = hebi.GroupCommand(num_joints)
        feedback = hebi.GroupFeedback(num_joints)

        self.__status = state.ACTIVE
        while (abs(joint_error[0]) >= 0.03) or (abs(joint_error[1]) >= 0.03) or (abs(joint_error[2]) >= 0.03) or (abs(joint_error[3]) >= 0.05):
            if (self.__interrupted):
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("[ExoArmExecute] Time Cancel   =", current_time)
                self.__status = state.PREEMPTED
                self.__interrupted = False
                return

            self.__group.get_next_feedback(reuse_fbk=feedback)
            current_joint_error = np.expand_dims(np.array(feedback.position), axis=-1)
            joint_error = selected_pt - current_joint_error
            torque_command = self.__exponential_stiffness(joint_error) * joint_error
            for i in range(num_joints):
                if abs(torque_command[i]) > 1.5:
                    torque_command[i] = 0.5 * torque_command[i]

            repulsive_torque = self.__obstacle_force_function(feedback.position)
            command.effort = torque_command + repulsive_torque
            self.__group.send_command(command)
            sleep(0.05)

        self.__status = state.SUCCEEDED

    def __decay_function(self, point, stiffness, decay_rate, t):
        starting_point = 15.0
        array_size = 100
        factor = 100
        offset = stiffness + 0.5
        decay_formula2 = []

        for i in range(array_size):
            decay = starting_point * pow((1 - decay_rate), t)
            decay_formula2.append(decay)
            starting_point = decay

        for i in range(array_size):
            decay_formula2[i] = offset + decay_formula2[i]

        index = decay_formula2[int(factor * point)]
        return index

    def __exponential_stiffness(self, joint_error):
        joint_base = abs(joint_error[0])
        joint_J1 = abs(joint_error[1])
        joint_J2 = abs(joint_error[2])
        joint_J3 = abs(joint_error[3])

        stiffness = np.empty((4, 1))
        stiffness_base = 1.4
        stiffness_J1 = 1.8
        stiffness_J2 = 2.0
        stiffness_J3 = 2.5

        min_error = 0.03
        max_error = 0.5

        if joint_base < min_error:
            base_stiffness = 10.0
        elif min_error <= joint_base < max_error:
            base_stiffness = self.__decay_function(abs(joint_base), stiffness_base, 0.03, 6)
        else:
            base_stiffness = stiffness_base

        if joint_J1 < min_error:
            J1_stiffness = 15.0
        elif min_error <= joint_J1 < max_error:
            J1_stiffness = self.__decay_function(abs(joint_J1), stiffness_J1, 0.05, 1.5)
        else:
            J1_stiffness = stiffness_J1

        if joint_J2 < min_error:
            J2_stiffness = 15.0
        elif min_error <= joint_J2 < max_error:
            J2_stiffness = self.__decay_function(abs(joint_J2), stiffness_J2, 0.01, 2)
        else:
            J2_stiffness = stiffness_J2

        if joint_J3 < min_error:
            J3_stiffness = 15.0
        elif min_error <= joint_J3 < max_error:
            J3_stiffness = self.__decay_function(abs(joint_J3), stiffness_J3, 0.015, 3)
        else:
            J3_stiffness = stiffness_J3

        stiffness[0] = base_stiffness
        stiffness[1] = J1_stiffness
        stiffness[2] = J2_stiffness
        stiffness[3] = J3_stiffness

        return stiffness

    def __obstacle_decay(self, point):
        starting_point = 1.5
        decay_rate = 0.025
        t = 15
        array_size = 500
        factor = 100
        decay_formula2 = []

        for i in range(array_size):
            decay = starting_point * pow((1 - decay_rate), t)
            decay_formula2.append(decay)
            starting_point = decay

        for i in range(array_size):
            decay_formula2[i] = decay_formula2[i]

        index = decay_formula2[int(factor * point)]
        return index

    def __obstacle_force_function(self, q_current):
        torque_repulsive = np.empty((4, 1))
        q_base = q_current[0]
        q_J1 = q_current[1]
        q_J2 = q_current[2]
        q_J3 = q_current[3]

        q_base_negative_limit = -1.5
        q_base_positive_limit = 1.5

        q_J1_negative_limit = -2.7
        q_J1_positive_limit = 2.7

        q_J2_negative_limit = pi/2
        q_J2_positive_limit = 3 * pi/4

        q_J3_negative_limit = -pi/2
        q_J3_positive_limit = pi/4

        if q_base >= 0:
            dist_base = q_base_positive_limit - q_base
            torque_repulsive_base = -self.__obstacle_decay(dist_base)
        elif q_base < 0:
            dist_base = q_base - q_base_negative_limit
            torque_repulsive_base = self.__obstacle_decay(dist_base)

        if q_J1 >= 0:
            dist_J1 = q_J1_positive_limit - q_J1
            torque_repulsive_J1 = -self.__obstacle_decay(dist_J1)
        elif q_J1 < 0:
            dist_J1 = q_J1 - q_J1_negative_limit
            torque_repulsive_J1 = self.__obstacle_decay(dist_J1)

        if q_J2 > 1.75:
            dist_J2 = q_J2_positive_limit - q_J2
            torque_repulsive_J2 = -self.__obstacle_decay(dist_J2)
        elif q_J2 <= 1.75:
            dist_J2 = q_J2 - q_J2_negative_limit
            torque_repulsive_J2 = self.__obstacle_decay(dist_J2)

        if q_J3 >= 0:
            dist_J3 = q_J3_positive_limit - q_J3
            torque_repulsive_J3 = -self.__obstacle_decay(dist_J3)
        elif q_J3 < 0:
            dist_J3 = q_J3 - q_J3_negative_limit
            torque_repulsive_J3 = self.__obstacle_decay(dist_J3)

        torque_repulsive[0] = torque_repulsive_base
        torque_repulsive[1] = torque_repulsive_J1
        torque_repulsive[2] = torque_repulsive_J2
        torque_repulsive[3] = torque_repulsive_J3

        return torque_repulsive


class ExoArm(ExoArmHandle):
    def __init__(self):
        super().__init__()


def rosmain():
    print("ExoArm ROS main")

    global logger
    import logging
    formatter = logging.Formatter('%(message)s')
    handler = logging.StreamHandler()
    handler.setLevel(LOG_LEVEL)
    handler.setFormatter(formatter)
    logger = logging.getLogger('local_logger')
    logger.setLevel(LOG_LEVEL)
    logger.addHandler(handler)


def main():
    print("ExoArm main")

    global logger
    import logging as logger
    logger.basicConfig(format='%(message)s', level=LOG_LEVEL)

    global status
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gain', type=str, default=gain_file)
    parser.add_argument('--hrdf', type=str, default=hrdf_file)
    parser.add_argument('--user', type=str, default=user_file)
    parser.add_argument('--log', type=bool, default=False)
    args = parser.parse_args()

    exoarm = ExoArm()
    exoarm.set_user_file(args.user)
    exoarm.set_gain_file(args.gain)
    exoarm.set_hrdf_file(args.hrdf)

    if not exoarm.exoarm_init():
        logger.error("ExoArm Initialization Failed")
        exit()

    exoarm.exoarm_home()
    target = input("Enter goal (x y z): ")
    x, y, z = map(float, target.split())
    result = exoarm.exoarm_execute(x, y, z)
    print("RESULT: ", result.name)


if __name__ == "__main__":
    main()
