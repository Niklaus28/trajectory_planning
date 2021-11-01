import hebi
from math import pi
from time import sleep, time
import numpy as np
from matplotlib import pyplot as plt

from math import pi
from time import sleep, time
import numpy as np
from matplotlib import pyplot as plt
gain_file = "gains/exoarm_gains.xml"
hrdf_file = "hrdf/exoarm_hrdf.hrdf"
def get_group():
    global group
    global model
    """
    Helper function to create a group from named modules, and set specified gains on the modules in that group.
    """

    families = ['rightarm']
    names = ['base', 'J1', 'J2', 'elbow']

    lookup = hebi.Lookup()
    #sleep(2.0)
    group = lookup.get_group_from_names(families, names)
    if group is None:
        return None, None

    # Set gains
    gains_command = hebi.GroupCommand(group.size)
    try:
        gains_command.read_gains(gain_file)
    except Exception as e:
        print('Failed to read gains: {0}'.format(e))
        return group, None
    if not group.send_command_with_acknowledgement(gains_command):
        print('Failed to receive ack from group')
        return group, None

    model = None
    try:
        model = hebi.robot_model.import_from_hrdf(hrdf_file)
    except Exception as e:
        print('Could not load hrdf: {0}'.format(e))
        return group, None

    return group, model

def get_point():
    num_joints = group.size
    feedback = hebi.GroupFeedback(num_joints)
    group.get_next_feedback(reuse_fbk=feedback)
    curr_angle = feedback.position
    print(curr_angle)

def main():
    get_group()
    get_point()

if __name__ == '__main__':
    main()