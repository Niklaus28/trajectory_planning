#!/usr/bin/env python3

from __future__ import print_function

import sys
import rospy
from hebi_code.srv import *

def target_client(x, y, z):
    rospy.wait_for_service('target_topic')
    try:
        target_coor = rospy.ServiceProxy('target_topic', Targets)
        resp1 = target_coor(x, y, z)
        return resp1.response
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def usage():
    return "%s [x y z]"%sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 4:
        x = float(sys.argv[1])
        y = float(sys.argv[2])
        z = float(sys.argv[3])
    else:
        print(usage())
        sys.exit(1)
    target_client(x,y,z)
    
    print("goal sent")