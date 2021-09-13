#! /usr/bin/env python

from enum import Enum

class RobotState(Enum):
    PENDING = 1
    RECEIVED = 2
    ACTIVE = 3
    PREEMPTED = 4
    SUCCEEDED = 5
    FAILED = 6
    ABORTED = 7
