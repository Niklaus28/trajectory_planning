#! /usr/bin/env python

from datetime import datetime
from exoarm_state import RobotState as state
from controller import setup,home_position



status = state.PENDING
interrupted = False;

def ExoArmTerminateAction():
    global interrupted
    interrupted = True

def ExoArmGetStatus():
    global status
    return status

def ExoArmExecute(x, y, z):
    global status
    global interrupted

    interrupted = False
    print("\n\n")
    print("[ExoArmExecute] Received pt: ", x, y, z);
    status = state.RECEIVED

    time.sleep(1)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("[ExoArmExecute] Time Start =", current_time)
    #status = state.ACTIVE
    
    
 
    if status == 2:
        try:
            xyz_targets = np.expand_dims(np.array([x,y,z]),axis=-1)
            setup(xyz_targets)
            status = state.SUCCEEDED
        except KeyboardInterrupt:
            home_position()
            status = state.ABORTED
            sys.exit()

    elif status == 6 or status == 7:
        home_position()
        interrupted = True
        status = state.ABORTED
    
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
