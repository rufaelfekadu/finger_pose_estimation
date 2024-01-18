import leap
import time
import pandas as pd
import numpy as np
import argparse
import sys
from threading import Thread
from datetime import datetime
import os
import json
import sys
sys.path.append('../')
from lock import lock
# Your Recording class import here
from .NeuroLeap import get_bone_core_angles
_TRACKING_MODES = {
    leap.TrackingMode.Desktop: "Desktop",
    leap.TrackingMode.HMD: "HMD",
    leap.TrackingMode.ScreenTop: "ScreenTop",
}

class LeapRecorder(Thread):
    def __init__(self, save_as: str):
        super().__init__()

        self.save_as = save_as
        self.data_dir = None
        self.is_connected = False
        self._init_client()
        self.exp_start_time = None

    def _init_client(self):
        self._listener = LeapListener()
        self._client = leap.Connection()
        self._client.add_listener(self._listener)

    def run(self):
        self.is_connected = True
        self.exp_start_time = datetime.utcnow()
        with self._client.open():
            self._client.set_tracking_mode(leap.TrackingMode.Desktop)
            while self.is_connected:
                time.sleep(1)
    
    def pause(self):
        self.is_connected = False
    
    def save_data(self):
        df = pd.DataFrame(self._listener.data, columns=self._listener.columns)
        df.to_csv(self.save_as)
        

    def stop(self):
        # append the start time to the log.json file
        filename = os.path.join(self.data_dir, 'log.json')
        with open(filename, 'r') as f:
            try:
                existing_data = json.load(f)
            except ValueError:
                existing_data = {}

        # Append the new data
        existing_data.update({'leap_start_time': str(self.exp_start_time), 'leap_end_time': str(datetime.utcnow())})

        # Write everything back to the file
        with lock:
            with open(filename, 'w') as f:
                json.dump(existing_data, f)

        self.is_connected = False
        self._client.remove_listener(self._listener)
        self.save_data()

class LeapListener(leap.Listener):

    JOINT_NAMES = ["Metacarpal", "Proximal", "Intermediate", "Distal"]
    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    POSITIONS = ["x", "y", "z"] 
    ROTATIONS = ["x", "y", "z", "w"]

    def __init__(self):
        super().__init__()

        self.background = np.zeros((30,30,3), np.uint8)
        self.make_columns()
        self.data = []

    def make_columns(self):
        self.columns = ["time","timestamp", "hand_id", "hand_type", "palm_x", "palm_y", "palm_z", "arm_x", "arm_y", "arm_z"]
        for finger in self.FINGER_NAMES:
            for joint in self.JOINT_NAMES:
                for pos in self.POSITIONS:
                    self.columns.append(f"{finger}_{joint}_position_{pos}")
                for rot in self.ROTATIONS:
                    self.columns.append(f"{finger}_{joint}_rotation_{rot}")
        
    def on_connection_event(self, event):
        print("Connected")
        return 

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")
    
    def on_connection_lost_event(self, event):
        print("Connection lost")
        


    def on_tracking_event(self, event):

        if len(event.hands) == 0:
            self.data.append([datetime.utcnow(), event.timestamp] + [np.nan for i in range(len(self.columns)-2)])
            
        for hand in event.hands:
            hand_type = "left" if str(hand.type) == "HandType.Left" else "right"
            
            if hand_type == "right":
                row = [
                    datetime.utcnow(),
                    event.timestamp,
                    hand.id,
                    hand_type,
                    hand.palm.position.x,
                    hand.palm.position.y,
                    hand.palm.position.z,
                    hand.arm.prev_joint.x,
                    hand.arm.prev_joint.y,
                    hand.arm.prev_joint.z,
                ]
                
                fingers_data = [
                    [
                        bone.prev_joint.__getattribute__(pos)
                        for pos in self.POSITIONS
                    ] + [
                        bone.rotation.__getattribute__(rot)
                        for rot in self.ROTATIONS
                    ]
                    for finger in hand.digits
                    for bone in finger.bones
                ]
                
                for finger_data in fingers_data:
                    row.extend(finger_data)

                self.data.append(row)

            # elif hand_type=="left":
            #     self.data.append([datetime.utcnow(), event.timestamp, hand.id, hand_type] + [np.nan for i in range(len(self.columns)-3)])

def main(args):

    save_dir = args.save_dir

    my_listener = LeapListener()

    connection = leap.Connection()
    connection.add_listener(my_listener)

    running = True

    print("Press Enter Key to stop recording:")
    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        while running:
            try:
                sys.stdin.readline()
            except KeyboardInterrupt:
                pass
            finally:
                df = pd.DataFrame(my_listener.data, columns=my_listener.columns)
                df.to_csv(save_dir)
                connection.remove_listener(my_listener)
                break



if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Leap Motion data collection")
    parser.add_argument("--save_dir", type=str, default="Leap_data.csv", help="Directory to save data")
    args = parser.parse_args()

    main(args)
