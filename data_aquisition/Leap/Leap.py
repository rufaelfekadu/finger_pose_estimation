import leap
import time
import pandas as pd
import numpy as np
import argparse
import sys
from threading import Thread
# Your Recording class import here

class LeapRecorder(Thread):
    def __init__(self, save_dir: str):
        super().__init__()

        self.save_dir = save_as
        self.is_connected = False
        self._init_client()

    def _init_client(self):

        self._listener = LeapListener()
        self._client = leap.Connection()
        self._client.add_listener(self._listener)

    def run(self):
        self.is_connected = True
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
        self.columns = ["time","timestamp", "hand_id", "hand_type", "palm_x", "palm_y", "palm_z"]
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

        # print(f"Frame {event.tracking_frame_id} with {len(event.hands)} hands.")
        # timestamp = datetime.utcfromtimestamp(event.timestamp).strftime('%YY-%MM-%DD %H:%M:%S.%f')
        start = time.time()
        #  if no hands are detected, append a row of zeros
        if len(event.hands) == 0:
            self.data.append([time.time(), event.timestamp] + [0 for i in range(len(self.columns)-2)])
            
        for hand in event.hands:
            hand_type = "left" if str(hand.type) == "HandType.Left" else "right"
            
            if hand_type == "right":
                row = [
                    time.time(),
                    event.timestamp,
                    hand.id,
                    hand_type,
                    hand.palm.position.x,
                    hand.palm.position.y,
                    hand.palm.position.z
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

        print(f"Time taken to write hands: {time.time() - start}")


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
