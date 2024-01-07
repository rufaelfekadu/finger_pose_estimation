from typing import Any
from .data import Data
from datetime import datetime   
import os
import json
import sys
sys.path.append('../')
from lock import lock
# print (sys.path)
class EMG(Data):

    def __init__(self,
                 host_name: str,
                 port: int,
                 timeout_secs: float = None,
                 verbose: bool = False,
                 save_as: str = None):

        super().__init__(host_name, port, timeout_secs, verbose, save_as)
        self.data_dir = None
        self.visualize = False
        self.exp_start_time = None
    
    def run(self):
        self.exp_start_time = datetime.utcnow()
        super().run()

    def stop(self):

        filename = os.path.join(self.data_dir, 'log.json')
        with open(filename, 'r') as f:
            try:
                existing_data = json.load(f)
            except ValueError:
                existing_data = []
        # Append the new data
        existing_data.update({'emg_start_time': str(self.exp_start_time), 'emg_end_time': str(datetime.utcnow())})

        # Write everything back to the file
        with lock:
            with open(filename, 'w') as f:
                json.dump(existing_data, f)

        super().stop()