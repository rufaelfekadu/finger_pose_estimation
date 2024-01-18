from typing import Any
from .data import Data
from datetime import datetime   
import pandas as pd
import numpy as np
import os
import json
import sys
sys.path.append('../')
sys.path.append('../../')
from lock import lock
from Leap import get_bone_core_angles, get_all_bone_angles_from_core
from util import build_leap

class EMG(Data):

    def __init__(self,
                 host_name: str,
                 port: int,
                 LeapClient: Any,
                 timeout_secs: float = None,
                 verbose: bool = False,
                 save_as: str = None):

        super().__init__(host_name, port, timeout_secs, verbose, save_as)
        self.data_dir = None
        self.visualize = False
        self.exp_start_time = None
        self._LeapClient = LeapClient
        self.leap_data = []
        self.leap_path = None

    def run(self):
        """
        Overrides Thread.run().
        Commands what to do when a Data thread is started: continuously receive data from the socket, parse it into
        Records, (print details of received packet if verbose==True), and add data to growing data matrix.
        """
        self.exp_start_time = datetime.utcnow()

        while self.is_connected:
            # Receive incoming record(s)
            records = self._parse_incoming_records()
            self._add_to_data(records)
            # get the frame from leap_client    
            data_ = self._get_leap_data()
            # append to the leap_data
            
            print(f'{datetime.utcnow()} {data_} ')
            self.leap_data.append(data_)
            # Add newly received data to main data matrix
        print('Connection terminated.')
        self.save_data()

    # @staticmethod
    # def _get_all_bone_angles_from_core(core_angles):
    #     thumb_tmc_fe = core_angles[0]
    #     thumb_tmc_aa = core_angles[1]
    #     thumb_mcp_fe = core_angles[2]
    #     thumb_mcp_aa = core_angles[3]
    #     index_mcp_fe = core_angles[4]
    #     index_mcp_aa = core_angles[5]
    #     index_pip = core_angles[6]
    #     middle_mcp_fe = core_angles[7]
    #     middle_mcp_aa = core_angles[8]
    #     middle_pip = core_angles[9]
    #     ring_mcp_fe = core_angles[10]
    #     ring_mcp_aa = core_angles[11]
    #     ring_pip = core_angles[12]
    #     little_mcp_fe = core_angles[13]
    #     little_mcp_aa = core_angles[14]
    #     little_pip = core_angles[15]

    #     return [

    #         0, 0, 0,
    #         thumb_tmc_fe, thumb_tmc_aa, 0,
    #         thumb_mcp_fe, thumb_mcp_aa, 0,
    #         .5*thumb_mcp_fe, 0, 0,

    #         0, 0, 0,
    #         index_mcp_fe, index_mcp_aa, 0,
    #         index_pip, 0, 0,
    #         (2/3)*index_pip, 0, 0,

    #         0, 0, 0,
    #         middle_mcp_fe, middle_mcp_aa, 0,
    #         middle_pip, 0, 0,
    #         (2/3)*middle_pip, 0, 0,

    #         0, 0, 0,
    #         ring_mcp_fe, ring_mcp_aa, 0,
    #         ring_pip, 0, 0,
    #         (2/3)*ring_pip, 0, 0,

    #         0, 0, 0,
    #         little_mcp_fe, little_mcp_aa, 0,
    #         little_pip, 0, 0,
    #         (2/3)*little_pip, 0, 0,
    #     ]
    
    def _get_leap_data(self):
        row = [datetime.utcnow()]
        data = get_bone_core_angles(self._LeapClient)
        
        if data is not None:
            row.extend(data)
        else:
            row.extend([np.nan]*16)
        return data
    
    def make_leap_columns(self):
        columns = ['timestamp']
        for i in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            for j in ['metacarpal', 'proximal', 'intermediate', 'distal']:
                for k in ['fe', 'ab', 'rl']:
                    columns.append(f'{i}_{j}_{k}')
        return columns
    
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

        self.is_connected = False
        # 
        if self.leap_path is not None:
            print(self.leap_data)
            leap_data = pd.DataFrame(self.leap_data, columns=build_leap(full=False))
            leap_data.to_csv(self.leap_path)
            print(f'Leap data saved to {self.leap_path}')
        super().stop()