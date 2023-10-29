# Description: utility functions for finger pose estimation
import os
import numpy as np
import pandas as pd
import mne
from scipy import signal

from datetime import datetime
class ExpTimes:
    refernce_time = datetime.strptime('2023-10-02 14:59:55.627000', '%Y-%m-%d %H:%M:%S.%f')
    manus_start_time = datetime.strptime('2023-10-02 14:59:20.799000', '%Y-%m-%d %H:%M:%S.%f')
    emg_start_time = datetime.strptime('2023-10-02 14:59:55.627000', '%Y-%m-%d %H:%M:%S.%f')
    video_Start_time = datetime.strptime('2023-10-02 14:59:55.628000', '%Y-%m-%d %H:%M:%S.%f')


def read_emg(path):
    '''
    read emg signal from edf file
    Input: path to edf file
    Output: emg dataframe
    '''
    raw = mne.io.read_raw_edf(path, preload=True)

    # to dataframe
    emg_df = raw.to_data_frame()

    # change time to datetime
    emg_df['time'] = pd.to_datetime(emg_df['time']*1000, unit='ms', origin=ExpTimes.emg_start_time)

    # filterout the time column for only ms resolution
    emg_df['time'] = emg_df['time'].dt.floor('ms')

    # resample emg data to 150 Hz
    emg_df = emg_df.set_index('time')
    emg_df = emg_df.resample('8ms', origin='end').mean()

    return emg_df

def read_manus(path):
    '''
    Read data from manus glove
    Input: path to csv file
    Output: manus dataframe
    '''

    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    key_points = ['MCP', 'DIP', 'PIP', 'CMC']
    movement = ['Spread', 'Flex']

    # Additional columns for manus
    pinch_columns = ['Pinch_ThumbToIndex', 'Pinch_ThumbToMiddle', 'Pinch_ThumbToRing', 'Pinch_ThumbToPinky']
    time_column = ['time']

    valid_columns = key_points + pinch_columns + time_column
    
    
    manus_df = pd.read_csv(path)

    #rename Elapsed_Time_In_Milliseconds to time
    manus_df.rename(columns={'Elapsed_Time_In_Milliseconds': 'time'}, inplace=True)

    # Convert time to datetime and drop values l
    manus_df['time'] = pd.to_datetime(manus_df['time'], unit='ms', origin=ExpTimes.manus_start_time)


    # remove acceleration and velocity columns
    acc_vel_col = [item for item in manus_df.columns if 'Acceleration' in item or 'Velocity' in item]
    manus_df.drop(columns=acc_vel_col, inplace=True)

    #drop unused columns
    unused_columns = ['Time', 'Frame'] + manus_df.filter(regex='_[X/Y/Z]', axis=1).columns.tolist()
    manus_df.drop(columns=unused_columns, inplace=True)

    assert sorted(list(manus_df.columns) )== sorted(valid_columns), 'Columns are not valid'

    # set time as index
    manus_df = manus_df.set_index('time')

    return manus_df 