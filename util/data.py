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



def build_manus_columns():
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    joints = ['CMC', 'MCP', 'PIP', 'DIP']
    movements = ['Flex', 'Spread']
    manus_columns = ['time']

    for finger in fingers:
        for joint in joints:
            for flex in movements:
                if (finger == 'Thumb' and joint == 'MCP') or (finger != 'Thumb' and joint == 'CMC'):
                    continue
                manus_columns.append(f'{finger}_{joint}_{flex}')
    return manus_columns

def build_leap_columns():
    
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    joints = ['CMC', 'MCP', 'PIP', 'DIP']
    rotations = ['x', 'y', 'z', 'w']
    leap_columns = ['time']

    for finger in fingers:
        for joint in joints:
            leap_columns.append(f'{finger}_{joint}_rotation_w')

    return leap_columns

def read_emg(path, start_time: datetime =None, end_time: datetime =None, fs: int=250):


    if start_time is None:
        start_time = header['meas_date']
        # convert to datetime from string
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')

    raw = mne.io.read_raw_edf(path, preload=True)

    #  get annotations
    annotaions = raw.annotations
    #  convert annotations time to datetime
    annotaions.onset = pd.to_datetime(annotaions.onset, unit='s', utc=True, origin=start_time)


    # to dataframe
    emg_df = raw.to_data_frame()

    # get header
    header = raw.info

    emg_df['time'] = pd.to_datetime(emg_df['time'], unit='s', utc=True, origin=start_time)
    emg_df.set_index('time', inplace=True)

    #  resample emg data to fs Hz
    emg_df = emg_df.resample(f'{int(1000/fs)}ms', origin='start').mean()
    return emg_df, annotaions, header

def create_windowed_dataset(df, w, s):
    # Convert window size and stride from seconds to number of rows
    w_rows = int(w * df.index.freq.delta.total_seconds())
    s_rows = int(s * df.index.freq.delta.total_seconds())

    data = []
    times = []
    for i in range(0, len(df) - w_rows, s_rows):
        window = df.iloc[i:i+w_rows]
        data.append(window.values)
        times.append(window.index[-1])

    data = np.array(data)
    times = np.array(times)

    # Reshape data to (N-w)/(S)*W*C
    data = data.reshape((-1, w_rows * df.shape[1]))

    return data, times

def read_manus(path, start_time=None, end_time=None):

    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    key_points = ['MCP', 'DIP', 'PIP', 'CMC']
    movement = ['Spread', 'Flex']

    if start_time is None:
        start_time = ExpTimes.manus_start_time
    else:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')

    # Additional columns for manus
    pinch_columns = ['Pinch_ThumbToIndex', 'Pinch_ThumbToMiddle', 'Pinch_ThumbToRing', 'Pinch_ThumbToPinky']
    time_column = ['time']

    valid_columns = time_column + build_manus_columns()
    
    
    manus_df = pd.read_csv(path)

    #rename Elapsed_Time_In_Milliseconds to time
    manus_df.rename(columns={'Elapsed_Time_In_Milliseconds': 'time'}, inplace=True)

    # Convert time to datetime and drop values l
    manus_df['time'] = pd.to_datetime(manus_df['time'], unit='ms', origin=start_time)


    # remove acceleration and velocity columns
    acc_vel_col = [item for item in manus_df.columns if 'Acceleration' in item or 'Velocity' in item or 'Spread' in item]
    manus_df.drop(columns=acc_vel_col, inplace=True)

    #drop unused columns
    unused_columns = ['Time', 'Frame'] + manus_df.filter(regex='_[X/Y/Z]', axis=1).columns.tolist()+pinch_columns
    manus_df.drop(columns=unused_columns, inplace=True)
    # assert sorted(list(manus_df.columns)) == sorted(valid_columns), 'Columns are not valid'

    # set time as index
    manus_df = manus_df.set_index('time')
    return manus_df, None, None

def read_leap(path, fs=125):

    leap_df = pd.read_csv(path)
    leap_df['time'] = pd.to_datetime(leap_df['time'], unit='s', origin=ExpTimes.video_Start_time)
    leap_df = leap_df.set_index('time')

    # drop null and duplicates
    leap_df.dropna(inplace=True)
    leap_df.drop_duplicates(inplace=True, subset=['time'])

    # convert to datetime
    leap_df['time'] = pd.to_datetime(leap_df['time'], utc=True)
    leap_df.set_index('time', inplace=True)

    # calculate relative position
    for i in leap_df.columns:
        if 'position_x' in i:
            leap_df[i] = leap_df[i] - leap_df['palm_x']
        elif 'position_y' in i:
            leap_df[i] = leap_df[i] - leap_df['palm_y']
        elif 'position_z' in i:
            leap_df[i] = leap_df[i] - leap_df['palm_z']
        else:
            continue
    
    leap_df = leap_df.resample(f'{int(1000/fs)}ms', origin='start').ffill()

    valid_columns = build_leap_columns()
    leap_df = leap_df[valid_columns]

    return leap_df, None, None