# Description: utility functions for finger pose estimation
import os
import numpy as np
import pandas as pd
import mne
from scipy import signal

from datetime import datetime

from concurrent.futures import ThreadPoolExecutor
import time

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

def build_leap_columns(positions=False, rotations=False):
    
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    joints = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    # rotations = ['x', 'y', 'z', 'w']
    # positions = ['x', 'y', 'z']
    leap_columns = []
    if positions:
        for finger in fingers:
            for joint in joints:
                leap_columns.append(f'{finger}_{joint}_position_x')
                leap_columns.append(f'{finger}_{joint}_position_y')
                leap_columns.append(f'{finger}_{joint}_position_z')
    if rotations:
        for finger in fingers:
            for joint in joints:
                leap_columns.append(f'{finger}_{joint}_rotation_w')
                # leap_columns.append(f'{finger}_{joint}_rotation_x')
                # leap_columns.append(f'{finger}_{joint}_rotation_y')
                # leap_columns.append(f'{finger}_{joint}_rotation_z')

    return leap_columns


def read_emg(path, start_time=None, end_time=None, fs: int=250):

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

    # get header
    header = raw.info

    if start_time is None:
        start_time = header['meas_date']
        # convert to pd.datetime from datetime.datetime
        start_time = pd.to_datetime(start_time).tz_localize(None)
        # remove 2 hours
        start_time = start_time - pd.to_timedelta(2, unit='h')
        print(start_time)
    
    #  get annotations
    annotations = raw.annotations
    annotations.onset = start_time + pd.to_timedelta(annotations.onset, unit='s')
    
    # get annotations as df
    to_append = []
    gestures_rep ={i:0 for i in set(annotations.description) if 'start' in i}
    for ind, (i,j) in enumerate(zip(annotations.onset, annotations.description)):
        if 'start_' in j:
            if 'end_' in annotations.description[ind+1] and j.replace('start_', '') == annotations.description[ind+1].replace('end_', ''):
                gestures_rep[j] += 1
                new_j = j.replace('start', f'{gestures_rep[j]}').strip('_')
                #  add 1 sec ofset to onset and append
                offset = pd.to_timedelta(1, unit='s')
                to_append.append([annotations.onset[ind]+offset, annotations.onset[ind+1]+offset, new_j])
    
    #  append rest periods
    for i in range(len(to_append)-1):
        if to_append[i][1] != to_append[i+1][0]:
            to_append.append([to_append[i][1], to_append[i+1][0], f'{i}_rest'])

    ann_df = pd.DataFrame(to_append, columns=['start_time', 'end_time', 'gesture'])
    # sort by start time
    ann_df.sort_values(by='start_time', inplace=True)
    #  if duration is greater than 10 sec, drop
    ann_df = ann_df[ann_df['end_time'] - ann_df['start_time'] < pd.to_timedelta(10, unit='s')]


    emg_df = raw.to_data_frame()
    emg_df['time'] = pd.to_datetime(emg_df['time'], unit='s', origin=start_time)
    emg_df.set_index('time', inplace=True)

    # start data from first annotation
    start_time = ann_df['start_time'].iloc[0]
    emg_df = emg_df[start_time:]
    
    #  resample emg data to fs Hz
    emg_df = emg_df.resample(f'{int(1000/fs)}ms', origin='start').mean()

    return emg_df, ann_df, header
# def read_emg(path, start_time: datetime =None, end_time: datetime =None, fs: int=250):


#     if start_time is None:
#         start_time = header['meas_date']
#         # convert to datetime from string
#         start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')

#     raw = mne.io.read_raw_edf(path, preload=True)

#     #  get annotations
#     annotaions = raw.annotations
#     #  convert annotations time to datetime
#     annotaions.onset = pd.to_datetime(annotaions.onset, unit='s', utc=True, origin=start_time)


#     # to dataframe
#     emg_df = raw.to_data_frame()

#     # get header
#     header = raw.info

#     emg_df['time'] = pd.to_datetime(emg_df['time'], unit='s', utc=True, origin=start_time)
#     emg_df.set_index('time', inplace=True)

#     #  resample emg data to fs Hz
#     emg_df = emg_df.resample(f'{int(1000/fs)}ms', origin='start').mean()
#     return emg_df, annotaions, header

# def process_window(df, i, w_rows):
#     window = df.iloc[i:i+w_rows]
#     return window.values, window.index[-1]

# def create_windowed_dataset(df, w, s, unit='sequence'):
#     # Convert window size and stride from seconds to number of rows
#     if unit == 'second':
#         w_rows = int(w * df.index.freq.delta.total_seconds())
#         s_rows = int(s * df.index.freq.delta.total_seconds())
#     elif unit == 'sequence':
#         w_rows = w
#         s_rows = s
#     else:
#         raise ValueError(f'unit must be second or sequence, got {unit}')
    
#     start_time = time.time()
#     with ThreadPoolExecutor(max_workers=2) as executor:
#         results = list(executor.map(process_window, [df]*len(range(0, len(df) - w_rows, s_rows)), range(0, len(df) - w_rows, s_rows), [w_rows]*len(range(0, len(df) - w_rows, s_rows))))

#     data = np.array([result[0] for result in results])
#     times = np.array([result[1] for result in results])

#     # Reshape data to (N-w)/(S)*W*C
#     data = data.reshape((-1, w_rows, df.shape[1]))
#     print(f'Time taken: {time.time() - start_time}')

#     return data, times

def get_gesture(time, ann_df):
    gesture_df = ann_df[(ann_df['start_time'] <= time) & (ann_df['end_time'] >= time)]
    if not gesture_df.empty:
        return gesture_df['gesture'].iloc[0]
    return '50_rest'

def find_closest(leap_data, times, annotations):
    start_time = time.time()
    index = []
    gestures = []   
    for idx, i in enumerate(times):
        #  find the time indeex closest to i
        index.append(leap_data.index.asof(i))
        gestures.append(get_gesture(i,annotations))
        #  find the gesture closest to i
    leap_closest = leap_data.loc[index]
    print(f'Time taken to find closest: {time.time() - start_time}')
    return leap_closest.to_numpy(), gestures

def create_windowed_dataset(df, label, annotations, w, s, unit='sequence'):
    # Convert window size and stride from seconds to number of rows
    if unit == 's':
        w_rows = int(w * df.index.freq.delta.total_seconds())
        s_rows = int(s * df.index.freq.delta.total_seconds())
    elif unit == 'sequence':
        w_rows = w
        s_rows = s
    else:
        raise ValueError(f'unit must be s or sequence, got {unit}')

    start_time = time.time()
    data = []
    times = []
    gestures = []
    leap_indexs = []
    for i in range(0, len(df) - w_rows, s_rows):
        window = df.iloc[i:i+w_rows]
        data.append(window.values)
        # times.append(window.index[-1])
        leap_indexs.append(label.index.asof(window.index[-1]))
        gestures.append(get_gesture(window.index[-1], annotations))

    data = np.array(data)
    # times = np.array(times)
    gestures = np.array(gestures)
    label = label.loc[leap_indexs].to_numpy()

    # Reshape data to (N-w)/(S)*W*C
    data = data.reshape((-1, w_rows, df.shape[1]))
    print(f'Time taken to create windowed dataset: {time.time() - start_time}')

    return data, label, gestures




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

def read_leap(path, fs=125, positions=True, rotations=False):

    leap_df = pd.read_csv(path, index_col=False)

    # drop null and duplicates
    leap_df.dropna(inplace=True)
    leap_df.drop_duplicates(inplace=True, subset=['time'])

    leap_df['time'] = pd.to_datetime(leap_df['time'])
    leap_df['time'] = leap_df['time'].dt.tz_localize(None)
    leap_df = leap_df.set_index('time')

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
    
    # leap_df = leap_df.resample(f'{int(1000/fs)}ms', origin='start').ffill()
    

    valid_columns = build_leap_columns(positions=positions, rotations=rotations)
    if len(valid_columns) != 0:
        leap_df = leap_df[valid_columns]

    if rotations and len(valid_columns) != 0 and not positions:
        leap_df = leap_df.apply(lambda x: np.rad2deg(x))
        
    #     # add offset value of 50 degrees to all angles
    #     leap_df = leap_df.apply(lambda x: x - 50)
    #     #  proximal columns
    #     proximal = [i for i in leap_df.columns if "proximal" in i.lower()]
    #     leap_df[proximal] = leap_df[proximal].apply(lambda x: x-45)
    #     # leap_df = leap_df.apply(lambda x: x - 180 if x > 180 else x)
    #     # mcp = [i for i in leap_df.columns if "metacarpal" in i.lower() and "thumb" not in i.lower()]
    #     # leap_df[mcp] = leap_df[mcp].apply(lambda x: 0)
    # #  convert radians to degrees
    # # leap_df = leap_df.apply(np.degrees)

    return leap_df, None, None

# def read_leap(path, fs=125):

#     leap_df = pd.read_csv(path)
#     leap_df['time'] = pd.to_datetime(leap_df['time'], unit='s', origin=ExpTimes.video_Start_time)
#     leap_df = leap_df.set_index('time')

#     # drop null and duplicates
#     leap_df.dropna(inplace=True)
#     leap_df.drop_duplicates(inplace=True, subset=['time'])

#     # convert to datetime
#     leap_df['time'] = pd.to_datetime(leap_df['time'], utc=True)
#     leap_df.set_index('time', inplace=True)

#     # calculate relative position
#     for i in leap_df.columns:
#         if 'position_x' in i:
#             leap_df[i] = leap_df[i] - leap_df['palm_x']
#         elif 'position_y' in i:
#             leap_df[i] = leap_df[i] - leap_df['palm_y']
#         elif 'position_z' in i:
#             leap_df[i] = leap_df[i] - leap_df['palm_z']
#         else:
#             continue
    
#     leap_df = leap_df.resample(f'{int(1000/fs)}ms', origin='start').ffill()

#     valid_columns = build_leap_columns()
#     leap_df = leap_df[valid_columns]

#     return leap_df, None, None