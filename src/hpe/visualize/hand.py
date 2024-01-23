from typing import Any
from peaceful_pie.unity_comms import UnityComms
from dataclasses import dataclass
from threading import Thread
import time
import torch
import os
import sys
sys.path.append('/Users/rufaelmarew/Documents/tau/finger_pose_estimation')
from config import cfg
from util import read_manus, read_leap, build_leap_columns
from data import make_exp_dataset, make_dataloader
from models import make_model
import numpy as np

@dataclass
class params:
    angles: list
    jointNames: list
    handName: str


['Thumb_CMC_Spread', 'Thumb_CMC_Flex', 'Thumb_PIP_Flex', 'Thumb_DIP_Flex',
    'Index_MCP_Spread', 'Index_MCP_Flex', 'Index_PIP_Flex', 'Index_DIP_Flex', 
    'Middle_MCP_Spread', 'Middle_MCP_Flex', 'Middle_PIP_Flex', 'Middle_DIP_Flex', 
    'Ring_MCP_Spread', 'Ring_MCP_Flex', 'Ring_PIP_Flex', 'Ring_DIP_Flex',
    'Pinky_MCP_Spread', 'Pinky_MCP_Flex', 'Pinky_PIP_Flex','Pinky_DIP_Flex']

class HandBase:
    def __init__(self):
        self.fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self.joints = ["CMC", "MCP", "PIP", "DIP"]
        pass

    def update(self, angles: list, unity_comms: UnityComms):
        self.params.angles = angles
        unity_comms.UpdateHand(angles=self.params)
    
    def make_joint_names(self):
        pass

    def reset(self, unity_comms: UnityComms):
        self.params.angles = [0 for i in range(len(self.num_joints))]
        unity_comms.UpdateHand(angles=self.params)

    def run_from_csv(self, cfg, sleep_time=1):
        pass

    def run_saved_data(self, cfg, sleep_time=1):
        pass

    def run_online(self, cfg, model, model_path, sleep_time=1):
        pass

class HandManus(HandBase):

    def __init__(self, cfg, hand_name="Prediction"):
        super().__init__()
        self.joints = ["CMC", "MCP", "PIP", "DIP"]
        self.rotations = ["Spread", "Flex"]
        self.joint_names = self.make_joint_names()
        self.params = params(angles=[0 for i in range(len(self.joint_names))], jointNames=self.joint_names, handName=hand_name)

    def make_joint_names(self):
        joint_names = []
        for i in self.fingers:
            for j in self.joints:
                if (j == "MCP" and i =="Thumb") or (j == "CMC" and i !="Thumb"):
                    continue
                # if j == "CMC" or j == "MCP":
                #     joint_names.append(f"{i}_{j}_Spread")
                joint_names.append(f"{i}_{j}_Flex")
        return joint_names



class HandLeap(HandBase):
    def __init__(self, cfg, hand_name="Prediction"):
        super().__init__()
        self.joint_names = build_leap_columns()
        # angles = self.convert_to_manus([0 for i in range(len(self.joint_names))])
        self.params = params(angles=[0 for i in range(12)], jointNames=self.joint_names, handName=hand_name)
        self.unity_comms = UnityComms(cfg.VISUALIZE.PORT)

    def convert_to_manus(self, keypoints: Any):
        #  convert anlges to list of lists with x,y,z coordinates
        new_keypoints = []
        for i in range(0, len(keypoints), 3):
            new_keypoints.append([keypoints[i], keypoints[i+1], keypoints[i+2]])
        return new_keypoints

    def reset(self):
        pass
    def update(self, keypoints: Any, unity_comms: UnityComms):
        #  convert to manus angles
        self.params.angles = keypoints
        unity_comms.UpdateLeapHands(angles=self.params)
    
    def read_csv(self, cfg, sleep_time=0.001):

        label_dir = cfg.VISUALIZE.LABEL_PATH
        file_name = [i for i in os.listdir(label_dir) if i.endswith(".csv")][0]
        label_path = os.path.join(label_dir, file_name)

        dataset,_,_ = read_leap(label_path, positions=False, rotations=True)
        self.joint_names = dataset.columns.tolist()
        self.params.jointNames = self.joint_names
        print("started visualisation with {} data points".format(len(dataset)))
        print("press enter to exit")
        for i in range(0, len(dataset)):
            angles = dataset.iloc[i].tolist()
            print(len(angles), len(self.params.jointNames))
            # self.params.angles = angles
            self.update(angles, self.unity_comms)
            time.sleep(sleep_time)

    def run_from_loader(self, cfg, sleep_time=0.001, dataloader=None):
        if dataloader is None:
            dataloader = make_exp_dataset(cfg, save=False)
        print("started visualisation with {} data points".format(len(dataloader)))
        self.joint_names = dataloader.dataset.dataset.label_columns
        self.params.jointNames = self.joint_names
        data_iter = iter(dataloader)
        for i in range(0, len(dataloader)):
            data, leap_data, gesture = next(data_iter)
            for j in range(0, len(leap_data)):
                print(gesture[j])
                angles = leap_data[j].tolist()
                self.update(angles, self.unity_comms)
                #  exit when enter is pressed
                if input() == "q":
                    return
                time.sleep(sleep_time)


class HandEMG(HandBase):
    def __init__(self):
        pass

    def update(self, keypoints: Any):
        pass

    def reset(self):
        pass   

class Hands:
    def __init__(self, cfg, model=None, data_loader=None):
        
        self.unity_comms = UnityComms(cfg.VISUALIZE.PORT)

        self.handPrediction = HandLeap(hand_name="Prediction")
        self.handLabel = HandLeap(hand_name="Label")
        self.mode = cfg.VISUALIZE.MODE
        self.model = model
        self.data_loader = data_loader

    def update(self, keypoints: Any):
        print(len(keypoints[0]), len(keypoints[1]))
        self.handPrediction.update(keypoints[0], self.unity_comms)
        self.handLabel.update(keypoints[1], self.unity_comms)
        
    def reset(self):
        self.handPrediction.__init__()
        self.handLabel.__init__()

    def run_from_csv(self, cfg, sleep_time=1):
        label_path = cfg.VISUALIZE.LABEL_PATH
        dataset = read_manus(label_path)
        dataset = dataset[3000:]
        print("started visualisation with {} data points".format(len(dataset)))
        print("press enter to exit")
        for i in range(0, len(dataset)):
            angles = dataset.iloc[i].tolist()
            self.handLabel.update(angles, self.unity_comms)
            time.sleep(sleep_time)
            if input() == "q":
                break
            time.sleep(sleep_time)

    def run_saved_data(self, cfg, sleep_time=1):

        print(f"reading from {cfg.SOLVER.LOG_DIR}")
        pred = torch.load(os.path.join(cfg.SOLVER.LOG_DIR, 'pred_cache.pth')).tolist()
        label = torch.load(os.path.join(cfg.SOLVER.LOG_DIR, 'label_cache.pth')).tolist()
        
        print("started visualisation with {} data points".format(len(pred)))
        print("press enter to exit")
        for i in range(0, len(pred)):
            self.update((pred[i], label[i]))
            #  exit when enter is pressed
            if input() == "q":
                break
            time.sleep(sleep_time)

    def run_from_dataloader(self, cfg, sleep_time=1):
        if self.data_loader is None:
            self.data_loaders = make_dataloader(cfg, save=False)

        if self.model is None:
            self.model = make_model(cfg)
            self.model.load_pretrained(cfg.SOLVER.PRETRAINED_PATH)

        print("started visualisation with {} data points".format(len(self.dataloader)))
        self.params.jointNames = self.dataloader.dataset.dataset.label_columns
        data_iter = iter(self.dataloader)
        for i in range(0, len(self.dataloader)):
            data, label, gesture = next(data_iter)
            output = self.model(data.unsqueeze(0))
            for j in range(0, len(label)):
                print(gesture[j])
                self.update((output.tolist()[0], label[j].tolist()))
                #  exit when enter is pressed
                if input() == "q":
                    return
                time.sleep(sleep_time)

    #  TODO: implement this  
    def run_online(self, cfg, model, model_path, sleep_time=1):
        model.load_pretrained(model_path)
        model.eval()
        pass

HAND_MODES = {
    "manus": HandManus,
    "leap": HandLeap,
}

def make_hands(mode):
    return HAND_MODES[mode]()

def main(cfg, data_loader):
    hands = HandLeap(cfg)
    hands.reset()
    hands.run_from_loader(cfg, sleep_time=0.01, dataloader=data_loader)
    # hands.read_csv(cfg, sleep_time=0.01, data_loader=data_loader)

if __name__ == "__main__":

    cfg.SOLVER.LOG_DIR = os.path.join(cfg.SOLVER.LOG_DIR, cfg.MODEL.NAME)
    cfg.DATA.EXP_SETUP = 'exp0'
    cfg.DATA.PATH = './dataset/FPE/S1/p3'
    cfg.DEBUG = False
    dataloaders = make_dataloader(cfg, save=False)

    main(cfg, dataloaders["train"])