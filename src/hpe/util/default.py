import logging
import os
from typing import Any
import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse_arg(disc="Train the model"):
    parser = argparse.ArgumentParser(description=disc)
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()
    return args
    
class AverageMeterList(object):
    def __init__(self,label_names) -> None:
        self.meters = [AverageMeter() for _ in range(len(label_names))]
        self.label_names = label_names

    def update(self, values: list, n: int = 1) -> None:
        assert len(values) == len(self.meters)
        for i, meter in enumerate(self.meters):
            meter.update(values[i].item(), n)  

    def reset(self) -> None:
        for meter in self.meters:
            meter.reset()

    def __str__(self) -> str:
        return ''.join([f'{name}: {meter.avg:.3f}\n' for name, meter in zip(self.label_names, self.meters)])
    
    def __call__(self) -> list:
        return [meter.avg for meter in self.meters]
    
    def save_to_json(self,dir):
        import json
        data = {}
        for name, meter in zip(self.label_names, self.meters):
            data[name] = meter.avg
        with open(os.path.join(dir,'losses.json'), 'w') as f:
            json.dump(data, f, indent=4)
    
    def plot(self,path):
        # update label names. replace 'position' by ''
        self.label_names = [name.replace('_position', '') for name in self.label_names]
        self.label_names = [name.replace('_rotation', '') for name in self.label_names]
        plt.figure(figsize=(10, 8))
        plt.xticks(range(len(self.label_names)), self.label_names)
        # rotate axis labels
        plt.setp(plt.gca().get_xticklabels(), rotation=90, horizontalalignment='right')
        bar = plt.bar(np.arange(len(self.label_names)), [meter.avg for meter in self.meters], align='center', )
        for i in range(0, len(self.label_names), 4):
            bar[i].set_color('coral')
            bar[i+1].set_color('olivedrab')
            bar[i+2].set_color('r')
        # set plot size
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def plot_finger(self, path):

        # update label names. replace 'position' by ''
        self.label_names = [name.replace('_position', '') for name in self.label_names]
        self.label_names = [name.replace('_rotation', '') for name in self.label_names]
        plt.figure(figsize=(10, 8))
        # group by finger and average
        finger_ = {'Thumb': [], 'Index': [], 'Middle': [], 'Ring': [], 'Pinky': []}
        for i,v in finger_.items():
            for j in range(len(self.label_names)):
                if i in self.label_names[j]:
                    finger_[i].append(self.meters[j].avg)
        plt.xticks(range(len(finger_)), finger_.keys())
        plt.setp(plt.gca().get_xticklabels(), rotation=90, horizontalalignment='right')
        bar = plt.bar(np.arange(len(finger_)), [np.mean(v) for v in finger_.values()], align='center', )
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
class AverageMeter(object):

    def __init__(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, value: Any, n: int = 1) -> None:
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
    
    def reset(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def __str__(self) -> str:
        return f"{self.avg:.3f}"
    
    def __call__(self) -> float:
        return self.avg
    
def create_logger(log_path):
    """
    Create a logger object with basic configuration.

    Args:
        log_path (str): The path to the log file.

    Returns:
        logger (logging.Logger): The configured logger object.
    """

    # Clear the log file
    open(log_path, 'w').close()

    # Create a logger object
    logger = logging.getLogger(__name__)

    # Set the log level
    logger.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger




