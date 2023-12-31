import logging
import os
from typing import Any

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

def create_logger(log_path):
    """
    Create a logger object with basic configuration.

    Args:
        log_path (str): The path to the log file.

    Returns:
        logger (logging.Logger): The configured logger object.
    """

    # Clear the log file if it exists
    if os.path.exists(log_path):
        open(log_path, 'w', ).close()
       
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





