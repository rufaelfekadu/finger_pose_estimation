import logging
import os

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





