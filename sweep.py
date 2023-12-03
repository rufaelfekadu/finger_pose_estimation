import ray
from ray import tune
from train import train, test, main
from config import cfg
import argparse
from util import create_logger
import os
# Define your training/validation function
def train_fn(config):
    # Perform training/validation using the hyperparameters in the config dictionary
    # Return the metric you want to optimize (e.g., accuracy, loss)
    
    pass
# Define the search space for hyperparameters

def parse_arg():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arg()

    # only for 
    if cfg.DEBUG:
        cfg.SOLVER.LOG_DIR = "../debug"
        # set the config attribute of args to 
        
    
    # load config file
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.LOG_DIR = os.path.join(cfg.SOLVER.LOG_DIR, cfg.MODEL.NAME)
    cfg.freeze()

    # setup logging
    logger = create_logger(os.path.join(cfg.SOLVER.LOG_DIR, 'train.log'))

    logger.info(f"Running using Config:\n{cfg}\n\n")

    
    search_space = {
        "learning_rate": tune.loguniform(0.001, 0.1),
        "batch_size": tune.choice([16, 32, 64]),
        "num_layers": tune.choice([1, 2, 3]),
    }

    # Create the configuration dictionary
    config = {
        "num_samples": 10,  # Number of trials to run
        "config": search_space,
        "stop": {"training_iteration": 100},  # Stopping criteria
    }

    # Start the hyperparameter tuning process
    analysis = tune.run(train_fn, **config)

    # Retrieve the best hyperparameters
    best_config = analysis.get_best_config(metric="loss", mode="min")


