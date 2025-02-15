from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

experiment = Experiment(
  api_key="placeholder",
  project_name="placeholder",
  workspace="placeholder",
  disabled=True
)

import os
import shutil
import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from utils.config import *
from trainer import *

def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        '--config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    arg_parser.add_argument(
        '--CUR_DIR',
        metavar='CUR_DIR location',
        default="None",
        help='Value CURRENT DIRECTORY OF CODE USE PWD')
    arg_parser.add_argument(
        '--SEED',
        metavar='SEED of experiment',
        default="None",
        help='Value of SEED of experiment')

    args = arg_parser.parse_args()

    print(args)
    config = process_config(args)
    experiment.log_parameters(config)
    experiment.set_name(str(config.exp_name))
    trainer_class = globals()[config.agent]
    trainer = trainer_class(config=config,comet_ml=experiment)
    trainer.run(config=config)
    trainer.finalize()


if __name__ == '__main__':
    main()
