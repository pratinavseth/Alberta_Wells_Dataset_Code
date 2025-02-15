import os
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
from datetime import datetime

import json
from easydict import EasyDict
from pprint import pprint

from utils.dirs import create_dirs


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def process_config(args):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    json_file = args.config
    #json_file = os.path.join(args.CUR_DIR,json_file)
    config, _ = get_config_from_json(json_file)

    if bool(args.SEED):
        config.seed = args.SEED
    else:
        config.seed = 123

    print("\nExperiment Seed : ",config.seed)

    name_path = str(config.exp_name)+"_"+str(config.seed)
    if config.task == "OD":
        path_logs = os.path.join(args.CUR_DIR,'logs','Object_Detection',name_path)
        os.makedirs(path_logs, exist_ok =True)
        config.experiments_logs_dir = path_logs
        config.train_hdf5_file = os.path.join(args.CUR_DIR,'downloads','adw-files-split','train_csv_file_filtered.csv')
        config.eval_hdf5_file = os.path.join(args.CUR_DIR,'downloads','adw-files-split','eval_csv_file_filtered.csv')
        config.test_hdf5_file = os.path.join(args.CUR_DIR,'downloads','adw-files-split','test_csv_file_filtered.csv')
        config.checkpoint_file = os.path.join(args.CUR_DIR,config.checkpoint_file)
    else:
        path_logs = os.path.join(args.CUR_DIR,'logs','Binary_Segmenation',name_path)
        os.makedirs(path_logs, exist_ok =True)
        config.experiments_logs_dir = path_logs
        config.train_hdf5_file = os.path.join(args.CUR_DIR,'downloads','adw-files-split','train_csv_file.csv')
        config.eval_hdf5_file = os.path.join(args.CUR_DIR,'downloads','adw-files-split','eval_csv_file.csv')
        config.test_hdf5_file = os.path.join(args.CUR_DIR,'downloads','adw-files-split','test_csv_file.csv')
        config.checkpoint_file = os.path.join(args.CUR_DIR,config.checkpoint_file)
    
    
    print(" THE Configuration of your experiment ..")
    pprint(config)

    # making sure that you have provided the exp_name.
    try:
        print(" *************************************** ")
        print("The experiment name is {}".format(config.exp_name))
        print(" *************************************** ")
    except AttributeError:
        print("ERROR!!..Please provide the exp_name in json file..")
        exit(-1)

    config.dir_name = str(config.exp_name)+str(datetime.utcnow().strftime('_%Y_%m_%d__%H_%M_%S'))
    config.summary_dir = os.path.join(config.experiments_logs_dir, config.dir_name, "summaries/")
    config.checkpoint_dir = os.path.join(config.experiments_logs_dir, config.dir_name, "checkpoints/")
    config.out_dir = os.path.join(config.experiments_logs_dir, config.dir_name, "out/")
    config.log_dir = os.path.join(config.experiments_logs_dir, config.dir_name, "logs/")

    create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])
    print(config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir)

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info("Hi, This is root.")
    logging.getLogger().info("After the configurations are successfully processed and dirs are created.")
    logging.getLogger().info("The pipeline of the project will begin now.")

    print(config)

    return config
