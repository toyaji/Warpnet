import argparse
import sys
import yaml
from easydict import EasyDict
from datetime import datetime


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
        config = EasyDict(config)
    return config

def load_config_from_args():
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", required=True, help="You can see the sample yaml template in /config folder.")
    args.add_argument("-n", "--name", type=str, help="You can put the name for the experiment, this will be used for log file name.")
    args = args.parse_args(sys.argv[1:])
    
    config = load_config(args.config)
    config.log.name = datetime.now().strftime("%Y%m%d%H%M")
    config.log.version = args.name

    return config