import argparse
import sys
import yaml
from easydict import EasyDict


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
        config = EasyDict(config)

    return config

def load_config_from_args():
    args = argparse.ArgumentParser()
    args.add_argument("config")
    args = args.parse_args(sys.argv[1:])

    return load_config(args.config)