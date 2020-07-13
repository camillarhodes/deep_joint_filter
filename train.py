import os
import shutil
import argparse
import torch

from src.parser import YamlParser
from src.deep_joint_filter import DeepJointFilter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./configs/djf_rgbnir256_gaussion25.yaml", help="config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = YamlParser(config_file=args.config_file)

    import ipdb; ipdb.set_trace()
    djf  = DeepJointFilter(config)
    djf.load()
    djf.train()

