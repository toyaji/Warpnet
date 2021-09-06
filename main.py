import warnings
from pathlib import Path

import torch
from torch.utils.data.dataset import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import WarpModel
from data.zoomdata import ZoomLZoomData

warnings.filterwarnings('ignore')


def main(config):

    # dataset settting
    train_set = ZoomLZoomData(config.dataset, train=True)
    test_set = ZoomLZoomData(config.dataset, train=False)

    length = [round(len(train_set)*0.8), round(len(train_set)*0.2)]
    train_set, val_set = random_split(train_set, length)

    # load pytorch lightning model - TODO 요 부분 argparser 로 모델명 받게하기
    model = WarpModel(config.model, config.dataloader)
    model.set_dataset(train_set, val_set, test_set)

    # instantiate trainer
    logger = TensorBoardLogger('logs/', log_graph=True, **config.log)
    example = torch.zeros(1, 3, 1024, 1024).cuda()
    logger.log_graph(model, (example, example))
    trainer = Trainer(logger=logger, **config.trainer)
    
    # start training!
    trainer.fit(model)

    
if __name__ == "__main__":
    from options import load_config
    config = load_config("config/warpnet_v2_template.yaml")
    main(config)