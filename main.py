import warnings

from torch.utils.data.dataset import ConcatDataset, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler
from model import WarpModel
from data.zoomdata import ZoomLZoomData

warnings.filterwarnings('ignore')


def main(config):

    # dataset settting
    train_data = [ZoomLZoomData(config.dataset, scale_idx=(1, i), train=True) for i in range(2, 8)]
    test_data = [ZoomLZoomData(config.dataset, scale_idx=(1, i), train=False) for i in range(2, 8)]
    train_set = ConcatDataset(train_data)
    test_set = ConcatDataset(test_data)

    length = [round(len(train_set)*0.8), round(len(train_set)*0.2)]
    train_set, val_set = random_split(train_set, length)

    # load pytorch lightning model - TODO 요 부분 argparser 로 모델명 받게하기
    model = WarpModel(config.model, config.dataloader)
    model.set_dataset(train_set, val_set, test_set)

    # instantiate trainer
    logger = TensorBoardLogger('logs/', **config.log)
    trainer = Trainer(logger=logger, **config.trainer)
    
    # start training!
    trainer.fit(model)

    
if __name__ == "__main__":
    from options import load_config
    config = load_config("config/warpnet_v2_template.yaml")
    main(config)