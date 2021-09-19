import torch
import pytorch_lightning as pl

from torch.nn import Sequential
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose, CenterCrop, Normalize

from .warpnet import WarpNet
from .transformation import GeometricTnf

class WarpModel(pl.LightningModule):
    def __init__(self, geometry, model_params, loader_params, opt_params) -> None:
        super().__init__()

        # transforme preprocess - it will crop the image before feed into main net
        self.preprocess = Compose([CenterCrop(model_params.size - model_params.buffer*2), 
                                   Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])
        # load the model
        self.model = WarpNet(**model_params)


        self.transformer = GeometricTnf(geometry, 
                                        size=model_params.size)

        # set dataloader paramters
        self.batch_size = loader_params.batch_size
        self.num_workers = loader_params.num_workers
        self.shuffle = loader_params.shuffle
        self.lr = opt_params.learning_rate
        self.l2_lambda = opt_params.l2_lambda
        self.weight_decay = opt_params.weight_decay
        self.geo_model = geometry
        
        # save hprams for log
        self.save_hyperparameters(model_params)
        self.save_hyperparameters(loader_params)

    def forward(self, x, y):
        x = self.preprocess(x)
        y = self.preprocess(y)
        return self.model(x, y)

    def configure_optimizers(self):
        # TODO adam parameter setting 좀 더 확인하기
        optimazier = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimazier, patience=7),
            'monitor': "val_loss",
            'name': 'leraning_rate'
        }
        return [optimazier], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        theta = self(x, y)
        aligned = self.transformer(x, theta)
        loss = F.mse_loss(aligned, y)

        # following code is for memory leak debug
        """
        f = open("obj_log/iter_{}.txt".format(batch_idx), "w")
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    if  len(obj.size()) > 0:
                        #print(type(obj), obj.size())
                        f.write("{}, {} \n".format(type(obj), obj.size()))
            except: pass
        f.close()
        """
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        theta = self(x, y)
        aligned = self.transformer(x, theta)
        loss = F.mse_loss(aligned, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        theta = self(x, y)
        aligned = self.transformer(x, theta)
        loss = F.mse_loss(aligned, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def set_dataset(self, train_set, val_set, test_set):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        dataloader = DataLoader(self.train_set, self.batch_size, self.shuffle, num_workers=self.num_workers)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.train_set, self.batch_size, self.shuffle, num_workers=self.num_workers)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.train_set, self.batch_size, self.shuffle, num_workers=self.num_workers)
        return dataloader