import torch
import pytorch_lightning as pl

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from .warpnet import WarpNet
from .transformation import GeometricTnf

class WarpModel(pl.LightningModule):
    def __init__(self, model_params, loader_params, loss_params) -> None:
        super().__init__()
        # load the model
        self.model = WarpNet(**model_params)
        self.transformer = GeometricTnf('affine', size=model_params.size)

        # set dataloader paramters
        self.batch_size = loader_params.batch_size
        self.num_workers = loader_params.num_workers
        self.shuffle = loader_params.shuffle
        self.lr = loader_params.learning_rate
        self.l2_lambda = loss_params.l2_lambda
        self.weight_decay = loss_params.weight_decay
        
        # save hprams for log
        self.save_hyperparameters(model_params)
        self.save_hyperparameters(loader_params)

    def configure_optimizers(self):
        # TODO adam parameter setting 좀 더 확인하기
        optimazier = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': LambdaLR(optimazier, lr_lambda=lambda step: 0.95*step),
            'name': 'leraning_rate'
        }
        return [optimazier], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        y, x = batch
        b = x.size()[0] 

        origin = x.clone().detach()
        theta = self.model(x, y)
        aligned = self.transformer(origin, theta)
        theta = theta.view(b, 2, 3)
        # to give l2 reglurize on theta
        l2_loss = self.l2_lambda * torch.norm(theta[:, :, :2], p=2)
        loss = F.mse_loss(aligned, y) + l2_loss

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
        y, x = batch
        origin = x.clone().detach()
        theta = self.model(x, y)
        aligned = self.transformer(origin, theta)
        loss = F.mse_loss(aligned, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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