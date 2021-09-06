from numpy import frexp
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import PSNR, SSIM
from model.warpnet import WarpNet
from model.transformation import GeometricTnf
import gc


class WarpModel(pl.LightningModule):
    def __init__(self, model_params, loader_params) -> None:
        super().__init__()
        # load the model
        self.model = WarpNet(**model_params)
        self.transformer = GeometricTnf('affine', size=model_params.size)

        # set dataloader paramters
        self.batch_size = loader_params.batch_size
        self.num_workers = loader_params.num_workers
        self.shuffle = loader_params.shuffle

        # set metrices to evaluate performence
        # TODO 다른 metrics 추가해야함... 모듈 만들던지 해서
        psnr = PSNR(); ssim = SSIM()
        self.train_psnr = psnr.clone()
        self.train_ssim = ssim.clone()
        self.valid_psnr = psnr.clone()
        self.valid_ssim = ssim.clone()

    def forward(self, x, y):
        return self.model(x, y)

    def configure_optimizers(self):
        # TODO params 분리되 되는듯... 여기다가 앞에 CNN gep 붙이는거 붙여되 되겠네
        optimazier = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimazier

    def training_step(self, batch, batch_idx):
        y, x = batch
        # my dataloader get 6 pairs of image crops per iteration
        if len(x.size()) > 4:
            b, ncrops, c, h, w = x.size()
            x = x.view(-1, c, h, w)
            y = y.view(-1, c, h, w)

        theta = self.model(x, y)
        aligned = self.transformer(x, theta)
        loss = F.mse_loss(aligned, y)
        self.train_psnr(aligned, y)
        self.train_ssim(aligned, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_psnr', self.train_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_ssim', self.train_ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y, x = batch
        # my dataloader get 6 pairs of image crops per iteration
        if len(x.size()) > 4:
            b, ncrops, c, h, w = x.size()
            x = x.view(-1, c, h, w)
            y = y.view(-1, c, h, w)

        theta = self.model(x, y)
        aligned = self.transformer(x, theta)
        loss = F.mse_loss(aligned, y)
        self.valid_psnr(aligned, y)
        self.valid_ssim(aligned, y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_psnr', self.valid_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_ssim', self.valid_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def set_dataset(self, train_set, val_set, test_set):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    # TODO data scale 별로 다 붙이기
    def train_dataloader(self):
        dataloader = DataLoader(self.train_set, self.batch_size, self.shuffle, num_workers=self.num_workers)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.train_set, self.batch_size, self.shuffle, num_workers=self.num_workers)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.train_set, self.batch_size, self.shuffle, num_workers=self.num_workers)
        return dataloader