import cv2
import numpy as np
from data import common
import torch
from pathlib import Path
from torch.utils.data import Dataset

#from data import common

class ZoomLZoomData(Dataset):
    """
    This data class return data pair which consist of 7 images having different resolution  
    
    """
    def __init__(self, args, scale_idx=(1,2), train: bool = True) -> None:
        super().__init__()
        self.patch_size = args.patch_size
        self.scale_idx = scale_idx
        self.get_from_dir = args.get_from_dir
        self.img_ext = args.img_ext
        self.gray_scale = args.gray_scale
        self.reduce_size = args.reduce_size
        self.train = train

        self._set_filesystem(args.data_dir)

    def __getitem__(self, idx):
        hr, lr = self._scan(idx)
        hr, lr = common.get_random_patch(hr, lr, self.patch_size)
        # to reduce computational cost, we can consider reduce the size of inputs
        if self.reduce_size != 1:
            hr = cv2.resize(hr, dsize=(0, 0), fx=self.reduce_size, fy=self.reduce_size, interpolation=cv2.INTER_LINEAR)
            lr = cv2.resize(lr, dsize=(0, 0), fx=self.reduce_size, fy=self.reduce_size, interpolation=cv2.INTER_LINEAR)
        return common.np2Tensor([hr, lr], 255)

    def __len__(self):
        return len(self.base_paths)

    def _set_filesystem(self, dir_data):
        if isinstance(dir_data, str):
            self.apath = Path(dir_data)
        # check for path exist
        assert self.apath.exists(), "Data dir path is wrong!"

        if self.train:
            self.base_paths = sorted(list((self.apath / "train").glob("*")))
        else:
            self.base_paths = sorted(list((self.apath / "test").glob("*")))
    
    def _scan(self, idx):
        (target_idx, source_idx) = self.scale_idx
        base_path = self.base_paths[idx] / self.get_from_dir
        target_path = base_path / "{:05d}.{}".format(target_idx, self.img_ext)
        source_path = base_path / "{:05d}.{}".format(source_idx, self.img_ext)
        # gray scale or RGB according to args setting
        if self.gray_scale:
            hr = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
            lr = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
            hr = np.expand_dims(hr, 2)
            lr = np.expand_dims(lr, 2)
        else:
            hr = cv2.imread(str(target_path))
            lr = cv2.imread(str(source_path)) 
        return hr, lr

    def _get_focalscale(self, idx):
        ref_paths = self.base_paths[idx].glob("*.JPG")
        focals = [common.readFocal_pil(p) for p in ref_paths]
        focal_scale = np.array(focals) / 240
        return focal_scale

