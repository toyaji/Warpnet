import numpy as np
from pathlib import Path
from PIL import Image
from data import common
import rawpy


class Crop:
    """
    Crop along the zoomed images and align them. Following codes are almost from 'Zoom-Learn-Zoom'
    :https://github.com/ceciliavision/zoom-learn-zoom
    
    """
    def __init__(self, base_path) -> None:        
        # path setting 
        if isinstance(base_path, Path):
            self.base_path = base_path
        else:
            self.base_path = Path(base_path)

        self.ref_path = self.base_path / '00001.JPG'
        self.tar_dir = self.base_path / 'cropped'
        self.tar_onlycrop_dir = self.base_path / 'onlycrop'

        if not self.tar_dir.exists():
            self.tar_dir.mkdir()
        if not self.tar_onlycrop_dir.exists():
            self.tar_onlycrop_dir.mkdir()

        self.ref_focal = common.readFocal_pil(self.ref_path)
    
    def crop_main(self, buffer=1., verbose=False):
        # crop loop for one imageset(consist of 7 images)
        for i in range(7):
            self.crop_one_img(i+1, buffer=buffer, verbose=verbose)
        
        if verbose:
            print("Set number {} has cropped.".format(self.base_path))

    def crop_one_img(self, file_no, buffer=1.1, save=True, verbose=False) -> np.ndarray:
        # crop just one image when the image number is given.
        # here, I modified it to add a buffer to the crop factor, 
        # due to that focal lengths vary in the dataset.
        file_name = "{:05d}.JPG".format(file_no)
        file_path = self.base_path / file_name
        focal_ratio = self.ref_focal / common.readFocal_pil(file_path)

        if verbose:
            print("Image {} / {} has focal ratio: {:2.4f} ".format(self.base_path.stem, file_path.stem, focal_ratio))
        
        img_rgb = Image.open(file_path)

        if file_no == 1:
            cropped = common.crop_fov(np.array(img_rgb), 1. / focal_ratio, buffer=1.)
        else:    
            cropped = common.crop_fov(np.array(img_rgb), 1. / focal_ratio, buffer=buffer)
        
        if save:
            rgb_cropped = Image.fromarray(cropped)
            rgb_cropped.save(self.tar_onlycrop_dir / "{:05d}.JPG".format(file_no))
            img_rgb_s = rgb_cropped.resize((int(rgb_cropped.width * focal_ratio), 
                                            int(rgb_cropped.height * focal_ratio)), Image.ANTIALIAS)
            img_rgb_s.save(self.tar_dir / "{:05d}.JPG".format(file_no))
            