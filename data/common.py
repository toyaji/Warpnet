import random
import rawpy
import numpy as np
import skimage.color as sc
from PIL import Image
from pathlib import Path
import torch

def get_identical_patches(imgs, patch_size):
    """Get patches of same fov from all scales of images"""
    ih, iw = imgs[0].shape[:2]
    tp = patch_size
    ix = np.random.randint(0, iw - patch_size)
    iy = np.random.randint(0, ih - patch_size)
    imgs = []
    for i in range(len(imgs)):
        imgs.append(imgs[i][iy:iy + tp, ix:ix + tp, :])
    return imgs

def get_random_patch(hr, lr, patch_size):
    # some input images have little bit different size so we need to consider that
    ih1, iw1 = hr.shape[:2]
    ih2, iw2  =lr.shape[:2]
    ih = min(ih1, ih2)
    iw = min(iw1, iw2)

    # get patch by random crop
    tp = patch_size
    ix = np.random.randint(0, iw - patch_size)
    iy = np.random.randint(0, ih - patch_size)
    hr = hr[iy:iy + tp, ix:ix + tp, :]
    lr = lr[iy:iy + tp, ix:ix + tp, :]
    return hr, lr

def get_random_patches(hr, lrs, patch_size):
    """Get patches of different random fov for each scale of image"""
    def _get_random_patch(hr, lr, patch_size):
        ih, iw = hr.shape[:2]
        tp = patch_size
        ix = np.random.randint(0, iw - patch_size)
        iy = np.random.randint(0, ih - patch_size)
        hr = hr[iy:iy + tp, ix:ix + tp, :]
        lr = lr[iy:iy + tp, ix:ix + tp, :]
        return hr, lr

    hrs = []
    for i, lr in enumerate(lrs):
        h, l = _get_random_patch(hr, lr, patch_size)
        hrs.append(h)
        lrs[i] = l
    return hrs, lrs


def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)
        return tensor

    return [_np2Tensor(_l) for _l in l]

def add_noise(x, noise='.'):
    if noise != '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(_l) for _l in l]

def readFocal_pil(image_path, focal_code=37386):
    if isinstance(image_path, Path):
        image_path = str(image_path)
    try:
        img = Image.open(image_path)
    except:
        print(image_path)
        return None
    exif_data = img._getexif()
    img.close()
    return float(exif_data[focal_code])

def crop_fov(image, ratio, buffer=1.):
    width, height = image.shape[:2]
    new_width = width * ratio * buffer
    new_height = height * ratio * buffer
    left = np.ceil((width - new_width)/2.)
    top = np.ceil((height - new_height)/2.)
    right = np.floor((width + new_width)/2.)
    bottom = np.floor((height + new_height)/2.)
    # print("Cropping boundary: ", top, bottom, left, right)
    cropped = image[int(left):int(right), int(top):int(bottom), ...]
    return cropped

# zoom-learn-zoom/utils.py
def get_bayer(path, black_lv=512, white_lv=16383):
    if isinstance(path, str) or isinstance(path, Path):
        raw = rawpy.imread(str(path))
    else:
        raw = path
    bayer = raw.raw_image_visible.astype(np.float32)
    bayer = (bayer - black_lv) / (white_lv - black_lv)  # subtract the black level
    return bayer

def get_4ch(bayer):
    h, w = bayer.shape[:2]
    rgba = np.zeros((h // 2, w // 2, 4), dtype=np.float32)
    rgba[:, :, 0] = bayer[0::2, 0::2]  # R
    rgba[:, :, 1] = bayer[1::2, 0::2]  # G1
    rgba[:, :, 2] = bayer[1::2, 1::2]  # B
    rgba[:, :, 3] = bayer[0::2, 1::2]  # G2

    return rgba

def get_1ch(raw):
    h, w = raw.shape[:2]
    bayer = np.zeros((h * 2, w * 2), dtype=raw.dtype)
    bayer[0::2, 0::2] = raw[..., 0]
    bayer[1::2, 0::2] = raw[..., 1]
    bayer[1::2, 1::2] = raw[..., 2]
    bayer[0::2, 1::2] = raw[..., 3]

    return bayer