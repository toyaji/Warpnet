from os import error
import numpy as np
import cv2
from pathlib import Path


class Align:
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

        self.img_paths =  sorted(list((self.base_path / 'cropped').glob("*")))
        self.tform_path = self.base_path / 'tform.txt'
        self.tar_dir = self.base_path / 'aligned'
        self.tar_compare_dir = self.base_path / 'compare'

        if not self.tar_dir.exists():
            self.tar_dir.mkdir()
        if not self.tar_compare_dir.exists():
            self.tar_compare_dir.mkdir()
        
        # set empty variables
        self.motion = None
        self.rsz = None
        self.num = None
        self.size = None

    def align_main(self, motion='affine', rsz=1, iteration=500, eps=1e-8, verbose=False):
        # set parameters
        self.motion = motion
        self.rsz = rsz
        
        assert self.motion in ['affine', 'homography', 'translation', 'euclidean']

        # find matrix and transform the give images
        imgs, imgs_ds, grays = self._scan()

        try:
            t, inv_t, _ = self.align_ecc(imgs_ds, grays, ecc_iter=iteration, eps=eps)
        except Exception as e:
            print("Following error occurs: ", e)
            return None

        t_imgs, rsz_t, inv_rsz_t = self.apply_transform(imgs, t, inv_t)

        # save tform as txt file in each folder
        self._tform_save(rsz_t)
        # crop and compare to check if aligned well.
        corner = self.get_aligned_corner(inv_rsz_t)
        self._crop(t_imgs, corner)
        sum_img_t, sum_img = self.align_check(imgs, t_imgs)

        cv2.imwrite(str(self.tar_compare_dir / 'aligned.jpg'), np.uint8(255.*sum_img_t))
        cv2.imwrite(str(self.tar_compare_dir /'orig.jpg'), np.uint8(255.*sum_img))

        if verbose:
            return cv2.cvtColor(sum_img_t, cv2.COLOR_BGR2RGB), cv2.cvtColor(sum_img, cv2.COLOR_BGR2RGB)
        else:
            return None
    
    def _scan(self):
        imgs_fsize = [cv2.imread(str(img), -1) for img in self.img_paths]
        imgs_fsize= [img.astype(np.float32) / 255 for img in imgs_fsize]
        imgs_set = [cv2.resize(img, None, fx=1./(2 ** self.rsz), fy=1./(2 ** self.rsz), interpolation=cv2.INTER_CUBIC) for img in imgs_fsize]
        gray_set = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs_set]

        self.size = imgs_fsize[-1].shape[:2]
        self.num = len(imgs_fsize)

        return imgs_fsize, imgs_set, gray_set

    def align_ecc(self, image_set, images_gray_set, ecc_iter=500, eps=1e-6, thre=1e-5):
        # set motion model 
        if self.motion == 'affine':
            warp_mode = cv2.MOTION_AFFINE
        elif self.motion == 'homography':
            warp_mode = cv2.MOTION_HOMOGRAPHY
        elif self.motion == 'translation':
            warp_mode = cv2.MOTION_TRANSLATION
        elif self.motion == 'euclidean':
            warp_mode = cv2.MOTION_EUCLIDEAN
        else:
            raise KeyError("Proper motion model should be given: \
                ('affine', 'homography', 'translation', 'euclidean')")

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if  warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
            tform_set_init = np.tile(np.eye(3, 3, dtype=np.float32), (len(image_set), 1, 1))
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            tform_set_init = np.tile(np.eye(2, 3, dtype=np.float32), (len(image_set), 1, 1))

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iter, eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        tforms = np.zeros_like(tform_set_init)
        inv_tforms = np.zeros_like(tform_set_init)
        for i in range(self.num):
            _, warp_matrix = cv2.findTransformECC(images_gray_set[0], images_gray_set[i], warp_matrix, warp_mode, criteria)
            tforms[i] = warp_matrix
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                inv_tforms[i] = np.linalg.inv(warp_matrix)
            else:
                inv_tforms[i] = cv2.invertAffineTransform(warp_matrix)

        l = self.rigid_regularizer(inv_tforms)
        valid_id = l < thre
        return tforms, inv_tforms, valid_id

    @staticmethod
    def rigid_regularizer(matrixs):
        def _rigid_regularizer(matrix):
            t_1 = matrix[0, 0] - matrix[1, 1]
            t_2 = matrix[0, 1] + matrix[1, 0]
            l = t_1 ** 2 + t_2 ** 2
            return l
        ls = []
        for i in range(7):
            l = _rigid_regularizer(matrixs[i])
            ls.append(l)
        return np.array(ls)

    def apply_transform(self, image_set, tform_set, tform_inv_set):
        rszed_tforms = tform_set.copy()
        inv_rszed_tforms = tform_inv_set.copy()
        transfored_imgs = np.zeros_like(image_set)
        
        r, c = image_set[0].shape[0:2]
        for i in range(self.num):
            rszed_tforms[i][0:2,2] *= 2 ** self.rsz 
            inv_rszed_tforms[i][0:2,2] *= 2** self.rsz
            if self.motion == "homography":
                image_i_transform = cv2.warpPerspective(image_set[i], rszed_tforms[i], (c, r),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            else:    
                image_i_transform = cv2.warpAffine(image_set[i], rszed_tforms[i], (c, r),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            transfored_imgs[i] = image_i_transform

        return transfored_imgs, rszed_tforms, inv_rszed_tforms
    
    def _tform_save(self, tform):
        with open(self.tform_path, 'w') as out:
            for i, t_i in enumerate(tform):
                out.write("%05d-%05d:"%(1, i+1) + '\n')
                np.savetxt(out, t_i, fmt="%.4f")

    def get_aligned_corner(self, inv_tforms):
        h, w = self.size
        corner = np.array([[0, 0, w, w],   
                           [0, h, 0, h],
                           [1, 1, 1, 1]])

        # pairing corners between ref image and transformed image
        for i in range(self.num):
            if self.motion == 'homography':
                corner_out = inv_tforms[i]
            else:
                corner_out = np.vstack([inv_tforms[i], [0,0,1]]).dot(corner)
            corner_out[0] /= corner_out[2]
            corner_out[1] /= corner_out[2]
            corner_out = corner_out[..., np.newaxis]
            if i == 0:
                corner_t = corner_out
            else:
                corner_t = np.append(corner_t,corner_out, axis=2)

        min_w = corner_t[0,[0,1],:].max()
        min_h = corner_t[1,[0,2],:].max()
        max_w = corner_t[0,[2,3],:].min()
        max_h = corner_t[1,[1,3],:].min()
        
        min_w, min_h, max_w, max_h = np.fix((min_w, min_h, max_w, max_h)).astype(int)

        with open(self.tform_path, 'a') as out:
            out.write("corner:" + '\n')
            out.write("%05d %05d %05d %05d"%(min_h, max_h, min_w, max_w))
            out.close()

        return (min_h, min_w, max_h, max_w)

    def _crop(self, transformed_imgs, corner):
        min_h, min_w, max_h, max_w = corner

        for i, img in enumerate(transformed_imgs):
            cropped_imgs = img[min_h:max_h,min_w:max_w,:]
            wt, ht = cropped_imgs.shape[:2]
            save_path = str(self.tar_dir / "{}.JPG".format(self.img_paths[i].stem))
            cv2.imwrite(save_path, np.uint8(255.*cropped_imgs))

    def align_check(self, image_set, image_aligned):
        sum_img = np.float32(image_set[0]) * 1. / len(image_aligned)
        sum_img_t = np.float32(image_aligned[0]) * 1. / len(image_aligned)
        identity_transform = np.eye(2, 3, dtype=np.float32)
        r, c = image_set[0].shape[0:2]
        for i in range(1, len(image_aligned)):
            sum_img_t += np.float32(image_aligned[i]) * 1. / len(image_aligned)
            image_set_i = cv2.warpAffine(image_set[i], identity_transform, (c, r),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            sum_img += np.float32(image_set_i) * 1. / len(image_aligned)
        return sum_img_t, sum_img
