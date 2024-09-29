import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread, get_transform_withoutNorm
import torchvision.transforms.functional as photometric
import pdb
import torch
import torch.nn.functional as F
# import cv2

class KITTIDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)

        self.dx_gt_filenames = self.dy_gt_filenames = None
            
        self.training = training
        # if self.training:
        #     assert self.disp_filenames is not None
        self.K = np.array([[0.58, 0, 0.5, 0],
                    [0, 1.92, 0.5, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.float32)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]

        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        elif len(splits[0]) == 3:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images
        elif len(splits[0]) == 4 and splits[0][2] == 'None':
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images


    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def load_dx_dy(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)
    
    def augment_intrinsic_matrices(self, intrinsics, img_size, resize, tx, ty):     
        intm_size_h = img_size[0]
        intm_size_w = img_size[1]
        scale_x = intm_size_w / resize[1]
        scale_y = intm_size_h / resize[0]
        
        ## Scaling        
        intrinsics[0, 0] = intrinsics[0, 0] / scale_x
        intrinsics[1, 1] = intrinsics[1, 1] / scale_y
        intrinsics[0, 2] = intrinsics[0, 2] / scale_x
        intrinsics[1, 2] = intrinsics[1, 2] / scale_y
        
        ## Cropping after scaling
        intrinsics[0, 2] -= float(tx)/intm_size_w
        intrinsics[1, 2] -= float(ty)/intm_size_h

        return intrinsics

    def __getitem__(self, index):
        inputs = {}
        if True:
            do_flip = False
            side = "l"
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

            
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None


        if self.training:
            w, h = left_img.size
            # crop_w, crop_h = 1152, 320 
            crop_w, crop_h = 512, 256 
            wider_pixels = 128

            x1 = random.randint(0, w - wider_pixels - crop_w)
            y1 = random.randint(0, h - crop_h)
            
            K = self.augment_intrinsic_matrices(self.K.copy(), (h,w), (h,w), x1, y1)
            for scale in range(1):
                K[0, :] *= crop_w
                K[1, :] *= crop_h

                inv_K = np.linalg.pinv(K)

                inputs[("K", scale)] = torch.from_numpy(K)
                inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
                
            ### flip
            if random.random()>0.5:
                left_img_flip = left_img.transpose(Image.FLIP_LEFT_RIGHT)
                right_img_flip = right_img.transpose(Image.FLIP_LEFT_RIGHT)
                left_img = right_img_flip
                right_img = left_img_flip

            left_img_wider = left_img.crop((x1, y1, x1 + crop_w + wider_pixels, y1 + crop_h))
            right_img_wider = right_img.crop((x1, y1, x1 + crop_w + wider_pixels, y1 + crop_h))
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            if disparity is not None:
                disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # processed1 = get_transform_withoutNorm()
            processed1 = get_transform()
            left_img1 = processed1(left_img)
            right_img1 = processed1(right_img)
            left_img_wider1 = processed1(left_img_wider)
            right_img_wider1 = processed1(right_img_wider)
            
            # photometric augmentation: brightness and contrast perturb
            sym_random_brt = np.random.uniform(0.5, 1.5)
            sym_random_cts = np.random.uniform(0.5, 1.5)
            asym_random_brt = np.random.uniform(0.95, 1.05, size=2)
            asym_random_cts = np.random.uniform(0.95, 1.05, size=2)
            # brightness
            left_img = photometric.adjust_brightness(left_img, sym_random_brt)
            right_img = photometric.adjust_brightness(right_img, sym_random_brt)
            left_img_wider = photometric.adjust_brightness(left_img_wider, sym_random_brt)
            right_img_wider = photometric.adjust_brightness(right_img_wider, sym_random_brt)
            left_img = photometric.adjust_brightness(left_img, asym_random_brt[0])
            right_img = photometric.adjust_brightness(right_img, asym_random_brt[1])
            left_img_wider = photometric.adjust_brightness(left_img_wider, asym_random_brt[1])
            right_img_wider = photometric.adjust_brightness(right_img_wider, asym_random_brt[1])
            # contrast
            left_img = photometric.adjust_contrast(left_img, sym_random_cts)
            right_img = photometric.adjust_contrast(right_img, sym_random_cts)
            left_img_wider = photometric.adjust_contrast(left_img_wider, sym_random_cts)
            right_img_wider = photometric.adjust_contrast(right_img_wider, sym_random_cts)
            left_img = photometric.adjust_contrast(left_img, asym_random_cts[0])
            right_img = photometric.adjust_contrast(right_img, asym_random_cts[1])
            left_img_wider = photometric.adjust_contrast(left_img_wider, asym_random_cts[1])
            right_img_wider = photometric.adjust_contrast(right_img_wider, asym_random_cts[1])
            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            left_img_wider = processed(left_img_wider)
            right_img_wider = processed(right_img_wider)
            # # random patch exchange of right image
            # patch_h = random.randint(10, 220)
            # patch_w = random.randint(10, 50)
            # patch1_x = random.randint(0, crop_h-patch_h)
            # patch1_y = random.randint(0, crop_w-patch_w)
            # patch2_x = random.randint(0, crop_h-patch_h)
            # patch2_y = random.randint(0, crop_w-patch_w)
            # # pdb.set_trace()
            # # print(right_img.shape)
            # img_patch = right_img[:, patch2_x:patch2_x+patch_h, patch2_y:patch2_y+patch_w]
            # right_img[:, patch1_x:patch1_x+patch_h, patch1_y:patch1_y+patch_w] = img_patch


            inputs[("color", 0, 0)] = left_img1
            inputs[("color", 's', 0)] = right_img1
            inputs["left_raw_wider"] = left_img_wider1
            inputs["right_raw_wider"] = right_img_wider1
            inputs["left_aug_wider"] = left_img_wider
            inputs["right_aug_wider"] = right_img_wider
            inputs["wider_pixels"] = torch.tensor(wider_pixels)
            inputs["left"] = left_img
            inputs["right"] = right_img
            if disparity is not None:
                disparity = torch.from_numpy(disparity)
                inputs["disparity"] = disparity

            return inputs
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1280x384
            top_pad = 384 - h
            right_pad = 1280 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='edge')
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='edge')
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)


            # if disparity is not None and dx_gt is not None and dy_gt is not None:
            if disparity is not None:
                disparity = torch.from_numpy(disparity)
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
            


class KITTI_Raw(KITTIDataset):
    def __init__(self, datapath, list_filename, training):
        super().__init__(datapath, list_filename, training)
        self.datapath = datapath+'kitti_raw'

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        folder = [x[0] for x in splits]
        frame_index = []
        side = []
        for i, x in enumerate(splits):
            if len(x) == 3:
                frame_index.append(int(x[1]))
                side.append(x[2])
            else:
                frame_index.append(0)
                side.append(None)
        left_images = []
        right_images = []
        for i, x in enumerate(side):
            f_str = "{:010d}{}".format(frame_index[i], '.png')
            left_images.append(os.path.join(folder[i],
                                            "image_0{}/data".format('2'),
                                            f_str))
            right_images.append(os.path.join(folder[i],
                                            "image_0{}/data".format('3'),
                                            f_str))
            
        disp_images = None
        return left_images, right_images, disp_images
    
    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)
    
