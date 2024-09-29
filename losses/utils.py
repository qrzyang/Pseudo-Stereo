import os, sys
import torch, torch.nn as nn
sys.path.append(os.path.dirname(__file__))
import torch.nn.functional as F
from einops.layers.torch import Rearrange

    
from torch_scatter import scatter_max
class Rendering(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def get_coordgrid(self, x):

        b, _, h, w = x.size()
        grid_h = torch.linspace(0.0, w - 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        grid_v = torch.linspace(0.0, h - 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        ones = torch.ones_like(grid_h)
        coordgrid = torch.cat((grid_h, grid_v, ones), dim=1).requires_grad_(False)

        return coordgrid
    
    def median_filter_2d(self, x, kernel_size=3):
        b, c, h, w = x.size()
        x_unfold = F.unfold(x.float(), kernel_size=kernel_size, padding=kernel_size//2)

        x_unfold = x_unfold.view(b, c, kernel_size*kernel_size, h, w)
        
        # x_unfold, _ = x_unfold.sort(dim=2)
        # median = x_unfold[:, :, kernel_size*kernel_size // 2, :, :].view(b, c, h, w)

        out = torch.prod(x_unfold, dim=2)
        out = F.pad(out[...,1:-1,1:-1],(1,1,1,1),"replicate")
        return out
        
    def rendering(self, x, pixel_coords_2d, sort_by = None, return_idx = False):
        B, C, H, W = x.size()
        image = torch.zeros_like(x)

        pixel_coords_2d = torch.round(pixel_coords_2d).long() 
        point_values = Rearrange('b c h w -> b (h w) c')(x)

        def shift_coord(coord, pixels:int=1):
            tmp_list = [coord]
            for i in range(pixels):
                pixel_coords_tmp0 = coord.clone()
                pixel_coords_tmp1 = coord.clone()
                
                pixel_coords_tmp0[..., [0]] = pixel_coords_tmp0[..., [0]] - (i+1)
                pixel_coords_tmp1[..., [0]] = pixel_coords_tmp1[..., [0]] + (i+1)
                
                tmp_list.append(pixel_coords_tmp0)
                tmp_list.append(pixel_coords_tmp1)
            return torch.cat(tmp_list, 1)
               
        # pixel_coords_2d = shift_coord(pixel_coords_2d)   

        mask = (pixel_coords_2d[..., [0]] >= 0) & (pixel_coords_2d[..., [0]] < W) & \
            (pixel_coords_2d[..., [1]] >= 0) & (pixel_coords_2d[..., [1]] < H)

        pixel_coords_2d = pixel_coords_2d * mask
        # point_values = point_values * mask
        
        if sort_by is not None:
            sort_by = Rearrange('b c h w -> b (h w) c')(sort_by)
            point_values_norm = sort_by
        else:
            point_values_norm = 1./(point_values[...,[-1]]+1e-8) if point_values.shape[-1]>1 else point_values
            point_values_norm *= point_values_norm<1e4
            
        # point_values_norm = point_values_norm.repeat(1, 3, 1)
        
        x_coords = pixel_coords_2d[..., 0]
        y_coords = pixel_coords_2d[..., 1]

        linear_idx = y_coords * W + x_coords  
        linear_idx = linear_idx[:,:,None]

        image_flat = Rearrange('b c h w -> b (h w) c')(image)
        image_flat = image_flat[...,[-1]]
        out, arg_max = scatter_max(point_values_norm, linear_idx, dim=-2, out=image_flat)
        
        arg_valid_mask = (arg_max>=0)*(arg_max<point_values_norm.shape[-2])
        arg_max = (arg_max*arg_valid_mask) % point_values.shape[-2]
        max_vals = torch.take_along_dim(point_values, arg_max, -2)
        max_vals = torch.where(arg_valid_mask, max_vals, torch.ones_like(max_vals)*1e8)
        
        
        max_vals[...,0,:] = 1e8
        image = Rearrange('b (h w) c -> b c h w', h=H, w=W)(max_vals)
        image *= image<1e4
        return image if not return_idx else image, arg_max
        
    def forward(self, x, flow_in, return_mask = False):
        """
        x: color or pts, souce
        
        out: color or pts, target
        """
        B, C, H, W = flow_in.size()
        assert (C in [1,2])
        # if input is disparity (value 0~1)
        if C == 1:
            flow = torch.zeros_like(flow_in).repeat(1,2,1,1)
            flow[:, [0], :, : ] = flow_in
        else:
            flow = flow_in
        
        init_pixelgrid = self.get_coordgrid(flow)[:,:2,...]
        pixel_coords_2d = init_pixelgrid + flow
        pixel_coords_2d = Rearrange('b c h w -> b (h w) c')(pixel_coords_2d)

        sort_by = None if C>1 else torch.abs(flow_in).clone()
        image, arg_max = self.rendering(x, pixel_coords_2d, sort_by=sort_by, return_idx=True)
        if not return_mask:
            return image
        else:
            mask = torch.zeros_like(arg_max,requires_grad=False).scatter_(1, arg_max*(arg_max>=0)*(arg_max<(H*W)), 1)
            mask[...,0,:] = 0.
            mask = Rearrange('b (h w) c -> b c h w', h=H, w=W)(mask)
            mask = self.median_filter_2d(mask)
            return image, mask
            
            
    
    def forward_with_norm_coord(self, x, coord, valid_inbound=None):
        B, C, H, W = x.size()
        pixel_coords_2d = coord.clone()
        pixel_coords_2d[:,0,...] = ((pixel_coords_2d[:,0,...]+1)/2) * W
        pixel_coords_2d[:,1,...] = ((pixel_coords_2d[:,1,...]+1)/2) * H
        pixel_coords_2d = Rearrange('b c h w -> b (h w) c')(pixel_coords_2d)
        image = self.rendering(x, pixel_coords_2d)
        return image
        
        
import random
class Fake_Flip():
    def __init__(self,device,args):
        self._rendering = Rendering()
        self.device = device
        self.args = args
    
    def generate_fake_flip(self, inputs:dict, model):
        ### ("color", 0, 0) without aug
        ### inputs["left"] with aug
        left_aug = inputs["left"].clone().to(self.device)
        right_aug = inputs["right"].clone().to(self.device)
        left = inputs[("color", 0, 0)].clone().to(self.device)
        right = inputs[("color", 's', 0)].clone().to(self.device)
        
        left_aug_wider = inputs["left_aug_wider"].clone().to(self.device)
        right_aug_wider = inputs["right_aug_wider"].clone().to(self.device)
        left_raw_wider = inputs["left_raw_wider"].clone().to(self.device)
        right_raw_wider = inputs["right_raw_wider"].clone().to(self.device)
        
        
        def get_fake_img(img_l_input, img_r_input, flip=False):
            with torch.no_grad():
                prop_disp_pyramid = model(img_l_input, img_r_input)
                outputs_h = {}
                max_i = 0
                for i, v in enumerate(prop_disp_pyramid):
                    outputs_h[("disp", i)] = v
                    max_i = i
            
            def pad_zeros(img, bg_img):
                pool_list = []
                list_len = 5
                def neighbor_mean(x):
                    b,c,h,w=x.shape
                    x_unfold = F.unfold(x, kernel_size=3, padding=1)
                    x_unfold = x_unfold.view(b, c, 3*3, h, w)
                    sum = torch.sum(x_unfold, 2)
                    nonzero_cont = torch.count_nonzero(x_unfold, 2) + 1e-8
                    y = sum / nonzero_cont
                    return y
                for i in range(list_len):
                    if i==0:
                        pool_list.append(neighbor_mean(img))
                    elif i<list_len-1:
                        pool_list.append(neighbor_mean(pool_list[i-1]))
                    else:
                        pool_list.append(bg_img)
                        
                for i in range(list_len):
                    img_valid_mask = img.sum(1,keepdim=True)!=0
                    img = torch.where(img_valid_mask, img, pool_list[i])
                
                return img
            disp = torch.flip(outputs_h[("disp",max_i)], [3]) if flip else outputs_h[("disp",max_i)]
            if flip:
                fake_image, valid_mask = self._rendering(right_aug_wider, -disp,return_mask=True)
                fake_image = pad_zeros(fake_image, right_aug_wider)
                fake_image = fake_image[...,:-inputs["wider_pixels"][0]]
            else:
                fake_image, valid_mask = self._rendering(left_aug_wider, -disp,return_mask=True)
                fake_image = pad_zeros(fake_image, left_aug_wider)
                fake_image = fake_image[...,:-inputs["wider_pixels"][0]]
            return fake_image
        
        for i, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[i] = v.to(self.device)
        
        # fake_image = get_fake_img(torch.flip(right, [3]), torch.flip(left, [3]), flip=True)
        # if self.args.global_step>0.:
        #     fake_image_no_flip = get_fake_img(left, right)
        #     fake_image_no_flip = fake_image_no_flip
        # else:
        #     fake_image_no_flip = inputs["right"]
            
        # flip_mask = torch.rand((left.shape[0], 1, 1, 1), device = left.device)>0.5
        
        ### 
        if random.random()>0.5:
            flip_mask = torch.ones((left.shape[0], 1, 1, 1), device = left.device)>0.5
            fake_image = get_fake_img(torch.flip(right_raw_wider, [3]), torch.flip(left_raw_wider, [3]), flip=True)
            fake_image_no_flip = inputs["right"]
        else:
            flip_mask = torch.zeros((left.shape[0], 1, 1, 1), device = left.device)>0.5
            fake_image = inputs["right"]
            if self.args.global_step>self.args.all_fake:
                fake_image_no_flip = get_fake_img(left_raw_wider, right_raw_wider)
            else:
                fake_image_no_flip = inputs["right"]
            # fake_image_no_flip = get_fake_img(left, right)
            
        inputs_fake = inputs.copy()
        inputs_fake[("color", 0, 0)] = torch.where(flip_mask, inputs[("color", 's', 0)], inputs[("color", 0, 0)])
        inputs_fake[("color", 's', 0)] = torch.where(flip_mask, inputs[("color", 0, 0)], inputs[("color", 's', 0)])
        inputs_fake["left"] = torch.where(flip_mask, inputs["right"], inputs["left"])
        inputs_fake["right"] = torch.where(flip_mask, fake_image, fake_image_no_flip)
        inputs_fake["flip"] = flip_mask
        
        if "edge" in inputs_fake.keys():
            del inputs_fake["edge"]
        return inputs_fake
     

def get_smooth_loss(disp, img, mask = None):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    mask: 0-1 tensor, ignore the 0 pixels
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    if mask is not None:
        b,c,h,w=mask.shape
        x_unfold = F.unfold(mask.float(), kernel_size=3, padding=1)
        x_unfold = x_unfold.view(b, c, 3*3, h, w)
        mask = torch.prod(x_unfold, dim=2)
        # mask = F.pad(mask[...,1:-1,1:-1],(1,1,1,1))
        
        grad_disp_x = grad_disp_x[mask[:, :, :, :grad_disp_x.shape[3]]==1]
        grad_disp_y = grad_disp_y[mask[:, :, :grad_disp_y.shape[2], :]==1]

    return grad_disp_x.mean() + grad_disp_y.mean()

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
                