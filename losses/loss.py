import torch
import torch.nn.functional as F

from .utils import *

class Loss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

        # checking height and width are multiples of 32
        assert self.args.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.args.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = self.args.device

        self.relu = torch.nn.ReLU()

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

    def forward(self, inputs, outputs):
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for key, ipt in inputs.items():
            if isinstance(ipt, torch.Tensor):
                inputs[key] = ipt.to(self.args.device)
        scale_list = [k[1] for k in outputs.keys() if k.__contains__('disp')]
            
        for scale in scale_list:
            disp = outputs[("disp", scale)]
            if "flip" in inputs.keys():
                disp = torch.where(inputs["flip"], -disp, disp)
            source_scale = 0

            outputs[("disp", scale)]=disp
            for i, frame_id in enumerate(['s']):
                warp_right = self.warp(
                    inputs[("color", frame_id, source_scale)],
                    disp)
                outputs[("color", frame_id, scale)] = warp_right
                outputs["warped"] = outputs[("color", frame_id, scale)].clone().detach()



    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
        
    def disocc_detection(self, disp_l, img=None):
        if not hasattr(self,'_rendering'):
            self._rendering = Rendering()
        
        if img is not None:
            img_r_warped, valid_mask = self._rendering(img, disp_l, return_mask=True)
            return valid_mask, img_r_warped
        else:
            disp_r_warped, valid_mask = self._rendering(disp_l, disp_l, return_mask=True)
            return valid_mask, disp_r_warped
    
       
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0
        global_loss_weights = self.args.loss_weights

        scale_list = [k[1] for k in outputs.keys() if k.__contains__('disp')]
        
        imgL = inputs[("color", 0, 0)]
        imgR = inputs[("color", 's', 0)]

        

        for scale in scale_list:
            loss = 0
            reprojection_losses = []

            source_scale = 0
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, source_scale)]
            target = inputs[("color", 0, source_scale)]
            pred = outputs[("color", 's', scale)]
            reprojection_loss = self.compute_reprojection_loss(pred, target)
                
            if self.args.occ_detect:
                occ_tensor, img_r_warped = self.disocc_detection(-disp.detach(), imgL)
                reprojection_loss *= occ_tensor
                
                if self.args.do_summary:
                    with torch.no_grad():
                        outputs['vis_occ'] = (occ_tensor>0).to(torch.float)
                        outputs['vis_img_r_warped'] = img_r_warped
            
            reprojection_loss = reprojection_loss.mean()

            losses['reproj_loss/{}'.format(scale)] = reprojection_loss * global_loss_weights[0]

            loss += reprojection_loss * global_loss_weights[0]  

            mean_disp = outputs[("disp", scale)].mean(2, True).mean(3, True)
            norm_disp = outputs[("disp", scale)] / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color) * 1e-0

            loss += smooth_loss * global_loss_weights[1]
            losses[f"loss_smooth{scale}"] = smooth_loss

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        losses["loss"] = total_loss

        return losses


    def warp(self, x, disp, padding_mode='border',mode='bilinear'):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        vgrid = torch.cat((xx, yy), 1).float()

        if padding_mode == 'reflection':
            tmp = vgrid[:,:1,:,:] - disp
            tmp[tmp<0] = vgrid[:,:1,:,:][tmp<0] * (-1) - disp[tmp<0]
            vgrid[:,:1,:,:] = tmp
        else:
            # vgrid = Variable(grid)
            vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, padding_mode=padding_mode, mode=mode,align_corners=True)
        
        mask = torch.ones_like(x, requires_grad=False)
        mask = F.grid_sample(mask, vgrid, padding_mode='zeros')
        mask = (mask >= 0.999)

        output = torch.where(mask, output, output.detach())
        return output











