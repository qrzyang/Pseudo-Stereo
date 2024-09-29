import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from utils import *
from torch.utils.data import DataLoader
import gc
import json
from datetime import datetime
from utils.saver import Saver
import losses.loss as total_loss
from torch.cuda import amp
from torch.optim.lr_scheduler import CosineAnnealingLR

cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class Trainer:
    def __init__(self) -> None:
        def get_args():
            parser = argparse.ArgumentParser()


            parser.add_argument('--maxdisp', type=int, default=192,
                                help='maximum disparity')
            parser.add_argument('--fea_c', type=list,
                                default=[32, 24, 24, 16, 16], help='feature extraction channels')

            parser.add_argument('--dataset', 
                                default="kitti",
                                help='dataset name', choices=__datasets__.keys())
            parser.add_argument('--datapath', help='data path', default="/mnt/win/DataSet/")

            parser.add_argument('--trainlist',
                                help='training list', 
                                default="filenames/kt2015_all_repeat.txt"
                                )
            parser.add_argument('--testlist',
                                help='testing list', 
                                default="filenames/kt2015_val.txt"
                                )

            parser.add_argument('--lr', type=float, default=1e-4,
                                help='base learning rate')
            parser.add_argument('--batch_size', type=int, 
                                default=6,
                                help='training batch size')
            parser.add_argument('--test_batch_size', type=int,
                                default=8, help='testing batch size')
            parser.add_argument('--epochs', type=int, default=8000,
                                help='number of epochs to train')
            parser.add_argument('--lrepochs', type=str, default="8000,15000:2,2",
                                help='the epochs to decay lr: the downscale rate')
            parser.add_argument('--ckpt_start_epoch', type=int, default=80,
                                help='the epochs at which the program start saving ckpt')

            parser.add_argument('--logdir', help='the directory to save logs and checkpoints',
                                default="./logs/fake_psm", 
                                type=str
                                )
            parser.add_argument(
                '--loadckpt', help='load the weights from a specific checkpoint')
            parser.add_argument('--resume', type=str, default="",
                help='continue training the model')
            parser.add_argument('--seed', type=int, default=1,
                                metavar='S', help='random seed (default: 1)')

            parser.add_argument('--summary_freq', type=int, 
                                default=501,
                                help='the frequency of saving summary')
            parser.add_argument('--save_freq', type=int, default=500,
                                help='the frequency of saving checkpoint')
            
            parser.add_argument('--model', type=str, default="PSMNet")
            parser.add_argument('--message', type=str, default="")
            parser.add_argument('--occ_detect', action='store_true', default=False)
            parser.add_argument('--all_fake', type=int, default=1e10,
                                help='All fake input after xx iterations')

            return parser.parse_args()
    
        ## set args
        args = get_args()
        args.device = 'cuda'
        
        self.args = args
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        os.makedirs(args.logdir, exist_ok=True)

        # create summary self.logger
        self.saver = Saver(args)
        print("creating new summary file")
        self.logger = SummaryWriter(self.saver.experiment_dir)
        self.logger.add_text('Train', self.args.message)

        self.logfilename = self.saver.experiment_dir + '/log.txt'

        with open(self.logfilename, 'a') as log:  # wrt running information to log
            log.write('\n\n\n\n')
            log.write('-------------------NEW RUN-------------------\n')
            log.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            log.write('\n')
            json.dump(args.__dict__, log, indent=2)
            log.write('\n')

        # dataset, dataloader
        self.TrainDataset = __datasets__[args.dataset]
        self.TestDataset = __datasets__[args.dataset]
        self.train_dataset = self.TrainDataset(args.datapath, args.trainlist, True)
        self.test_dataset = self.TestDataset(args.datapath, args.testlist, False)
        # collate_fn = MyCollator()
        self.TrainImgLoader = DataLoader(self.train_dataset, args.batch_size, shuffle=True, 
                                         num_workers=6, drop_last=True, pin_memory=True, 
                                         prefetch_factor=4
                                         )
        self.TestImgLoader = DataLoader(self.test_dataset, args.test_batch_size, shuffle=False, 
                                        num_workers=4, drop_last=False, pin_memory=True)

        # self.model, self.optimizer
        from models.PSMNet import PSMNet
        self.model = PSMNet(args.maxdisp)

        self.model.to(args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        self.scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=2000, eta_min=5e-5)
        self.enable_amp = False
        self.scaler = amp.GradScaler(enabled = self.enable_amp)

        # load parameters
        self.start_epoch = 0
        if args.resume:
            print("loading the lastest self.model in logdir: {}".format(args.resume))
            self.state_dict = torch.load(args.resume)
            self.model.load_state_dict(self.state_dict['self.model'])
            self.optimizer.load_state_dict(self.state_dict['self.optimizer'])
            self.start_epoch = self.state_dict['epoch'] + 1
        elif args.loadckpt:
            # load the checkpoint file specified by args.loadckpt
            print("loading self.model {}".format(args.loadckpt))
            self.state_dict = torch.load(args.loadckpt)
            self.model.load_state_dict(self.state_dict['self.model'])
        print("start at epoch {}".format(self.start_epoch))


    def train(self):
        args = self.args
        min_dic={'min_EPE': args.maxdisp, 'min_D1': 1, 'min_Thres3': 1, 'minD1_epoch':0, 'minEPE_epoch':0}

        def save_test_block(min_dic):
            # saving checkpoints
            if (epoch_idx + 1) % args.save_freq == 0 and epoch_idx >= args.ckpt_start_epoch:
                checkpoint_data = {'epoch': epoch_idx, 'self.model': self.model.state_dict(), 'self.optimizer': self.optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(self.saver.experiment_dir, epoch_idx))

            if (epoch_idx + 1) % 1 == 0:
                checkpoint_data = {'epoch': epoch_idx, 'self.model': self.model.state_dict(), 'self.optimizer': self.optimizer.state_dict()}
                torch.save(checkpoint_data, f"{self.saver.experiment_dir}/latest_checkpoint.ckpt")
            gc.collect()

            # testing
            avg_test_scalars = AverageMeterDict()
            for batch_idx, sample in enumerate(self.TestImgLoader):
                start_time = time.time()
                do_summary = (self.global_step + batch_idx) % 21 == 0
                scalar_outputs, image_outputs = self.test_sample(sample, compute_metrics=do_summary)
                if do_summary:
                    save_scalars(self.logger, 'test', scalar_outputs, self.global_step + batch_idx)
                    save_images(self.logger, 'test', image_outputs, self.global_step + batch_idx)
                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs, image_outputs
                print('Epoch {}/{}, Iter {}/{}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                    batch_idx,
                                                                                    len(self.TestImgLoader), 
                                                                                    time.time() - start_time))
                with open(self.logfilename, 'a') as log:
                    log.write('Epoch {}/{}, Iter {}/{}, time = {:.3f}\n'.format(epoch_idx, args.epochs,
                                                                                                batch_idx,
                                                                                                len(self.TestImgLoader),
                                                                                    
                                                                                                time.time() - start_time))
            avg_test_scalars = avg_test_scalars.mean()
            if avg_test_scalars['EPE'][-1] < min_dic['min_EPE']:
                min_dic['min_EPE'] = avg_test_scalars['EPE'][-1]
                min_dic['minEPE_epoch'] = epoch_idx
                checkpoint_data = {'epoch': epoch_idx, 'self.model': self.model.state_dict(), 'self.optimizer': self.optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/bestEPE_checkpoint.ckpt".format(self.saver.experiment_dir))
            if avg_test_scalars['D1'][-1] < min_dic['min_D1']:
                min_dic['min_D1'] = avg_test_scalars['D1'][-1]
                min_dic['minD1_epoch'] = epoch_idx
                checkpoint_data = {'epoch': epoch_idx, 'self.model': self.model.state_dict(), 'self.optimizer': self.optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/bestD1_checkpoint.ckpt".format(self.saver.experiment_dir))
            if avg_test_scalars['Thres3'][-1] < min_dic['min_Thres3']:
                min_dic['min_Thres3'] = avg_test_scalars['Thres3'][-1]
                minThres3_epoch = epoch_idx
                checkpoint_data = {'epoch': epoch_idx, 'self.model': self.model.state_dict(), 'self.optimizer': self.optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/bestThres3_checkpoint.ckpt".format(self.saver.experiment_dir))
            save_scalars(self.logger, 'fulltest', avg_test_scalars, self.global_step)
            print("avg_test_scalars", avg_test_scalars)
            with open(self.logfilename, 'a') as log:
                js = json.dumps(avg_test_scalars)
                log.write(js)
                log.write('\n')
            gc.collect()

        for epoch_idx in range(self.start_epoch, args.epochs):
            self.args.epoch_idx = epoch_idx
            
            # training
            for batch_idx, sample in enumerate(self.TrainImgLoader):
                global_step = len(self.TrainImgLoader) * epoch_idx + batch_idx
                self.global_step = self.args.global_step = global_step
                start_time = time.time()
                do_summary = global_step % args.summary_freq == 0
                
                #### random flip with fake flip inputs
                self.model.train()
                if self.global_step>500:
                    if not hasattr(self, 'fake_flip'):
                        from losses.utils import Fake_Flip
                        self.fake_flip = Fake_Flip(device=self.args.device,args=self.args)
                    sample = self.fake_flip.generate_fake_flip(sample, self.model)
                
                loss, scalar_outputs, image_outputs = self.train_sample(sample, compute_metrics=do_summary)
                if do_summary:
                    save_scalars(self.logger, 'train', scalar_outputs, global_step)
                    save_images(self.logger, 'train', image_outputs, global_step)
                del scalar_outputs, image_outputs
                print('Epoch {}/{}, Iter {}/{}, train loss = {}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                        batch_idx,
                                                                                        len(self.TrainImgLoader), loss,
                                                                                        time.time() - start_time))
                with open(self.logfilename, 'a') as log:
                    log.write('Epoch {}/{}, Iter {}/{}, train loss = {}, time = {:.3f}\n'.format(epoch_idx, args.epochs,
                                                                                                    batch_idx,
                                                                                                    len(self.TrainImgLoader),
                                                                                                    loss,
                                                                                                    time.time() - start_time))
                if args.dataset=="kitti_raw" and (global_step + 1) % 501 == 0:
                    save_test_block(min_dic)

            save_test_block(min_dic)
            with open(self.logfilename, 'a') as log:
                log.write('min_EPE: {}/{}; min_D1: {}/{}'.format(min_dic['min_EPE'], min_dic['minEPE_epoch'], min_dic['min_D1'], min_dic['minD1_epoch']))


    # train one sample
    def train_sample(self, sample, compute_metrics=False):
        self.args.do_summary = compute_metrics
        args = self.args
        args.height = sample['left'].shape[2]
        args.width = sample['left'].shape[3]
        self.model.train()

        if ('disparity' not in sample): 
            disp_gt = None
            mask = True
        else:
            disp_gt = sample['disparity']
            disp_gt = disp_gt.to(args.device).unsqueeze(1)
            mask = ((disp_gt < args.maxdisp) & (disp_gt > 0)).cpu()
        imgL, imgR = sample['left'], sample['right']
        imgL = imgL.to(args.device)
        imgR = imgR.to(args.device)

        self.optimizer.zero_grad()

        with amp.autocast(enabled=self.enable_amp):
            outputs = self.model(imgL, imgR)
            
            prop_disp_pyramid = outputs
            outputs_h = {}

            for i, v in enumerate(prop_disp_pyramid):
                outputs_h[("disp", i)] = v

            gs = self.global_step
            
            ### loss weights
            smooth_weight = 0.5 * (gs/1e4)**2
            smooth_weight = np.clip(smooth_weight, 0.001, 0.5)
            args.loss_weights = [1, smooth_weight]    
            
            loss_fun = total_loss.Loss(args)
            outputs_h, losses = loss_fun(sample, outputs_h)
            loss = losses["loss"]
            
        with torch.no_grad():
            scalar_outputs = {"weighted_loss_sum": loss}
            img_warped = outputs_h["warped"]
            
            image_outputs = {key:img for key,img in outputs_h.items() if 'vis' in key}
            image_outputs.update({"disp_est": prop_disp_pyramid,  
                            "img_warped": img_warped, 
                            "imgL": imgL, "imgR": imgR
                            })

        if compute_metrics:
            with torch.no_grad():
                for i, v in losses.items():
                    scalar_outputs[i] = v
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


    # test one sample
    @make_nograd_func
    def test_sample(self, sample, compute_metrics=True):
        args = self.args
        self.model.eval()

        imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
        imgL = imgL.to(args.device)
        imgR = imgR.to(args.device)
        disp_gt = disp_gt.to(args.device).unsqueeze(1)

        outputs = self.model(imgL, imgR)
        if self.args.model == "PSMNet":
            prop_disp_pyramid = [outputs]
        else:
            prop_disp_pyramid = outputs["prop_disp_pyramid"]
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

        scalar_outputs = {}

        image_outputs = {"disp_est": prop_disp_pyramid, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

        scalar_outputs["D1"] = [D1_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1)) for disp_est in prop_disp_pyramid]
        scalar_outputs["EPE"] = [EPE_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1)) for disp_est in prop_disp_pyramid]
        scalar_outputs["Thres1"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 1.0) for disp_est in prop_disp_pyramid]
        scalar_outputs["Thres2"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 2.0) for disp_est in prop_disp_pyramid]
        scalar_outputs["Thres3"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 3.0) for disp_est in prop_disp_pyramid]

        if compute_metrics:
            image_outputs["errormap"] = [disp_error_image_func.apply(disp_est.squeeze(1), disp_gt.squeeze(1)) for disp_est in prop_disp_pyramid]

        return tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
