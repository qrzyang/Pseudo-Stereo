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
import imageio
from models.PSMNet import PSMNet
cudnn.benchmark = True

class Tester:
    def __init__(self) -> None:
        def get_args():
            parser = argparse.ArgumentParser()


            parser.add_argument('--maxdisp', type=int, default=192,
                                help='maximum disparity')
            parser.add_argument('--fea_c', type=list,
                                default=[32, 24, 24, 16, 16], help='feature extraction channels')

            parser.add_argument('--dataset', default="kitti",
                                help='dataset name', choices=__datasets__.keys())
            parser.add_argument('--datapath', help='data path', default="/mnt/win/DataSet/")

            parser.add_argument('--testlist',
                                help='testing list', default="filenames/kt15_testing.txt")


            parser.add_argument('--lr', type=float, default=0.,
                                help='base learning rate')
            parser.add_argument('--batch_size', type=int, default=1,
                                help='training batch size')
            parser.add_argument('--test_batch_size', type=int,
                                default=1, help='testing batch size')


            parser.add_argument('--logdir', default="./logs/test",
                                help='the directory to save logs and checkpoints')
            parser.add_argument(
                '--loadckpt', help='load the weights from a specific checkpoint')
            parser.add_argument('--resume', type=str, 
                help='read the model')
            parser.add_argument('--seed', type=int, default=1,
                                metavar='S', help='random seed (default: 1)')

            parser.add_argument('--summary_freq', type=int, default=20,
                                help='the frequency of saving summary')


            parser.add_argument('--save_disp_to_file', action="store_true",
                                help='save estimated disp maps to files')
            parser.add_argument('--message', type=str, default="")
            return parser.parse_args()

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
        self.StereoDataset = __datasets__[args.dataset]

        self.test_dataset = self.StereoDataset(args.datapath, args.testlist, False)

        self.TestImgLoader = DataLoader(self.test_dataset, args.test_batch_size, shuffle=False, 
                                        num_workers=8, drop_last=False, pin_memory=True)

        # self.model, self.optimizer
        self.model = PSMNet(args.maxdisp)
        self.model.to(args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))

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
        
        now = datetime.now()
        self.formatted_datetime = now.strftime("%Y-%m-%d-%H-%M-%S")


    def run(self):
        args = self.args

        
        epoch_idx = self.start_epoch
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(self.TestImgLoader):

            global_step = len(self.TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = (global_step) % args.summary_freq== 0
            scalar_outputs, image_outputs = self.test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(self.logger, 'test', scalar_outputs, global_step)
                save_images(self.logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}, Iter {}/{}, time = {:3f}'.format(epoch_idx,batch_idx,
                                                                                len(self.TestImgLoader), 
                                                                                time.time() - start_time))
            with open(self.logfilename, 'a') as log:
                log.write('Epoch {}, Iter {}/{}, time = {:.3f}\n'.format(epoch_idx,batch_idx,
                                                                                            len(self.TestImgLoader),
                                                                                            time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()

        print("avg_test_scalars", avg_test_scalars)
        with open(self.logfilename, 'a') as log:
            js = json.dumps(avg_test_scalars)
            log.write(js)
            log.write('\n')
        gc.collect()

    # test one sample
    @make_nograd_func
    def test_sample(self, sample, compute_metrics=True):
        args = self.args
        args.height = sample['left'].shape[2]
        args.width = sample['left'].shape[3]
        self.model.eval()

        if ('disparity' not in sample): 
            disp_gt = None
            mask = True
        else:
            disp_gt = sample['disparity']
            disp_gt = disp_gt.to(args.device)
            mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        imgL, imgR = sample['left'], sample['right']
        
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        if disp_gt is not None:
            disp_gt = disp_gt.cuda().unsqueeze(1)


        prop_disp_pyramid = self.model(imgL, imgR)

        scalar_outputs = {}

        image_outputs = {"disp_est": prop_disp_pyramid, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR,}
        if disp_gt is not None:
            scalar_outputs["D1"] = [D1_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1)) for disp_est in prop_disp_pyramid]
            scalar_outputs["EPE"] = [EPE_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1)) for disp_est in prop_disp_pyramid]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 1.0) for disp_est in prop_disp_pyramid]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 2.0) for disp_est in prop_disp_pyramid]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 3.0) for disp_est in prop_disp_pyramid]
        
        if args.save_disp_to_file:
            self.save_disp_tofile(prop_disp_pyramid, sample)

        if compute_metrics & (disp_gt is not None):
            image_outputs["errormap"] = [disp_error_image_func.apply(disp_est.squeeze(1), disp_gt.squeeze(1)) for disp_est in prop_disp_pyramid]

        return tensor2float(scalar_outputs), image_outputs
    
    @make_nograd_func
    def save_disp_tofile(self, disp, sample):
        filename = sample["left_filename"][0].split('/')[-1]
        path = self.args.datapath

        full_path = os.path.join(path,f'kitti_2015/testing/{self.formatted_datetime}/disp_0',filename)
        top = sample["top_pad"]
        right = sample["right_pad"]
        disp_np=np.array(disp[:,:,top:,:-right].squeeze().detach().cpu()*256).astype(np.uint16)
        os.makedirs(os.path.dirname(full_path), exist_ok=True) 
        imageio.imsave(full_path,disp_np)

if __name__ == '__main__':
    tester = Tester()
    tester.run()
