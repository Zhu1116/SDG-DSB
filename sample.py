import os
import copy
import argparse
import random
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np
import scipy.io as sio

import torch
from torch.multiprocessing import Process
from torch_ema import ExponentialMovingAverage

from logger import Logger
import distributed_util as dist_util
from i2sb import Runner, download_ckpt, ckpt_util
from corruption import build_corruption
from utils.evaluation import MetricsCal
from i2sb.estR import estR
from i2sb.unmixing import unmix

RESULT_DIR = Path("results")


def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def get_recon_imgs_fn(opt, nfe):
    if opt.use_cddb_deep:
        suffix = f"samples_cddb_deep_nfe{nfe}_step{opt.step_size}"
        sample_dir = RESULT_DIR / opt.ckpt / suffix
    elif opt.use_cddb:
        suffix = f"samples_cddb_nfe{nfe}_step{opt.step_size}"
        sample_dir = RESULT_DIR / opt.ckpt / suffix
    else:
        suffix = f"samples_nfe{nfe}_clip_1k"
        sample_dir = RESULT_DIR / opt.ckpt / suffix
    os.makedirs(sample_dir, exist_ok=True)

    recon_imgs_fn = sample_dir / "recon{}.pt".format(
        "" if opt.partition is None else f"_{opt.partition}"
    )
    return recon_imgs_fn, sample_dir

# @torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval-1

    # build runner
    runner = Runner(ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to()  # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99)  # re-init ema with fp16 weight

    # create save folder
    recon_imgs_fn, sample_dir = get_recon_imgs_fn(opt, nfe)

    log.info(f"Recon images will be saved to {sample_dir}!")
    log_count = 10

    # build corruption  method
    corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type)

    # load data
    data_type = 'pavia'
    data = sio.loadmat(os.path.join(opt.dataset_dir, data_type, data_type + '.mat'))
    lr_hsi = (data['LRHSI'] * 2 - 1).astype(np.float32)
    hr_msi = (data['HRMSI'] * 2 - 1).astype(np.float32)
    h, w, C = lr_hsi.shape

    unmix_file_path = os.path.join(opt.dataset_dir, data_type, data_type + '_unmixing_p' + str(opt.rank) + '.mat')
    if not os.path.exists(unmix_file_path):
        print("start unmixing ..........")
        E_hat, A_hat = unmix(lr_hsi, p=opt.rank)
        print("finish unmixing ..........")
        sio.savemat(unmix_file_path, {"E_hat": E_hat, "A_hat": A_hat})
    else:
        unmix_data = sio.loadmat(unmix_file_path)
        E_hat = unmix_data['E_hat']
        A_hat = unmix_data['A_hat']

    A_hat = A_hat.reshape(A_hat.shape[0], h, w)

    corrupt_img_y = torch.from_numpy(A_hat).unsqueeze(0).float().to(opt.device)
    upsample = torch.nn.Upsample(scale_factor=4, mode='bicubic')
    corrupt_img = upsample(corrupt_img_y)
    x1 = corrupt_img
    x1_pinv = corrupt_img
    x1_forw = {}
    x1_forw['E_hat'] = E_hat
    x1_forw['MSI'] = hr_msi
    mask = None
    cond = None

    # est srf
    srf_path = os.path.join(opt.dataset_dir, data_type, 'R.mat')
    if not os.path.exists(srf_path):
        print("estimate srf ..........")
        est = estR(lr_hsi, hr_msi, device=opt.device)
        srf = est.start_est()
        print("finish estimate srf ..........")
        sio.savemat(srf_path, {'R': srf})
    else:
        srf = sio.loadmat(srf_path)['R']
    x1_forw['srf'] = srf

    if opt.use_cddb_deep:
        xs, pred_x0s = runner.cddb_deep_sampling(
            ckpt_opt, x1, x1_pinv, x1_forw, mask=mask, corrupt_type=corrupt_type,
            corrupt_method=corrupt_method, cond=cond, clip_denoise=opt.clip_denoise,
            nfe=nfe, verbose=opt.n_gpu_per_node==1, log_count=log_count, step_size=opt.step_size,
            results_dir=sample_dir
        )
    elif opt.use_cddb:
        xs, pred_x0s = runner.cddb_sampling(
            ckpt_opt, x1, x1_pinv, x1_forw, mask=mask, corrupt_type=corrupt_type,
            corrupt_method=corrupt_method, cond=cond, clip_denoise=opt.clip_denoise,
            nfe=nfe, verbose=opt.n_gpu_per_node==1, log_count=log_count, step_size=opt.step_size,
            results_dir=sample_dir,
        )
    else:
        xs, pred_x0s = runner.ddpm_sampling(
            ckpt_opt, x1, x1_pinv, x1_forw, mask=mask, cond=cond, clip_denoise=opt.clip_denoise,
            nfe=nfe, verbose=opt.n_gpu_per_node==1, log_count=log_count
        )
    recon_img = xs[:, 0, ...]

    assert recon_img.shape == corrupt_img.shape

    recon_hsi = torch.from_numpy(E_hat).float() @ recon_img.reshape(recon_img.shape[1], -1)
    recon_hsi = recon_hsi.reshape(recon_hsi.shape[0], recon_img.shape[2], recon_img.shape[3]).numpy()
    recon_hsi = (np.transpose(recon_hsi, (1, 2, 0)) + 1) / 2

    rmse, psnr, sam, ergas, ssim, uiqi = MetricsCal(data['HRHSI'].astype(np.float32), recon_hsi, 4)
    print('rmse: ', rmse)
    print('psnr: ', psnr)
    print('sam:  ', sam)
    print('ergas:', ergas)
    print('ssim: ', ssim)
    print('uiqi: ', uiqi)

    sio.savemat(os.path.join(sample_dir, 'recon_info.mat'),
                {
                    'rmse': rmse,
                    'psnr': psnr,
                    'sam': sam,
                    'ergas': ergas,
                    'ssim': ssim,
                    'uiqi': uiqi,
                    'HRHSI': data['HRHSI'].astype(np.float32),
                    'recon': recon_hsi,
                })

    del runner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size",     type=int,  default=256)
    parser.add_argument("--dataset-dir",    type=Path, default="dataset",  help="path to LMDB dataset")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")
    parser.add_argument("--add-noise",      action="store_true",            help="If true, add small gaussian noise to y")

    # sample
    parser.add_argument("--batch-size",     type=int,  default=1)
    parser.add_argument("--ckpt",           type=str,  default='sr4x-bicubic')
    parser.add_argument("--nfe",            type=int,  default=100,         help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")
    parser.add_argument("--eta",            type=float, default=1.0,        help="ddim stochasticity. 1.0 recovers ddpm")
    parser.add_argument("--use-cddb-deep",  action="store_true",            help="use cddb-deep")
    parser.add_argument("--use-cddb",       action="store_true",            help="use cddb")
    parser.add_argument("--step-size",      type=float, default=1.0,        help="step size for gradient descent")
    parser.add_argument("--prob_mask",      type=float, default=0.35,       help="probability of masking")
    parser.add_argument("--rank",           type=int,   default=6,          help="rank")


    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    # one-time download: ADM checkpoint
    download_ckpt("data/")

    set_seed(opt.seed)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        # dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
        main(opt)
