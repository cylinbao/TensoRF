import os
from tqdm.auto import tqdm
from opt import config_parser

import json, random
from eval import evaluation, evaluation_sr, evaluation_path
from train_loop import train_loop, train_loop_sr
from models.tensoRF import TensorVM, TensorCP, TensorVMSplit
from models.superresolution import SR_Module, Interpolation
from models.samplers import SimpleSampler, ImageSampler 
import torch.nn.functional as F
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from super_image import EdsrModel

from dataLoader import dataset_dict
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_test, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    eval_func = evaluation
    # logfolder = os.path.dirname(args.ckpt)
    logfolder = f'{args.basedir}/{args.expname}'
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_train = eval_func(train_dataset,tensorf, args, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_train)} <========================')

    eval_func = evaluation_edsr
    args.sr_ratio = 4
    tensorf.sr_module = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=args.sr_ratio)
    # tensorf.sr_module = Interpolation(sr_ratio=args.sr_ratio).to(device)

    tensorf.save(f'{logfolder}/{args.expname}.th')
    breakpoint()

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        PSNRs_test= eval_func(test_dataset,tensorf, args, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    # if args.render_path:
    #     c2ws = test_dataset.render_path
    #     os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
    #     evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
    #                             N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    # if args.render_profile:
    #     # os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
    #     evaluation_profile(test_dataset, tensorf, args, renderer, N_vis=args.N_vis,
    #                        N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device)

def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct,
                    use_sr=args.use_sr, sr_method=args.sr_method, sr_ratio=args.sr_ratio)

    # tensorf.sr_module = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=args.sr_ratio)
    # tensorf.sr_module = SR_Module(device=device, sr_ratio=args.sr_ratio)

    if args.use_sr == False:
        tensorf = train_loop(args, tensorf, train_dataset, test_dataset,
                             nSamples, summary_writer, logfolder, device)
    else:
        tensorf = train_loop_sr(args, tensorf, train_dataset, test_dataset,
                                nSamples, summary_writer, logfolder, device)

    # if args.render_train:
    #     os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
    #     train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    #     PSNRs_test = eval_func(train_dataset, tensorf, args, f'{logfolder}/imgs_train_all/',
    #                             N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
    #     print(f'======> {args.expname} all psnr on Train Dataset: {np.mean(PSNRs_test)} <========================')

    # if args.render_test:
    #     os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
    #     PSNRs_test = eval_func(test_dataset, tensorf, args, f'{logfolder}/imgs_test_all/',
    #                             N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
    #     summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=args.n_iters)
    #     print(f'======> {args.expname} all psnr on Test Dataset: {np.mean(PSNRs_test)} <========================')

    tensorf.save(f'{logfolder}/{args.expname}.th')
    exit()

    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_test, is_stack=True)
    eval_func = evaluation_sr
    args.sr_ratio = 2
    # tensorf.sr_module = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
    tensorf.sr_module = Interpolation(sr_ratio=args.sr_ratio).to(device)

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = eval_func(test_dataset, tensorf, args, f'{logfolder}/imgs_test_all_edsr/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=args.n_iters)
        print(f'======> {args.expname} all psnr on Test Dataset: {np.mean(PSNRs_test)} <========================')

    tensorf.save(f'{logfolder}/{args.expname}.th')

    # if args.render_path:
    #     c2ws = test_dataset.render_path
    #     # c2ws = test_dataset.poses
    #     print('========>',c2ws.shape)
    #     os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
    #     evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
    #                             N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path or args.render_profile):
        render_test(args)
    else:
        reconstruction(args)

