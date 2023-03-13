import os
from tqdm.auto import tqdm
from opt import config_parser


import json, random
from renderer import OctreeRender_trilinear_fast, OctreeRender_trilinear_fast_with_SR, evaluation, evaluation_sr, evaluation_path
from train_loop import train_loop_sr
from models.tensoRF import TensorVM, TensorCP, TensorVMSplit
from models.superresolution import SR_Module
from models.samplers import SimpleSampler, ImageSampler 
import torch.nn.functional as F
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

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
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
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

    # logfolder = os.path.dirname(args.ckpt)
    logfolder = f'{args.basedir}/{args.expname}'
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_profile:
        # os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation_profile(test_dataset, tensorf, args, renderer, N_vis=args.N_vis,
                           N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device)

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
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)

    renderer_sr = OctreeRender_trilinear_fast_with_SR
    sr_module = SR_Module(device=device, sr_ratio=args.sr_ratio)
    tensorf, sr_module = train_loop_sr(args, tensorf, sr_module, train_dataset, test_dataset, renderer_sr,
                                       nSamples, summary_writer, logfolder, device)
    # tensorf.save(f'{logfolder}/{args.expname}.th')

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation_sr(train_dataset, tensorf, sr_module, args, renderer_sr, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} all psnr on Train Dataset: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation_sr(test_dataset, tensorf, sr_module, args, renderer_sr, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=args.n_iters)
        print(f'======> {args.expname} all psnr on Test Dataset: {np.mean(PSNRs_test)} <========================')
    exit()

    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)
    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    # grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    # optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    train_rays, train_rgbs = train_dataset.all_rays, train_dataset.all_rgbs
    
    # further donwsample the training data if sr is used
    if args.sr_ratio > 1:
        ds_train_rays = interpolate_image_data(train_rays, float(1/args.sr_ratio))
        ds_train_rgbs = interpolate_image_data(train_rgbs, float(1/args.sr_ratio))

    allrays, allrgbs = ds_train_rays.view(-1, 6), ds_train_rgbs.view(-1,3)
    W, H = train_dataset.img_wh
    ds_W, ds_H = int(W/args.sr_ratio), int(H/args.sr_ratio)
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    randomSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    # Training NeRF
    renderer = OctreeRender_trilinear_fast
    # pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    # for iteration in pbar:
    #     ray_idx = randomSampler.nextids()
    #     rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

    #     #rgb_map, alphas_map, depth_map, weights, uncertainty
    #     rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, img_wh=(ds_W, ds_H), chunk=args.batch_size,
    #                             N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

    #     loss = torch.mean((rgb_map - rgb_train) ** 2)

    #     total_loss = loss
    #     if Ortho_reg_weight > 0:
    #         loss_reg = tensorf.vector_comp_diffs()
    #         total_loss += Ortho_reg_weight*loss_reg
    #         summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
    #     if L1_reg_weight > 0:
    #         loss_reg_L1 = tensorf.density_L1()
    #         total_loss += L1_reg_weight*loss_reg_L1
    #         summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

    #     if TV_weight_density>0:
    #         TV_weight_density *= lr_factor
    #         loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
    #         total_loss = total_loss + loss_tv
    #         summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
    #     if TV_weight_app>0:
    #         TV_weight_app *= lr_factor
    #         loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
    #         total_loss = total_loss + loss_tv
    #         summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

    #     optimizer.zero_grad()
    #     total_loss.backward()
    #     optimizer.step()

    #     loss = loss.detach().item()
    #     
    #     PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
    #     summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
    #     summary_writer.add_scalar('train/mse', loss, global_step=iteration)

    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = param_group['lr'] * lr_factor

    #     # Print the current values of the losses.
    #     if iteration % args.progress_refresh_rate == 0:
    #         pbar.set_description(
    #             f'Iteration {iteration:05d}:'
    #             + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
    #             + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
    #             + f' mse = {loss:.6f}'
    #         )
    #         PSNRs = []


    #     if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
    #         PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
    #                                 prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
    #         summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

    # # if args.render_train:
    # #     os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
    # #     # train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    # #     PSNRs_test = evaluation(train_dataset, tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
    # #                             N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
    # #     print(f'======> {args.expname} all psnr on Train Dataset: {np.mean(PSNRs_test)} <========================')

    # if args.render_test:
    #     os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
    #     PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
    #                             N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
    #     summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
    #     print(f'======> {args.expname} all psnr on Test Dataset: {np.mean(PSNRs_test)} <========================')

    # Training NeRF + SR
    sr_module = SR_Module(device=device, sr_ratio=args.sr_ratio)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    grad_vars += [{'params':sr_module.mapping.parameters(), 'lr':0.01}]
    grad_vars += [{'params':sr_module.sr.parameters(), 'lr':0.01}]

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app

    renderer_sr = OctreeRender_trilinear_fast_with_SR
    imageSampler = ImageSampler(train_dataset.poses.shape[0], batch=1)
    data_intrinsics = train_dataset.intrinsics.to(device)

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        img_idx = imageSampler.nextids()
        rgbs = train_rgbs[img_idx].to(device).reshape(-1,3)
        cam2world_pose = train_dataset.poses[img_idx].to(device)
        cam_params = torch.cat([cam2world_pose.reshape(-1, 16), data_intrinsics.reshape(-1, 9)], 1)

        ds_rays, ds_rgbs = ds_train_rays[img_idx], ds_train_rgbs[img_idx].to(device)
        ds_rays, ds_rgbs = ds_rays.reshape(-1, 6), ds_rgbs.reshape(-1,3)

        # rgb_map, alphas_map, depth_map, weights, uncertainty
        sr_map, alphas_map, depth_map, weights, uncertainty = renderer_sr(ds_rays, cam_params, tensorf, sr_module, img_wh=(ds_W, ds_H), chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        sr_loss = torch.mean((sr_map - rgbs) ** 2)

        # loss
        total_loss = sr_loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = total_loss
        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation_sr(test_dataset, tensorf, sr_module, args, renderer_sr, f'{logfolder}/imgs_vis_sr/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
        
    tensorf.save(f'{logfolder}/{args.expname}.th')

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation_sr(train_dataset, tensorf, sr_module, args, renderer_sr, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} all psnr on Train Dataset: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation_sr(test_dataset, tensorf, sr_module, args, renderer_sr, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} all psnr on Test Dataset: {np.mean(PSNRs_test)} <========================')

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

