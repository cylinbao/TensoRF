import torch
from tqdm.auto import tqdm
import numpy as np
from utils import interpolate_image_data, TVLoss
from renderer import OctreeRender_trilinear_fast, OctreeRender_trilinear_fast_with_SR
from renderer import evaluation, evaluation_sr
from models.samplers import SimpleSampler, ImageSampler 
import sys
from torch.utils.tensorboard import SummaryWriter

def train_loop_sr(args, tensorf, sr_module, train_dataset, test_dataset, renderer_sr, 
                  nSamples, summary_writer, logfolder, device):
    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    train_rays, train_rgbs = train_dataset.all_rays, train_dataset.all_rgbs
    
    # further donwsample the training data for sr setting 
    ds_train_rays = interpolate_image_data(train_rays, float(1/args.sr_ratio))
    ds_train_rgbs = interpolate_image_data(train_rgbs, float(1/args.sr_ratio))

    W, H = train_dataset.img_wh
    ds_W, ds_H = int(W/args.sr_ratio), int(H/args.sr_ratio)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    grad_vars += [{'params':sr_module.mapping.parameters(), 'lr':0.01}]
    grad_vars += [{'params':sr_module.sr.parameters(), 'lr':0.01}]

    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app

    imageSampler = ImageSampler(train_dataset.poses.shape[0], batch=1)
    data_intrinsics = train_dataset.intrinsics.to(device)

    white_bg = train_dataset.white_bg

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
                                N_samples=nSamples, white_bg=white_bg, ndc_ray=args.ndc_ray, device=device, is_train=True)

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

        # update learning rate
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
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg=white_bg, ndc_ray=args.ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

    return tensorf, sr_module