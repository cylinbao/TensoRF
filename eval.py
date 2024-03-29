import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
from models.render_modules import positional_encoding
import torch.nn.functional as F
from super_image import EdsrModel
from renderer import OctreeRender_trilinear_fast, OctreeRender_trilinear_fast_SR


@torch.no_grad()
def evaluation(test_dataset, tensorf, args, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    renderer = OctreeRender_trilinear_fast

    W, H = test_dataset.img_wh
    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx in tqdm(idxs, file=sys.stdout):
        rays = test_dataset.all_rays[[idx]].to(device).reshape(-1,6)

        rgb_map, depth_map = renderer(rays, tensorf, chunk=args.batch_size, N_samples=N_samples,
                                               ndc_ray=ndc_ray, white_bg = white_bg, device=device)

        rgb_map = rgb_map.clamp(0,1)
        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[[idx]].reshape(H, W, 3).cpu()

            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            rgb_map = np.concatenate((rgb_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs


@torch.no_grad()
def evaluation_sr(test_dataset, tensorf, args, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, sr_maps = [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    renderer = OctreeRender_trilinear_fast_SR

    data_intrinsics = test_dataset.intrinsics.to(device)
    W, H = test_dataset.img_wh
    ds_W, ds_H = int(W/args.sr_ratio), int(H/args.sr_ratio)
    # near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx in tqdm(idxs, file=sys.stdout):
        rays = test_dataset.all_rays[[idx]]
        rays = interpolate_image_data(rays, float(1/args.sr_ratio)).reshape(-1, 6)

        cam2world_pose = test_dataset.poses[idx].to(device)
        cam_params = torch.cat([cam2world_pose.reshape(-1, 16), data_intrinsics.reshape(-1, 9)], 1)

        sr_map, depth_map = renderer(rays, cam_params, tensorf, (ds_W, ds_H), chunk=args.batch_size, N_samples=N_samples,
                                               ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        sr_map = sr_map.clamp(0,1)
        sr_map = sr_map.reshape(H, W, 3).cpu()

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[[idx]].view(H, W, 3).cpu()
            loss = torch.mean((sr_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(sr_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), sr_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), sr_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        sr_map = (sr_map.cpu().numpy() * 255).astype('uint8')
        sr_maps.append(sr_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', sr_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(sr_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs


@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs


@torch.no_grad()
def evaluation_profile(test_dataset, tensorf, args, renderer, N_vis=5, N_samples=-1,
                       white_bg=False, ndc_ray=False, device='cuda'):
    try:
        tqdm._instances.clear()
    except Exception:
        pass

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    prof_time = []

    print("Runing warm up iterations...")
    rays = test_dataset.all_rays[0]
    for _ in range(5):
        renderer(rays, tensorf, chunk=args.batch_size, N_samples=N_samples,
                 ndc_ray=ndc_ray, white_bg = white_bg, device=device)

    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        start.record()
        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=args.batch_size, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        end.record()
        torch.cuda.synchronize()
        prof_time.append(start.elapsed_time(end))
        print(f"Render time for test image {idx}: {prof_time[-1]}")

    mean_t = np.mean(prof_time)
    print(f"Mean render time for each test image: {mean_t}")

    return
