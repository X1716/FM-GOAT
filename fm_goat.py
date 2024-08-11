import os, sys
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
transform = transforms.Compose([
        transforms.ToTensor()
    ]) 
import torch
import tensorly as tl
from pytorch_wavelets import DWTForward, DWTInverse
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure
from lpips import LPIPS
from copy import deepcopy
DEVICE = 'cuda:0'
model_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
model_lpips = LPIPS().to(DEVICE)
model_mse = torch.nn.MSELoss().to(DEVICE)

def fm_goat_attack(im_tensor, model, ssim_model=model_ssim, lpips_model=model_lpips, mse_model=model_mse, device='cuda:0', verbose=False, iters=10, LPIPS_THRESH=0.00, SSIM_THRESH=0.00, lr=1e-2, alpha=1,
                      wavetype='db2', wavemod='reflect', J=4, metric_range=100, dwt_scale_factor=None):
    '''
    args: 
        im_tensor: torch.tensor() in rgb [0,1] range with shape 1x3xHxW - image to attack 
        model: torch model - target NR-IQA model 
        metric_range: float - range of model's values 
        device: torch.device/str 
        iters: number of attack iterations 
        alpha: float - attack strength parameter, learning rate analog 
        LPIPS_THRESH, SSIM_THRESH: float - parameters for optional bootstrap stage, value <= 0 in one of these parameters turns off bootstrap (corresponds to attack algorithm described in paper) 
        lr: float - parameter for optional bootstrap stage, controls learning rate for optimizer during bootstrap
    wavetype, wavemod, J, dwt_scale_factor: str, str, int, float - additional parameters for DWT (high-frequency masking stage). dwt_scale_factor, if != None, allows to downscale image before mask computation (more efficient for high resolution) 
        ssim_model, lpips_model, mse_model: torch model - models for computing correponding FR metrics. Models should support gradient propagation. 
    returns:
        torch.tensor of shape 1x3xHxW - attacked image in rgb [0,1] range
    '''
    if len(im_tensor.shape) > 3 and im_tensor.shape[0] == 1:
        im_tensor = im_tensor.squeeze()

    # Create HF mask
    image_wl = im_tensor.clone().unsqueeze(0).to(device)
    if dwt_scale_factor is not None:
        image_wl = torch.nn.functional.interpolate(image_wl, size=(image_wl.shape[2]//dwt_scale_factor, image_wl.shape[3]//dwt_scale_factor), mode='bilinear')
    xfm = DWTForward(J=J, mode=wavemod, wave=wavetype).to(device)
    Yl, Yh = xfm(image_wl)
    res = transforms.Resize([im_tensor.shape[1], im_tensor.shape[2]])
    blur = transforms.GaussianBlur(5,1.0)
    mask = Yh[1].squeeze().sum(dim=0).sum(dim=0).detach().abs()
    mask2 = Yh[0].squeeze().sum(dim=0).sum(dim=0).detach().abs()
    mask = 2 * mask / mask.max()
    mask2 = 2 * mask2 / mask2.max()
    mask = torch.from_numpy(mask.clamp(0,1).cpu().numpy())
    mask2 = torch.from_numpy(mask2.clamp(0,1).cpu().numpy())
    resized_mask = blur(res(mask.unsqueeze(0))).squeeze().to(device) + blur(res(mask2.unsqueeze(0))).squeeze().to(device)
    resized_mask = torch.clamp(resized_mask, 0, 1)

    optimized_im_tensor = im_tensor.clone().to(device)
    add_part = torch.zeros_like(optimized_im_tensor).to(device)
    add_part = torch.autograd.Variable(add_part).to(device)
    add_part.requires_grad = True
    optim_full_image = torch.optim.AdamW([add_part], lr=lr)
    i = 0
    # Additional optional step: bootstrap the attack with few iterations of direct optimization, w/o orthogonalization.
    # Bootstrap stage ends, if visual losses (LPIPS and SSIM) exceed certain limit (LPIPS_THRESH and SSIM_THRESH):
    # then the last iteration is rolled back and orthogonalization attack starts
    if LPIPS_THRESH > 0.0 and SSIM_THRESH > 0.0:
        for i in range(iters):
            optim_full_image.zero_grad()
            saved_opt_state = deepcopy(optim_full_image.state_dict())
            prev_add_state = add_part.data.clone()
            tmp = optimized_im_tensor + resized_mask * add_part
            tmp = torch.clamp(tmp, 0, 1)
            val = 1.0 - model(torch.unsqueeze(tmp, 0).to('cuda:0')) / metric_range
            val.backward()
            optim_full_image.step()
            with torch.no_grad():
                tmp = optimized_im_tensor + resized_mask * add_part
                tmp = torch.clamp(tmp, 0, 1)
                ssim_loss_val = 1.0 - ssim_model(torch.unsqueeze(tmp, 0).to('cuda:0'), torch.unsqueeze(im_tensor, 0).to('cuda:0')).item()
                lpips_loss_val = lpips_model(torch.unsqueeze(tmp, 0).to('cuda:0'), torch.unsqueeze(im_tensor, 0).to('cuda:0')).item()
            if lpips_loss_val > LPIPS_THRESH or ssim_loss_val > SSIM_THRESH:
                add_part.data = prev_add_state.clone()
                optim_full_image.load_state_dict(saved_opt_state)
                #print(f'break on {i} iter')
                break

    # Orthogonal attack
    prev_iter_num = i
    def proj(b,a, eps=1e-10):
        # check if b == zero vector
        if (b * b).sum() <= eps:
            return b
        return b * (b * a).sum() / (b * b).sum()
    
    for i in range(iters - prev_iter_num):
        gradients = {}
        if add_part.grad is not None:
            add_part.grad.fill_(0)
        tmp = optimized_im_tensor + resized_mask * add_part
        tmp = torch.clamp(tmp, 0, 1)
        val = 1.0 - model(torch.unsqueeze(tmp, 0).to('cuda:0'))
        val.backward()
        gradients['metric'] = add_part.grad.data.clone()

        add_part.grad.fill_(0)
        tmp = optimized_im_tensor + resized_mask * add_part
        tmp = torch.clamp(tmp, 0, 1)
        ssim_loss = 1.0 - ssim_model(torch.unsqueeze(tmp, 0).to('cuda:0'), torch.unsqueeze(im_tensor, 0).to('cuda:0'))
        ssim_loss.backward()
        gradients['SSIM'] = add_part.grad.data.clone()

        add_part.grad.fill_(0)
        tmp = optimized_im_tensor + resized_mask * add_part
        tmp = torch.clamp(tmp, 0, 1)
        lpips_loss = lpips_model(torch.unsqueeze(tmp, 0).to('cuda:0'), torch.unsqueeze(im_tensor, 0).to('cuda:0'))
        lpips_loss.backward()
        gradients['LPIPS'] = add_part.grad.data.clone()

        add_part.grad.fill_(0)
        tmp = optimized_im_tensor + resized_mask * add_part
        tmp = torch.clamp(tmp, 0, 1)
        mse_loss = mse_model(torch.unsqueeze(tmp, 0).to('cuda:0'), torch.unsqueeze(im_tensor, 0).to('cuda:0'))
        mse_loss.backward()
        gradients['MSE'] = add_part.grad.data.clone()
        add_part.grad.fill_(0)
        if verbose:
            print('Metric: ', -(val.item() - 1.0))
            print('SSIM: ',ssim_loss.item())
            print('LPIPS: ',lpips_loss.item())
            print('MSE: ',mse_loss.item())

        normals = []
        order = ['LPIPS', 'SSIM', 'MSE', 'metric']
        for metric in order:
            cur_normal = gradients[metric].clone()
            for prev_norm in normals:
                cur_normal -= proj(prev_norm, gradients[metric])
            normals.append(cur_normal)

        normal_prj = normals[-1]
        add_part.data -= normal_prj * (alpha / metric_range)

    result = optimized_im_tensor + resized_mask * add_part
    result = torch.clamp(result, 0, 1)

    if verbose:
        with torch.no_grad():
            print('Metric: ',model(torch.unsqueeze(result, 0).to('cuda:0')))
            print('SSIM: ', ssim_model(torch.unsqueeze(result, 0).to('cuda:0'), torch.unsqueeze(im_tensor, 0).to('cuda:0')))
            print('LPIPS: ', lpips_model(torch.unsqueeze(result, 0).to('cuda:0'), torch.unsqueeze(im_tensor, 0).to('cuda:0')))

    return result.unsqueeze(0)


