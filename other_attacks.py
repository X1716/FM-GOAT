import numpy as np
import torchvision.transforms as transforms
import torch
import cv2
from scipy import ndimage
from torchvision import transforms
from IQA_pytorch import DISTS
dists_model_zhang = DISTS()

transform = transforms.Compose([
        transforms.ToTensor()
    ]) 
def ifgsm_attack(im_tensor, ref_image=None, model=None, metric_range=100, device='cuda:0', iters=10, eps=5./255, alpha=10./255, return_diff=False,
                 ssim_model=None,lpips_model=None, mse_model=None, NUM_ITER=10, dev='cuda:0'):
    return_np = False
    if type(im_tensor) == np.ndarray:
        im_tensor = transform(im_tensor).unsqueeze(0)
        return_np = True
    if im_tensor.shape[0] != 1:
        im_tensor = im_tensor.unsqueeze(0)
    #before_image = compress_image.clone().to(device)
    im_tensor = torch.autograd.Variable(im_tensor.clone().to(device), requires_grad=True)
    
    p = torch.zeros_like(im_tensor).to(device)
    p = torch.autograd.Variable(p, requires_grad=True)
    #sign = -1 if model.lower_better else 1
    sign = 1
    for i in range(iters):
        res = im_tensor + p
        res.data.clamp_(0., 1.)
        #score = model(ref_image.to(device), res.to(device)) if ref_image is not None else model(res.to(device))
        score = model(res.to(device))
        loss = 1 - score * sign / metric_range
        loss.backward() 
        g = p.grad
        #print(g)
        g = torch.sign(g)
        #print(g)
        p.data -= alpha * g
        p.data.clamp_(-eps, +eps)
        #grad_to_print = g.detach().cpu().numpy()
        #print((grad_to_print == 0).sum(), '/ ', grad_to_print.reshape(-1).shape[0] )
        p.grad.zero_()
        im_tensor.grad.zero_()
    res_image = im_tensor + p

    res_image = (res_image).data.clamp_(min=0, max=1)
    res_image = res_image.detach()
    
    #diff = res_image - before_image

    res_image = res_image
    #diff = diff.squeeze()
    #res_image = res_image.squeeze()
    if return_np:
        res_image = np.clip(res_image.permute(1,2,0).detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        if return_diff:
            diff = diff.permute(1,2,0).detach().cpu().numpy() * 255.0
    
    if return_diff:
        return res_image, diff
    return res_image


def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0
    return im_ycbcr

def makeSpatialActivityMap(im):
  im = im.cpu().detach().permute(0, 2, 3, 1).numpy()[0]
  #H = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8  
  im = rgb2ycbcr(im)
  im_sob = np.sqrt(ndimage.sobel(im[:,:,0], 0)**2 + ndimage.sobel(im[:,:,0], 1)**2 )
  im_zero = np.zeros_like(im_sob)
  im_zero[1:-1, 1:-1] = im_sob[1:-1, 1:-1]

  maxval = im_zero.max()

  if maxval == 0:
    im_zero = im_zero + 1
    maxval = 1
  
  im_sob = im_zero /maxval

  DF = np.array([[0, 1, 1, 1, 0], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [0, 1, 1, 1, 0]]).astype('uint8')
  
  out_im = cv2.dilate(im_sob, DF)
  return out_im
          

def korhonen_attack(im_tensor, ref_image=None, model=None, metric_range=100, device='cuda:0', iters = 10, lr = 0.005, return_diff=False, 
                    ssim_model=None,lpips_model=None, mse_model=None):
    #print(lr)
    return_np = False
    if type(im_tensor) == np.ndarray:
        im_tensor = transform(im_tensor).unsqueeze(0)
        # print(compress_image.shape)
        return_np = True
    if im_tensor.shape[0] != 1:
        im_tensor = im_tensor.unsqueeze(0)
        
    im_tensor = im_tensor.to(device)
    before_image = im_tensor.clone().to(device)


    sp_map = makeSpatialActivityMap(im_tensor * 255)
    sp_map = sp_map / 255
    sp_map = transforms.ToTensor()(sp_map.astype(np.float32))
    sp_map = sp_map.unsqueeze_(0)
    sp_map = sp_map.to(device)

    
    im_tensor = torch.autograd.Variable(im_tensor, requires_grad=True)
    opt = torch.optim.Adam([im_tensor], lr = lr)
    
    #sign = -1 if model.lower_better else 1
    sign = 1
    for i in range(iters):
        score = model(ref_image.to(device), im_tensor.to(device)) if ref_image is not None else model(im_tensor.to(device))
        loss = 1 - score * sign / metric_range
        loss.backward() 
        im_tensor.grad *= sp_map
            
        #print(compress_image.grad)
        #if i == 0:
        grad_to_print = im_tensor.grad.detach().cpu().squeeze().permute(1,2,0).numpy()
        #print((grad_to_print == 0).sum(), '/ ', grad_to_print.reshape(-1).shape[0])

        opt.step()
        im_tensor.data.clamp_(0., 1.)
        opt.zero_grad()
    #plt.imshow((grad_to_print - grad_to_print.min()) * 300000,interpolation='none')
    #plt.colorbar()
    res_image = im_tensor.data.clamp_(min=0, max=1)

    res_image = res_image.detach()
    
    diff = res_image - before_image

    #res_image = res_image.squeeze()
    diff = diff.squeeze()

    if return_np:
        res_image = np.clip(res_image.permute(1,2,0).detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        if return_diff:
            diff = diff.permute(1,2,0).detach().cpu().numpy() * 255.0
    
    if return_diff:
        return res_image, diff
    return res_image


def zhang_dists_attack(im_tensor, ref_image=None, model=None, metric_range=100, device='cpu', lower_better=False, iters = 10, lr = 0.005):
    loss_f = dists_model_zhang.to(device)
    im_tensor = torch.autograd.Variable(im_tensor.clone().to(device), requires_grad=True)
    in_image = torch.autograd.Variable(im_tensor.clone().to(device), requires_grad=False)
    optimizer = torch.optim.Adam([im_tensor], lr=lr)
    sign = -1 if lower_better else 1
    for i in range(iters):
        score = model(ref_image.to(device), im_tensor.to(device)) if ref_image is not None else model(im_tensor.to(device))
        loss = loss_f(im_tensor, in_image, as_loss=True).to(device) - score.to(device) * sign / metric_range
        loss.backward() 
        im_tensor.grad.data[torch.isnan(im_tensor.grad.data)] = 0
        optimizer.step()
        im_tensor.data.clamp_(min=0, max=1)
        im_tensor.data[torch.isnan(im_tensor.data)] = 0
        optimizer.zero_grad() 

    res_image = im_tensor.data.clamp_(min=0, max=1)
    return res_image


