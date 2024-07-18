import torch
import os
import torch.optim as optim
import  torch.nn as nn
import numpy as np
import scipy.io as sio
import argparse
from tqdm import tqdm
import cvxpy as cp
import time
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TVTF
from skimage.measure import compare_psnr, compare_ssim
from skimage import img_as_float, img_as_ubyte
from model.TV_denoising import TV_denoising, TV_denoising3d
from utils.utils import clip, ssim, psnr,load_state_dict_cpu
from utils.ani import save_ani,weights_init_kaiming,CAVEDatasetTest
from model.ffdnet import FFDNet
from matplotlib import pyplot as plt
from scipy.io import savemat
import scipy.io
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Select device')
parser.add_argument('--device', default=0)
# parser.add_argument('--level', default=0)
args = parser.parse_args()
device_num = args.device
# level = float(args.level)
device = 'cuda:{}'.format(device_num)
print('using device:', device)

torch.no_grad()
######network_ffdnet
# model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R').to(device)
# model.load_state_dict(torch.load('pretrained_models/ffdnet_gray.pth'))

####my_ffdnet
# checkpoint = torch.load('./pretrained_models/model_state_40')
# model = FFDNet()
# model = torch.nn.DataParallel(model).to(device)
# model.load_state_dict(checkpoint)

#####ffdnet
model = FFDNet(num_input_channels=1).to(device)
model.apply(weights_init_kaiming)

model = nn.DataParallel(model, device_ids=[0]).cuda()
model.load_state_dict(torch.load('pretrained_models/net.pth'))


model.eval()

test_set = CAVEDatasetTest(mode='test')  ########### load test image
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

def normalize(data, max_val, min_val):
    return (data - min_val) / (max_val - min_val)

def ffdnet_denosing(x, sigma, flag):
    image_m, image_n, image_c = x.shape
    if flag:
        x_min = x.min().item()
        x_max = x.max().item()
        scale = 0.7
        shift = (1 - scale) / 2
        x = (x - x_min) / (x_max - x_min)
        x = x * scale + shift
        sigma = torch.tensor(sigma / (x_max - x_min) * scale, device=device)
    else:
        sigma = torch.tensor(sigma, device=device)

    frame_list = []
    with torch.no_grad():
        for j in range(image_c):
            temp_x = x[:, :, j].view(1, 1, image_m, image_n)
            # estimate_img = model(temp_x, sigma.view(1, 1, 1, 1))
            pred_noise = model(temp_x, sigma.view(1, 1, 1, 1))
            estimate_img = temp_x - pred_noise  ########pay attention to the output of network
            frame_list.append(estimate_img[0, 0, :, :])
        x = torch.stack(frame_list, dim=2)

    if flag:
        x = (x - shift) / scale
        x = x * (x_max - x_min) + x_min
    return x

def run():
    save_path = './data/test_Harvard'

    PSNR_ffd=0.0
    SSIM_ffd=0.0
    PSNR_tv = 0.0
    SSIM_tv = 0.0
    PSNR_x = 0.0
    SSIM_x = 0.0
    test_number = 0
    for t, im in enumerate(test_loader):
        #########
        #######simulation
        # image_data = sio.loadmat('/work/work_ysw/SSPSR/DeepInverse-Pytorch/CAVE/Hyper_Train_Clean/chart_and_stuffed_toy_ms.mat')
        mat_data = sio.loadmat('./data/toy31_cassi.mat')
        # im_orig = image_data['image']
        im=im.squeeze(0)
        im_orig =im.to(device)
        image_m, image_n, image_c = im_orig.shape
        image_seq = []
        # ---- load mask matrix ----
        #####Harvard
        mask_data = sio.loadmat('./data/mask1040.mat')
        mask = torch.from_numpy(mask_data['mask'].astype(np.float32)).to(device)
        ##########CAVE
        #mask = torch.from_numpy(mat_data['mask'].astype(np.float32)).to(device)

        y = torch.sum(im_orig * mask, dim=2)
        y = y.type(torch.float32).to(device)
        x = y.unsqueeze(2).expand_as(mask) * mask
        mask_sum = torch.sum(mask ** 2, dim=2)
        mask_sum[mask_sum == 0] = 1
        flag = True
        y1 = torch.zeros_like(y, dtype=torch.float32, device=device)


        sigma_ = 50 / 255
        for i in tqdm(range(100)):
            if i == 10: flag = False
            yb = torch.sum(mask * x, dim=2)
            # no Acceleration
            # temp = (y - yb) / (mask_sum)
            # x = x + 1 * (temp.unsqueeze(2).expand_as(mask) * mask)
            y1 = y1 + (y - yb)
            temp = (y1 - yb) / mask_sum
            x = x + 1 * (temp.unsqueeze(2).expand_as(mask) * mask)

            if i < 20:
                x = ffdnet_denosing(x, 50./255, flag)
            else:
                ffdnet_hypara_list = [100., 80., 60., 40., 20., 10., 5.]
                ffdnet_num = len(ffdnet_hypara_list)
                tv_hypara_list = [10, 0.01]
                tv_num = len(tv_hypara_list)
                ffdnet_list = [ffdnet_denosing(x, level/255., flag).clamp(0, 1) for level in ffdnet_hypara_list]
                tv_list = [TV_denoising(x, level, 5).clamp(0, 1) for level in tv_hypara_list]

                ffdnet_mat = np.stack(
                    [x_ele[:, :, :].cpu().numpy().reshape(-1).astype(np.float64) for x_ele in ffdnet_list],
                    axis=0)
                tv_mat = np.stack(
                    [x_ele[:, :, :].cpu().numpy().reshape(-1).astype(np.float64) for x_ele in tv_list],
                    axis=0)
                w = cp.Variable(ffdnet_num + tv_num)
                P = np.zeros((ffdnet_num + tv_num, ffdnet_num + tv_num))
                P[:ffdnet_num, :ffdnet_num] = ffdnet_mat @ ffdnet_mat.T
                P[:ffdnet_num, ffdnet_num:] = -ffdnet_mat @ tv_mat.T
                P[ffdnet_num:, :ffdnet_num] = -tv_mat @ ffdnet_mat.T
                P[ffdnet_num:, ffdnet_num:] = tv_mat @ tv_mat.T
                one_vector_ffdnet = np.ones((1, ffdnet_num))
                one_vector_tv = np.ones((1, tv_num))
                objective = cp.quad_form(w, P)
                problem = cp.Problem(
                    cp.Minimize(objective),
                    [one_vector_ffdnet @ w[:ffdnet_num] == 1,
                        one_vector_tv @ w[ffdnet_num:] == 1,
                        w >= 0])
                problem.solve()
                w_value = w.value
                x_ffdnet, x_tv = 0, 0
                for idx in range(ffdnet_num):
                    x_ffdnet += w_value[idx] * ffdnet_list[idx]
                for idx in range(tv_num):
                    x_tv += w_value[idx + ffdnet_num] * tv_list[idx]
                x = 0.5 * (x_ffdnet + x_tv)

        plt.subplot(131)
        plt.imshow(x_ffdnet[:,:,11].cpu().numpy())
        plt.title('ffdnet_re Image')

        plt.subplot(132)
        plt.imshow(x_tv[:,:,21].cpu().numpy())
        plt.title('tv_re Image')

        plt.subplot(133)
        plt.imshow(x[:,:,1].cpu().numpy())
        plt.title('x_re Image')
        plt.show()



        x.clamp_(0, 1)

        # Savemat
        test_number += 1

        mat_dir = os.path.join(save_path,str(test_number) +'_'+'x_ffdnet.mat')
        scipy.io.savemat(mat_dir, {'ffdnet_result': x_ffdnet.cpu().numpy()})

        mat_dir = os.path.join(save_path, str(test_number) +'_'+'x_tv.mat')
        scipy.io.savemat(mat_dir, {'x_tv_result': x_tv.cpu().numpy()})

        mat_dir = os.path.join(save_path, str(test_number) +'_'+ 'x.mat')
        scipy.io.savemat(mat_dir, {'x_result': x.cpu().numpy()})
        # save_ani(image_seq, filename='infrared_candle_HSI.mp4', fps=fps)
        psnr_ffd = [psnr(x_ffdnet[..., kv], im_orig[..., kv]) for kv in range(image_c)]
        ssim_ffd = [ssim(x_ffdnet[..., kv], im_orig[..., kv]) for kv in range(image_c)]

        psnr_tv = [psnr(x_tv[..., kv], im_orig[..., kv]) for kv in range(image_c)]
        ssim_tv = [ssim(x_tv[..., kv], im_orig[..., kv]) for kv in range(image_c)]

        psnr_x = [psnr(x[..., kv], im_orig[..., kv]) for kv in range(image_c)]
        ssim_x = [ssim(x[..., kv], im_orig[..., kv]) for kv in range(image_c)]

        print('{:.2f}, {:.4f},{:.2f}, {:.4f},{:.2f}, {:.4f}'.format(np.mean(psnr_ffd), np.mean(ssim_ffd),np.mean(psnr_tv), np.mean(ssim_tv),np.mean(psnr_x), np.mean(ssim_x)))

        PSNR_ffd += np.mean(psnr_ffd)
        SSIM_ffd += np.mean(ssim_ffd)
        PSNR_tv  += np.mean(psnr_tv )
        SSIM_tv  += np.mean(ssim_tv )
        PSNR_x+=np.mean(psnr_x)
        SSIM_x+=np.mean(ssim_x)


    return PSNR_ffd / test_number,SSIM_ffd/test_number,PSNR_tv/test_number,SSIM_tv/test_number,PSNR_x/test_number,SSIM_x/test_number
begin_time = time.time()
psnr_ffd,ssim_ffd,psnr_tv,ssim_tv,psnr_res, ssim_res = run()
end_time = time.time()
running_time = end_time - begin_time
print('{:.2f}, {:.4f},{:.2f}, {:.4f},{:.2f}, {:.4f}, {:.2f}'.format(psnr_ffd,ssim_ffd,psnr_tv,ssim_tv,psnr_res, ssim_res, running_time))
