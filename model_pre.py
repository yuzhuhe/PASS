import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import misc as sf
from os.path import expanduser
import torchvision.models as models
from fftc import *
from mask_pre import *

home = expanduser("~")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cos_sim(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

# CG
def myCG(A,rhs,cgIter,cgTol):
    cond=lambda i,rTr,*_: torch.eq(torch.lt(i.to(device=device),cgIter),torch.abs(rTr.to(device=device))>cgTol)
    fn=lambda a,b:torch.sum(torch.conj(a)*b)
    def body(i,rTr,x,r,p):
        Ap=A(p)
        alpha = rTr / fn(p,Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = fn(r,r)
        beta = rTrNew / rTr
        p = r + beta * p
        return i+1,rTrNew,x,r,p
    x=torch.zeros_like(rhs)
    i,r,p=torch.as_tensor(0),torch.as_tensor(rhs),torch.as_tensor(rhs)
    rTr =  fn(r,r)
    while cond(i,rTr):
        i,rTr,x,r,p=body(i,rTr,x,r,p)
    out=x
    return out

# A'A
def myAtA(img,csm,mask):
    cimg = csm*img
    tmp = ifft2c_new(mask * fft2c_new(sf.torch_c2r(cimg)))
    coilComb = torch.sum(sf.torch_r2c(tmp)*torch.conj(csm), dim=-3)
    return coilComb

def consFn(x):
    y = torch.clamp(x, 0, 1)
    return y

#
class DC1(nn.Module):
    def __init__(self,lamT,cgIter,cgTol):
        super(DC1, self).__init__()
        self.a = lamT
        self.b = cgIter
        self.c = cgTol
    def forward(self,z,atbT,AtA1):
        z = sf.torch_r2c(z)  # BS*320*320
        rhs = atbT + self.a * z  # A'b+*lamTz
        def fn(inp):
            B = lambda x: AtA1(x) + self.a * x  # (A'A+lamT)x
            y = myCG(B, inp, self.b, self.c)
            return y
        rec = fn(rhs)
        rec = sf.torch_c2r(rec)
        return rec

# ADMM
class DC2(nn.Module):
    def __init__(self,lamT,cgIter,cgTol,rho):
        super(DC2, self).__init__()
        self.a = lamT
        self.b = cgIter
        self.c = cgTol
        self.d = rho
    def forward(self,z1,z2,atbT,y,AtA1):
        z1 = sf.torch_r2c(z1)  # BS*320*320
        z2 = sf.torch_r2c(z2)
        y = sf.torch_r2c(y)
        rhs = atbT + self.a * z1 + self.d * (z2-y/self.d)
        def fn(inp):
            B = lambda x: AtA1(x) + self.a * x + self.d * x  # (A'A+lamT+rho)x
            y = myCG(B, inp, self.b, self.c)
            return y
        rec = fn(rhs)
        rec = sf.torch_c2r(rec)
        return rec

#
class DC3(nn.Module):
    def __init__(self):
        super(DC3, self).__init__()

    def forward(self, x, org, csm, mask):
        raw_kspace = fft2c_new(sf.torch_c2r(org*csm))  # BS*num_coil*H*W*2
        pre_kspace = fft2c_new(sf.torch_c2r(sf.torch_r2c(x)*csm))  # BS*num_coil*H*W*2
        out_kspace = mask * raw_kspace + (1 - mask) * pre_kspace  # BS*num_coil*H*W*2
        out_multi_img = ifft2c_new(out_kspace)  # BS*num_coil*H*W*2
        out = torch.sum(sf.torch_r2c(out_multi_img)*torch.conj(csm), dim=-3)  # BS*num_coil*H*W
        out = sf.torch_c2r(out)  # BS*H*W*2
        return out

#
class Net_D1(nn.Module):
    def __init__(self):
        super(Net_D1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(2),
            nn.GroupNorm(num_groups=2, num_channels=2)
        )

    def forward(self, x):
        # BS*320*320*2
        a = x
        x = x.permute(0, 3, 1, 2)
        out = self.layer(x)
        out = out.permute(0, 2, 3, 1)
        out = out + a
        return out

#
class Net_G(nn.Module):
    def __init__(self):
        super(Net_G, self).__init__()
        self.guide_map = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.img_feature = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU()
        )

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(2)
            nn.GroupNorm(num_groups=2, num_channels=2)
        )

    def forward(self, x, AD_map):
        # BS*320*320*2,AD_map：BS*1*320*320
        a = x
        x = x.permute(0, 3, 1, 2)
        guide_map = self.guide_map(AD_map)
        out = self.img_feature(x)
        out = out * guide_map
        out = self.layer(out)
        out = out.permute(0, 2, 3, 1)
        out = out + a
        return out

#
class Net_M(nn.Module):
    def __init__(self, mask_loss1, mask_loss2):
        super(Net_M, self).__init__()
        self.mask_loss1 = mask_loss1
        self.mask_loss2 = mask_loss2

    def forward(self, x, org, csm, map, mask1):
        '''
        Args:
            x:BS*H*W*2
            org: BS*H*W
            csm: BS*num_coil*H*W
            map: BS*H*W
            mask:BS*H*W*1
        Returns:
            mask loss
        '''
        tmp0 = torch.abs(sf.torch_r2c(x) - org)
        loss1 = self.mask_loss1 * torch.mean(torch.pow(tmp0, 2))
        tmp1 = map * (myAtA(org, csm, mask1) - myAtA(sf.torch_r2c(x), csm, mask1))
        loss2 = self.mask_loss2 * torch.mean(torch.pow(tmp1, 2))
        mask_loss = loss1 + loss2
        return mask_loss

#
class Net_Grad(nn.Module):
    def __init__(self, grad_shape):
        super(Net_Grad, self).__init__()
        self.shape = grad_shape
        self.high = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.img = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.extraction = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.linear1 = nn.Linear(320*320, self.shape[0]*self.shape[1], bias=True)
        self.linear2 = nn.Linear(self.shape[0]*self.shape[1], self.shape[0]*self.shape[1], bias=True)

    def forward(self, x, csm, AD_map, mask1):
        # BS*320*320*2,AD_map：BS*1*320*320
        tmp1 = AD_map * (myAtA(sf.torch_r2c(x), csm, mask1))  # 1,1,320,320 complex
        tmp1 = tmp1.squeeze(1)  # 1,320,320 complex
        tmp1 = sf.torch_c2r(tmp1)  # 1,320,320,2
        tmp1 = tmp1.permute(0, 3, 1, 2) # 1,2,320,320
        tmp2 = AD_map * x.permute(0, 3, 1, 2)# 1,2,320,320

        feature_high = self.high(tmp1)
        feature_img = self.img(tmp2)

        feature_grad = torch.concat([feature_high, feature_img], dim=1)  # 1,2,320,320
        feature_grad = self.extraction(feature_grad)  # 1,1,320,320

        out = feature_grad.flatten()
        out = self.linear1(out)
        out = self.linear2(out)

        out = out.view(self.shape)
        return out

# AD model
class CLIP_guided_model(nn.Module):
    def __init__(self, clip_model, text_features, seg_mem_features):
        super(CLIP_guided_model, self).__init__()
        # self.AD_pth = AD_pth
        self.clip_model = clip_model
        # self.checkpoint = torch.load(self.AD_pth)
        # self.clip_model.seg_adapters.load_state_dict(self.checkpoint["seg_adapters"])
        # self.clip_model.det_adapters.load_state_dict(self.checkpoint["det_adapters"])
        self.text_features = text_features
        self.seg_mem_features = seg_mem_features

    def forward(self, x):
        '''

        Args:
            x: BS*320*320*2
            seg_mem_features

        Returns:
            AD_map：BS*1*320*320

        '''

        image = torch.abs(sf.torch_r2c(x)).unsqueeze(1)
        image = image.repeat(1, 3, 1, 1)
        _, _, HH, _ = image.shape
        self.clip_model = self.clip_model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
        # with torch.no_grad():
            _, seg_patch_tokens, _ = self.clip_model(image)
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]

            # few-shot, seg head
            anomaly_maps_few_shot = []
            for idx, p in enumerate(seg_patch_tokens):
                cos = cos_sim(self.seg_mem_features[idx], p)
                height = int(np.sqrt(cos.shape[1]))
                anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                     size=HH, mode='bilinear', align_corners=True)
                anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
            score_map_few = np.sum(anomaly_maps_few_shot, axis=0)

            # zero-shot, seg head
            anomaly_maps = []
            for layer in range(len(seg_patch_tokens)):
                seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * seg_patch_tokens[layer] @ self.text_features).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=HH, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            score_map_zero = np.sum(anomaly_maps, axis=0)

        score_map_few = (score_map_few - score_map_few.min()) / (
                score_map_few.max() - score_map_few.min())
        score_map_zero = (score_map_zero - score_map_zero.min()) / (
                score_map_zero.max() - score_map_zero.min())

        AD_map = 0.5 * score_map_few + 0.5 * score_map_zero
        AD_map = torch.from_numpy(AD_map).to(device)
        return AD_map.unsqueeze(1)

class model(nn.Module):
    def __init__(self, M, N, acc_factor, scale, lam1, lam2, sigma, cgIter, cgTol, ADMM_Iter, ADMM_tol, ADMM_rho, K, mask_loss1, mask_loss2, lr, clip_model, text_features, seg_mem_features):
        super(model, self).__init__()
        self.M = M
        self.N = N
        self.acc_factor = acc_factor
        self.scale = scale
        self.lamT1 = nn.Parameter(torch.tensor(lam1, dtype=torch.float32))
        self.lamT2 = nn.Parameter(torch.tensor(lam2, dtype=torch.float32))
        self.sigmaT = torch.tensor(sigma, dtype=torch.float32)
        self.cgIter = cgIter
        self.cgTol = cgTol
        self.ADMM_Iter = ADMM_Iter
        self.ADMM_tol = ADMM_tol
        self.ADMM_rho = ADMM_rho
        self.clip_model = clip_model
        self.text_features = text_features
        self.seg_mem_features = seg_mem_features
        self.K = K
        self.create_mask1 = create_mask(self.M, self.N, sparsity=self.acc_factor, scale=self.scale)
        self.Net_D = Net_D1()
        self.Net_G = Net_G()
        self.Net_AD = CLIP_guided_model(self.clip_model, self.text_features, self.seg_mem_features)
        self.DC_1 = DC1(self.lamT1, self.cgIter, self.cgTol)
        self.DC_2 = DC2(self.lamT1, self.cgIter, self.cgTol, self.ADMM_rho)

    def forward(self, orgT, csmT):
        # orgT:BS*320*320; csmT:BS*num_coil*320*320
        if orgT.shape[1] == 1:
            orgT = orgT.squeeze(1)
        orgT = orgT.to(torch.complex64)
        csmconj = torch.conj(csmT)
        kspace = fft2c_new(sf.torch_c2r(orgT*csmT))

        mask1, mask2, up_mask = self.create_mask1(sf.torch_c2r(orgT))
        mask = consFn(mask1+mask2)
        mask_out1 = mask
        under_kspace = (mask*kspace).to(device=device)
        shp = under_kspace.shape
        noiseT = torch.randn(shp) * self.sigmaT
        noiseT = noiseT.to(device=device)
        under_kspace = under_kspace + noiseT

        atbT = ifft2c_new(under_kspace)
        atbT = torch.sum(csmconj * sf.torch_r2c(atbT), dim=-3)

        del kspace
        del under_kspace
        del noiseT
        # torch.cuda.empty_cache()


        z = torch.zeros_like(atbT)  # BS*320*320
        r = torch.zeros_like(atbT)  # BS*320*320
        y = torch.zeros_like(atbT)
        x_out1 = torch.zeros_like(atbT)
        AD_map = torch.zeros_like(atbT)  # BS*320*320
        z = sf.torch_c2r(z)
        r = sf.torch_c2r(r)
        y = sf.torch_c2r(y)  # ADMM
        x_out1 = sf.torch_c2r(x_out1)
        # AtA
        AtA1 = lambda x: myAtA(x, csmT, mask)

        x = self.DC_1(z, atbT, AtA1) #BS,320,320,2

        for i in range(self.K):
            z = self.Net_D(x)
            x = self.DC_1(z, atbT, AtA1)
            x_out1 = x
            # Anomaly map
            AD_map = self.Net_AD(x)
            r = x
            # ADMM
            for j in range(self.ADMM_Iter):
                x = self.DC_2(self.Net_D(x), r, atbT, y, AtA1)

                r = self.Net_G(r, AD_map)
                r = (self.lamT2 * r + self.ADMM_rho * x + y) / (self.ADMM_rho + self.lamT2)
                # y
                y = y + self.ADMM_rho * (x - r)
                #
                if torch.dist(x, r, 2) < self.ADMM_tol:
                    break

        return atbT, x_out1, x, AD_map

#
class model_line(nn.Module):
    def __init__(self, M, N, acc_factor, scale, lam1, lam2, sigma, cgIter, cgTol, ADMM_Iter, ADMM_tol, ADMM_rho, K, mask_loss1, mask_loss2, lr, clip_model, text_features, seg_mem_features):
        super(model_line, self).__init__()
        self.M = M
        self.N = N
        self.acc_factor = acc_factor
        self.scale = scale
        #
        self.lamT1 = nn.Parameter(torch.tensor(lam1, dtype=torch.float32))
        self.lamT2 = nn.Parameter(torch.tensor(lam2, dtype=torch.float32))
        #
        self.sigmaT = torch.tensor(sigma, dtype=torch.float32)
        #
        self.cgIter = cgIter
        self.cgTol = cgTol
        #
        self.ADMM_Iter = ADMM_Iter
        self.ADMM_tol = ADMM_tol
        self.ADMM_rho = ADMM_rho
        # AD model
        self.clip_model = clip_model
        self.text_features = text_features
        self.seg_mem_features = seg_mem_features
        #
        self.K = K
        # mask
        self.create_mask1 = create_mask_line(self.M, self.N, sparsity=self.acc_factor, scale=self.scale)
        self.Net_D = Net_D1()
        self.Net_G = Net_G()
        self.Net_AD = CLIP_guided_model(self.clip_model, self.text_features, self.seg_mem_features)
        # DC1
        self.DC_1 = DC1(self.lamT1, self.cgIter, self.cgTol)
        # ADMM DC
        self.DC_2 = DC2(self.lamT1, self.cgIter, self.cgTol, self.ADMM_rho)

    def forward(self, orgT, csmT):
        # orgT:BS*320*320; csmT:BS*num_coil*320*320
        if orgT.shape[1] == 1:
            orgT = orgT.squeeze(1)
        orgT = orgT.to(torch.complex64)
        csmconj = torch.conj(csmT)
        kspace = fft2c_new(sf.torch_c2r(orgT*csmT))

        mask1, mask2, up_mask = self.create_mask1(sf.torch_c2r(orgT))
        mask = consFn(mask1+mask2)
        mask_out1 = mask
        # mask = torch.clamp(mask1+mask2, max=1.)

        under_kspace = (mask*kspace).to(device=device)
        shp = under_kspace.shape
        noiseT = torch.randn(shp) * self.sigmaT
        noiseT = noiseT.to(device=device)
        under_kspace = under_kspace + noiseT

        atbT = ifft2c_new(under_kspace)
        atbT = torch.sum(csmconj * sf.torch_r2c(atbT), dim=-3)

        del kspace
        del under_kspace
        del noiseT
        # torch.cuda.empty_cache()

        #
        z = torch.zeros_like(atbT)  # BS*320*320
        r = torch.zeros_like(atbT)  # BS*320*320
        y = torch.zeros_like(atbT)
        x_out1 = torch.zeros_like(atbT)
        AD_map = torch.zeros_like(atbT)  # BS*320*320
        z = sf.torch_c2r(z)
        r = sf.torch_c2r(r)
        y = sf.torch_c2r(y)  # ADMM
        x_out1 = sf.torch_c2r(x_out1)
        # AtA
        AtA1 = lambda x: myAtA(x, csmT, mask)

        x = self.DC_1(z, atbT, AtA1) #BS,320,320,2

        for i in range(self.K):
            #
            z = self.Net_D(x)
            x = self.DC_1(z, atbT, AtA1)
            x_out1 = x
            # Anomaly map
            AD_map = self.Net_AD(x)
            r = x
            # ADMM
            for j in range(self.ADMM_Iter):
                # x
                x = self.DC_2(self.Net_D(r), r, atbT, y, AtA1)
                # r
                r = self.Net_G(r, AD_map)
                r = (self.lamT2 * r + self.ADMM_rho * x + y) / (self.ADMM_rho + self.lamT2)
                # y
                y = y + self.ADMM_rho * (x - r)
                if torch.dist(x, r, 2) < self.ADMM_tol:
                    break

        return atbT, x_out1,  x, AD_map

