import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize01(img):
    """
    Normalize the image between 0 and 1
    """
    shp=img.shape
    if np.ndim(img)>=3:
        nimg=np.prod(shp[0:-2])
    elif np.ndim(img)==2:
        nimg=1
    img=np.reshape(img,(nimg,shp[-2],shp[-1]))
    eps=1e-15
    img2=np.empty_like(img)
    for i in range(nimg):
        mx=img[i].max()
        mn=img[i].min()
        img2[i]= (img[i]-mn)/(mx-mn+eps)
    img2=np.reshape(img2,shp)
    return img2

def norm_torch(img):
    """
    Normalize the image between 0 and 1
    """
    shp=img.shape
    if len(shp)>=3:
        nimg=torch.prod(torch.tensor(shp[0:-2]))
    elif len(shp)==2:
        nimg=1
    img=torch.reshape(img,(nimg,shp[-2],shp[-1]))
    eps=1e-15
    img2=torch.empty_like(img)
    for i in range(nimg):
        mx=img[i].max()
        mn=img[i].min()
        img2[i]= (img[i]-mn)/(mx-mn+eps)
    img2=torch.reshape(img2,shp)
    return img2

def myPSNR(org,recon):
    sqrError=np.abs(org-recon)**2
    N=np.prod(org.shape[-2:])
    mse=np.sum(sqrError,axis=(-1,-2))/N
    maxval=np.max(org,axis=(-1,-2)) + 1e-15
    psnr=10*np.log10(maxval**2/(mse+1e-15 ))
    return psnr

def torch_r2c(inp):
    return torch.complex(inp[...,0],inp[...,1])

def torch_c2r(inp):
    return torch.stack([torch.real(inp),torch.imag(inp)],dim=-1)

def torch_c3r(inp):
    c = torch.abs(inp)
    return torch.stack([c,c,c],dim=-1)

def SSIM(org,rec):
    xbar, ybar = org.mean(), rec.mean()

def ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
    Returns:
        float: ssim result.
    """
    K1 = 0.01
    K2 = 0.03
    C1 = (K1)**2
    C2 = (K2)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # ux
    ux = img1.mean()
    # uy
    uy = img2.mean()
    # ux^2
    ux_sq = ux**2
    # uy^2
    uy_sq = uy**2
    # ux*uy
    uxuy = ux * uy
    # ox、oy方差计算
    ox_sq = img1.var()
    oy_sq = img2.var()
    ox = np.sqrt(ox_sq)
    oy = np.sqrt(oy_sq)
    oxoy = ox * oy
    oxy = np.mean((img1 - ux) * (img2 - uy))
    # 公式二计算
    ssim = ((2*ux*uy+C1)*(2*oxy+C2))/((ux_sq+uy_sq+C1)*(ox_sq+oy_sq+C2))
    # 验证结果输出
    # print('ssim:', ssim, ",L:", L, ",C:", C, ",S:", S)
    return ssim

class MS_SSIM_L1_LOSS(nn.Module):
    """
    Have to use cuda, otherwise the speed is too slow.
    Both the group and shape of input image should be attention on.
    I set 255 and 1 for gray image as default.
    """

    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=255.0,
                 K=(0.01, 0.03),  # c1,c2
                 alpha=0.84,  # weight of ssim and l1 loss
                 compensation=200.0,  # final factor for total loss
                 cuda_dev=0,  # cuda device choice
                 channel=1):  # RGB image should set to 3 and Gray image should be set to 1
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.channel = channel
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros(
            (self.channel * len(gaussian_sigmas), 1, filter_size, filter_size))  # 创建了(3*5, 1, 33, 33)个masks
        for idx, sigma in enumerate(gaussian_sigmas):
            if self.channel == 1:
                # only gray layer
                g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            elif self.channel == 3:
                # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
                g_masks[self.channel * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size,
                                                                                   sigma)  # 每层mask对应不同的sigma
                g_masks[self.channel * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
                g_masks[self.channel * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            else:
                raise ValueError
        self.g_masks = g_masks.cuda(cuda_dev)  # 转换为cuda数据类型

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)
        # Outer product of input and vec2. If input is a vector of size nn and vec2 is a vector of size mm,
        # then out must be a matrix of size (n \times m)(n×m).

    def forward(self, x, y):
        b, c, h, w = x.shape
        assert c == self.channel

        mux = F.conv2d(x, self.g_masks, groups=c, padding=self.pad)  # 图像为96*96，和33*33卷积，出来的是64*64，加上pad=16,出来的是96*96
        muy = F.conv2d(y, self.g_masks, groups=c, padding=self.pad)  # groups 是分组卷积，为了加快卷积的速度

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=c, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=c, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=c, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        if self.channel == 3:
            lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]  # 亮度对比因子
            PIcs = cs.prod(dim=1)
        elif self.channel == 1:
            lM = l[:, -1, :, :]
            PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, C, H, W]
        # average l1 loss in num channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.channel, length=self.channel),
                               groups=c, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()
