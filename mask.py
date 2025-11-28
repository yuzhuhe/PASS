import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import misc as sf
from os.path import expanduser
import torchvision.models as models


class ProbMask(nn.Module):
    def __init__(self, slope, M, N, scale):
        super(ProbMask, self).__init__()

        # higher slope means a more step-function-like logistic function
        # note: slope is converted to a tensor so that we can update it during training if necessary
        self.slope = slope
        self.M = M
        self.N = N
        lst = [self.M, self.N, 1]
        input_shape_h = tuple(lst)
        self.mask_shape = input_shape_h
        self.eps = 0.01
        self.scale = scale
        self.para = torch.rand(self.mask_shape) * (1 - 2 * self.eps) + self.eps
        self.mult = - torch.log(1. / self.para - 1.) / self.slope

        self.mult[int(self.M//2-self.M*self.scale/2):int(self.M//2+self.M*self.scale/2),int(self.N//2-self.N*self.scale/2):int(self.N//2+self.N*self.scale/2), :] = -10.0


    def forward(self, x):
        logit_weights = 0 * x[..., 0:1] + self.mult.to(x.device)
        return torch.sigmoid(self.slope * logit_weights)

class ProbMask_line(nn.Module):
    def __init__(self, slope, M, N, scale):
        super(ProbMask_line, self).__init__()

        # higher slope means a more step-function-like logistic function
        # note: slope is converted to a tensor so that we can update it during training if necessary
        self.slope = slope
        self.M = M
        self.N = N
        lst = [1, self.N, 1]
        input_shape_h = tuple(lst)
        self.mask_shape = input_shape_h
        self.eps = 0.01
        self.scale = scale
        self.para = torch.rand(self.mask_shape) * (1 - 2 * self.eps) + self.eps
        self.mult = - torch.log(1. / self.para - 1.) / self.slope

        self.mult[:, int(self.N//2-self.N*self.scale/2):int(self.N//2+self.N*self.scale/2), :] = -10.0

    def forward(self, x):
        return torch.sigmoid(self.slope * self.mult)

class RescaleProbMap(nn.Module):
    def __init__(self, sparsity):
        super(RescaleProbMap, self).__init__()
        self.sparsity = sparsity
    def forward(self, x):
        xbar = torch.mean(x)
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)

        # compute adjucement
        le = torch.le(r, 1).to(torch.float32)
        return le * x * r + (1 - le) * (1 - (1 - x) * beta)

class RandomMask(nn.Module):
    def __init__(self):
        super(RandomMask, self).__init__()
    def forward(self, x):
        input_shape = x.shape
        threshs = torch.rand(input_shape)
        return (0*x) + threshs.to((0*x).device)

    def compute_output_shape(self, input_shape):
        return input_shape

class ThresholdRandomMask(nn.Module):
    def __init__(self, slope):
        super(ThresholdRandomMask, self).__init__()
        # higher slope means a more step-function-like logistic function
        # note: slope is converted to a tensor so that we can update it during training if necessary
        self.slope = slope

    def forward(self, x):
        inputs = x[0]
        thresh = x[1]
        if self.slope is not None:
            return torch.sigmoid(self.slope * (inputs-thresh))
        else:
            return inputs > thresh

class create_mask(nn.Module):
    def __init__(self,M,N,pmask_slope=1,sparsity=0.01,sample_slope=200.0,scale=0.1):
        super(create_mask, self).__init__()
        """
        Sparse: desired sparse acceleration coefficient
        Pmask_stope: slope of logistic parameter in probability mask
        Sample_strop: combine the uniform distribution to obtain the slope value of the actual mask
        Scale: Take the proportion of the middle block
        
        return:
        mask: BS*H*W*1
        """
        self.M = M
        self.N = N
        self.pmask_slope = pmask_slope
        self.sparsity = sparsity
        self.sample_slope = sample_slope
        self.scale = scale
        self.ProbMask = ProbMask(self.pmask_slope,self.M,self.N,self.scale)
        self.RescaleProbMap = RescaleProbMap(self.sparsity)
        self.RandomMask = RandomMask()
        self.ThresholdRandomMask = ThresholdRandomMask(self.sample_slope)

    def forward(self,x):
        prob_mask_tensor = self.ProbMask(x)

        prob_mask_tensor = self.RescaleProbMap(prob_mask_tensor)

        # Realization of probability mask
        thresh_tensor = self.RandomMask(prob_mask_tensor)  # U(0,1)
        high_frequency_mask = self.ThresholdRandomMask([prob_mask_tensor, thresh_tensor])

        low_frequency_mask = torch.zeros_like(high_frequency_mask)
        low_frequency_mask[:,int(self.M//2-self.M*self.scale/2):int(self.M//2+self.M*self.scale/2),
                    int(self.N//2-self.N*self.scale/2):int(self.N//2+self.N*self.scale/2), :] = 1.0

        update_mask = torch.ones_like(high_frequency_mask)
        update_mask[:,int(self.M//2-self.M*self.scale/2):int(self.M//2+self.M*self.scale/2),
                    int(self.N//2-self.N*self.scale/2):int(self.N//2+self.N*self.scale/2), :] = 0.0

        return low_frequency_mask, high_frequency_mask, update_mask.squeeze(0)

class create_mask_line(nn.Module):
    def __init__(self,M,N,pmask_slope=1,sparsity=0.01,sample_slope=200.0,scale=0.1):
        super(create_mask_line, self).__init__()
        """
        Sparse: desired sparse acceleration coefficient
        Pmask_stope: slope of logistic parameter in probability mask
        Sample_strop: combine the uniform distribution to obtain the slope value of the actual mask
        Scale: Take the proportion of the middle block
        
        return:
        mask: BS*H*W*1
        """
        self.M = M
        self.N = N
        self.pmask_slope = pmask_slope
        self.sparsity = sparsity
        self.sample_slope = sample_slope
        self.scale = scale
        self.ProbMask = ProbMask_line(self.pmask_slope,self.M,self.N,self.scale)
        self.RescaleProbMap = RescaleProbMap(self.sparsity)
        self.RandomMask = RandomMask()
        self.ThresholdRandomMask = ThresholdRandomMask(self.sample_slope)

    def forward(self,x):
        lst = [1, self.N, 1]
        input_shape_h = tuple(lst)

        prob_mask_tensor = self.ProbMask(x)

        prob_mask_tensor = self.RescaleProbMap(prob_mask_tensor.unsqueeze(0))

        # Realization of probability mask
        thresh_tensor = self.RandomMask(prob_mask_tensor)  # U(0,1)
        high_frequency_mask = self.ThresholdRandomMask([prob_mask_tensor, thresh_tensor])

        low_frequency_mask = torch.zeros_like(high_frequency_mask)
        low_frequency_mask[:, :, int(self.N//2-self.N*self.scale/2):int(self.N//2+self.N*self.scale/2), :] = 1.0


        update_mask = torch.ones(input_shape_h)
        update_mask[:,int(self.N//2-self.N*self.scale/2):int(self.N//2+self.N*self.scale/2), :] = 0.0

        return low_frequency_mask.repeat(1, self.M, 1, 1), high_frequency_mask.repeat(1, self.M, 1, 1), update_mask
