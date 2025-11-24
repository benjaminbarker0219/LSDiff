import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from mpl_toolkits.mplot3d import Axes3D
from .lse import gradient,gradient_sobel
import torch

import cv2
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn.functional as F
def compute_vector_field(phi_t, normalize=True):

    vy, vx = gradient_sobel(phi_t, split=True)
    vf = torch.cat([vx, vy], dim=1)
    if normalize:
        vf = F.normalize(vf, p=2, dim=1)
    return vf

def compute_curvature(phi):

    phi_smooth = gaussian_blur(phi, kernel_size=5, sigma=1.0) 
    phi_y, phi_x = gradient_sobel(phi_smooth, split=True)
    phi_x_padded = F.pad(phi_x, (0, 0, 1, 1), mode="replicate")
    phi_y_padded = F.pad(phi_y, (1, 1, 0, 0), mode="replicate")
    phi_xx = phi_x_padded[:, :, 2:, :] - 2 * phi_x_padded[:, :, 1:-1, :] + phi_x_padded[:, :, :-2, :]
    phi_yy = phi_y_padded[:, :, :, 2:] - 2 * phi_y_padded[:, :, :, 1:-1] + phi_y_padded[:, :, :, :-2]
    curvature = phi_xx + phi_yy
    
    return curvature


def gaussian_kernel(kernel_size=5, sigma=1.0):

    x = torch.arange(kernel_size).float() - kernel_size // 2
    y = torch.arange(kernel_size).float() - kernel_size // 2
    x, y = torch.meshgrid(x, y, indexing="ij")
    
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum() 
    return kernel.view(1, 1, kernel_size, kernel_size) 

def gaussian_blur(img, kernel_size=5, sigma=1.0):
    kernel = gaussian_kernel(kernel_size, sigma).to(img.device)
    padding = kernel_size // 2 
    blurred_img = F.conv2d(img, kernel, padding=padding, groups=img.shape[1])  
    return blurred_img

