#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from PIL import Image
from torch import optim
from scipy.misc import face
import random
from torchvision import transforms
import time
import matplotlib.pyplot as plt


# In[193]:


import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

J = 1j
EPS = 1e-20

class MapTools:
    def __init__(self, N_grid, theta_max):
        self.set_map_properties(N_grid, theta_max)
        self.set_fft_properties(N_grid, self.theta_max)
        self.imag_indices = self.get_imag_indices()

    def do_fwd_KS(self, kappa_l):
        kappa_l = self.symmetrize_fourier(kappa_l)
        kappa_l_complex = kappa_l[0] + J * kappa_l[1] 

        F_gamma_1 = (self.ell_x**2 - self.ell_y**2) * kappa_l_complex / (self.ell**2)
        F_gamma_2 = 2. * self.ell_x * self.ell_y    * kappa_l_complex / (self.ell**2)
        
        gamma_1 =  np.fft.irfftn(F_gamma_1) / self.PIX_AREA
        gamma_2 =  np.fft.irfftn(F_gamma_2) / self.PIX_AREA
        
        return gamma_1, gamma_2    
    
    def do_KS_inversion(self, eps):        
        A_ell = ((self.ell_x_full**2 - self.ell_y_full**2) - J * (2 * self.ell_x_full * self.ell_y_full)) \
                                            /(self.ell_full**2)
        
        eps_1, eps_2 = eps
        eps_ell = self.PIX_AREA * np.fft.fftn(eps_1 + J * eps_2)
        kappa_ell = A_ell * eps_ell
        kappa_map_KS = np.fft.ifftn(kappa_ell).real /  self.PIX_AREA
        return kappa_map_KS
    
    def map2fourier(self, x_map):
        Fx_complex =  self.PIX_AREA * np.fft.rfftn(x_map)
        return np.array([Fx_complex.real, Fx_complex.imag])
    
    
    def fourier2map(self, Fx):
        Fx         = self.symmetrize_fourier(Fx)
        Fx_complex = (Fx[0] + J * Fx[1])
        return np.fft.irfftn(Fx_complex) /  self.PIX_AREA
    
    def symmetrize_fourier(self, Fx):
        return np.array([(self.fourier_symm_mask * Fx[0]) + 
                          (~self.fourier_symm_mask * Fx[0,self.fourier_symm_flip_ind])
                         ,(self.fourier_symm_mask * Fx[1]) - 
                          (~self.fourier_symm_mask * Fx[1,self.fourier_symm_flip_ind])])

    def set_map_properties(self, N_grid, theta_max):
        self.N_grid      = N_grid
        self.theta_max   = theta_max
        self.Omega_s     = self.theta_max**2
        self.PIX_AREA    = self.Omega_s / self.N_grid**2
        self.Delta_theta = theta_max / N_grid
        
    def set_fft_properties(self, N_grid, theta_max):
        lx = 2*np.pi*np.fft.fftfreq(N_grid, d=theta_max / N_grid)
        ly = 2*np.pi*np.fft.fftfreq(N_grid, d=theta_max / N_grid)

        N_Y = (N_grid//2 +1)
        self.N_Y = N_Y
        
        # mesh of the 2D frequencies
        self.ell_x = np.tile(lx[:, None], (1, N_Y))       
        self.ell_y = np.tile(ly[None, :N_Y], (N_grid, 1))
        self.ell = np.sqrt(self.ell_x**2 + self.ell_y**2)
        self.ell[0,0] = 1.
        
        self.ell_x_full = np.tile(lx[:, None], (1, N_grid))       
        self.ell_y_full = np.tile(ly[None, :], (N_grid, 1))
        self.ell_full   = np.sqrt(self.ell_x_full**2 + self.ell_y_full**2)
        self.ell_full[0,0] = 1.
        
        fourier_symm_mask = np.ones((N_grid, self.N_Y))
        fourier_symm_mask[(self.N_Y):,0]  = 0
        fourier_symm_mask[(self.N_Y):,-1] = 0
        fourier_symm_mask[0,0]            = 0
        self.fourier_symm_mask = fourier_symm_mask.astype(bool)        
        
        fourier_symm_mask_imag = fourier_symm_mask.copy()
        fourier_symm_mask_imag[0,-1]        = 0
        fourier_symm_mask_imag[self.N_Y-1,0]  = 0
        fourier_symm_mask_imag[self.N_Y-1,-1] = 0
        self.fourier_symm_mask_imag = fourier_symm_mask_imag.astype(bool)
        
        fourier_symm_flip_ind      = np.arange(N_grid)
        fourier_symm_flip_ind[1:]  = fourier_symm_flip_ind[1:][::-1]
        self.fourier_symm_flip_ind = fourier_symm_flip_ind

# ================== 1D array to Fourier plane ===============================             
    def set_fourier_plane_face(self, F_x, x):
        N = self.N_grid
        F_x[:,:,1:-1] = x[:N**2 - 2*N].reshape(2,N,N//2-1)
        return F_x

    def set_fourier_plane_edge(self, F_x, x):
        N = self.N_grid
        N_Y = N//2+1
        N_edge = N//2-1    
        
        F_x[:,1:N_Y-1,0]  = x[N**2 - 2*N:N**2 - 2*N+2*N_edge].reshape((2,-1))
        F_x[:,1:N_Y-1,-1] = x[N**2 - 2*N+2*N_edge:-3].reshape((2,-1))
        return F_x

    def set_fourier_plane_corner(self, F_x, x):    
        N = self.N_grid
        N_Y = N//2+1
               
        F_x[0,N_Y-1,-1] = x[-3]
        F_x[0,0,-1]     = x[-2]
        F_x[0,N_Y-1,0]  = x[-1]
        return F_x
    
    def array2fourier_plane(self, x):
        N = self.N_grid
        N_Y = N//2+1
        N_edge = N//2-1    

        F_x_plane = np.zeros((2,N,N_Y))
        F_x_plane = self.set_fourier_plane_face(F_x_plane, x)
        F_x_plane = self.set_fourier_plane_edge(F_x_plane, x)
        F_x_plane = self.set_fourier_plane_corner(F_x_plane, x)

        F_x_plane = self.symmetrize_fourier(F_x_plane)        
        return F_x_plane
    
    def fourier_plane2array(self, Fx):
        N = self.N_grid
        N_Y = N//2+1
        N_edge = N//2-1    

        x = np.zeros(shape=N*N-1)

        x[:N**2 - 2*N]                    = Fx[:,:,1:-1].reshape(-1)
        x[N**2 - 2*N:N**2 - 2*N+2*N_edge] = Fx[:,1:N_Y-1,0].reshape(-1)
        x[N**2 - 2*N+2*N_edge:-3]         = Fx[:,1:N_Y-1,-1].reshape(-1)
        
        x[-3] = Fx[0,N_Y-1,-1]
        x[-2] = Fx[0,0,-1]
        x[-1] = Fx[0,N_Y-1,0]
        
        return x
    
    def get_imag_indices(self):
        x0 = np.zeros(self.N_grid**2-1)
        Fx = np.array(self.array2fourier_plane(x0))
        Fx[1] = 1

        x0 = np.array(self.fourier_plane2array(Fx)).astype(int)

        indices = np.arange(x0.shape[0])
        imag_indices_1d = indices[(x0 == 1)]

        return imag_indices_1d

class TorchMapTools:
    def __init__(self, N_grid, theta_max):
        self.set_map_properties(N_grid, theta_max)
        self.set_fft_properties(N_grid, self.theta_max)
        self.imag_indices = self.get_imag_indices()

    def do_fwd_KS(self, kappa_l):
        kappa_l = self.symmetrize_fourier(kappa_l)
        kappa_l_complex = kappa_l[0] + 1j * kappa_l[1] 

        F_gamma_1 = (self.ell_x**2 - self.ell_y**2) * kappa_l_complex / (self.ell**2)
        F_gamma_2 = 2. * self.ell_x * self.ell_y    * kappa_l_complex / (self.ell**2)
        
        gamma_1 =  torch.fft.irfftn(F_gamma_1, s=(self.N_grid, self.N_grid)) / self.PIX_AREA
        gamma_2 =  torch.fft.irfftn(F_gamma_2, s=(self.N_grid, self.N_grid)) / self.PIX_AREA
        
        return gamma_1, gamma_2    
    
    def do_KS_inversion(self, eps):        
        A_ell = ((self.ell_x_full**2 - self.ell_y_full**2) - 1j * (2 * self.ell_x_full * self.ell_y_full)) \
                                            /(self.ell_full**2)
        
        eps_1, eps_2 = eps
        eps_ell = self.PIX_AREA * torch.fft.fftn(eps_1 + 1j * eps_2, s=(self.N_grid, self.N_grid))
        kappa_ell = A_ell * eps_ell
        kappa_map_KS = torch.fft.ifftn(kappa_ell, s=(self.N_grid, self.N_grid)).real /  self.PIX_AREA
        return kappa_map_KS
    
    def map2fourier(self, x_map):
        Fx_complex =  self.PIX_AREA * torch.fft.rfftn(x_map, s=(self.N_grid, self.N_grid))
        return torch.stack([Fx_complex.real, Fx_complex.imag])
    
    
    def fourier2map(self, Fx):
        Fx         = self.symmetrize_fourier(Fx)
        Fx_complex = Fx[0] + 1j * Fx[1]
        return torch.fft.irfftn(Fx_complex, s=(self.N_grid, self.N_grid)) /  self.PIX_AREA
    
    def symmetrize_fourier(self, Fx):
        return torch.stack([(self.fourier_symm_mask * Fx[0]) + 
                            (~self.fourier_symm_mask * Fx[0,self.fourier_symm_flip_ind])
                         ,(self.fourier_symm_mask * Fx[1]) - 
                            (~self.fourier_symm_mask * Fx[1,self.fourier_symm_flip_ind])])

    def set_map_properties(self, N_grid, theta_max):
        self.N_grid      = N_grid
        self.theta_max   = theta_max
        self.Omega_s     = self.theta_max**2
        self.PIX_AREA    = self.Omega_s / self.N_grid**2
        self.Delta_theta = theta_max / N_grid
        
    def set_fft_properties(self, N_grid, theta_max):
        lx = 2 * torch.pi * torch.fft.fftfreq(N_grid, d=theta_max / N_grid)
        ly = 2 * torch.pi * torch.fft.fftfreq(N_grid, d=theta_max / N_grid)
    
        N_Y = (N_grid // 2 + 1)
        self.N_Y = N_Y
        
        # mesh of the 2D frequencies
        self.ell_x = torch.tile(lx[:, None], (1, N_Y))       
        self.ell_y = torch.tile(ly[None, :N_Y], (N_grid, 1))
        self.ell = torch.sqrt(self.ell_x**2 + self.ell_y**2)
        self.ell[0,0] = 1.
        
        self.ell_x_full = torch.tile(lx[:, None], (1, N_grid))       
        self.ell_y_full = torch.tile(ly[None, :], (N_grid, 1))
        self.ell_full   = torch.sqrt(self.ell_x_full**2 + self.ell_y_full**2)
        self.ell_full[0,0] = 1.
        
        fourier_symm_mask = torch.ones((N_grid, self.N_Y), dtype=bool)
        fourier_symm_mask[(self.N_Y):,0]  = 0
        fourier_symm_mask[(self.N_Y):,-1] = 0
        fourier_symm_mask[0,0]            = 0
        self.fourier_symm_mask = fourier_symm_mask
        
        fourier_symm_mask_imag = fourier_symm_mask.clone()
        fourier_symm_mask_imag[0,-1]        = 0
        fourier_symm_mask_imag[self.N_Y-1,0]  = 0
        fourier_symm_mask_imag[self.N_Y-1,-1] = 0
        self.fourier_symm_mask_imag = fourier_symm_mask_imag
        
        fourier_symm_flip_ind = torch.arange(N_grid)
        fourier_symm_flip_ind[1:] = torch.flip(fourier_symm_flip_ind[1:], dims=[0])
        self.fourier_symm_flip_ind = fourier_symm_flip_ind

    def set_fourier_plane_face(self, F_x, x):
        N = self.N_grid
        F_x[:,:,1:-1] = x[:N**2 - 2*N].view(2, N, N//2 - 1)
        return F_x

    def set_fourier_plane_edge(self, F_x, x):
        N = self.N_grid
        N_Y = N//2 + 1
        N_edge = N//2 - 1    
        
        F_x[:,1:N_Y-1,0]  = x[N**2 - 2*N:N**2 - 2*N+2*N_edge].view(2, -1)
        F_x[:,1:N_Y-1,-1] = x[N**2 - 2*N+2*N_edge:-3].view(2, -1)
        return F_x

    def set_fourier_plane_corner(self, F_x, x):    
        N = self.N_grid
        N_Y = N//2 + 1
               
        F_x[0,N_Y-1,-1] = x[-3]
        F_x[0,0,-1]     = x[-2]
        F_x[0,N_Y-1,0]  = x[-1]
        return F_x
    
    def array2fourier_plane(self, x):
        N = self.N_grid
        N_Y = N//2 + 1
        N_edge = N//2 - 1    

        F_x_plane = torch.zeros((2, N, N_Y))
        F_x_plane = self.set_fourier_plane_face(F_x_plane, x)
        F_x_plane = self.set_fourier_plane_edge(F_x_plane, x)
        F_x_plane = self.set_fourier_plane_corner(F_x_plane, x)

        F_x_plane = self.symmetrize_fourier(F_x_plane)        
        return F_x_plane
    
    def fourier_plane2array(self, Fx):
        N = self.N_grid
        N_Y = N // 2 + 1
        N_edge = N // 2 - 1    
    
        x = torch.zeros(N * N - 1)
    
        x[:N**2 - 2 * N] = Fx[:, :, 1:-1].reshape(-1)
        x[N**2 - 2 * N:N**2 - 2 * N + 2 * N_edge] = Fx[:, 1:N_Y - 1, 0].reshape(-1)
        x[N**2 - 2 * N + 2 * N_edge:-3] = Fx[:, 1:N_Y - 1, -1].reshape(-1)
        
        x[-3] = Fx[0, N_Y - 1, -1]
        x[-2] = Fx[0, 0, -1]
        x[-1] = Fx[0, N_Y - 1, 0]
        
        return x
    
    def get_imag_indices(self):
        x0 = torch.zeros(self.N_grid**2 - 1)
        Fx = self.array2fourier_plane(x0)
        Fx[1] = 1

        x0 = self.fourier_plane2array(Fx).int()

        indices = torch.arange(x0.shape[0])
        imag_indices_1d = indices[(x0 == 1)]

        return imag_indices_1d


# In[194]:


def add_noise_to_shear(shear_map, std_map):
    """
    Adds Gaussian noise to a shear map while ensuring gradients are tracked.

    Parameters:
    shear_map (torch.Tensor): The input shear map with shape (256, 256).
    std_map (torch.Tensor, optional): A 256x256 tensor specifying the standard deviation at each pixel.

    Returns:
    torch.Tensor: The shear map with added Gaussian noise.
    """
    assert isinstance(shear_map, torch.Tensor), "shear_map must be a torch.Tensor"
    
    if not shear_map.requires_grad:
        shear_map.requires_grad = True

    noise = torch.randn_like(shear_map) * std_map
    noisy_shear_map = shear_map + noise

    # Ensure noisy_shear_map tracks gradients
    if not noisy_shear_map.requires_grad:
        noisy_shear_map.requires_grad = True

    return noisy_shear_map


# In[195]:


from PIL import Image
from torchvision import transforms

image_files = []
path = '/Users/michaeljacob/Diffusion/data/data_images_grey/' + f"WLconv_z2.00_{31:04d}r" + ".png"
image_files.append(path)
images = [Image.open(image_file) for image_file in image_files]
transform = transforms.Compose([transforms.ToTensor()])

# Apply the transformation
image_tensor = transform(images[0].convert('L')).squeeze(0)

def kappa_to_shear(kappa, N_grid = 256, theta_max = 12., J = 1j, EPS = 1e-20): 
    map_tool  = MapTools(N_grid, theta_max)
    kappa_fourier = map_tool.map2fourier(kappa)
    y_1, y_2   = map_tool.do_fwd_KS(kappa_fourier)
    y_1 = torch.from_numpy(y_1)
    y_2 = torch.from_numpy(y_2)
    shear_map = torch.stack((y_1, y_2))
    return shear_map

def torch_kappa_to_shear(kappa, N_grid = 256, theta_max = 12., J = 1j, EPS = 1e-20): 
    torch_map_tool  = TorchMapTools(N_grid, theta_max)
    kappa_fourier = torch_map_tool.map2fourier(kappa)
    y_1, y_2   = torch_map_tool.do_fwd_KS(kappa_fourier)
    shear_map = torch.stack((y_1, y_2))
    return shear_map

def torch_shear_to_kappa(shear, N_grid = 256, theta_max = 12., J = 1j, EPS = 1e-20): 
    torch_map_tool  = TorchMapTools(N_grid, theta_max)
    kappa = map_tool.do_KS_inversion(shear)
    return kappa
    
shear_1 = kappa_to_shear(image_tensor)
shear_2 = torch_kappa_to_shear(image_tensor)

plt.imshow(shear_1[0])
plt.show()
plt.imshow(shear_1[1])
plt.show()
plt.imshow(shear_2[0])
plt.show()
plt.imshow(shear_2[1])

print('Difference in First Shear', torch.sum(torch.abs(shear_1[0] - shear_2[0])))
print('Difference in Second Shear', torch.sum(torch.abs(shear_1[1] - shear_2[1])))


# In[196]:


def compute_complex_log_likelihood(noisy_pred, noisy_shear_map, std_map):
    # Extract real and imaginary parts
    y_1 = noisy_pred[0]
    y_2 = noisy_pred[1]

    y_1_sim = noisy_shear_map[0]
    y_2_sim = noisy_shear_map[1]

    # Compute MSE for real and imaginary parts separately
    real_mse = -0.5 * (y_1 - y_1_sim)**2 / std_map**2
    imag_mse = -0.5 * (y_2 - y_2_sim)**2 / std_map**2

    # Sum the MSEs for real and imaginary parts to get a real-valued loss
    mse = real_mse + imag_mse
    return (-1 * mse.sum())

input_tensor = torch.rand((256,256))
input_tensor.requires_grad = True

data = add_noise_to_shear(torch_kappa_to_shear(image_tensor), .1)
data = torch_kappa_to_shear(image_tensor)

learn = .01
optimizer = optim.Adam([input_tensor], lr=learn)

max_iter = 10000
best_img = None
best_loss = float("inf")

for epoch in range(1, max_iter + 1):
    optimizer.zero_grad()
    
    # Compute new coefficients while maintaining gradient flow

    shear_map = torch_kappa_to_shear(input_tensor)
    loss = compute_complex_log_likelihood(shear_map, data, .1)
    if(epoch % 1000 == 0):
        print('Loss: ', loss.detach().numpy())
    # Perform optimization step
    loss.backward()
    optimizer.step()
    # Track the best image and loss
    if loss < best_loss:
        best_loss = loss.item()
        best_img = input_tensor.clone().detach().cpu().numpy()


plt.imshow(best_img)


# In[197]:


kappa_KS = torch_shear_to_kappa(data)


# In[203]:


def plot_images(original, image2, image3, image4):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    
    # Set a common color map and normalize the images to have the same color range
    vmin = original.min()
    vmax = original.max()
    
    # Plot the first image
    cax1 = axs[0][0].imshow(original, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0][0].set_title('Original Kappa')
    axs[0][0].axis('off')  # Hide the axis
    
    # Plot the second image
    cax2 = axs[1][0].imshow(image2, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1][0].set_title('Adam Optimizer')
    axs[1][0].axis('off')  # Hide the axis
    
    # Plot the third image
    cax2 = axs[0][1].imshow(image2, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0][1].set_title('Reverse Kaiser Squires')
    axs[0][1].axis('off')  # Hide the axis
    
    # Plot the third image
    cax2 = axs[1][1].imshow(image2, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1][1].set_title('PLACEHOLDER IGNORE')
    axs[1][1].axis('off')  # Hide the axis
    
    
    
    # Add a color bar with a common scale
    fig.colorbar(cax1, ax=axs, orientation='vertical', fraction=.1)
    
    plt.show()

image_tensor_alt = image_tensor - image_tensor.mean()

best_img_alt = best_img - best_img.mean()
plot_images(image_tensor_alt, best_img_alt, kappa_KS, kappa_KS)

def printStats(image):
    if isinstance(image, np.ndarray):
       image = torch.tensor(image)
    print('Mean: ', torch.mean(image))
    print('Std: ', image.std())
    print('Max: ', torch.max(image))
    print('Min: ', torch.min(image))

printStats(image_tensor_alt)
printStats(best_img_alt)
printStats(kappa_KS)


# In[186]:


image_tensor.std(), kappa_KS.std(), best_img.std()


# In[ ]:




