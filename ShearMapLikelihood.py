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
from MapTools import TorchMapTools


# In[193]:


import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

J = 1j
EPS = 1e-20

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
path = 'PATH' + f"WLconv_z2.00_{31:04d}r" + ".png"
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




