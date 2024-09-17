#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import os as os
from PIL import Image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from matplotlib.animation import FuncAnimation
import torchvision.transforms as transforms
import imageio
from FieldAnalysis import PowerSpectrumCalculator, FieldCorrelations
from MapTools import TorchMapTools
import math


def compare_tensors_with_tolerance(tensor1, tensor2, tolerance=.0039):
    difference = torch.abs(tensor1 - tensor2)
    difference_with_tolerance = torch.where(difference <= tolerance, torch.tensor(0.0), difference)
    print(difference_with_tolerance)

def torch_kappa_to_shear(kappa, N_grid = 256, theta_max = 12., J = 1j, EPS = 1e-20): 
    torch_map_tool  = TorchMapTools(N_grid, theta_max)
    kappa_fourier = torch_map_tool.map2fourier(kappa)
    y_1, y_2   = torch_map_tool.do_fwd_KS(kappa_fourier)
    shear_map = torch.stack((y_1, y_2))
    return shear_map

def torch_shear_to_kappa(shear, N_grid = 256, theta_max = 12., J = 1j, EPS = 1e-20): 
    torch_map_tool  = TorchMapTools(N_grid, theta_max)
    kappa = torch_map_tool.do_KS_inversion(shear)
    return kappa

def add_noise_to_shear(shear_map, std_map):
    if not shear_map.requires_grad:
        shear_map.requires_grad = True

    noise = torch.randn_like(shear_map) * std_map
    noisy_shear_map = shear_map + noise

    # Ensure noisy_shear_map tracks gradients
    if not noisy_shear_map.requires_grad:
        noisy_shear_map.requires_grad = True

    return noisy_shear_map

def neff2noise(neff, pix_area):
    """
    :neff: Effective number density of galaxies per arcmin^2
    :pix_area: pixel area in arcmin^2
    """
    N = neff * pix_area    # avg. number of galaxies per pixel
    sigma_e = 0.26      # avg. shape noise per galaxy
    total_noise = sigma_e / math.sqrt(N)
    return total_noise

def unnorm_kappa(kappa, kappa_min = -0.08201675, kappa_max = 0.7101586):
    kappa_unnorm = (kappa * (kappa_max - kappa_min)) + kappa_min
    return kappa_unnorm


#Hyperparameters
Delta_theta = 3.5 / 256 * 60.      # Pixel side in arcmin
pix_area    = Delta_theta**2
ddim_sampling_eta = 1
zeta = .5
batch_size = 1
neff = 10
type_of_output = 1 #This specifies whether to give a statistics plot, the kappa map and corresponding shear maps, or video

sigma_noise = neff2noise(neff, pix_area)
print('Noise', sigma_noise)
if(type_of_output == 3):
    return_all_timesteps= True 
else:
    return_all_timesteps = False

#Prep Noisy Data Measurement
path = '/home2/mgjacob/Diffusion/data/data_images_grey/WLconv_z2.00_0002r.png'

normed_map = Image.open(path)
transform = transforms.Compose([transforms.ToTensor()])

# Apply the transformation
normed_map = transform(normed_map.convert('L')).squeeze(0)
kappa_map = normed_map #Unnorm kappa map normally

kappa_map.to('cuda:0')
kappa_map.requires_grad = False

shear_map = torch_kappa_to_shear(kappa_map)
noisy_shear_map = add_noise_to_shear(shear_map, sigma_noise)
noisy_shear_map = noisy_shear_map.detach()

KS_inverse = torch_shear_to_kappa(noisy_shear_map)

name = "Conditioning"  # Replace 'NAME' with the desired folder name
samples_root = os.path.join("./samples", name)
os.makedirs(samples_root, exist_ok=True)
len_samples = len(os.listdir(samples_root))

current_image = Image.fromarray((noisy_shear_map[0].detach().cpu().numpy() * 255).astype('uint8').squeeze(), mode='L')
file_name = f"Target_gamma_1_noisy.png"
current_image.save(os.path.join(samples_root, file_name))

current_image = Image.fromarray((noisy_shear_map[1].detach().cpu().numpy() * 255).astype('uint8').squeeze(), mode='L')
file_name = f"Target_gamma_2_noisy.png"
current_image.save(os.path.join(samples_root, file_name))


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False, 
    channels = 1
).cuda()

noisy_shear_map = noisy_shear_map.unsqueeze(0)
diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 1000, 
    noisy_image = noisy_shear_map,
    sigma_noise = sigma_noise,
    ddim_sampling_eta = ddim_sampling_eta
).cuda()

trainer = Trainer(
    diffusion,
    '/home2/mgjacob/Diffusion/data/single_channel/data_images_grey',
    train_batch_size = 16,
    train_lr = 8e-5,
    save_and_sample_every = 20000,
    num_samples = 100, 
    train_num_steps = 200000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False              # whether to calculate fid during training
)

trainer.load('3-NORM')

sampled_images_posterior = diffusion.sample_posterior(batch_size = batch_size, return_all_timesteps= return_all_timesteps)
x, y, *_ = sampled_images_posterior.shape


sampled_images = diffusion.sample(batch_size = 4, return_all_timesteps = False)
sampled_images_trimmed = []
for i in sampled_images: 
    prenorm = i.detach().cpu().squeeze()
    normed = (prenorm - prenorm.min())/(prenorm.max() - prenorm.min())
    sampled_images_trimmed.append(i.detach().cpu().squeeze())

name = "DPS"  # Replace 'NAME' with the desired folder name
samples_root = os.path.join("./samples", name)
os.makedirs(samples_root, exist_ok=True)
len_samples = len(os.listdir(samples_root))

if(type_of_output == 1):
    for i in range(sampled_images_posterior.size(0)):
        index = i
        current_image_tensor = sampled_images_posterior[i].detach().cpu().squeeze(0)
        #current_image_tensor = (current_image_tensor - kappa_map.min())/(kappa_map.max() - kappa_map.min())
        #current_image_tensor = (current_image_tensor - current_image_tensor.min())/(current_image_tensor.max() - current_image_tensor.min())
        diffusion_output = unnorm_kappa(current_image_tensor)
        file_name = f"comparison_plot_{index + len_samples}"
        file_name = os.path.join(samples_root, file_name)
        FieldCorrelations(current_image_tensor, kappa_map, 256, file_name, comp_fields = sampled_images_trimmed, comparison = True)#, KS_inverse = kappa_comparison)#, comparison = False)
elif(type_of_output == 2):
    for i in range(sampled_images.size(0)):
        index = int((len_samples - 5)/3)
        print('Saving images')
        current_image_tensor = sampled_images[i]
        print(current_image_tensor.shape)
        current_image_tensor = current_image_tensor.squeeze(0)
        current_image = Image.fromarray((current_image_tensor.detach().cpu().numpy() * 255).astype('uint8').squeeze(), mode='L')
        file_name = f"output_image_{index}.png"
        current_image.save(os.path.join(samples_root, file_name))
        current_image_tensor.to('cuda:0')

        current_image_tensor = current_image_tensor.squeeze(0)
        shear_map = torch_kappa_to_shear(current_image_tensor)
        
        gamma_1 = shear_map[0]
        gamma_2 = shear_map[1]
        gamma_1 = (gamma_1 - gamma_1.min()) / (gamma_1.max() - gamma_1.min())
        gamma_2 = (gamma_2 - gamma_2.min()) / (gamma_2.max() - gamma_2.min())
        current_image = Image.fromarray((gamma_1.detach().cpu().numpy() * 255).astype('uint8').squeeze(), mode='L')
        file_name = f"gamma_1_{index}.png"
        current_image.save(os.path.join(samples_root, file_name))
        
        current_image = Image.fromarray((gamma_2.detach().cpu().numpy() * 255).astype('uint8').squeeze(), mode='L')
        file_name = f"gamma_2_{index}.png"
        current_image.save(os.path.join(samples_root, file_name))
else:
    frames = []

    for i in range(y):
        current_image_tensor = sampled_images[0][i]
        current_image = Image.fromarray((current_image_tensor.detach().cpu().numpy() * 255).astype('uint8').squeeze(), mode='L')
        frames.append(current_image)

    print('Number of frames', len(frames))
    name = "Conditioning/"  # Replace 'NAME' with the desired folder name
    samples_root = os.path.join("./samples", name)
    output_video = samples_root + 'normed_conditioned_no_divide_output_video.mp4'
    fps = 10
    imageio.mimsave(output_video, frames, fps=fps)

print("All samples are saved in folder")
