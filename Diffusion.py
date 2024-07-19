#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#Hyperparameters varied are UNET structure(dim = 16, mults = (1,2,4)), and sampling_timesteps(250 vs 500 vs 1000)

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import numpy as np
from torchvision import transforms
from PIL import Image
import psutil


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False, 
    channels = 1
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 250
).cuda()

trainer = Trainer(
    diffusion,
    '/Users/michaeljacob/Diffusion/png_data/',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 70000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False # whether to calculate fid during training
)

trainer.train()


