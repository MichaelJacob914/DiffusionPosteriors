#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import Statements
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from scipy.interpolate import make_interp_spline
from astropy.io import fits
from PIL import Image
from torchvision import transforms
from scipy.stats import gaussian_kde


# In[3]:


class PowerSpectrumCalculator:
    def __init__(self, N_grid, L):
        self.set_map_properties(N_grid, L)
        self.set_fft_properties(N_grid, L)
            
    def set_map_properties(self, N_grid, L):
        self.N_grid   = N_grid
        self.L        = L
        self.Area     = self.L**2          # Area of the map  
        self.PIX_AREA = self.Area / self.N_grid**2

    def set_fft_properties(self, N_grid, L):
        kx = 2 * np.pi * np.fft.fftfreq(N_grid, d=L / N_grid)
        ky = 2 * np.pi * np.fft.fftfreq(N_grid, d=L / N_grid)

        self.N_Y = (N_grid//2 +1)
        
        # mesh of the 2D frequencies
        self.kx = np.tile(kx[:, None], (1, self.N_Y))
        self.ky = np.tile(ky[None, :self.N_Y], (N_grid, 1))
        self.k  = np.sqrt(self.kx**2 + self.ky**2)

        self.kmax = self.k.max()
        self.kmin = np.sort(self.k.flatten())[1]
        
        fourier_symm_mask = np.ones((N_grid, self.N_Y))
        fourier_symm_mask[(self.N_Y):,0]  = 0
        fourier_symm_mask[(self.N_Y):,-1] = 0
        fourier_symm_mask[0,0]            = 0
        self.fourier_symm_mask = fourier_symm_mask.astype(bool)
        
    def map2fourier(self, x_map):
        Fx_complex =  self.PIX_AREA * np.fft.rfftn(x_map) / self.Area
        return np.array([Fx_complex.real, Fx_complex.imag])

    def get_k_bins(self, N_bins):
        return np.logspace(np.log10(self.kmin), np.log10(self.kmax), N_bins)
    
    def set_k_bins(self, N_bins):
        self.k_bins = self.get_k_bins(N_bins)

    def binned_Pk(self, delta1, delta2=None):
        cross_Pk_bins = []
        k_bin_centre = []
        delta_ell_1 = self.map2fourier(delta1)
        if delta2 is not None:
            delta_ell_2 = self.map2fourier(delta2)
        else:
            delta_ell_2 = delta_ell_1
        for i in range(len(self.k_bins) - 1):
            select_k = (self.k > self.k_bins[i]) & (self.k < self.k_bins[i+1]) & self.fourier_symm_mask
            k_bin_centre.append(np.mean(self.k[select_k]))
            # The factor of 2 needed because there are both real and imaginary modes in the l selection!
            cross_Pk = 2. * np.mean(delta_ell_1[:,select_k] * delta_ell_2[:,select_k]) * self.Area
            cross_Pk_bins.append(cross_Pk)


        """
        
        # Check for NaN or inf values and replace them with zeros
        invalid_indices = np.isnan(cross_Pk_bins) | np.isinf(cross_Pk_bins)
        if np.any(invalid_indices):
            print(cross_Pk_bins[invalid_indices])
            cross_Pk_bins[invalid_indices] = 0.0
        

        return k_bin_centre, cross_Pk_bins
        """
        return np.array(k_bin_centre), np.array(cross_Pk_bins)


# In[9]:


class ImageAnalysis: 
    """
    This is a class to streamline the process of calculating specific statistical metrics for a field, 
    specifically the power spectrum, pdf, peak counts, and void counts. 
    """
    
    def __init__(self, field, field_length, mask = np.ones((256, 256), dtype=bool), is_tensor = True):
        if(is_tensor):
            self.field = field.detach().cpu().numpy()
        else:
            self.field = field
        print(self.field.shape)
        self.field_length = field_length #This is a measure of the length of any individual field, not the number of fields produced
        self.mask = mask
        self.is_tensor = is_tensor
    
    def get_neighbor_maps(self, flat_map):
        print(flat_map.shape)
        n, m = flat_map.shape
        neighbor_maps = []
        
        # Define the shifts for neighbors (8 directions)
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Top-Left, Top-Right, Bottom-Left, Bottom-Right
    
        for dx, dy in shifts:
            shifted_map = np.zeros_like(flat_map)
            for i in range(n):
                for j in range(m):
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < n and 0 <= nj < m:
                        shifted_map[i, j] = flat_map[ni, nj]
                    else:
                        shifted_map[i, j] = 0  # Or some other boundary value
            neighbor_maps.append(shifted_map)
        
        return np.array(neighbor_maps)

    def get_kappa_peaks(self,flat_map, mask):
        neighbor_maps = self.get_neighbor_maps(flat_map)
        max_neighbour_map = np.max(neighbor_maps, axis=0)
        select_peaks = (flat_map > max_neighbour_map) & mask
        return flat_map[select_peaks]
    
    def get_kappa_voids(self,flat_map, mask):
        neighbor_maps = self.get_neighbor_maps(flat_map)
        max_neighbour_map = np.max(neighbor_maps, axis=0)
        select_peaks = (flat_map < max_neighbour_map) & mask
        return flat_map[select_peaks]
        
    def plot_kde_peaks(self):
        all_peaks = self.get_kappa_peaks(self.field, self.mask)
        plt.figure(figsize=(10, 6))
        sns.kdeplot(all_peaks, fill=None, alpha=0.7, linestyle = 'dashed')
        plt.title('Kappa Peaks', fontsize=18)
        plt.xlabel('Peak Value', fontsize=15)
        plt.ylabel('Density', fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def plot_kde_voids(self):
        all_peaks = self.get_kappa_voids(self.field, self.mask)
        plt.figure(figsize=(10, 6))
        sns.kdeplot(all_peaks, fill=None, alpha=0.7, linestyle = 'dashed')
        plt.title('Kappa Voids', fontsize=18)
        plt.xlabel('Void Value', fontsize=15)
        plt.ylabel('Density', fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def plot_PDF(self):
        # Flatten the data to 1D array
        data_flattened = self.field.flatten()
        # Plotting the PDF using seaborn
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data_flattened, fill=None, alpha=0.7, linestyle = 'dashed')
        plt.title('Smooth Probability Density Function (PDF)', fontsize=18)
        plt.xlabel('Values', fontsize=15)
        plt.ylabel('Density', fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    # Main function to calculate and plot power spectrum
    def plot_power_spectrum(self,N_grid = 256):
        # Define parameters
          # Number of grid points along one dimension of the map
        L = 3.5     # Length of the square map (assuming units)
    
        # Load or generate flat sky map
    
        # Instantiate PowerSpectrumCalculator
        psc = PowerSpectrumCalculator(N_grid, L)
        psc.set_k_bins(50)
        
        # Calculate binned power spectrum
        ps = []
        k_bin_centre, cross_Pk_bins = psc.binned_Pk(self.field)
        #k_bin_centre = np.nan_to_num(k_bin_centre, nan=0.0, posinf=0.0, neginf=0.0)
        #cross_Pk_bins = np.nan_to_num(cross_Pk_bins, nan=0.0, posinf=0.0, neginf=0.0) 
        plt.loglog(k_bin_centre, cross_Pk_bins, marker='o', linestyle='-', color='b')
        plt.title('Power Spectrum of Flat Sky Map', fontsize=18)
        plt.xlabel('k (Spatial Frequency)', fontsize=15)
        plt.ylabel('k * P(k) (Power Spectrum)', fontsize=15)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.show()
    


# In[15]:


class FieldAnalysis: 
    """This is a class to handle comparisons of many fields"""
    
    def __init__(self, field_length, mask = np.ones((256, 256), dtype=bool), is_tensor = True):
        self.is_tensor = is_tensor
        self.field_length = field_length #This is a measure of the length of any individual field, not the number of fields produced
        self.mask = mask

    def plot_median_power_spectrum(self, fields, N_grid = 256, comparison_fields = None):
        # Instantiate PowerSpectrumCalculator
        L = 3.5
        psc = PowerSpectrumCalculator(N_grid, L)
        psc.set_k_bins(50)
    
        # Calculate power spectrum for each field
        all_power_spectra = []
    
        for field in fields:
            k_bin_centre, cross_Pk_bins = psc.binned_Pk(field)
            all_power_spectra.append(cross_Pk_bins)
    
        # Convert list to numpy array for easier median and percentile calculation
        all_power_spectra = np.array(all_power_spectra)
    
        # Calculate median power spectrum
        median_power_spectrum = np.median(all_power_spectra, axis=0)
        median_power_spectrum = median_power_spectrum * k_bin_centre
        # Calculate 2.5th and 97.5th percentiles
        lower_bound = np.percentile(all_power_spectra, 2.5, axis=0) * k_bin_centre
        upper_bound = np.percentile(all_power_spectra, 97.5, axis=0) * k_bin_centre
        if(comparison_fields != None):
            all_power_spectra_comp = []
            for field in comparison_fields:
                k_bin_centre, cross_Pk_bins = psc.binned_Pk(field)
                all_power_spectra_comp.append(cross_Pk_bins)
                
            # Convert list to numpy array for easier median and percentile calculation
            all_power_spectra_comp = np.array(all_power_spectra_comp)
        
            # Calculate median power spectrum
            median_power_spectrum_comp = np.median(all_power_spectra_comp, axis=0)
            median_power_spectrum_comp = median_power_spectrum_comp * k_bin_centre
            plt.loglog(k_bin_centre, median_power_spectrum_comp, marker='o', linestyle='-', color='r', label='Simulation Median Power Spectrum')

        
        # Plot median power spectrum
        plt.loglog(k_bin_centre, median_power_spectrum, marker='o', linestyle='-', color='b', label='Diffusion Sample Median Power Spectrum')
    
        
        # Fill between the upper and lower bounds
        plt.fill_between(k_bin_centre, lower_bound, upper_bound, color='blue', alpha=0.3, label='95% Confidence Interval')
    
        # Plot formatting
        plt.title('Median Power Spectrum with 95% Confidence Interval', fontsize=18)
        plt.xlabel('k (Spatial Frequency)', fontsize=15)
        plt.ylabel('kP(k) (Power Spectrum)', fontsize=15)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

    def get_neighbor_maps(self, flat_map):
        n, m = flat_map.shape
        neighbor_maps = []

        # Define the shifts for neighbors (8 directions)
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Top-Left, Top-Right, Bottom-Left, Bottom-Right

        for dx, dy in shifts:
            shifted_map = np.zeros_like(flat_map)
            for i in range(n):
                for j in range(m):
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < n and 0 <= nj < m:
                        shifted_map[i, j] = flat_map[ni, nj]
                    else:
                        shifted_map[i, j] = 0  # Or some other boundary value
            neighbor_maps.append(shifted_map)

        return np.array(neighbor_maps)

    def get_kappa_peaks(self, flat_map, mask):
        neighbor_maps = self.get_neighbor_maps(flat_map)
        max_neighbour_map = np.max(neighbor_maps, axis=0)
        select_peaks = (flat_map > max_neighbour_map) & mask
        return flat_map[select_peaks]

    def plot_median_kde_peaks(self, fields, comparison_fields=None):
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # Create two subplots side by side

        def plot_kde_peaks(ax, fields, label, color, confidence_interval = False):
            all_peaks = []
            for field in fields:
                peaks = self.get_kappa_peaks(field, self.mask)
                all_peaks.append(peaks)
            all_peaks = np.concatenate(all_peaks)
        
            all_kdes = []
            x_grid = np.linspace(0, np.max(all_peaks), 1000)
            for field in fields:
                peaks = self.get_kappa_peaks(field, self.mask)
                if len(peaks) > 0:
                    kde = gaussian_kde(peaks)
                    all_kdes.append(kde(x_grid))
        
            all_kdes = np.array(all_kdes)
            median_kde = np.median(all_kdes, axis=0)
            lower_bound = np.percentile(all_kdes, 2.5, axis=0)
            upper_bound = np.percentile(all_kdes, 97.5, axis=0)
            
            ax.plot(x_grid, median_kde, label=label, color=color)
            if(confidence_interval):
                ax.fill_between(x_grid, lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')
            ax.set_title('Median Peak Counts with 95% Confidence Interval', fontsize=18)
            ax.set_xlabel('Peak Value', fontsize=15)
            ax.set_ylabel('Density', fontsize=15)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        # Plot the KDE peaks for comparison_fields if provided
        if comparison_fields is not None:
            plot_kde_peaks(axs[0], comparison_fields, 'Simulation Median', 'red')
            plot_kde_peaks(axs[1], comparison_fields, 'Simulation Median', 'red')
    
        # Plot the KDE peaks for the main fields
        plot_kde_peaks(axs[0], fields, 'Diffusion Sample Median Peaks', 'blue', confidence_interval = True)
        plot_kde_peaks(axs[1], fields, 'Diffusion Sample Median Peaks', 'blue', confidence_interval = True)
        
        # Adjust y-axis for the first plot to be on a log scale
        axs[0].set_yscale('log')
        axs[0].set_ylim(1e-10, 1e+1)  # Set y-axis limits for the log plot
    
        plt.show()

    def get_kappa_voids(self,flat_map, mask):
        neighbor_maps = self.get_neighbor_maps(flat_map)
        min_neighbour_map = np.min(neighbor_maps, axis=0)
        select_voids = (flat_map[1:-1,1:-1] < min_neighbour_map[1:-1,1:-1]) #& mask
        return (flat_map[1:-1,1:-1])[select_voids]
        
    def plot_median_kde_voids(self, fields, comparison_fields = None):
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # Create two subplots side by side

        def plot_kde_voids(ax, fields, label, color, confidence_interval = False):
            all_voids = []
            for field in fields:
                voids = self.get_kappa_voids(field, self.mask)
                all_voids.append(voids)
            all_voids = np.concatenate(all_voids)
        
            all_kdes = []
            x_grid = np.linspace(0, np.max(all_voids), 1000)
            for field in fields:
                voids = self.get_kappa_voids(field, self.mask)
                if len(voids) > 0:
                    kde = gaussian_kde(voids)
                    all_kdes.append(kde(x_grid))
        
            all_kdes = np.array(all_kdes)
            median_kde = np.median(all_kdes, axis=0)
            lower_bound = np.percentile(all_kdes, 2.5, axis=0)
            upper_bound = np.percentile(all_kdes, 97.5, axis=0)
            
            ax.plot(x_grid, median_kde, label=label, color=color)
            if(confidence_interval):
                ax.fill_between(x_grid, lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')
            ax.set_title('Median Void Counts with 95% Confidence Interval', fontsize=18)
            ax.set_xlabel('Void Value', fontsize=15)
            ax.set_ylabel('Density', fontsize=15)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        # Plot the KDE voids for comparison_fields if provided
        if comparison_fields is not None:
            plot_kde_voids(axs[0], comparison_fields, 'Simulation Median', 'red')
            plot_kde_voids(axs[1], comparison_fields, 'Simulation Median', 'red')
    
        # Plot the KDE voids for the main fields
        plot_kde_voids(axs[0], fields, 'Diffusion Sample Median Voids', 'blue', confidence_interval = True)
        plot_kde_voids(axs[1], fields, 'Diffusion Sample Median Voids', 'blue', confidence_interval = True)
        
        # Adjust y-axis for the first plot to be on a log scale
        axs[0].set_yscale('log')
        axs[0].set_ylim(1e-6, 16)  # Set y-axis limits for the log plot
    
        plt.show()

    def plot_median_pdf(self, fields, comp_fields=None):
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # Create two subplots side by side
    
        def plot_pdf(ax, fields, label, color):
            flattened_fields = [field.flatten() for field in fields]
            all_kdes = []
            x_grid = np.linspace(np.min(flattened_fields), np.max(flattened_fields), 1000)
            for field in flattened_fields:
                kde = gaussian_kde(field)
                all_kdes.append(kde(x_grid))
            all_kdes = np.array(all_kdes)
            median_kde = np.median(all_kdes, axis=0)
            lower_bound = np.percentile(all_kdes, 2.5, axis=0)
            upper_bound = np.percentile(all_kdes, 97.5, axis=0)
            
            ax.plot(x_grid, median_kde, label=label, color=color)
            ax.fill_between(x_grid, lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')
            ax.set_title('Median PDF with 95% Confidence Interval', fontsize=18)
            ax.set_xlabel('Value', fontsize=15)
            ax.set_ylabel('Density', fontsize=15)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
    
        if comp_fields is not None:
            plot_pdf(axs[0], comp_fields, 'Simulation Median PDF', 'red')
            plot_pdf(axs[1], comp_fields, 'Simulation Median PDF', 'red')
    
        plot_pdf(axs[0], fields, 'Diffusion Sample Median PDF', 'blue')
        plot_pdf(axs[1], fields, 'Diffusion Sample Median PDF', 'blue')
        
        axs[0].set_yscale('log')
        axs[0].set_ylim(1e-10, 1e+1)  # Set y-axis limits for the log plot
    
        plt.show()


# In[19]:


def get_image(image_path):
    # Open the image
    image_path = image_path + '.png'
    with Image.open(image_path) as img:
        # Get the dimensions
        width, height = img.size
        #print(f"Width: {width}, Height: {height}")
        
        # Define the transformations: Convert to grayscale and then to tensor
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        # Apply the transformations to the image
        img_tensor = transform(img)
        
        # Print the shape of the tensor
        
        return img_tensor.squeeze(0)

def get_all_slices(img_tensor, grid_size=5, slice_size=256, padding=2):
    slices = []
    for i in range(grid_size):
        for j in range(grid_size):
            start_x = i * (slice_size + padding) + padding
            start_y = j * (slice_size + padding) + padding
            slice = img_tensor[start_x:start_x + slice_size, start_y:start_y + slice_size]
            slices.append(slice)
    return slices

def preprocess(list):
    for i in range(len(list)):
        list[i] = list[i].detach().cpu().numpy()
    return list

comp_fields = []
for i in range(30, 55):
    path = '/Users/michaeljacob/Diffusion/data/data_images_grey/'
    img_tensor = get_image(path + f"WLconv_z2.00_{i:04d}r")
    comp_fields.append(img_tensor)

#plot_median_power_spectrum(fields, 256, comparison_fields = comp_fields)

# Path to your image
image_path = '/Users/michaeljacob/Diffusion/results/NET16-sample-2'
img_tensor = get_image(image_path)
fields = get_all_slices(img_tensor)


# Path to your image
image_path = '/Users/michaeljacob/Diffusion/results/sample-60'
img_tensor = get_image(image_path)
fields_new = get_all_slices(img_tensor)


field_length = 256
fa = FieldAnalysis(field_length, mask = np.ones((field_length, field_length), dtype=bool), is_tensor = False)
fa.plot_median_power_spectrum(fields, 256, comparison_fields = comp_fields)
fa.plot_median_power_spectrum(fields_new, 256, comparison_fields = comp_fields)
fa.plot_median_pdf(fields, comp_fields)
fa.plot_median_pdf(fields_new, comp_fields)
fields = preprocess(fields)
comp_fields = preprocess(comp_fields)
fields_new = preprocess(fields_new)
fa.plot_median_kde_peaks(fields, comp_fields)
fa.plot_median_kde_peaks(fields_new, comp_fields)
fa.plot_median_kde_voids(fields, comp_fields)
fa.plot_median_kde_voids(fields_new, comp_fields)


# In[ ]:


def arrange_images(images, grid_size=(5, 5), padding=10, bg_color=(0, 0, 0)):
    # Number of images
    num_images = len(images)
    assert num_images == grid_size[0] * grid_size[1], "Number of images must match the grid size"

    # Get dimensions of a single image
    img_width, img_height = images[0].size

    # Calculate the dimensions of the full grid image
    grid_width = grid_size[0] * img_width + (grid_size[0] + 1) * padding
    grid_height = grid_size[1] * img_height + (grid_size[1] + 1) * padding

    # Create a new image with the calculated dimensions and a white background
    grid_image = Image.new('RGB', (grid_width, grid_height), bg_color)

    # Paste each image into the grid image
    for idx, img in enumerate(images):
        row = idx // grid_size[0]
        col = idx % grid_size[0]
        x = padding + col * (img_width + padding)
        y = padding + row * (img_height + padding)
        grid_image.paste(img, (x, y))

    return grid_image

image_files = []
for i in range(30, 55):
    path = '/Users/michaeljacob/Diffusion/data/data_images_grey/' + f"WLconv_z2.00_{i:04d}r" + ".png"
    image_files.append(path)
images = [Image.open(image_file) for image_file in image_files]
grid_image = arrange_images(images, grid_size=(5, 5), padding=2, bg_color=(0, 0, 0))

grid_image.save("grid_image_2.png")


# In[ ]:




