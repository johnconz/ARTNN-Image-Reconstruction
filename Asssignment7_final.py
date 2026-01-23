# Connor Prikkel
# ECE 595 (Artificial Neural Networks)
# Assignment 7
# 11/19/24
# Reference and code from: "Single Layer Perceptroon Te3mplate.ipynb", "Assignment6.py (my code)", and "rle.m" https://www.mathworks.com/matlabcentral/fileexchange/31123-rle-run-length-encoding

# Used torch whenever possible to simplify
# handle imports
from __future__ import print_function     # for printing out information
import argparse                           # allows us to parse in information
import numpy as np
import torch                              # base tensor library
import torch.nn as nn                     # base neural network
import torch.nn.functional as F           # access to neural network components
import torch.optim as optim               # access to optimizers
from torchvision import datasets, transforms    # access to all of the computer vision datasets and loading
from torch.autograd import Variable       # base class for all data
from torch.utils.data import Subset
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from skimage.transform import resize            # access to resize fn
from skimage import data                        # access to imgs
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse      # access to fns for computing metrics
from sklearn.metrics import mean_squared_error



##############################################################################################################################
# NN and Other Methods Definitions
##############################################################################################################################

# Speed up computations by running on gpu
#torch.set_default_device('cuda')

# ART param
alpha = 0.001
vigilance = 0.9
lr = 0.75

# Def global var for smoll value
smoll_val = 1e-6

# Method to split image into blocks that fill space
# Each 'block' is treated as its own separate unit
def split_into_blocks(img, block_size):

    # Split image vertically along dim 0 and horizontally along dim 1
    img_blocks = img.unfold(0, block_size[0], block_size[0]).unfold(1, block_size[1], block_size[1])
    
    # Store num of blocks in each row and column
    num_row, num_col = img_blocks.shape[:2]

    # Flatten block into a 2D tensor of size n x n
    # Use '.contiguous()' to make a copy of tensor so its ele are in the expected loc in memory in spite of shape changes
    return img_blocks.contiguous().view(-1, block_size[0] * block_size[1]), num_row, num_col


# Method to reconstruct compressed img from encoded cluster_idx and codebook
def reconstruct_img(codebook, block_size, cluster_idx, num_row, num_col):

    h_block, w_block = block_size

    # Def reconstructed blocks based on cluster idx
    r_blocks = codebook[:, cluster_idx].T

    # Assemble recostructed img

    # Reshape blocks back to the image grid
    re_blocks = r_blocks.view(num_row, num_col, h_block, w_block)

    # Use '.permute()' to flip orientation of reconstructed img to match original 
    r_img = re_blocks.permute(0, 2, 1, 3).contiguous().view(num_row * block_size[0], num_col * block_size[1])

    # Ensure the reconstructed image shape matches the original to avoid size errors
    # Slice to original shape
    #r_img = r_img[:og_shape[0], :og_shape[1]]  

    return r_img
    

# Adaptive Resonance Theory (ART) Neural Network Def
# Easier to define as a method instead of a 'Net' for this application
def ARTNN(data, alpha, vigilance, lr, max_iter):

    num_samples, num_features = data.shape

    # Each column contains cluster's centroid- initially only one set to 0
    weights = torch.zeros(num_features, 1, dtype=torch.float32)
    cluster_idx = torch.full((num_samples,), -1, dtype=torch.long)

    # Start the outer loop for multiple training iterations
    for iteration in range(max_iter):
        # Save previous weights before updating
        prev_weights = weights.clone().detach()

        # Iterate thru all input samples
        for i in range(num_samples):
            xi = data[i]

            # Define choice function
            CFj = torch.sum(torch.min(weights, xi.unsqueeze(1)), dim=0) / (alpha + torch.sum(weights, dim=0))

            # Determine winner
            CFJ = torch.argmax(CFj)

            # Define vigilance function
            # Check vigilance fn denominator to ensure not dividing by 0
            denom_vf = torch.sum(xi)

            #if torch.isclose(denom_vf, torch.tensor(0.0)):
            #    denom_vf += smoll_val
            VFJ = torch.sum(torch.min(weights[:, CFJ], xi)) / denom_vf

            # While failing the vigilance test, try new nodes
            while VFJ < vigilance:
                if CFJ == weights.shape[1] - 1:
                    # Create a new cluster if no existing cluster satisfies vigilance
                    weights = torch.cat((weights, xi.unsqueeze(1)), dim=1)
                    break

                # Reset winner function
                CFj[CFJ] = 0
                CFJ = torch.argmax(CFj)
                VFJ = torch.sum(torch.min(weights[:, CFJ], xi)) / denom_vf

            # If the sample passes the test
            if VFJ >= vigilance:

                # Update the winning cluster's weights
                weights[:, CFJ] = lr * torch.min(weights[:, CFJ], xi) + (1 - lr) * weights[:, CFJ]
                cluster_idx[i] = CFJ

        # Check for convergence
        converged = True
        min_clusters = min(weights.shape[1], prev_weights.shape[1])

        # Compare only the overlapping clusters
        for i in range(min_clusters):
            if not torch.allclose(weights[:, i], prev_weights[:, i]):
                converged = False
                break

        # If the number of clusters differs, consider it as not converged
        if weights.shape[1] != prev_weights.shape[1]:
            converged = False

        if converged:
            print(f"Converged after {iteration + 1} iterations")
            break

    return cluster_idx, weights
    
# Method to compute RLE: Run-Length Encoding
# Based on 'rle.m'
def rle(seq):
    encoded_seq = []

    # Initialize var to track val being counted, set to first val of seq
    prev_val = seq[0]
    occurance = 1

    # Iterate thru the rest of the seq
    for current_val in seq[1:]:
    
        # If the current val is equal to the last occuranceed
        if current_val == prev_val:
            # The val has been seen before so increment occurance
            occurance += 1
        else:
            # Add prev val and its # of occurances
            encoded_seq.append([prev_val.item(), occurance])

            # Reset occurance # and prev_val to new value
            prev_val = current_val
            occurance = 1
    
    # Add the last run to encoded seq
    encoded_seq.append([prev_val.item(), occurance])
    return encoded_seq

# Method to compute RLD: Run-Length Decoding
def rld(seq, length):
    decoded_seq = []

    for current_val, occurance in seq:
        decoded_seq.extend([current_val] * occurance)

    return torch.tensor(decoded_seq[:length], dtype=torch.long)

# Method to compress images
def maniupulate_img(img, alpha, vigilance, lr, max_iter, block_size):

    # Convert to torch tensor
    img = torch.tensor(img, dtype=torch.float32)

    # Split img into equal size blocks
    blocks, num_row, num_col = split_into_blocks(img, block_size)

    print(f"Blocks Shape: {blocks.shape}, Num Rows: {num_row}, Num Cols: {num_col}")

    # Train on ARTNN
    cluster_idx, codebook = ARTNN(blocks, alpha, vigilance, lr, max_iter)

    # Plot cluster mapping using '.heatmap()'
    plt.figure()
    plt.title("Cluster Assignment Map")
    cluster_map = cluster_idx.view(num_row, num_col).numpy()
    sns.heatmap(cluster_map, annot=True, fmt="d", cmap="tab10")
    plt.show()

    #r_blocks = codebook[:, cluster_idx].T
    #print(f"Reconstructed Blocks Shape: {r_blocks.shape}")

    # Apply RLE
    encoded_blocks = rle(cluster_idx)
    #print(f"Encoded Blocks: {encoded_blocks}")

    # Apply RLD and decode the encoded idx
    decoded_idx = rld(encoded_blocks, len(cluster_idx))

    # Reconstruct the image
    r_img = reconstruct_img(codebook, block_size, decoded_idx, num_row, num_col)

    # Compute evaluation metrics and display them

    # '.numel()' returns the total # of pixels in an img
    og_pixels = img.numel()

    # Calculate compressed img size + compression ratio
    comp_pixels = codebook.numel() + len(encoded_blocks)
    comp_ratio = og_pixels / comp_pixels

    # Calculate the mse between og and reconstructed img
    mse_val = mean_squared_error(np.array(img), np.array(r_img))

    # Calculate peak signal-to-noise ratio
    psnr_val = psnr(np.array(img), np.array(r_img))

    print(f"Compression Ratio: {comp_ratio:2f}")
    print(f"MSE: {mse_val:.4f} ")
    print(f"PSNR: {psnr_val:.2f} dB")

    return img, r_img


##############################################################################################################################
# Test with Low, Medium, and High Complexity Images
##############################################################################################################################

# COMMENT OUT OTHER IMAGES- CAN ONLY RECONSTRUCT ONE AT A TIME
# 'resize()' automatically normalizes
# Low Complexity:
#image = resize(data.clock(), (256, 256), anti_aliasing=True)

# Medium Complexity:
#image = resize(data.eagle(), (256, 256), anti_aliasing=True)

# High Complexity:
image = resize(data.camera(), (256, 256), anti_aliasing=True)

og_img, r_img = maniupulate_img(image, alpha, vigilance, lr, 30, (4, 4))

# Convert to numpy array for plotting
og_img_np = np.array(og_img)
r_img_np = np.array(r_img)

# Display the images
plt.figure(figsize=(10, 5))
    
# Show original image
plt.subplot(1, 2, 1)
plt.imshow(og_img_np, cmap='gray')
plt.title("Original Image")
plt.axis("off")
    
# Show reconstructed image
plt.subplot(1, 2, 2)
plt.imshow(r_img_np, cmap='gray')
plt.title("Reconstructed Image")
plt.axis("off")
    
# Display both images
plt.show()



