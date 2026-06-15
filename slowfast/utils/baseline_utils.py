#!/usr/bin/env python3
"""Utility functions for baseline gaze prediction methods."""

import json
import torch
import numpy as np


def create_gaussian_heatmap(height, width, center_x, center_y, sigma):
    """
    Create a 2D Gaussian heatmap centered at (center_x, center_y).

    Args:
        height (int): Height of the heatmap
        width (int): Width of the heatmap
        center_x (float): X coordinate of the center (normalized 0-1 or pixel coords)
        center_y (float): Y coordinate of the center (normalized 0-1 or pixel coords)
        sigma (float): Standard deviation of the Gaussian in pixels

    Returns:
        torch.Tensor: Gaussian heatmap of shape (height, width)
    """
    # Convert normalized coordinates to pixel coordinates if needed
    if center_x <= 1.0 and center_y <= 1.0:
        center_x = center_x * width
        center_y = center_y * height

    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

    # Compute Gaussian
    gaussian = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))

    return gaussian


def generate_random_baseline_heatmaps(batch_size, num_frames, height, width, sigma, device='cpu'):
    """
    Generate random baseline heatmaps with Gaussian blobs at random locations.

    Args:
        batch_size (int): Batch size
        num_frames (int): Number of frames
        height (int): Heatmap height
        width (int): Heatmap width
        sigma (float): Gaussian standard deviation in pixels
        device (str): Device to create tensors on

    Returns:
        torch.Tensor: Random heatmaps of shape (batch_size, 1, num_frames, height, width)
    """
    heatmaps = torch.zeros(batch_size, num_frames, height, width, device=device)

    for b in range(batch_size):
        for f in range(num_frames):
            # Generate random center point (normalized coordinates)
            rand_x = torch.rand(1).item()
            rand_y = torch.rand(1).item()

            # Create Gaussian heatmap
            gaussian = create_gaussian_heatmap(height, width, rand_x, rand_y, sigma)
            heatmaps[b, f] = gaussian.to(device)

    # Add channel dimension: (B, T, H, W) -> (B, 1, T, H, W)
    return heatmaps.unsqueeze(1)


def generate_center_baseline_heatmaps(batch_size, num_frames, height, width, sigma, device='cpu'):
    """
    Generate center baseline heatmaps with Gaussian blobs at frame center (0.5, 0.5).

    Args:
        batch_size (int): Batch size
        num_frames (int): Number of frames
        height (int): Heatmap height
        width (int): Heatmap width
        sigma (float): Gaussian standard deviation in pixels
        device (str): Device to create tensors on

    Returns:
        torch.Tensor: Center heatmaps of shape (batch_size, 1, num_frames, height, width)
    """
    # Create single center Gaussian heatmap
    center_gaussian = create_gaussian_heatmap(height, width, 0.5, 0.5, sigma)

    # Replicate for all batches and frames: (H, W) -> (B, 1, T, H, W)
    heatmaps = center_gaussian.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, num_frames, 1, 1)

    return heatmaps.to(device)


def load_dataset_prior(prior_path, dataset_name):
    """
    Load precomputed dataset gaze prior from JSON file.

    Args:
        prior_path (str): Path to JSON file with dataset priors
        dataset_name (str): Name of the dataset

    Returns:
        dict: Dictionary with keys 'mean_x', 'mean_y', 'std_x', 'std_y'
    """
    with open(prior_path, 'r') as f:
        priors = json.load(f)

    if dataset_name not in priors:
        raise ValueError(f"Dataset '{dataset_name}' not found in prior file {prior_path}")

    return priors[dataset_name]


def generate_dataset_prior_heatmaps(batch_size, num_frames, height, width, sigma,
                                     prior_x, prior_y, device='cpu'):
    """
    Generate dataset prior baseline heatmaps with Gaussian blobs at the prior location.

    Args:
        batch_size (int): Batch size
        num_frames (int): Number of frames
        height (int): Heatmap height
        width (int): Heatmap width
        sigma (float): Gaussian standard deviation in pixels
        prior_x (float): Prior mean x coordinate (normalized 0-1)
        prior_y (float): Prior mean y coordinate (normalized 0-1)
        device (str): Device to create tensors on

    Returns:
        torch.Tensor: Prior heatmaps of shape (batch_size, 1, num_frames, height, width)
    """
    # Create single prior Gaussian heatmap
    prior_gaussian = create_gaussian_heatmap(height, width, prior_x, prior_y, sigma)

    # Replicate for all batches and frames: (H, W) -> (B, 1, T, H, W)
    heatmaps = prior_gaussian.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, num_frames, 1, 1)

    return heatmaps.to(device)
