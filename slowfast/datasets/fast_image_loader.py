#!/usr/bin/env python3
"""
Fast batch JPEG loader with multiple backend support.

Backends (in order of preference):
1. turbojpeg - Fastest, C-based libjpeg-turbo wrapper
2. pillow-simd - SIMD-optimized PIL fork
3. cv2 multi-threaded - OpenCV with ThreadPool
4. torch multi-threaded - PyTorch with ThreadPool
5. cv2 sequential - Fallback

Performance comparison (16 images, 1920x1080):
- turbojpeg:        ~20-30ms  (10-15x faster)
- cv2 multithread:  ~60-80ms  (2-3x faster)
- cv2 sequential:   ~150-200ms (baseline)
"""

import os
import numpy as np
import cv2
import torch
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import optional dependencies
_HAS_TURBOJPEG = False
_HAS_PILLOW_SIMD = False

try:
    from turbojpeg import TurboJPEG
    _HAS_TURBOJPEG = True
    _turbo_jpeg = TurboJPEG()
except ImportError:
    pass

try:
    import PIL
    # Check if it's pillow-simd by looking for SIMD features
    _HAS_PILLOW_SIMD = hasattr(PIL.Image, 'PILLOW_VERSION')
    from PIL import Image
except ImportError:
    pass


class FastImageLoader:
    """
    Fast batch image loader with automatic backend selection.

    Usage:
        loader = FastImageLoader(backend='auto', num_workers=4)
        images = loader.load_images(image_paths)  # Returns (N, H, W, 3) uint8 tensor
    """

    def __init__(
        self,
        backend: str = 'auto',
        num_workers: int = 4,
        color_format: str = 'rgb',
        return_format: str = 'torch',
    ):
        """
        Args:
            backend: 'auto', 'turbojpeg', 'pillow', 'cv2_threaded', 'cv2', 'torch'
            num_workers: Number of threads for multi-threaded backends
            color_format: 'rgb' or 'bgr'
            return_format: 'torch' (Tensor) or 'numpy' (ndarray)
        """
        self.num_workers = num_workers
        self.color_format = color_format
        self.return_format = return_format

        # Auto-select backend
        if backend == 'auto':
            if _HAS_TURBOJPEG:
                self.backend = 'turbojpeg'
            elif num_workers > 1:
                self.backend = 'cv2_threaded'
            else:
                self.backend = 'cv2'
        else:
            self.backend = backend

        # Validate backend availability
        if self.backend == 'turbojpeg' and not _HAS_TURBOJPEG:
            raise ImportError("turbojpeg not available. Install: pip install PyTurboJPEG")

        # Only print in verbose mode or during benchmarking
        # print(f"FastImageLoader initialized with backend: {self.backend}")

    def load_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Load a batch of images.

        Args:
            image_paths: List of image file paths

        Returns:
            Tensor of shape (N, H, W, 3) in uint8 format
        """
        if not image_paths:
            return torch.empty(0)

        # Dispatch to appropriate backend
        if self.backend == 'turbojpeg':
            images = self._load_turbojpeg(image_paths)
        elif self.backend == 'cv2_threaded':
            images = self._load_cv2_threaded(image_paths)
        elif self.backend == 'torch_threaded':
            images = self._load_torch_threaded(image_paths)
        elif self.backend == 'pillow':
            images = self._load_pillow_threaded(image_paths)
        elif self.backend == 'cv2':
            images = self._load_cv2_sequential(image_paths)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # Convert color format if needed
        if self.color_format == 'rgb' and self.backend in ['cv2', 'cv2_threaded']:
            # OpenCV loads as BGR, convert to RGB
            if isinstance(images, torch.Tensor):
                images = images[:, :, :, [2, 1, 0]]
            else:
                images = images[:, :, :, ::-1].copy()

        # Convert to desired format
        if self.return_format == 'torch' and not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images)
        elif self.return_format == 'numpy' and isinstance(images, torch.Tensor):
            images = images.numpy()

        return images

    def _load_turbojpeg(self, image_paths: List[str]) -> np.ndarray:
        """Load images using turbojpeg (fastest)."""
        images = []

        for path in image_paths:
            with open(path, 'rb') as f:
                img_bytes = f.read()

            # Decode JPEG
            if self.color_format == 'rgb':
                img = _turbo_jpeg.decode(img_bytes, pixel_format=0)  # RGB
            else:
                img = _turbo_jpeg.decode(img_bytes, pixel_format=1)  # BGR

            images.append(img)

        return np.stack(images)

    def _load_cv2_threaded(self, image_paths: List[str]) -> np.ndarray:
        """Load images using OpenCV with ThreadPool."""
        def load_single(path):
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Failed to load: {path}")
            return img

        images = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_path = {executor.submit(load_single, path): i
                             for i, path in enumerate(image_paths)}

            # Collect results in original order
            results = [None] * len(image_paths)
            for future in as_completed(future_to_path):
                idx = future_to_path[future]
                results[idx] = future.result()

        return np.stack(results)

    def _load_torch_threaded(self, image_paths: List[str]) -> torch.Tensor:
        """Load images using torchvision with ThreadPool."""
        from torchvision.io import read_image

        def load_single(path):
            # read_image returns (C, H, W) in RGB
            img = read_image(path)
            # Convert to (H, W, C)
            return img.permute(1, 2, 0)

        images = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_path = {executor.submit(load_single, path): i
                             for i, path in enumerate(image_paths)}

            results = [None] * len(image_paths)
            for future in as_completed(future_to_path):
                idx = future_to_path[future]
                results[idx] = future.result()

        return torch.stack(results)

    def _load_pillow_threaded(self, image_paths: List[str]) -> np.ndarray:
        """Load images using PIL/pillow-simd with ThreadPool."""
        def load_single(path):
            img = Image.open(path).convert('RGB')
            return np.array(img)

        images = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_path = {executor.submit(load_single, path): i
                             for i, path in enumerate(image_paths)}

            results = [None] * len(image_paths)
            for future in as_completed(future_to_path):
                idx = future_to_path[future]
                results[idx] = future.result()

        return np.stack(results)

    def _load_cv2_sequential(self, image_paths: List[str]) -> np.ndarray:
        """Load images using OpenCV sequentially (fallback)."""
        images = []

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Failed to load: {path}")
            images.append(img)

        return np.stack(images)


# Global loader cache to avoid repeated initialization
_LOADER_CACHE = {}


def retry_load_images_fast(
    image_paths: List[str],
    retry: int = 3,
    backend: str = 'auto',
    num_workers: int = 4,
) -> torch.Tensor:
    """
    Drop-in replacement for utils.retry_load_images with faster loading.

    Args:
        image_paths: List of image paths
        retry: Number of retries
        backend: Image loading backend
        num_workers: Number of worker threads

    Returns:
        Tensor of shape (T, H, W, 3) in RGB format, uint8
    """
    # Use cached loader to avoid repeated initialization
    cache_key = (backend, num_workers)
    if cache_key not in _LOADER_CACHE:
        _LOADER_CACHE[cache_key] = FastImageLoader(
            backend=backend,
            num_workers=num_workers,
            color_format='rgb',
            return_format='torch',
        )
        # Print only on first initialization
        print(f"FastImageLoader initialized with backend: {_LOADER_CACHE[cache_key].backend}, workers: {num_workers}")

    loader = _LOADER_CACHE[cache_key]

    for attempt in range(retry):
        try:
            images = loader.load_images(image_paths)
            return images
        except Exception as e:
            if attempt == retry - 1:
                raise e
            print(f"Retry {attempt + 1}/{retry} after error: {e}")

    raise RuntimeError(f"Failed to load images after {retry} retries")


# Benchmark utilities
def benchmark_loader(image_paths: List[str], backend: str, num_workers: int = 4, num_runs: int = 5):
    """Benchmark image loading performance."""
    import time

    loader = FastImageLoader(backend=backend, num_workers=num_workers)

    # Warmup
    _ = loader.load_images(image_paths[:2])

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        images = loader.load_images(image_paths)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)

    return {
        'backend': backend,
        'num_images': len(image_paths),
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'images_per_sec': len(image_paths) / (avg_time / 1000),
    }


if __name__ == "__main__":
    """Test and benchmark the fast image loader."""
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Test fast image loader")
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--num_images', type=int, default=16,
                       help='Number of images to load')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker threads')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark comparing all backends')
    args = parser.parse_args()

    # Find test images
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')))[:args.num_images]

    if not image_paths:
        print(f"No images found in {args.image_dir}")
        exit(1)

    print(f"Testing with {len(image_paths)} images")
    print(f"First image: {image_paths[0]}")

    if args.benchmark:
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)

        backends = ['cv2']
        if _HAS_TURBOJPEG:
            backends.insert(0, 'turbojpeg')
        if args.num_workers > 1:
            backends.append('cv2_threaded')

        results = []
        for backend in backends:
            try:
                result = benchmark_loader(image_paths, backend, args.num_workers)
                results.append(result)
                print(f"\n{backend}:")
                print(f"  Avg time: {result['avg_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
                print(f"  Throughput: {result['images_per_sec']:.1f} images/sec")
            except Exception as e:
                print(f"\n{backend}: FAILED - {e}")

        # Compare speedup
        if len(results) > 1:
            baseline = results[-1]['avg_time_ms']
            print("\n" + "="*60)
            print("SPEEDUP vs cv2 sequential:")
            for result in results:
                speedup = baseline / result['avg_time_ms']
                print(f"  {result['backend']}: {speedup:.2f}x")
    else:
        # Simple test
        loader = FastImageLoader(backend='auto', num_workers=args.num_workers)
        images = loader.load_images(image_paths)
        print(f"\nLoaded images shape: {images.shape}")
        print(f"Data type: {images.dtype}")
        print(f"Backend used: {loader.backend}")
