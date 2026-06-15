#!/usr/bin/env python3
import ctypes
import os
import site

# Keep native math libraries from over-subscribing threads on login/CI nodes.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _preload_cublas_pair():
    """Preload libcublas + libcublasLt, preferring system CUDA libs when the
    pip nvidia-cublas-cu12 version exceeds what the GPU driver supports.
    """
    if os.environ.get("GLC_SKIP_CUBLAS_PRELOAD", "0") == "1":
        return

    def _driver_cuda_ver():
        try:
            from pynvml import nvmlInit, nvmlSystemGetCudaDriverVersion_v2
            nvmlInit()
            v = nvmlSystemGetCudaDriverVersion_v2()
            return (v // 1000, (v % 1000) // 10)
        except Exception:
            return None

    def _pip_cublas_ver():
        try:
            from importlib.metadata import version as pkg_ver
            v = pkg_ver("nvidia-cublas-cu12")
            p = v.split(".")
            return (int(p[0]), int(p[1]))
        except Exception:
            return None

    def _load_pair(cublas, cublaslt):
        ctypes.CDLL(cublas, mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(cublaslt, mode=ctypes.RTLD_GLOBAL)

    def _find_pip_cublas():
        for p in site.getsitepackages() + [site.getusersitepackages()]:
            d = os.path.join(p, "nvidia", "cublas", "lib")
            c, lt = os.path.join(d, "libcublas.so.12"), os.path.join(d, "libcublasLt.so.12")
            if os.path.isfile(c) and os.path.isfile(lt):
                return c, lt
        return None, None

    _SYS_CUDA_DIRS = [
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
    ]

    def _find_sys_cublas():
        for d in _SYS_CUDA_DIRS:
            c, lt = os.path.join(d, "libcublas.so.12"), os.path.join(d, "libcublasLt.so.12")
            if os.path.isfile(c) and os.path.isfile(lt):
                return c, lt
        return None, None

    pip_c, pip_lt = _find_pip_cublas()
    sys_c, sys_lt = _find_sys_cublas()
    drv = _driver_cuda_ver()
    pip_ver = _pip_cublas_ver()

    use_system = drv and pip_ver and pip_ver > drv

    if use_system and sys_c:
        _load_pair(sys_c, sys_lt)
    elif pip_c:
        _load_pair(pip_c, pip_lt)
    elif sys_c:
        _load_pair(sys_c, sys_lt)


_preload_cublas_pair()

import torch  # import torch early to surface CUDA/library setup issues before launching jobs.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from test_gaze_net import test
from train_gaze_net import train


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    if os.environ.get("GLC_DISABLE_CUDNN", "0") == "1":
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        print("[run_net] GLC_DISABLE_CUDNN=1 -> cuDNN disabled for stability.")

    # In MIG environments, CUDA_VISIBLE_DEVICES is commonly a MIG UUID rather
    # than a numeric GPU id.  Use device_count() as the source of truth here:
    # torch.cuda.is_available() can be conservative before CUDA is initialized,
    # while device_count() still reports the scheduler-visible MIG device.
    available_gpus = torch.cuda.device_count()
    if cfg.NUM_GPUS > available_gpus:
        if available_gpus == 0:
            raise RuntimeError(
                "NUM_GPUS is set to {} but no CUDA devices are visible. "
                "Check your CUDA setup or lower NUM_GPUS to zero. "
                "CUDA_VISIBLE_DEVICES: {}. "
                "torch cuda available: {}. "
                "torch cuda version: {}.".format(
                    cfg.NUM_GPUS,
                    os.environ.get("CUDA_VISIBLE_DEVICES"),
                    torch.cuda.is_available(),
                    torch.version.cuda,
                )
            )

        print(
            "[run_net] Requested {} GPUs but only {} available. "
            "Overriding NUM_GPUS to {}.".format(
                cfg.NUM_GPUS, available_gpus, available_gpus
            )
        )
        cfg.defrost()
        cfg.NUM_GPUS = available_gpus
        cfg.freeze()

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
