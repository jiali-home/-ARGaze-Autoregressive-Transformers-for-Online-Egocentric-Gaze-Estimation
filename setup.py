#!/usr/bin/env python3

from setuptools import find_packages, setup


setup(
    name="argaze",
    version="0.1.0",
    description="ARGaze core release for online egocentric gaze estimation.",
    packages=find_packages(exclude=("configs", "docs", "scripts", "data", "checkpoints")),
)
