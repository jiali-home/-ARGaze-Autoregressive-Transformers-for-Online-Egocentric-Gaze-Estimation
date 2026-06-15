#!/usr/bin/env python3

"""Dataset registry exports for the public ARGaze release."""

from .build import DATASET_REGISTRY, build_dataset  # noqa

# Importing these modules registers the datasets used by the paper release.
from .egtea_gaze import Egteagaze  # noqa
from .ego4d_gaze import Ego4dgaze  # noqa
from .egoexo4d_gaze import Egoexo4dgaze  # noqa
