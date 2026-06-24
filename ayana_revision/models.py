from __future__ import annotations

import numpy as np

from neurosned.models.segmentation.attention_sneddy_unet import AttentionSneddyUnet
from neurosned.models.segmentation.factorization_unet import FactorizationSneddyUnet
from neurosned.models.segmentation.inception import EEGInceptionSeg1D
from neurosned.models.segmentation.reccurent_unet import RecurrentSneddyUnet
from neurosned.models.segmentation.sneddy_unet import SneddySegUNet1D


MODEL_REGISTRY = {
    "AttentionSneddyUnet": AttentionSneddyUnet,
    "FactorizationSneddyUnet": FactorizationSneddyUnet,
    "EEGInceptionSeg1D": EEGInceptionSeg1D,
    "RecurrentSneddyUnet": RecurrentSneddyUnet,
    "SneddySegUNet1D": SneddySegUNet1D,
}


CHANNELS_LIST_OVERRIDES = {
    "RecurrentSneddyUnet": np.array([idx for idx in range(128) if idx not in (9, 121)]),
}


def build_model(model_name: str, model_config: dict, device):
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**model_config).to(device)


def channels_for_model(model_name: str, model) -> np.ndarray:
    return CHANNELS_LIST_OVERRIDES.get(model_name, np.arange(model.n_chans))

