import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class PseudoDevice:
    """
    Represents a pseudodevice composed of a selection of channels.
    """
    def __init__(self, channels, region_mapping=None, reference_channel=None, reference_group=None, default_max=False, seed=42):
        """
        channels: List of channel indices or names that form this pseudodevice.
        region_mapping: (Optional) mapping from channel to region for bookkeeping/inspection.
        reference_channel: (Optional) reference channel; if None, chosen by strategy.
        reference_group: (Optional) region/group to select a reference channel from. If provided and reference_channel is None, a reference is selected from this group.
        default_max: If True and no other reference conditions, use largest channel id, else random.
        seed: Optional seed for random selection (for reproducibility).
        """
        self.channels = sorted(channels)
        self.region_mapping = region_mapping
        self.default_max = default_max
        rng = np.random.RandomState(seed)

        if reference_channel is not None:
            self.reference_channel = reference_channel 
        else:
            # If reference_group is provided, choose ref from this region
            if reference_group is not None and self.region_mapping is not None:
                region2chs = [ch for ch in self.channels if self.region_mapping[ch] == reference_group]
                self.reference_channel = max(region2chs) if default_max else rng.choice(region2chs)
            else:
                self.reference_channel = max(self.channels) if default_max else rng.choice(self.channels)

        # Map original channel ids to indices in the device (after dropping ref -> -1).
        self.channel_map = self._build_channel_map(drop_ref=True)
        self.channel_to_device = self.channel_map  # backward-friendly alias

    def __len__(self) -> int:
        return len(self.channels)

    def _build_channel_map(self, drop_ref: bool = True):
        """
        Build mapping from original channel id to position in the device output.
        If drop_ref is True, the reference channel maps to -1 (removed).
        """
        mapping = {}
        out_idx = 0
        for ch in self.channels:
            if drop_ref and ch == self.reference_channel:
                mapping[ch] = -1
                continue
            mapping[ch] = out_idx
            out_idx += 1
        return mapping

    def region_counts(self):
        """Return a Series with the count of channels per region for this device."""
        if self.region_mapping is None:
            raise ValueError("No region mapping was provided.")
        return pd.Series([self.region_mapping[ch] for ch in self.channels]).value_counts().sort_index()

    def region2id(self):
        """Map each region to a list of channel ids in self.channels (order preserved)."""
        if self.region_mapping is None:
            raise ValueError("No region mapping was provided.")
        region2chs = {}
        for ch in self.channels:
            region = self.region_mapping[ch]
            region2chs.setdefault(region, []).append(ch)
        return region2chs

    def transform(self, eeg: np.array, drop_ref: bool=True):
        """Re-reference the eeg data to the selected reference channel."""
        ch_axis = -2
        device_chs = np.array(self.channels)
        eeg_arr = eeg.values if isinstance(eeg, pd.DataFrame) else eeg

        eeg_device = np.take(eeg_arr, device_chs, axis=ch_axis)

        ref_idx = self.reference_channel
        ref_in_sel = int(np.where(device_chs == ref_idx)[0][0]) 
        ref_data = np.take(eeg_device, ref_in_sel, axis=ch_axis)
        ref_data = np.expand_dims(ref_data, axis=ch_axis)

        eeg_device = eeg_device - ref_data

        if drop_ref:
            keep = np.arange(len(device_chs)) != ref_in_sel
            eeg_device = np.take(eeg_device, np.where(keep)[0], axis=ch_axis)

        return eeg_device

    def transform_batch(self, eeg, drop_ref: bool = True):
        """
        Re-reference a torch tensor batch.

        Parameters
        ----------
        eeg : torch.Tensor
            Shape (B, n_channels, T) (или (..., n_channels, T)).
            Канальная ось = -2.
        drop_ref : bool
            Если True — удаляем ref-канал из результата.

        Returns
        -------
        torch.Tensor
            Shape (B, len(self.channels) - drop_ref, T)
        """
        ch_axis = -2
        device_chs = torch.as_tensor(self.channels, device=eeg.device, dtype=torch.long)

        # select device channels: (..., n_dev_ch, T)
        eeg_dev = eeg.index_select(ch_axis, device_chs)

        # find ref index inside selected channels
        ref_idx = int(self.reference_channel)
        # (n_dev_ch,)
        ref_in_sel = int((device_chs == ref_idx).nonzero(as_tuple=False).item())

        # reference data: (..., 1, T) via keepdim
        ref = eeg_dev.select(dim=ch_axis, index=ref_in_sel).unsqueeze(ch_axis)

        eeg_dev = eeg_dev - ref

        if drop_ref:
            keep_idx = torch.arange(eeg_dev.size(ch_axis), device=eeg.device)
            keep_idx = keep_idx[keep_idx != ref_in_sel]
            eeg_dev = eeg_dev.index_select(ch_axis, keep_idx)

        return eeg_dev

    def convert(self, base_dataset, drop_ref: bool = True):
        """
        Wrap a base dataset so that each sample is transformed by this pseudodevice.

        base_dataset: any Dataset that yields either
            - a tensor/ndarray shaped (C, T), or
            - a tuple/list where the first element is the EEG array.
        Returns a Dataset with the same structure, but with the EEG re-referenced
        to this pseudodevice (optionally dropping the reference channel).
        """
        return PseudoDeviceDataset(base_dataset, self, drop_ref=drop_ref)

    def __repr__(self):
        """String representation listing region and channel detail."""
        n_channels = len(self.channels)
        region2chs = self.region2id()
        msg = [f"<PseudoDevice(n_channels={n_channels}, reference_channel={self.reference_channel})>"]
        for region in sorted(region2chs.keys()):
            chs_in_order = region2chs[region]
            chs_main_str = ", ".join(str(ch) for ch in chs_in_order)
            msg.append(f"  {region}: {len(chs_in_order)} -> {chs_main_str}")
        return "\n".join(msg)


class PseudoDeviceDataset(Dataset):
    """
    Dataset wrapper that applies a PseudoDevice transform to each sample
    while preserving the remaining fields.
    """
    def __init__(self, base_dataset, device: PseudoDevice, drop_ref: bool = True):
        self.base_dataset = base_dataset
        self.device = device
        self.drop_ref = drop_ref

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if isinstance(sample, (tuple, list)):
            x = sample[0]
            rest = sample[1:]
        else:
            x = sample
            rest = tuple()

        x_t = torch.as_tensor(x)
        if x_t.ndim != 2:
            raise ValueError(f"Expected EEG shaped (C, T), got shape {tuple(x_t.shape)}.")

        x_dev = self.device.transform_batch(x_t, drop_ref=self.drop_ref)

        if rest:
            return (x_dev, *rest)
        return x_dev
