from neurosned.foundation.pseudo_devices import PseudoDevice
import numpy as np

class DeviceSplitter:
    def __init__(self, region_mapping, random_seed=None):
        """
        region_mapping: dict mapping channel index (or name) to region ("C", "F", etc.)
        random_seed: optional random seed for reproducibility
        """
        self.region_mapping = region_mapping
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def uniform_split(self, uniform_receipt, n_devices=3, repeat=False, reference_group='C'):
        """
        uniform_receipt: dict, e.g. {"C": 8, 'F': 8, "O":8, "P":7, "TL":5, 'TR':5}
        n_devices: number of pseudodevices ("splits")
        repeat: if True, sample with replacement; if False, without replacement across devices.
        reference_group: region/group (default: 'C') from which to select the reference channel for each PseudoDevice
        Returns: list of PseudoDevice objects (one per pseudodevice)
        """
        # Group channels by region
        region_channels = {}
        for ch, region in self.region_mapping.items():
            region_channels.setdefault(region, []).append(ch)

        devices = []
        if repeat:
            for _ in range(n_devices):
                channels = []
                for region, n in uniform_receipt.items():
                    region_chs = region_channels.get(region, [])
                    channels.extend(self.rng.choice(region_chs, size=n, replace=True).tolist())
                devices.append(PseudoDevice(channels, region_mapping=self.region_mapping, reference_group=reference_group))
        else:
            # Shuffle region channel lists once for reproducibility
            shuffled = {region: np.array(chs) for region, chs in region_channels.items()}
            for chs in shuffled.values():
                self.rng.shuffle(chs)
            ptr = {region: 0 for region in uniform_receipt}
            for _ in range(n_devices):
                channels = []
                for region, n in uniform_receipt.items():
                    chs = shuffled.get(region, [])
                    take = chs[ptr[region]:ptr[region]+n].tolist()
                    remain = n - len(take)
                    if remain > 0:
                        fill = self.rng.choice(chs, size=remain, replace=False).tolist()
                        take.extend(fill)
                    channels.extend(take)
                    ptr[region] += n
                devices.append(PseudoDevice(channels, region_mapping=self.region_mapping, reference_group=reference_group))
        return devices

    def random_split(self, n_devices=3, n_channels_per_device=32):
        """
        Generate pseudodevices by randomly sampling channels (from all channels) for each device.
        n_devices: number of pseudodevices to generate.
        n_channels_per_device: number of channels to assign to each pseudodevice.
        Returns: list of PseudoDevice objects.
        Note: The reference channel for each PseudoDevice is selected at random.
        Sampling is always performed without replacement; if not enough channels, an error will be raised.
        """
        all_channels = np.array(list(self.region_mapping.keys()))
        devices = []
        for _ in range(n_devices):
            chosen = self.rng.choice(all_channels, size=n_channels_per_device, replace=False).tolist()
            devices.append(PseudoDevice(chosen, region_mapping=self.region_mapping))
        return devices

    def worst_split(self, n_devices=3, n_channels_per_device=41, reference_group=None):
        reg2chs = {}
        for ch, reg in self.region_mapping.items():
            reg2chs.setdefault(reg, []).append(ch)

        regs = sorted(reg2chs, key=lambda r: len(reg2chs[r]), reverse=True)
        for r in regs:
            self.rng.shuffle(reg2chs[r])

        flat = [ch for r in regs for ch in reg2chs[r]]

        devices = []
        for i in range(n_devices):
            chs = flat[i*n_channels_per_device:(i+1)*n_channels_per_device]
            ref_reg = reference_group if reference_group is not None else self.region_mapping[chs[0]]
            devices.append(PseudoDevice(chs, region_mapping=self.region_mapping, reference_group=ref_reg))
        return devices