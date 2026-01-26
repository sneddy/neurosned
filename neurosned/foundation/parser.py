import pandas as pd
import numpy as np

class SFPParser:
    """Parse .sfp EEG montage file and assign anatomical regions to each electrode."""
    def __init__(self, sfp_path):
        self.sfp_path = sfp_path
        self.sfp_df, self.coords = self._parse_sfp()
        self.q3_absx = self._compute_q3_absx()
        self.y_quartiles = None
        self._assign_regions()

    def _parse_sfp(self):
        """Parse .sfp file into DataFrame and extract coordinates."""
        df = pd.read_csv(self.sfp_path, sep=r"\s+", names=["name", "x", "y", "z"])
        df_ECz = df[df['name'].str.match(r"^E\d+$") | (df['name'] == 'Cz')].reset_index(drop=True)
        coords = {row["name"]: (float(row["x"]), float(row["y"]), float(row["z"])) for _, row in df_ECz.iterrows()}
        return df_ECz, coords

    def _compute_q3_absx(self):
        """Compute 75th percentile of absolute x coordinates."""
        self.sfp_df["abs_x"] = np.abs(self.sfp_df["x"])
        return np.percentile(self.sfp_df["abs_x"], 75)

    @staticmethod
    def _electrode_region(row, q3_absx):
        """Assign TL/TR for lateral electrodes, else None."""
        if row["abs_x"] >= q3_absx:
            if row["x"] < 0:
                return "TL"
            elif row["x"] > 0:
                return "TR"
        return None

    def _assign_others(self, y):
        """Assign F/C/P/O region based on quartiles of y coordinate."""
        if self.y_quartiles is None:
            raise ValueError("y_quartiles not computed.")
        if y >= self.y_quartiles[2]:
            return "F"
        elif y >= self.y_quartiles[1]:
            return "C"
        elif y >= self.y_quartiles[0]:
            return "P"
        else:
            return "O"

    def _assign_regions(self):
        """Assign all regions (temporal and non-temporal) to electrodes."""
        self.sfp_df["region"] = self.sfp_df.apply(lambda row: self._electrode_region(row, self.q3_absx), axis=1)
        not_temporal = self.sfp_df["region"].isna()
        self.y_quartiles = np.percentile(self.sfp_df.loc[not_temporal, "y"], [25, 50, 75])
        self.sfp_df.loc[not_temporal, "region"] = self.sfp_df.loc[not_temporal, "y"].apply(self._assign_others)

    def get_coord_region(self):
        """Return a dictionary mapping electrode name to region."""
        return {row["name"]: row["region"] for _, row in self.sfp_df.iterrows()}

    def channel_id_to_region(self):
        """Return a dictionary mapping channel numeric index (row) to region."""
        return {idx: row["region"] for idx, row in self.sfp_df.iterrows()}
