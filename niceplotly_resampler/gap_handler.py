from typing import Optional, Tuple
import numpy as np

class AbstractGapHandler:
    def _get_gap_mask(self, x_agg: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError


class MedDiffGapHandler(AbstractGapHandler):
    def _calc_med_diff(self, x_agg: np.ndarray) -> Tuple[float, np.ndarray]:
        x_diff = np.diff(x_agg, prepend=x_agg[0])
        n_blocks = 128
        if x_agg.shape[0] > 5 * n_blocks:
            blck_size = x_diff.shape[0] // n_blocks
            sid_v: np.ndarray = x_diff[: blck_size * n_blocks].reshape(n_blocks, -1)
            med_diff = np.median(np.mean(sid_v, axis=1))
        else:
            med_diff = np.median(x_diff)
        return med_diff, x_diff

    def _get_gap_mask(self, x_agg: np.ndarray) -> Optional[np.ndarray]:
        if x_agg.size < 2:
            return None
        med_diff, x_diff = self._calc_med_diff(x_agg)
        gap_mask = x_diff > 4.1 * med_diff
        if not any(gap_mask):
            return None
        return gap_mask