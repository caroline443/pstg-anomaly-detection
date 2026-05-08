"""
Dynamic Thresholding for Online Anomaly Detection
===================================================
Adapts the anomaly threshold based on recent prediction residual statistics,
following the Telemanom-style dynamic thresholding approach.
"""

import numpy as np
import torch


class DynamicThreshold:
    """
    Online dynamic threshold estimator.

    Computes a smoothed error sequence and adaptively sets the anomaly
    threshold based on recent residual statistics (mean + k*std).

    Args:
        smoothing_base (int): Base factor n_s for smoothing window size.
        test_batch_size (int): Test batch size B_s.
        tuning_percentage (float): Tuning percentage p_s.
    """

    def __init__(
        self,
        smoothing_base: int = 30,
        test_batch_size: int = 70,
        tuning_percentage: float = 0.05,
    ):
        self.n_s = smoothing_base
        self.B_s = test_batch_size
        self.p_s = tuning_percentage
        self._window_size = int(np.sqrt(self.n_s * self.B_s))

    def smooth(self, errors: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing to error sequence."""
        w = self._window_size
        kernel = np.ones(w) / w
        return np.convolve(errors, kernel, mode='same')

    def compute_threshold(self, errors: np.ndarray) -> float:
        """
        Compute anomaly threshold from error sequence.

        Args:
            errors: 1-D array of prediction residuals.

        Returns:
            threshold: Scalar anomaly threshold.
        """
        smoothed = self.smooth(errors)
        mean = np.mean(smoothed)
        std = np.std(smoothed)
        # Adaptive multiplier based on tuning percentage
        z = self._compute_z(smoothed, mean, std)
        return mean + z * std

    def _compute_z(
        self,
        smoothed: np.ndarray,
        mean: float,
        std: float,
    ) -> float:
        """Find z-score multiplier such that p_s fraction of points are anomalous."""
        if std < 1e-8:
            return 3.0
        z = 1.0
        while z < 10.0:
            threshold = mean + z * std
            anomaly_ratio = np.mean(smoothed > threshold)
            if anomaly_ratio <= self.p_s:
                break
            z += 0.5
        return z

    def detect(self, errors: np.ndarray) -> np.ndarray:
        """
        Detect anomalies from error sequence.

        Args:
            errors: 1-D array of prediction residuals.

        Returns:
            labels: Binary array, 1 = anomaly, 0 = normal.
        """
        threshold = self.compute_threshold(errors)
        smoothed = self.smooth(errors)
        return (smoothed > threshold).astype(int)
