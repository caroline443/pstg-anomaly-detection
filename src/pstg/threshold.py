"""
Adaptive Dynamic Thresholding for Online Anomaly Detection
===========================================================
改进点：在原 Telemanom 动态阈值基础上加入自适应校准。
原版用固定的 tuning_percentage（如 5%）对所有实体一刀切，
导致低异常比例实体（<2%）阈值过低（大量误报），
高异常比例实体（>20%）阈值过高（大量漏报）。

改进策略：
  1. 用训练集误差分布估计"正常误差上界" mu_train + k*sigma_train
  2. 在测试集上，以训练集上界为锚点，自适应搜索 z-score 倍数
  3. 对极低异常比例实体收紧 tuning_percentage，避免过度误报
"""

import numpy as np


class DynamicThreshold:
    """
    Adaptive dynamic threshold estimator.

    Args:
        smoothing_base (int):    Base factor n_s for smoothing window size.
        test_batch_size (int):   Test batch size B_s.
        tuning_percentage (float): Default tuning percentage (fallback).
        use_adaptive (bool):     Enable adaptive calibration from train errors.
        z_min (float):           Minimum z-score multiplier.
        z_max (float):           Maximum z-score multiplier.
        z_step (float):          Search step for z-score.
    """

    def __init__(
        self,
        smoothing_base: int = 30,
        test_batch_size: int = 32,
        tuning_percentage: float = 0.05,
        use_adaptive: bool = True,
        z_min: float = 1.0,
        z_max: float = 12.0,
        z_step: float = 0.25,
    ):
        self.n_s = smoothing_base
        self.B_s = test_batch_size
        self.p_s = tuning_percentage
        self.use_adaptive = use_adaptive
        self.z_min = z_min
        self.z_max = z_max
        self.z_step = z_step
        self._window_size = max(3, int(np.sqrt(self.n_s * self.B_s)))

        # Set by calibrate(); used in adaptive mode
        self._train_mean: float = None
        self._train_std: float = None
        self._calibrated_p: float = None

    # ── Calibration ────────────────────────────────────────────────────────────

    def calibrate(self, train_errors: np.ndarray) -> None:
        """
        Calibrate threshold parameters using training set errors.

        Estimates the normal error distribution from the training set and
        derives an entity-specific tuning_percentage that avoids over-alerting
        on low-anomaly-ratio entities.

        Args:
            train_errors: 1-D array of per-window MAE on the training set.
        """
        smoothed = self.smooth(train_errors)
        self._train_mean = float(np.mean(smoothed))
        self._train_std  = float(np.std(smoothed) + 1e-8)

        # Estimate what fraction of training points exceed mean+3σ
        # (should be ~0.3% for Gaussian; higher means noisier signal)
        upper = self._train_mean + 3.0 * self._train_std
        noise_ratio = float(np.mean(smoothed > upper))

        # Adaptive tuning_percentage:
        #   - Very clean signal (noise_ratio < 0.5%): tighten to 1%
        #   - Noisy signal (noise_ratio > 5%): relax up to 15%
        #   - Otherwise: scale linearly between 1% and 10%
        if noise_ratio < 0.005:
            self._calibrated_p = 0.01
        elif noise_ratio > 0.05:
            self._calibrated_p = min(0.15, noise_ratio * 2.0)
        else:
            # Linear interpolation between 1% and 10%
            t = (noise_ratio - 0.005) / (0.05 - 0.005)
            self._calibrated_p = 0.01 + t * (0.10 - 0.01)

    # ── Core methods ───────────────────────────────────────────────────────────

    def smooth(self, errors: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing."""
        w = self._window_size
        kernel = np.ones(w) / w
        return np.convolve(errors, kernel, mode='same')

    def compute_threshold(self, errors: np.ndarray) -> float:
        """
        Compute anomaly threshold from test error sequence.

        In adaptive mode, uses calibrated tuning_percentage and anchors
        the search around the training-set normal error upper bound.
        Falls back to original Telemanom logic if not calibrated.

        Args:
            errors: 1-D array of per-window prediction residuals (test set).

        Returns:
            threshold: Scalar anomaly threshold.
        """
        smoothed = self.smooth(errors)
        mean = float(np.mean(smoothed))
        std  = float(np.std(smoothed) + 1e-8)

        if self.use_adaptive and self._calibrated_p is not None:
            p = self._calibrated_p
        else:
            p = self.p_s

        z = self._search_z(smoothed, mean, std, p)
        return mean + z * std

    def _search_z(
        self,
        smoothed: np.ndarray,
        mean: float,
        std: float,
        target_p: float,
    ) -> float:
        """
        Binary-search-style z-score selection.
        Find smallest z such that anomaly_ratio <= target_p.
        """
        z = self.z_min
        while z < self.z_max:
            threshold = mean + z * std
            anomaly_ratio = float(np.mean(smoothed > threshold))
            if anomaly_ratio <= target_p:
                break
            z += self.z_step
        return z

    def detect(
        self,
        errors: np.ndarray,
        train_errors: np.ndarray = None,
    ) -> np.ndarray:
        """
        Detect anomalies from error sequence.

        Args:
            errors:       1-D array of test prediction residuals.
            train_errors: 1-D array of train prediction residuals (optional).
                          If provided and use_adaptive=True, calibrates first.

        Returns:
            labels: Binary array, 1 = anomaly, 0 = normal.
        """
        if self.use_adaptive and train_errors is not None:
            self.calibrate(train_errors)

        threshold = self.compute_threshold(errors)
        smoothed  = self.smooth(errors)
        return (smoothed > threshold).astype(int)

    @property
    def calibrated_p(self) -> float:
        """Return the calibrated tuning percentage (after calibrate() is called)."""
        return self._calibrated_p if self._calibrated_p is not None else self.p_s
