# surrogate.py
# Ensemble surrogate with isotonic calibration and robust input validation

import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression

class EnsembleCalibratedSurrogate:
    def __init__(self, n_models=6, rf_estimators=50, random_state=42):
        # Numerical safety
        self.min_std = 1e-3
        self.jitter = 1e-8

        self.model = RandomForestRegressor(
            n_estimators=rf_estimators,
            random_state=random_state
        )
        self.iso_list = []
        self.is_trained = False
        self.X = None
        self.Y = None

    def fit(self, X, Y):
        """
        X: (n_samples, n_params)
        Y: (n_samples, n_outputs)
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Fit RF (sklearn supports multi-output)
        self.model.fit(X, Y)

        # Build raw ensemble stds (or fallback)
        try:
            preds = np.stack([tree.predict(X) for tree in self.model.estimators_], axis=0)
            raw_stds = preds.std(axis=0)  # shape (n_samples, n_outputs)
        except Exception:
            raw_preds = self.model.predict(X)
            raw_stds = np.abs(raw_preds - Y).mean(axis=0, keepdims=True).repeat(X.shape[0], axis=0)

        # Build isotonic calibrators per output
        n_out = Y.shape[1]
        self.iso_list = []
        for j in range(n_out):
            try:
                true_err = np.abs(self.model.predict(X)[:, j] - Y[:, j])
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(raw_stds[:, j], true_err)
            except Exception:
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
            self.iso_list.append(iso)

        self.is_trained = True
        self.X = X
        self.Y = Y

    def _ensure_feature_dim(self, X):
        """Make X have the same number of features as training if possible.
        Pads with zeros if too short, truncates if too long.
        """
        if not hasattr(self.model, "n_features_in_"):
            return X  # no info; assume OK

        expected = int(self.model.n_features_in_)
        if X.shape[1] == expected:
            return X
        warnings.warn(f"Surrogate input feature dim {X.shape[1]} != expected {expected}. "
                      "Auto-adjusting by pad/truncate.", UserWarning)
        if X.shape[1] < expected:
            # pad with zeros to expected
            pad = np.zeros((X.shape[0], expected - X.shape[1]), dtype=X.dtype)
            X2 = np.hstack([X, pad])
            return X2
        else:
            # truncate extra features
            return X[:, :expected]

    def predict_mean_std(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not hasattr(self, "is_trained") or not self.is_trained:
            # Conservative cold start: single-output fallback
            n = X.shape[0]
            return np.zeros((n, 1)), np.full((n, 1), 0.25, dtype=float)

        # Ensure feature dimension matches what RF expects
        X = self._ensure_feature_dim(X)

        # Mean prediction
        mean_preds = self.model.predict(X)  # (n_samples, n_outputs)

        # Ensemble-predicted std (across trees)
        all_preds = np.stack([tree.predict(X) for tree in self.model.estimators_], axis=0)
        raw_stds = all_preds.std(axis=0)  # (n_samples, n_outputs)

        # jitter and calibration
        raw_stds = np.sqrt(raw_stds**2 + self.jitter)
        calibrated_stds = np.zeros_like(raw_stds)
        for j, iso in enumerate(self.iso_list):
            try:
                calibrated_stds[:, j] = iso.predict(raw_stds[:, j])
            except Exception:
                calibrated_stds[:, j] = raw_stds[:, j]

        calibrated_stds = np.maximum(calibrated_stds, self.min_std)

        return mean_preds, calibrated_stds
