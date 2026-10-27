"""Small CPU-first PyTorch autoencoder with train-only preprocessing."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .context import CONTEXT_FEATURES
from .detection import RESULT_COLUMNS, empty_results, normalize_input
from .isolation import FEATURES


class AutoencoderDetector:
    """Dense bottleneck autoencoder; reconstruction error is a score, not probability."""

    def __init__(
        self,
        hidden_dims=(16, 8),
        *,
        epochs=100,
        batch_size=128,
        learning_rate=1e-3,
        seed=42,
        threshold_quantile=0.99,
        threshold=None,
        patience=10,
        device="cpu",
        use_contextual_features=False,
        measurement_features=None,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.threshold_quantile = threshold_quantile
        self.threshold = threshold
        self.patience = patience
        self.device = device
        self.use_contextual_features = use_contextual_features
        self.configured_measurement_features = tuple(measurement_features or FEATURES)
        self.features = []
        self.measurement_features = []
        self.medians = None
        self.means = None
        self.stds = None
        self.model = None
        self.loss_history = []

    @staticmethod
    def _torch():
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError(
                "Install SensorGuard dependencies, including PyTorch, to use AutoencoderDetector"
            ) from exc
        return torch, nn

    def _matrix(self, frame, fit=False):
        data = normalize_input(frame).reset_index(drop=True)
        if fit:
            self.measurement_features = [
                f for f in self.configured_measurement_features if f in data
            ]
            if not self.measurement_features:
                raise ValueError("none of the configured measurement features are present")
            self.features = list(self.measurement_features)
            if self.use_contextual_features:
                context_features = [f for f in CONTEXT_FEATURES if f in data]
                if not context_features:
                    raise ValueError(
                        "contextual features were enabled but not supplied to the frame"
                    )
                self.features.extend(context_features)
        x = (
            data[self.features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=float)
        )
        if fit:
            self.medians = np.nanmedian(x, axis=0)
            self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
            x = np.where(np.isfinite(x), x, self.medians)
            self.means = x.mean(axis=0)
            raw = x.std(axis=0)
            self.stds = np.where(raw > 1e-12, raw, 1.0)
        else:
            x = np.where(np.isfinite(x), x, self.medians)
        return data, (x - self.means) / self.stds

    def _build(self):
        torch, nn = self._torch()
        dims = [len(self.features), *self.hidden_dims]
        enc = []
        for a, b in zip(dims, dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        dec = []
        rev = list(reversed(dims))
        for a, b in zip(rev, rev[1:]):
            dec.append(nn.Linear(a, b))
            if b != len(self.features):
                dec.append(nn.ReLU())
        self.model = nn.Sequential(*enc, *dec).to(self.device)

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame | None = None):
        torch, _ = self._torch()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available() and self.device == "auto":
            self.device = "cuda"
        if validation is None:
            ordered = normalize_input(train).sort_values("timestamp", kind="stable")
            cut = max(1, int(len(ordered) * 0.9))
            validation = ordered.iloc[cut:]
            train = ordered.iloc[:cut]
        _, x = self._matrix(train, True)
        if len(x) < 4:
            raise ValueError("autoencoder needs at least four training rows")
        valx = self._matrix(validation)[1] if validation is not None and len(validation) else None
        if valx is None:
            valx = x[-1:]
        if len(valx) == 0:
            valx = x[-1:]
        self._build()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lossfn = torch.nn.MSELoss()
        train_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        val_t = torch.tensor(valx, dtype=torch.float32, device=self.device)
        best = float("inf")
        best_state = None
        waiting = 0
        self.loss_history = []
        generator = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(train_t), generator=generator)
            total = 0.0
            for ix in perm.split(self.batch_size):
                batch = train_t[ix.to(self.device)]
                opt.zero_grad()
                loss = lossfn(self.model(batch), batch)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(batch)
            self.model.eval()
            with torch.no_grad():
                val_loss = float(lossfn(self.model(val_t), val_t))
            self.loss_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total / len(train_t),
                    "validation_loss": val_loss,
                }
            )
            if val_loss < best - 1e-7:
                best = val_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                waiting = 0
            else:
                waiting += 1
            if waiting >= self.patience:
                break
        if best_state:
            self.model.load_state_dict(best_state)
        if self.threshold is None:
            self.threshold = float(
                np.quantile(self._errors_from_x(valx)[0], self.threshold_quantile)
            )
        return self

    def _errors_from_x(self, x):
        torch, _ = self._torch()
        self.model.eval()
        with torch.no_grad():
            recon = (
                self.model(torch.tensor(x, dtype=torch.float32, device=self.device)).cpu().numpy()
            )
        errors = (recon - x) ** 2
        return errors.mean(axis=1), errors

    def score_samples(self, frame):
        """Return per-row aggregate reconstruction errors in input row order."""
        if self.model is None:
            raise RuntimeError("detector must be fitted")
        data, x = self._matrix(frame)
        aggregate, _ = self._errors_from_x(x)
        return data, aggregate

    def predict(self, frame, threshold=None):
        if self.model is None:
            raise RuntimeError("detector must be fitted")
        data, x = self._matrix(frame)
        aggregate, errors = self._errors_from_x(x)
        cutoff = self.threshold if threshold is None else threshold
        rows = []
        for i, r in data.iterrows():
            for f in self.measurement_features:
                j = self.features.index(f)
                if pd.notna(r[f]):
                    rows.append(
                        {
                            "timestamp": r.timestamp,
                            "sensor_id": r.sensor_id,
                            "feature": f,
                            "observed_value": r[f],
                            "anomaly_score": float(errors[i, j]),
                            "is_anomaly": bool(aggregate[i] >= cutoff),
                            "detector": "autoencoder",
                            "reason": f"Feature reconstruction error {errors[i, j]:.5g}; aggregate {aggregate[i]:.5g} vs threshold {cutoff:.5g}.",
                            "severity": "high" if aggregate[i] >= cutoff else "low",
                        }
                    )
        return pd.DataFrame(rows, columns=RESULT_COLUMNS) if rows else empty_results()

    def save(self, path):
        torch, _ = self._torch()
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hidden_dims": self.hidden_dims,
                "features": self.features,
                "measurement_features": self.measurement_features,
                "medians": self.medians,
                "means": self.means,
                "stds": self.stds,
                "threshold": self.threshold,
                "loss_history": self.loss_history,
                "params": {
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "seed": self.seed,
                    "threshold_quantile": self.threshold_quantile,
                    "patience": self.patience,
                    "device": self.device,
                    "use_contextual_features": self.use_contextual_features,
                    "measurement_features": self.configured_measurement_features,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu"):
        torch, _ = cls._torch()
        payload = torch.load(path, map_location=map_location, weights_only=False)
        obj = cls(hidden_dims=payload["hidden_dims"], **payload["params"])
        obj.features = payload["features"]
        obj.measurement_features = payload.get(
            "measurement_features", [name for name in obj.features if name in FEATURES]
        )
        obj.configured_measurement_features = tuple(
            payload.get("params", {}).get("measurement_features", obj.measurement_features)
        )
        obj.medians = payload["medians"]
        obj.means = payload["means"]
        obj.stds = payload["stds"]
        obj.threshold = payload["threshold"]
        obj.loss_history = payload["loss_history"]
        obj._build()
        obj.model.load_state_dict(payload["state_dict"])
        obj.model.eval()
        return obj
