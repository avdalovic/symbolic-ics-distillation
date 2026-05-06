from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectTrajectoryPredictor(nn.Module):
    """
    Direct sensor-space trajectory predictor for short-horizon forecasting.

    Args:
        x_window: [B, num_tags, history_len]
    Returns:
        y_hat: [B, horizon, n_sensors]
    """

    def __init__(
        self,
        sensor_idx: List[int],
        actuator_idx: List[int],
        num_tags: int,
        history_len: int,
        horizon: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        architecture: str = "gru",
        score_aggregation: str = "mean",
        horizon_loss_weights: Optional[List[float]] = None,
        horizon_weighting: str = "uniform",
        horizon_gamma: float = 0.9,
        prediction_mode: str = "deterministic",
        logvar_min: float = -6.0,
        logvar_max: float = 3.0,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 128,
        ae_hidden_1: int = 64,
        ae_hidden_2: int = 16,
    ) -> None:
        super().__init__()
        self.sensor_idx = [int(i) for i in sensor_idx]
        self.actuator_idx = [int(i) for i in actuator_idx]
        self.num_tags = int(num_tags)
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.architecture = str(architecture or "gru").lower()
        self.score_aggregation = str(score_aggregation or "mean").lower()
        self.horizon_weighting = str(horizon_weighting or "uniform").lower()
        self.horizon_gamma = float(horizon_gamma)
        self.prediction_mode = str(prediction_mode or "deterministic").lower()
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.transformer_heads = int(transformer_heads)
        self.transformer_ff_dim = int(transformer_ff_dim)
        self.ae_hidden_1 = int(ae_hidden_1)
        self.ae_hidden_2 = int(ae_hidden_2)

        if self.num_tags <= 0:
            raise ValueError("num_tags must be positive")
        if self.history_len <= 0:
            raise ValueError("history_len must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not self.sensor_idx:
            raise ValueError("sensor_idx must be non-empty")
        if self.architecture not in {
            "gru",
            "mlp",
            "linear",
            "transformer",
            "autoencoder_sensors",
            "autoencoder_full",
        }:
            raise ValueError(
                "architecture must be one of: gru, mlp, linear, transformer, "
                "autoencoder_sensors, autoencoder_full"
            )
        if self.score_aggregation not in {"mean", "weighted_mean", "max"}:
            raise ValueError("score_aggregation must be one of: mean, weighted_mean, max")
        if self.prediction_mode not in {"deterministic", "gaussian"}:
            raise ValueError("prediction_mode must be one of: deterministic, gaussian")
        if self.logvar_min > self.logvar_max:
            raise ValueError("logvar_min must be <= logvar_max")
        if self.architecture == "transformer":
            if self.transformer_heads <= 0:
                raise ValueError("transformer_heads must be positive")
            if self.hidden_dim % self.transformer_heads != 0:
                raise ValueError(
                    f"hidden_dim ({self.hidden_dim}) must be divisible by transformer_heads ({self.transformer_heads})"
                )
            if self.transformer_ff_dim <= 0:
                raise ValueError("transformer_ff_dim must be positive")
        if self.architecture in {"autoencoder_sensors", "autoencoder_full"}:
            if self.ae_hidden_1 <= 0 or self.ae_hidden_2 <= 0:
                raise ValueError("ae_hidden_1 and ae_hidden_2 must be positive")
            if self.prediction_mode != "deterministic":
                raise ValueError("autoencoder architectures currently support prediction_mode=deterministic only")

        self.register_buffer(
            "sensor_idx_tensor",
            torch.tensor(self.sensor_idx, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "actuator_idx_tensor",
            torch.tensor(self.actuator_idx, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "horizon_weight_tensor",
            self._build_horizon_weight_tensor(
                self.horizon,
                horizon_loss_weights=horizon_loss_weights,
                horizon_weighting=self.horizon_weighting,
                horizon_gamma=self.horizon_gamma,
            ),
            persistent=False,
        )
        out_multiplier = 2 if self.prediction_mode == "gaussian" else 1
        if self.architecture == "autoencoder_full":
            out_dim = self.num_tags * out_multiplier
        elif self.architecture == "autoencoder_sensors":
            out_dim = len(self.sensor_idx) * out_multiplier
        else:
            out_dim = self.horizon * len(self.sensor_idx) * out_multiplier
        if self.architecture == "gru":
            rnn_dropout = self.dropout if self.num_layers > 1 else 0.0
            self.temporal = nn.GRU(
                input_size=self.num_tags,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
            self.flat_head = nn.Sequential(
                nn.Linear(self.hidden_dim + len(self.actuator_idx), self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, out_dim),
            )
        elif self.architecture == "mlp":
            in_dim = self.num_tags * self.history_len
            self.temporal = None
            self.flat_head = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, out_dim),
            )
        elif self.architecture == "linear":
            self.temporal = None
            in_dim = self.num_tags * self.history_len
            self.flat_head = nn.Linear(in_dim, out_dim)
        elif self.architecture == "transformer":
            self.input_proj = nn.Linear(self.num_tags, self.hidden_dim)
            self.positional = nn.Parameter(
                torch.zeros(1, self.history_len, self.hidden_dim)
            )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.transformer_heads,
                dim_feedforward=self.transformer_ff_dim,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.temporal = nn.TransformerEncoder(
                enc_layer,
                num_layers=self.num_layers,
            )
            self.flat_head = nn.Linear(self.hidden_dim, out_dim)
        elif self.architecture in {"autoencoder_sensors", "autoencoder_full"}:
            self.temporal = None
            self.ae_encoder = nn.Sequential(
                nn.Linear(self.num_tags, self.ae_hidden_1),
                nn.ReLU(),
                nn.Linear(self.ae_hidden_1, self.ae_hidden_2),
            )
            decoder_out_dim = len(self.sensor_idx) if self.architecture == "autoencoder_sensors" else self.num_tags
            self.ae_decoder = nn.Sequential(
                nn.Linear(self.ae_hidden_2, self.ae_hidden_1),
                nn.ReLU(),
                nn.Linear(self.ae_hidden_1, self.ae_hidden_1),
                nn.ReLU(),
                nn.Linear(self.ae_hidden_1, decoder_out_dim),
            )
        else:
            raise RuntimeError(f"Unsupported architecture: {self.architecture}")

    @staticmethod
    def _build_horizon_weight_tensor(
        horizon: int,
        *,
        horizon_loss_weights: Optional[List[float]],
        horizon_weighting: str,
        horizon_gamma: float,
    ) -> torch.Tensor:
        h = int(horizon)
        if horizon_loss_weights is not None:
            if len(horizon_loss_weights) != h:
                raise ValueError(
                    f"horizon_loss_weights must have length={h}, got {len(horizon_loss_weights)}"
                )
            w = torch.tensor([float(v) for v in horizon_loss_weights], dtype=torch.float32)
        elif horizon_weighting == "exp_decay":
            gamma = float(horizon_gamma)
            if gamma <= 0.0:
                raise ValueError("horizon_gamma must be positive for exp_decay weighting")
            exponents = torch.arange(h, dtype=torch.float32)
            w = gamma ** exponents
        else:
            w = torch.ones(h, dtype=torch.float32)

        w_sum = float(w.sum().item())
        if w_sum <= 0.0:
            raise ValueError("horizon weights must sum to a positive value")
        return w / w_sum

    @staticmethod
    def _ensure_finite(tensor: torch.Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} has NaN/Inf values")

    def reconstruct_current(self, x_window: torch.Tensor) -> torch.Tensor:
        """
        Autoencoder forward path on the current observation x_t (last token in window).
        Returns sensor-only or full reconstruction depending on architecture.
        """
        if self.architecture not in {"autoencoder_sensors", "autoencoder_full"}:
            raise RuntimeError("reconstruct_current is only valid for autoencoder architectures")
        x_curr = x_window[:, :, -1]  # [B, num_tags]
        z = self.ae_encoder(x_curr)
        return self.ae_decoder(z)

    def _raw_head(self, x_window: torch.Tensor) -> torch.Tensor:
        if x_window.ndim != 3:
            raise ValueError(f"x_window must be [B, num_tags, history_len], got shape={tuple(x_window.shape)}")
        if x_window.shape[1] != self.num_tags:
            raise ValueError(f"Expected num_tags={self.num_tags}, got {x_window.shape[1]}")

        if self.architecture == "gru":
            seq = x_window.transpose(1, 2).contiguous()  # [B, history_len, num_tags]
            _, hidden = self.temporal(seq)
            state_t = hidden[-1]  # [B, hidden_dim]

            if len(self.actuator_idx) > 0:
                u_t = x_window.index_select(dim=1, index=self.actuator_idx_tensor)[:, :, -1]
            else:
                u_t = x_window.new_zeros((x_window.shape[0], 0))

            features = torch.cat([state_t, u_t], dim=1)
            return self.flat_head(features)

        if self.architecture == "transformer":
            seq = x_window.transpose(1, 2).contiguous()  # [B, history_len, num_tags]
            tok = self.input_proj(seq) + self.positional[:, : seq.shape[1], :]
            enc = self.temporal(tok)
            state_t = enc[:, -1, :]
            return self.flat_head(state_t)

        if self.architecture in {"autoencoder_sensors", "autoencoder_full"}:
            return self.reconstruct_current(x_window)

        flat_in = x_window.reshape(x_window.shape[0], -1)
        return self.flat_head(flat_in)

    def predict_distribution(self, x_window: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x_window: [B, num_tags, history_len]
        Returns:
            mu: [B, horizon, n_sensors]
            log_var: [B, horizon, n_sensors] for gaussian mode, else None
        """
        raw = self._raw_head(x_window)
        batch = x_window.shape[0]
        n_sensors = len(self.sensor_idx)

        if self.architecture == "autoencoder_sensors":
            mu = raw.view(batch, 1, n_sensors)
            self._ensure_finite(mu, "mu")
            return mu, None

        if self.architecture == "autoencoder_full":
            sensor_mu = raw.index_select(dim=1, index=self.sensor_idx_tensor).view(batch, 1, n_sensors)
            self._ensure_finite(sensor_mu, "mu")
            return sensor_mu, None

        if self.prediction_mode == "gaussian":
            raw = raw.view(batch, self.horizon, 2 * n_sensors)
            mu, log_var = torch.chunk(raw, 2, dim=2)
            log_var = torch.clamp(log_var, min=self.logvar_min, max=self.logvar_max)
            self._ensure_finite(mu, "mu")
            self._ensure_finite(log_var, "log_var")
            return mu, log_var

        mu = raw.view(batch, self.horizon, n_sensors)
        self._ensure_finite(mu, "mu")
        return mu, None

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible forward pass that returns deterministic mean prediction.
        """
        mu, _ = self.predict_distribution(x_window)
        return mu

    def _gaussian_nll_terms(
        self,
        mu: torch.Tensor,
        y_true: torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_var = torch.clamp(log_var, min=self.logvar_min, max=self.logvar_max)
        var = torch.exp(log_var)
        sq_err = (y_true - mu).pow(2)
        log_term = 0.5 * log_var
        sq_term = 0.5 * (sq_err / var)
        nll = log_term + sq_term
        self._ensure_finite(var, "var")
        self._ensure_finite(nll, "nll")
        return nll, log_term, sq_term

    def compute_prediction_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: [B, H, n_sensors] mean prediction
            y_true: [B, H, n_sensors]
            log_var: [B, H, n_sensors] for Gaussian mode
        Returns:
            scalar prediction loss
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"y_pred and y_true must have same shape, got {tuple(y_pred.shape)} vs {tuple(y_true.shape)}"
            )

        use_gaussian = log_var is not None
        if self.prediction_mode == "gaussian" and log_var is None:
            raise ValueError("Gaussian prediction_mode requires log_var for loss computation")

        if use_gaussian:
            if log_var is None or log_var.shape != y_true.shape:
                raise ValueError("log_var must match y_true shape for Gaussian loss")
            nll, _, _ = self._gaussian_nll_terms(y_pred, y_true, log_var)
            per_h = nll.mean(dim=2)  # [B, H]
        else:
            per_h = (y_true - y_pred).pow(2).mean(dim=2)  # [B, H]

        weights = self.horizon_weight_tensor.to(device=y_pred.device, dtype=y_pred.dtype)
        weighted = per_h * weights.view(1, -1)
        return weighted.sum(dim=1).mean()

    def _aggregate_scores(self, per_h_metric: torch.Tensor) -> torch.Tensor:
        """
        Args:
            per_h_metric: [B, H] per-horizon metric (MSE or NLL)
        Returns:
            score_pred: [B]
        """
        if self.score_aggregation == "mean":
            return per_h_metric.mean(dim=1)
        if self.score_aggregation == "weighted_mean":
            w = self.horizon_weight_tensor.to(device=per_h_metric.device, dtype=per_h_metric.dtype)
            return (per_h_metric * w.view(1, -1)).sum(dim=1)
        if self.score_aggregation == "max":
            return per_h_metric.max(dim=1).values
        raise RuntimeError(f"Unsupported score_aggregation: {self.score_aggregation}")

    def compute_anomaly_scores(self, x_window: torch.Tensor, y_true: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x_window: [B, num_tags, history_len]
            y_true: [B, H, n_sensors]
        Returns:
            score dict:
              - score_pred: [B]
              - score_lat: [B] (zeros for compatibility)
              - score_combined: [B] (equals score_pred)
              - residuals_horizon: [B, H, n_sensors]
              - residuals: [B, H*n_sensors] flattened residual matrix for NPZ compatibility
              - per_h_mse: [B, H]
              - per_h_nll: [B, H] (Gaussian mode; zeros otherwise)
        """
        mu, log_var = self.predict_distribution(x_window)
        if self.architecture in {"autoencoder_sensors", "autoencoder_full"}:
            x_curr = x_window[:, :, -1]
            y_true = x_curr.index_select(dim=1, index=self.sensor_idx_tensor).unsqueeze(1)
        residuals_horizon = y_true - mu
        per_h_mse = residuals_horizon.pow(2).mean(dim=2)  # [B, H]
        per_h_nll = torch.zeros_like(per_h_mse)
        nll_logvar_term = torch.zeros_like(per_h_mse)
        nll_sqerr_over_var_term = torch.zeros_like(per_h_mse)
        if self.prediction_mode == "gaussian":
            if log_var is None:
                raise RuntimeError("Gaussian mode must produce log_var")
            nll, log_term, sq_term = self._gaussian_nll_terms(mu, y_true, log_var)
            per_h_nll = nll.mean(dim=2)
            nll_logvar_term = log_term.mean(dim=2)
            nll_sqerr_over_var_term = sq_term.mean(dim=2)
            score_pred = self._aggregate_scores(per_h_nll)
        else:
            score_pred = self._aggregate_scores(per_h_mse)

        score_lat = torch.zeros_like(score_pred)
        score_combined = score_pred
        residuals = residuals_horizon.reshape(residuals_horizon.shape[0], -1)
        return {
            "score_pred": score_pred,
            "score_lat": score_lat,
            "score_combined": score_combined,
            "residuals_horizon": residuals_horizon,
            "residuals": residuals,
            "per_h_mse": per_h_mse,
            "per_h_nll": per_h_nll,
            "nll_logvar_term": nll_logvar_term,
            "nll_sqerr_over_var_term": nll_sqerr_over_var_term,
            "mu": mu,
            "log_var": log_var,
            "y_hat": mu,
        }
