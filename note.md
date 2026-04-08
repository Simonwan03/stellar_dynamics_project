# Autoregressive Seq2Seq Survival Model for Stellar Dynamics

This note translates the modeling direction in [task_b.ipynb](/home/wan/stellar_dynamics/task_b.ipynb) into a cleaner sequence-to-sequence survival formulation.

The notebook currently explores trajectory-prefix classification and soft time-to-instability targets from the first `T=100` steps. The design below keeps that prefix-based setup, but changes the target to a **future survival probability curve**:

- Input: `(batch_size, T_in, num_features)` with `T_in = 100`
- Output: `(batch_size, K)` survival probabilities at future checkpoints such as `t = [200, 400, 600, 800, 1000]`
- Physics constraint: the predicted curve is **monotonically non-increasing**

## Design Assumptions

To stay close to the feature conventions in `task_b.ipynb`, the recommended `num_features = 15` can be formed from the orbital descriptors:

- `aratios`
- `mratios`
- `eccs`
- `incs`
- `APs`
- `LANs`

with angular variables encoded using sine/cosine:

- `aratios`: 1
- `mratios`: 2
- `eccs`: 2
- `incs`: 2
- `sin(APs), cos(APs)`: 4
- `sin(LANs), cos(LANs)`: 4

Total: `1 + 2 + 2 + 2 + 4 + 4 = 15`

This is intentionally lighter than the full 36-channel feature stack in `task_b.ipynb`, which also includes masses, energy, angular momentum, geometric invariants, and an optional time channel.

## Model Overview

The model has three parts:

1. `Encoder`
   Processes the observed prefix `x[:, :100, :]` and returns a context vector.

2. `AutoregressiveDecoder`
   Starts from `P_0 = 1.0` and predicts one attenuation factor per future step:
   `P_k = P_{k-1} * alpha_k`, where `alpha_k in [0, 1]`.

3. `TemporalSurvivalLoss`
   Compares predicted and target survival curves with optional time-decay weighting.

Because every `alpha_k` is bounded to `[0, 1]`, the resulting prediction is always:

- in `[0, 1]`
- monotonically non-increasing

## Reference PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class GRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]
        return self.proj(h_last)


class CNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock1D(input_dim, 64, kernel_size=7, dropout=dropout),
            ConvBlock1D(64, 128, kernel_size=5, dropout=dropout),
            ConvBlock1D(128, hidden_dim, kernel_size=3, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        h = self.backbone(x).squeeze(-1)
        return h


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        channels=(64, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks = []
        prev = input_dim
        for i, ch in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_ch=prev,
                    out_ch=ch,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
            prev = ch

        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(prev, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = self.tcn(x)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class Encoder(nn.Module):
    """
    Wrapper to make encoder choice easy to swap:
      - "gru"
      - "cnn"
      - "tcn"
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        encoder_type: str = "gru",
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_type = encoder_type.lower()
        if encoder_type == "gru":
            self.encoder = GRUEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                dropout=dropout,
                bidirectional=False,
            )
        elif encoder_type == "cnn":
            self.encoder = CNNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif encoder_type == "tcn":
            self.encoder = TCNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AutoregressiveDecoder(nn.Module):
    """
    Monotone decoder:
      P_0 = 1
      alpha_k = sigmoid(f(h_k)) in [0, 1]
      P_k = P_{k-1} * alpha_k
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_size=1, hidden_size=hidden_dim)
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        context: torch.Tensor,
        horizon: int,
        teacher_forcing: bool = False,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        context: (B, H)
        targets: optional (B, K), only used if teacher_forcing=True
        returns: survival curve (B, K)
        """
        batch_size = context.size(0)
        hidden = context
        prev_p = torch.ones(batch_size, 1, device=context.device, dtype=context.dtype)

        preds = []
        for step in range(horizon):
            hidden = self.gru_cell(prev_p, hidden)
            alpha = torch.sigmoid(self.alpha_head(hidden))
            p_t = prev_p * alpha
            preds.append(p_t)

            if teacher_forcing and targets is not None:
                prev_p = targets[:, step:step + 1]
            else:
                prev_p = p_t

        return torch.cat(preds, dim=1)


class Seq2SeqSurvivalModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        horizon: int = 5,
        encoder_type: str = "gru",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            dropout=dropout,
        )
        self.decoder = AutoregressiveDecoder(hidden_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing: bool = False,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, T_in, F)
        returns: (B, K)
        """
        context = self.encoder(x)
        return self.decoder(
            context=context,
            horizon=self.horizon,
            teacher_forcing=teacher_forcing,
            targets=targets,
        )


class TemporalSurvivalLoss(nn.Module):
    """
    Weighted survival-curve loss.

    Supports:
      - loss_type="mse"
      - loss_type="bce"

    time_decay < 1.0 gives slightly larger weight to earlier horizons.
    Example for K=5 and time_decay=0.9:
      weights ~ [1.0, 0.9, 0.81, 0.729, 0.6561]
    """
    def __init__(
        self,
        loss_type: str = "mse",
        time_decay: float = 0.95,
        eps: float = 1e-6,
    ):
        super().__init__()
        if loss_type not in {"mse", "bce"}:
            raise ValueError("loss_type must be 'mse' or 'bce'")
        self.loss_type = loss_type
        self.time_decay = time_decay
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: (B, K)
        y_true: (B, K)
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")

        batch_size, horizon = y_pred.shape
        device = y_pred.device
        dtype = y_pred.dtype

        weights = torch.tensor(
            [self.time_decay ** k for k in range(horizon)],
            device=device,
            dtype=dtype,
        )
        weights = weights / weights.sum()

        if self.loss_type == "mse":
            per_step = (y_pred - y_true) ** 2
        else:
            y_pred = y_pred.clamp(self.eps, 1.0 - self.eps)
            per_step = F.binary_cross_entropy(y_pred, y_true, reduction="none")

        loss = (per_step * weights.view(1, horizon)).sum(dim=1).mean()
        return loss


def make_dummy_monotone_targets(batch_size: int, horizon: int, device: torch.device) -> torch.Tensor:
    """
    Generates monotone non-increasing target survival curves for a toy example.
    """
    alpha = torch.sigmoid(torch.randn(batch_size, horizon, device=device))
    y = torch.cumprod(alpha, dim=1)
    return y


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 32
    t_in = 100
    num_features = 15
    k = 5
    hidden_dim = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy input shaped like a trajectory prefix:
    # (batch, T_in, num_features)
    x = torch.randn(batch_size, t_in, num_features, device=device)

    # Dummy monotone target survival curves:
    # (batch, K)
    y_true = make_dummy_monotone_targets(batch_size=batch_size, horizon=k, device=device)

    model = Seq2SeqSurvivalModel(
        input_dim=num_features,
        hidden_dim=hidden_dim,
        horizon=k,
        encoder_type="gru",  # swap to "cnn" or "tcn"
        dropout=0.1,
    ).to(device)

    criterion = TemporalSurvivalLoss(loss_type="mse", time_decay=0.95)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    optimizer.zero_grad()

    y_pred = model(x)  # (32, 5)
    loss = criterion(y_pred, y_true)

    loss.backward()
    optimizer.step()

    print("x shape     :", x.shape)
    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)
    print("loss        :", float(loss.item()))
    print("monotone check:",
          bool(torch.all(y_pred[:, 1:] <= y_pred[:, :-1] + 1e-6).item()))
```

## Notes for Integration with `task_b.ipynb`

If you want to wire this into the current notebook pipeline, the cleanest path is:

1. Keep the existing prefix length `T=100`.
2. Replace the current single-label or 64-bin target with a compact survival target of size `K=5`.
3. Build `y_true[:, k]` as the fraction of perturbed trajectories still stable at each future checkpoint.
4. Feed a `(B, 100, 15)` tensor to the model instead of the current wider handcrafted channel stack.

For example, if the future checkpoints are:

- `t = 200`
- `t = 400`
- `t = 600`
- `t = 800`
- `t = 1000`

then each label row should look like:

```text
[P(survive to 200), P(survive to 400), P(survive to 600), P(survive to 800), P(survive to 1000)]
```

and must satisfy:

```text
1 >= P_1 >= P_2 >= ... >= P_K >= 0
```

## Why This Formulation Fits the Physics Better

Compared with a direct unconstrained regression head, this autoregressive attenuation form is useful because:

- it enforces physical monotonicity by construction
- it interprets each decoder step as a conditional survival retention rate
- it avoids impossible predictions such as late-time survival becoming larger than early-time survival
- it matches the project’s shift from hard classification toward survival/distribution forecasting

## Practical Training Suggestions

- Start with `encoder_type="gru"` as the most stable baseline.
- Use `loss_type="mse"` first, since the targets are continuous survival fractions.
- Try `hidden_dim` in `[64, 128, 256]`.
- Add teacher forcing only if optimization is unstable; for this formulation it is optional.
- Evaluate both:
  - per-horizon MAE/MSE
  - monotonic calibration of the full curve

If needed, the next extension would be to replace the simple encoder input with the richer `task_b.ipynb` feature builder and then compare:

- 15 orbital features only
- 15 orbital features + geometric invariants
- full notebook feature stack
