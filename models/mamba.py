import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except ImportError:
    # Fallback to causal-conv1d implementation which has better compatibility
    from mamba_ssm.modules.mamba_simple import Mamba
from typing import Optional

class MambaEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        depth: int,
        ssm_state_size: int = 16,
        expand_factor: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=ssm_state_size,
                d_conv=4,
                expand=expand_factor,
                dropout=dropout
            ) for _ in range(depth)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

class EncoderClassifierMamba(nn.Module):
    def __init__(
        self,
        device="cpu",
        pooling="mean",
        num_classes=2,
        sensors_count=37,
        static_count=8,
        layers=1,
        d_model=256,
        ssm_state_size=16,
        expand_factor=2,
        dropout=0.2,
        **kwargs
    ):
        super().__init__()

        self.pooling = pooling
        self.device = device
        self.sensors_count = sensors_count
        self.static_count = static_count

        # Input dimension for sensor data (doubled to account for mask values)
        self.sensor_axis_dim_in = 2 * self.sensors_count
        self.sensor_axis_dim = d_model
        self.static_out = self.static_count + 4

        # Mamba encoder layers
        self.mamba_layers = MambaEncoder(
            d_model=self.sensor_axis_dim,
            depth=layers,
            ssm_state_size=ssm_state_size,
            expand_factor=expand_factor,
            dropout=dropout
        )

        # Input projections
        self.sensor_embedding = nn.Linear(self.sensor_axis_dim_in, self.sensor_axis_dim)
        self.static_embedding = nn.Linear(self.static_count, self.static_out)

        # Output layers
        self.nonlinear_merger = nn.Linear(
            self.sensor_axis_dim + self.static_out,
            self.sensor_axis_dim + self.static_out,
        )
        self.classifier = nn.Linear(
            self.sensor_axis_dim + self.static_out, 
            num_classes
        )

        # Time encoding (similar to transformer but simpler)
        self.time_encoding = nn.Linear(1, self.sensor_axis_dim)

    def _add_time_encoding(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Reshape time to (batch, seq_len, 1)
        time = time.unsqueeze(-1)
        # Project time to d_model dimension
        time_embedding = self.time_encoding(time)
        # Add to input
        return x + time_embedding

    def forward(self, x, static, time, sensor_mask, **kwargs):
        # Prepare input similar to transformer version
        x_time = torch.clone(x)  # (N, F, T)
        x_time = torch.permute(x_time, (0, 2, 1))  # (N, T, F)
        
        # Create mask for non-zero values
        mask = torch.count_nonzero(x_time, dim=2) > 0

        # Add missing sensor value indicators
        x_sensor_mask = torch.clone(sensor_mask)  # (N, F, T)
        x_sensor_mask = torch.permute(x_sensor_mask, (0, 2, 1))  # (N, T, F)
        x_time = torch.cat([x_time, x_sensor_mask], axis=2)  # (N, T, 2F)

        # Project sensor data to model dimension
        x_time = self.sensor_embedding(x_time)  # (N, T, d_model)

        # Add time encoding
        x_time = self._add_time_encoding(x_time, time)

        # Apply Mamba layers
        x_time = self.mamba_layers(x_time, mask=mask)

        # Pooling operations
        if self.pooling == "mean":
            x_time = masked_mean_pooling(x_time, mask)
        elif self.pooling == "max":
            x_time = masked_max_pooling(x_time, mask)
        elif self.pooling == "sum":
            x_time = torch.sum(x_time, dim=1)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        # Process static features and combine with sequence features
        static = self.static_embedding(static)
        x_merged = torch.cat((x_time, static), axis=1)
        
        # Final classification
        nonlinear_merged = self.nonlinear_merger(x_merged).relu()
        return self.classifier(nonlinear_merged)

def masked_mean_pooling(datatensor, mask):
    """Compute masked mean pooling."""
    mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()
    data_summed = torch.sum(datatensor * mask_expanded, dim=1)
    data_counts = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return data_summed / data_counts

def masked_max_pooling(datatensor, mask):
    """Compute masked max pooling."""
    mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()
    datatensor[mask_expanded == 0] = -1e9
    return torch.max(datatensor, 1)[0]