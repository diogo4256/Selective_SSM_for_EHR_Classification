import sys
sys.path.append("/zhome/15/3/203515/ehr/Odyssey/odyssey/models/ehr_mamba2")
import torch
import torch.nn as nn
from model import Mamba2Pretrain
from typing import Optional

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

        # Initialize Mamba2Pretrain model with corrected dimensions
        self.mamba_model = Mamba2Pretrain(
            vocab_size=0,  # Not used since we're using inputs_embeds
            embedding_size=self.sensor_axis_dim,
            state_size=ssm_state_size,
            num_hidden_layers=layers,
            expand=expand_factor,
            dropout_prob=dropout,
            num_heads=1,  # Mamba doesn't use attention heads
            head_dim=self.sensor_axis_dim,  # Set to full dimension
            max_seq_length=5000,
            padding_idx=0,
            cls_idx=1,
            eos_idx=2,
            n_groups=1,
            chunk_size=self.sensor_axis_dim,  # Match embedding size
        )

        # Rest of the initialization remains the same
        self.sensor_embedding = nn.Linear(self.sensor_axis_dim_in, self.sensor_axis_dim)
        self.static_embedding = nn.Linear(self.static_count, self.static_out)
        self.nonlinear_merger = nn.Linear(
            self.sensor_axis_dim + self.static_out,
            self.sensor_axis_dim + self.static_out,
        )
        self.classifier = nn.Linear(
            self.sensor_axis_dim + self.static_out, 
            num_classes
        )
        self.time_encoding = nn.Linear(1, self.sensor_axis_dim)

    def _add_time_encoding(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Reshape time to (batch, seq_len, 1)
        time = time.unsqueeze(-1)
        # Project time to d_model dimension
        time_embedding = self.time_encoding(time)
        # Add to input
        return x + time_embedding

    def forward(self, x, static, time, sensor_mask, **kwargs):
        # x: [batch_size, sensors_count, seq_len]
        # static: [batch_size, static_count]
        # time: [batch_size, seq_len]
        # sensor_mask: [batch_size, sensors_count, seq_len]

        # Prepare input
        x_time = x.permute(0, 2, 1)  # [batch_size, seq_len, sensors_count]
        x_sensor_mask = sensor_mask.permute(0, 2, 1).float()  # [batch_size, seq_len, sensors_count]
        x_time = torch.cat([x_time, x_sensor_mask], dim=2)  # [batch_size, seq_len, 2 * sensors_count]

        # Create mask
        mask = torch.sum(x_time != 0, dim=2) > 0  # [batch_size, seq_len]
        mask = mask.long()  # Convert to LongTensor

        # Sensor embedding
        x_time = self.sensor_embedding(x_time)  # [batch_size, seq_len, d_model]

        # Add time encoding
        x_time = self._add_time_encoding(x_time, time)  # [batch_size, seq_len, d_model]

        # Debugging: Print tensor shapes
        print(f"x_time shape: {x_time.shape}")
        print(f"mask shape: {mask.shape}")

        # Apply Mamba model
        outputs = self.mamba_model.model(
            inputs_embeds=x_time,
            attention_mask=mask,
            return_dict=True,
        )
        last_hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, d_model]

        # Pooling
        if self.pooling == "mean":
            x_pooled = masked_mean_pooling(last_hidden_states, mask)
        elif self.pooling == "max":
            x_pooled = masked_max_pooling(last_hidden_states, mask)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        # Static embedding
        static_embedded = self.static_embedding(static)  # [batch_size, static_out]

        # Combine features
        x_combined = torch.cat([x_pooled, static_embedded], dim=1)  # [batch_size, d_model + static_out]
        x_combined = self.nonlinear_merger(x_combined).relu()  # [batch_size, d_model + static_out]

        # Classification
        logits = self.classifier(x_combined)  # [batch_size, num_classes]
        return logits
    
class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self._num_timescales = d_model // 2

    def forward(self, P_time):
        # P_time: [batch_size, time_steps]
        # Implement positional encoding logic here
        # Return: [batch_size, time_steps, d_model]
        pe = self.getPE(P_time)
        return pe

    def getPE(self, P_time):
        batch_size, seq_len = P_time.shape
        P_time = P_time.float()
        timescales = self.max_len ** torch.linspace(0, 1, self._num_timescales, device=P_time.device)
        scaled_time = P_time.unsqueeze(-1) / timescales
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return pe

def masked_mean_pooling(data_tensor, mask):
    mask_expanded = mask.unsqueeze(-1).expand(data_tensor.size()).float()
    data_summed = torch.sum(data_tensor * mask_expanded, dim=1)
    data_counts = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return data_summed / data_counts

def masked_max_pooling(data_tensor, mask):
    mask_expanded = mask.unsqueeze(-1).expand(data_tensor.size()).float()
    data_tensor = data_tensor.masked_fill(mask_expanded == 0, float('-inf'))
    return torch.max(data_tensor, dim=1).values