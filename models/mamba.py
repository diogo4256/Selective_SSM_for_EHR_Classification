import sys
import logging
import torch
import torch.nn as nn
from transformers import Mamba2Config, Mamba2ForCausalLM
from typing import Optional

class EncoderClassifierMamba(nn.Module):
    def __init__(
        self, 
        device="cpu",
        pooling="mean",
        num_classes=2,
        sensors_count=37,
        static_count=8,
        layers=3,
        d_model=256,
        ssm_state_size=16,
        expand_factor=2,
        dropout=0.2,
        **kwargs
    ):
        super().__init__()
        
        # Core model dimensions
        self.pooling = pooling
        self.device = device
        self.sensors_count = sensors_count
        self.static_count = static_count
        self.sensor_axis_dim_in = 2 * self.sensors_count  # 74
        self.sensor_axis_dim = d_model  # 256
        self.static_out = self.static_count  # Keep original dimension 8

        # Mamba configuration
        self.config = Mamba2Config(
            vocab_size=0,
            hidden_size=self.sensor_axis_dim,
            state_size=ssm_state_size,
            num_hidden_layers=layers,
            expand=expand_factor,
            conv_kernel=4,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            n_groups=1,
            chunk_size=self.sensor_axis_dim,
            dropout=dropout,
            num_heads=1,
            head_dim=self.sensor_axis_dim,
            max_position_embeddings=5000,
            tie_word_embeddings=False,  # Add this
        )

        # Initialize Mamba model
        self.mamba_model = Mamba2ForCausalLM(config=self.config)

        # Projection layers
        self.sensor_embedding = nn.Linear(self.sensor_axis_dim_in, self.sensor_axis_dim)  # 74 -> 256
        self.static_embedding = nn.Linear(self.static_count, self.static_out)  # 8 -> 8
        
        # Merger and classifier layers
        merged_dim = self.sensor_axis_dim + self.static_out  # 256 + 8 = 264
        self.nonlinear_merger = nn.Linear(merged_dim, merged_dim)  # 264 -> 264
        
        self.classifier = nn.Linear(merged_dim, num_classes)  # 264 -> 2
        self.time_encoding = nn.Linear(1, self.sensor_axis_dim)  # 1 -> 256

    def forward(self, x, static, time, sensor_mask, **kwargs):
        # Prepare sequence input
        x_time = x.permute(0, 2, 1)
        x_sensor_mask = sensor_mask.permute(0, 2, 1).float()
        x_time = torch.cat([x_time, x_sensor_mask], dim=2)

        # Create attention mask
        mask = torch.sum(x_time != 0, dim=2) > 0
        mask = mask.long()

        # Project to model dimension
        x_time = self.sensor_embedding(x_time)
        x_time = self._add_time_encoding(x_time, time)

        # Apply Mamba model
        outputs = self.mamba_model(
            inputs_embeds=x_time,
            attention_mask=mask,
            return_dict=True,
            output_hidden_states=True,
        )

        # Use input embeddings if model output is empty
        if outputs.logits.size(-1) == 0:
            last_hidden_states = x_time  # Use input embeddings: [16, seq_len, 256]
        else:
            last_hidden_states = outputs.logits
    
        # Pool sequence dimension
        if self.pooling == "mean":
            x_pooled = masked_mean_pooling(last_hidden_states, mask)
        elif self.pooling == "max":
            x_pooled = masked_max_pooling(last_hidden_states, mask)
        
        # Process static features  
        static_embedded = self.static_embedding(static)
        
        # Combine features
        x_combined = torch.cat([x_pooled, static_embedded], dim=1)
        x_combined = self.nonlinear_merger(x_combined).relu()

        return self.classifier(x_combined)

    def _add_time_encoding(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time = time.unsqueeze(-1)
        time_embedding = self.time_encoding(time)
        return x + time_embedding

# Masked pooling functions
def masked_mean_pooling(data_tensor, mask):
    mask_expanded = mask.unsqueeze(-1).expand(data_tensor.size()).float()
    data_summed = torch.sum(data_tensor * mask_expanded, dim=1)
    data_counts = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return data_summed / data_counts

def masked_max_pooling(data_tensor, mask):
    mask_expanded = mask.unsqueeze(-1).expand(data_tensor.size()).float()
    data_tensor = data_tensor.masked_fill(mask_expanded == 0, float('-inf'))
    return torch.max(data_tensor, dim=1).values