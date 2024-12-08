import sys
import logging
import torch
import torch.nn as nn
from transformers import Mamba2Config, Mamba2ForCausalLM
from typing import Optional

class EncoderClassifierMamba(nn.Module):
    def __init__(
        self,
        device: str = "cpu",
        pooling: str = "mean",
        num_classes: int = 2,
        sensors_count: int = 37,
        static_count: int = 8,
        layers: int = 4,
        d_model: int = 256,
        ssm_state_size: int = 16,
        expand_factor: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Validate inputs
        assert d_model % 8 == 0, "d_model must be divisible by 8"
        assert pooling in ["mean", "max"]
        
        # Fixed relationships with expansion factor
        self.sensor_axis_dim_in = 2 * sensors_count    
        self.base_dim = d_model
        self.expanded_dim = d_model * expand_factor
        self.static_out = self.expanded_dim  # Match expanded dimension
        self.expand = expand_factor                    

        # Mamba configuration with expanded dimensions
        self.config = Mamba2Config(
            vocab_size=0,
            hidden_size=self.expanded_dim,  # Use expanded dimension
            state_size=ssm_state_size,
            num_hidden_layers=layers,
            expand=self.expand,
            conv_kernel=4,
            dropout=dropout,
            num_heads=1,
            head_dim=self.expanded_dim,  # Use expanded dimension
            max_position_embeddings=5000,
            chunk_size=self.expanded_dim,  # Use expanded dimension
            n_groups=1,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            use_cache=False,
        )

        # Initialize models with gradient checkpointing
        self.mamba_model = Mamba2ForCausalLM(config=self.config)
        self.mamba_model.gradient_checkpointing_enable()

        # Projection layers with expanded dimensions
        self.sensor_embedding = nn.Linear(self.sensor_axis_dim_in, self.expanded_dim)
        self.static_embedding = nn.Linear(static_count, self.expanded_dim)  # Match expanded dimension
        self.time_encoding = nn.Linear(1, self.expanded_dim)
        
        # Output layers with matched dimensions
        merged_dim = self.expanded_dim * 2  # Both inputs are expanded_dim
        self.nonlinear_merger = nn.Linear(merged_dim, merged_dim)
        self.classifier = nn.Linear(merged_dim, num_classes)
        
        self.pooling = pooling
        self.device = device
        self.to(device)

    def forward(self, x, static, time, sensor_mask, **kwargs):
        # Prepare sequence input
        x_time = x.permute(0, 2, 1)                     # [B, T, F]
        x_sensor_mask = sensor_mask.permute(0, 2, 1).float()
        x_time = torch.cat([x_time, x_sensor_mask], dim=2)  # [B, T, 2F]

        # Create attention mask and project
        mask = torch.sum(x_time != 0, dim=2) > 0
        mask = mask.long()
        x_time = self.sensor_embedding(x_time)          # [B, T, D*expand]
        x_time = self._add_time_encoding(x_time, time)

        # Apply Mamba
        outputs = self.mamba_model(
            inputs_embeds=x_time,
            attention_mask=mask,
            return_dict=True,
        )

        # Get hidden states (maintaining expanded dimension)
        last_hidden_states = (x_time if outputs.logits.size(-1) == 0 
                            else outputs.logits)

        # Pool sequence with dynamic dimension handling
        x_pooled = (masked_mean_pooling(last_hidden_states, mask) if self.pooling == "mean"
                    else masked_max_pooling(last_hidden_states, mask))

        # Process static features and combine
        static_embedded = self.static_embedding(static)  # Now outputs expanded_dim
        x_combined = torch.cat([x_pooled, static_embedded], dim=1)  # Both tensors have expanded_dim
        x_combined = self.nonlinear_merger(x_combined).relu()

        return self.classifier(x_combined)

    def _add_time_encoding(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Add time encoding to input tensor.
        Args:
            x: Input tensor [B, T, D*expand]
            time: Time tensor [B, T]
        Returns:
            Tensor with time encoding added [B, T, D*expand]
        """
        time = time.unsqueeze(-1)  # [B, T, 1]
        time_embedding = self.time_encoding(time)  # [B, T, D*expand]
        return x + time_embedding  # [B, T, D*expand]

def masked_mean_pooling(data_tensor, mask):
    """Apply masked mean pooling to input tensor.
    Args:
        data_tensor: Input tensor [B, T, D]
        mask: Boolean mask [B, T]
    Returns:
        Pooled tensor [B, D]
    """
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, data_tensor.size(-1)).float()
    data_summed = torch.sum(data_tensor * mask_expanded, dim=1)
    data_counts = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return data_summed / data_counts

def masked_max_pooling(data_tensor, mask):
    """Apply masked max pooling to input tensor.
    Args:
        data_tensor: Input tensor [B, T, D]
        mask: Boolean mask [B, T]
    Returns:
        Pooled tensor [B, D]
    """
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, data_tensor.size(-1)).float()
    data_tensor = data_tensor.masked_fill(mask_expanded == 0, float('-inf'))
    return torch.max(data_tensor, dim=1).values