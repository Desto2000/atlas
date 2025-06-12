"""
Thought Diffusion Embedding - Fixed Dimension Issues
====================================================

Transforms discrete tokens into continuous cognitive patterns through
diffusion processes, enabling thought-native computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
import numpy as np

from .device import device_consistent_operation, get_device_from_module, ensure_tensor_device


class CognitiveNoiseModel(nn.Module):
    """Structured cognitive noise that respects conceptual relationships."""

    def __init__(self, embedding_dim: int, concept_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.concept_dim = concept_dim

        # Concept relationship encoder - outputs concept_dim
        self.concept_encoder = nn.Sequential(
            nn.Linear(embedding_dim, concept_dim),
            nn.Tanh(),
            nn.Linear(concept_dim, concept_dim)
        )

        # Bridge from concept space back to embedding space for covariance computation
        self.concept_to_embedding = nn.Sequential(
            nn.Linear(concept_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Network to generate concept-aware covariance - now takes embedding_dim input
        self.covariance_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim * embedding_dim)
        )

    @device_consistent_operation
    def forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate concept-aware noise covariance matrix."""
        batch_size, seq_len, embed_dim = x.shape
        device = x.device

        # Encode conceptual structure
        concept_repr = self.concept_encoder(x)  # [B, S, concept_dim]

        # Include temporal and context information
        if context is not None:
            context = ensure_tensor_device(context, device)
            # Project context to concept dimension if needed
            if context.shape[-1] != self.concept_dim:
                context_proj = nn.Linear(context.shape[-1], self.concept_dim, device=device)
                context = context_proj(context)
            concept_repr = concept_repr + context.unsqueeze(1)

        # Time-dependent scaling
        time_scale = torch.exp(-t.unsqueeze(-1).unsqueeze(-1))
        concept_repr = concept_repr * time_scale

        # Transform concept representation back to embedding space
        embedding_repr = self.concept_to_embedding(concept_repr)  # [B, S, embedding_dim]

        # Generate covariance parameters using embedding-space representation
        cov_input = embedding_repr.mean(dim=1)  # [B, embedding_dim]
        cov_params = self.covariance_network(cov_input)  # [B, embedding_dim²]
        cov_matrix = cov_params.view(batch_size, embed_dim, embed_dim)

        # Ensure positive semi-definiteness
        L = torch.tril(cov_matrix)
        epsilon = 1e-6
        Sigma = torch.bmm(L, L.transpose(-2, -1)) + epsilon * torch.eye(embed_dim, device=device)

        return Sigma

    @device_consistent_operation
    def sample_noise(self,
                     x: torch.Tensor,
                     t: torch.Tensor,
                     context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample structured cognitive noise."""
        Sigma = self.forward(x, t, context)
        device = x.device

        # Sample from multivariate normal with learned covariance
        batch_size, seq_len, embed_dim = x.shape
        standard_noise = torch.randn_like(x)

        # Transform noise using Cholesky decomposition
        try:
            L = torch.linalg.cholesky(Sigma)
            structured_noise = torch.einsum('bij,bsj->bsi', L, standard_noise)
        except RuntimeError:
            # Fallback to diagonal noise if cholesky fails
            diagonal_noise = torch.sqrt(torch.diagonal(Sigma, dim1=-2, dim2=-1)).unsqueeze(1)
            structured_noise = standard_noise * diagonal_noise

        return structured_noise


class ThoughtTransformer(nn.Module):
    """Transformer architecture for thought-level processing."""

    def __init__(self,
                 embed_dim: int = 1024,
                 num_heads: int = 16,
                 num_layers: int = 12,
                 feedforward_dim: int = 4096):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Ensure num_heads divides embed_dim evenly
        if embed_dim % num_heads != 0:
            self.num_heads = 8  # Fallback to 8 heads
            print(f"Warning: Adjusted num_heads to {self.num_heads} for embed_dim {embed_dim}")

        # Time embedding for diffusion timestep
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=self.num_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    @device_consistent_operation
    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process thoughts through transformer with time conditioning."""
        device = x.device

        # Time embedding
        time_emb = self.time_embed(self._timestep_embedding(t, self.embed_dim, device))

        # Add time embedding to input
        x = x + time_emb.unsqueeze(1)

        # Add context if provided
        if context is not None:
            context = ensure_tensor_device(context, device)
            # Project context to correct dimension if needed
            if context.shape[-1] != self.embed_dim:
                context_proj = nn.Linear(context.shape[-1], self.embed_dim, device=device)
                context = context_proj(context)
            x = x + context.unsqueeze(1)

        # Transformer processing
        output = self.transformer(x, src_key_padding_mask=mask)

        # Output projection
        return self.output_proj(output)

    def _timestep_embedding(self, timesteps: torch.Tensor, dim: int, device: torch.device) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device=device) / half
        )

        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding


class ThoughtDiffusionScheduler:
    """Manages the diffusion schedule for thought transformation."""

    def __init__(self,
                 num_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 schedule_type: str = "cosine"):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type

        # Create noise schedule
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For posterior q(x_{t-1}|x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _cosine_schedule(self, timesteps: int) -> torch.Tensor:
        """Cosine noise schedule for smoother diffusion."""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean data according to the schedule."""
        device = x_start.device

        # Ensure scheduler tensors are on correct device
        if self.alphas_cumprod.device != device:
            self.betas = self.betas.to(device)
            self.alphas = self.alphas.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
            self.posterior_variance = self.posterior_variance.to(device)

        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[timesteps].flatten()

        # Broadcast to match input dimensions
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


class ThoughtDiffusionEmbedding(nn.Module):
    """
    Main thought diffusion system that transforms tokens into continuous thought patterns.

    This system enables thought-native computation by operating on continuous
    cognitive representations rather than discrete tokens.
    """

    def __init__(self,
                 vocab_size: int = 50000,
                 embed_dim: int = 1024,
                 thought_dim: int = 2048,
                 num_timesteps: int = 1000,
                 num_transformer_layers: int = 12):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.thought_dim = thought_dim
        self.num_timesteps = num_timesteps

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(8192, embed_dim)  # Max sequence length

        # Thought space projection
        self.token_to_thought = nn.Linear(embed_dim, thought_dim)
        self.thought_to_token = nn.Linear(thought_dim, embed_dim)

        # Diffusion components - use thought_dim for noise model
        self.noise_model = CognitiveNoiseModel(thought_dim, concept_dim=min(thought_dim//4, 512))
        self.scheduler = ThoughtDiffusionScheduler(num_timesteps)

        # Calculate appropriate number of heads for thought dimension
        num_heads = 16
        while thought_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        self.thought_transformer = ThoughtTransformer(
            embed_dim=thought_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            feedforward_dim=thought_dim * 2
        )

        # Context integration - match thought_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(thought_dim, thought_dim // 2),
            nn.ReLU(),
            nn.Linear(thought_dim // 2, thought_dim)
        )

        print(f"ThoughtDiffusionEmbedding initialized:")
        print(f"  embed_dim: {embed_dim}, thought_dim: {thought_dim}")
        print(f"  noise_model concept_dim: {self.noise_model.concept_dim}")
        print(f"  transformer heads: {num_heads}")

    @device_consistent_operation
    def tokenize_to_thoughts(self,
                           token_ids: torch.Tensor,
                           context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert discrete tokens to continuous thought representations."""
        batch_size, seq_len = token_ids.shape
        device = get_device_from_module(self)

        # Ensure input is on correct device
        token_ids = ensure_tensor_device(token_ids, device)

        # Token and position embeddings
        token_emb = self.token_embedding(token_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        # Combined embedding
        combined_emb = token_emb + pos_emb

        # Project to thought space
        thought_repr = self.token_to_thought(combined_emb)

        # Apply diffusion transformation
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Generate context-aware noise
        noise = self.noise_model.sample_noise(thought_repr, timesteps, context)

        # Add structured noise
        noisy_thoughts = self.scheduler.add_noise(thought_repr, noise, timesteps)

        # Denoise to get final thought representation
        denoised_thoughts = self.thought_transformer(noisy_thoughts, timesteps, context)

        return denoised_thoughts

    @device_consistent_operation
    def thoughts_to_tokens(self,
                          thoughts: torch.Tensor,
                          target_vocab_size: Optional[int] = None) -> torch.Tensor:
        """Convert continuous thoughts back to discrete token representations."""
        device = thoughts.device

        # Project back to token embedding space
        token_logits = self.thought_to_token(thoughts)

        # Final projection to vocabulary
        if target_vocab_size is None:
            target_vocab_size = self.vocab_size

        vocab_proj = nn.Linear(self.embed_dim, target_vocab_size, device=device)
        logits = vocab_proj(token_logits)

        return logits

    @device_consistent_operation
    def diffuse_thoughts(self,
                        thoughts: torch.Tensor,
                        num_steps: int = 50,
                        guidance_scale: float = 1.0,
                        context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform thought-level diffusion for exploration and refinement."""
        batch_size = thoughts.shape[0]
        device = thoughts.device

        # Start from noise
        current_thoughts = torch.randn_like(thoughts)

        # Reverse diffusion process
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)

        for t in timesteps:
            t_batch = t.repeat(batch_size)

            # Predict noise
            predicted_noise = self.thought_transformer(current_thoughts, t_batch, context)

            # DDPM sampling step
            alpha_t = self.scheduler.alphas[t]
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t]
            beta_t = self.scheduler.betas[t]

            # Compute x_{t-1}
            if t > 0:
                noise = torch.randn_like(current_thoughts)
                sigma_t = torch.sqrt(self.scheduler.posterior_variance[t])
            else:
                noise = 0
                sigma_t = 0

            current_thoughts = (1 / torch.sqrt(alpha_t)) * (
                current_thoughts - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            ) + sigma_t * noise

        return current_thoughts

    def compute_thought_similarity(self, thoughts1: torch.Tensor, thoughts2: torch.Tensor) -> torch.Tensor:
        """Compute semantic similarity between thought representations."""
        # Normalize thoughts
        norm1 = F.normalize(thoughts1, dim=-1)
        norm2 = F.normalize(thoughts2, dim=-1)

        # Compute cosine similarity
        similarity = torch.sum(norm1 * norm2, dim=-1)

        return similarity

    def interpolate_thoughts(self,
                           thought1: torch.Tensor,
                           thought2: torch.Tensor,
                           alpha: float = 0.5) -> torch.Tensor:
        """Interpolate between two thought representations."""
        # Spherical linear interpolation (slerp) for better semantic interpolation
        dot_product = torch.sum(thought1 * thought2, dim=-1, keepdim=True)

        # Clamp to avoid numerical issues
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        theta = torch.acos(torch.abs(dot_product))
        sin_theta = torch.sin(theta)

        # Handle case where thoughts are nearly identical
        mask = sin_theta < 1e-6

        # Linear interpolation for nearly identical thoughts
        linear_interp = (1 - alpha) * thought1 + alpha * thought2

        # Spherical interpolation
        w1 = torch.sin((1 - alpha) * theta) / sin_theta
        w2 = torch.sin(alpha * theta) / sin_theta
        spherical_interp = w1 * thought1 + w2 * thought2

        # Use linear interpolation where spherical is unstable
        result = torch.where(mask, linear_interp, spherical_interp)

        return result

    def get_thought_trajectory(self,
                             start_thoughts: torch.Tensor,
                             end_thoughts: torch.Tensor,
                             num_steps: int = 10) -> torch.Tensor:
        """Generate a trajectory through thought space."""
        trajectories = []

        for i in range(num_steps + 1):
            alpha = i / num_steps
            interpolated = self.interpolate_thoughts(start_thoughts, end_thoughts, alpha)
            trajectories.append(interpolated)

        return torch.stack(trajectories, dim=1)  # [batch, steps, seq_len, thought_dim]
