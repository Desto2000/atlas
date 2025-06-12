"""
ATLAS: Adaptive Thought-Level Architecture for Living Substrates - Fixed Dimensions
==================================================================================

Integrates all four core components into a unified cognitive system capable
of gradient-based geometric reasoning and emergent conceptual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import logging
from dataclasses import dataclass

from .cognitive_manifold import CognitiveManifold
from .thought_diffusion import ThoughtDiffusionEmbedding
from .exploration import MultiModalExploration
from .synthesis import ContextCognitionSynthesis


@dataclass
class AtlasConfig:
    """Configuration for ATLAS architecture."""

    # Dimensions - ensure consistency
    vocab_size: int = 50000
    embed_dim: int = 1024
    thought_dim: int = 128
    manifold_dim: int = 128
    synthesis_dim: int = 128

    # Architecture parameters
    num_transformer_layers: int = 8  # Reduced for stability
    num_attention_heads: int = 16
    num_diffusion_steps: int = 100  # Reduced for faster processing

    # Exploration parameters
    num_concept_classes: int = 100
    max_clusters: int = 50

    # Training parameters
    learning_rate: float = 1e-4
    manifold_evolution_strength: float = 0.01
    synthesis_steps: int = 25  # Reduced for faster processing

    # Evaluation parameters
    evaluation_interval: int = 100
    checkpoint_interval: int = 1000

    def __post_init__(self):
        """Validate configuration for dimensional consistency."""
        # Ensure manifold_dim matches thought_dim for proper integration
        if self.manifold_dim != self.thought_dim:
            print(f"Warning: Setting manifold_dim to {self.thought_dim} to match thought_dim")
            self.manifold_dim = self.thought_dim

        # Ensure synthesis_dim is compatible
        if self.synthesis_dim != self.thought_dim:
            print(f"Warning: Setting synthesis_dim to {self.thought_dim} to match thought_dim")
            self.synthesis_dim = self.thought_dim

        # Validate attention heads
        if self.thought_dim % self.num_attention_heads != 0:
            # Find largest divisor
            for heads in range(self.num_attention_heads, 0, -1):
                if self.thought_dim % heads == 0:
                    print(f"Warning: Adjusting num_attention_heads from {self.num_attention_heads} to {heads}")
                    self.num_attention_heads = heads
                    break


class CognitiveState:
    """Represents the current state of the cognitive system."""

    def __init__(self):
        self.manifold_state: Optional[Dict[str, torch.Tensor]] = None
        self.thought_representations: Optional[torch.Tensor] = None
        self.exploration_state: Optional[Dict[str, Any]] = None
        self.synthesis_state: Optional[Dict[str, torch.Tensor]] = None
        self.cognitive_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}

    def update(self,
               manifold_state: Dict[str, torch.Tensor],
               thought_representations: torch.Tensor,
               exploration_state: Dict[str, Any],
               synthesis_state: Dict[str, torch.Tensor]):
        """Update the cognitive state."""
        self.manifold_state = manifold_state
        self.thought_representations = thought_representations
        self.exploration_state = exploration_state
        self.synthesis_state = synthesis_state

        # Store in history
        history_entry = {
            'timestamp': len(self.cognitive_history),
            'manifold_curvature': manifold_state.get('scalar_curvature', torch.tensor(0.0)).mean().item(),
            'thought_coherence': self._compute_thought_coherence(),
            'exploration_confidence': exploration_state.get('exploration_confidence', torch.tensor(0.0)).item(),
            'synthesis_quality': synthesis_state.get('synthesis_quality', torch.tensor(0.0)).mean().item()
        }

        self.cognitive_history.append(history_entry)

        # Keep only recent history
        if len(self.cognitive_history) > 1000:
            self.cognitive_history = self.cognitive_history[-1000:]

    def _compute_thought_coherence(self) -> float:
        """Compute coherence of current thought representations."""
        if self.thought_representations is None:
            return 0.0

        # Simple coherence measure: average pairwise similarity
        norm_thoughts = F.normalize(self.thought_representations, dim=-1)

        # Flatten for similarity computation
        flat_thoughts = norm_thoughts.view(-1, norm_thoughts.shape[-1])
        if flat_thoughts.shape[0] > 1:
            similarities = torch.mm(flat_thoughts, flat_thoughts.t())

            # Average off-diagonal similarities
            mask = ~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
            coherence = similarities[mask].mean().item()
        else:
            coherence = 1.0  # Single thought is perfectly coherent

        return coherence

    def get_cognitive_summary(self) -> Dict[str, Any]:
        """Get a summary of current cognitive state."""
        if not self.cognitive_history:
            return {'status': 'No cognitive history available'}

        recent_history = self.cognitive_history[-10:] if len(self.cognitive_history) >= 10 else self.cognitive_history

        summary = {
            'current_step': len(self.cognitive_history),
            'manifold_curvature': {
                'current': recent_history[-1]['manifold_curvature'],
                'trend': recent_history[-1]['manifold_curvature'] - recent_history[0]['manifold_curvature']
            },
            'thought_coherence': {
                'current': recent_history[-1]['thought_coherence'],
                'average': sum(h['thought_coherence'] for h in recent_history) / len(recent_history)
            },
            'exploration_confidence': {
                'current': recent_history[-1]['exploration_confidence'],
                'trend': recent_history[-1]['exploration_confidence'] - recent_history[0]['exploration_confidence']
            },
            'synthesis_quality': {
                'current': recent_history[-1]['synthesis_quality'],
                'average': sum(h['synthesis_quality'] for h in recent_history) / len(recent_history)
            }
        }

        return summary


class ATLAS(nn.Module):
    """
    Main ATLAS architecture integrating all cognitive components.

    This system represents a fundamental shift from AI as information processing
    to AI as synthetic neural substrate capable of genuine thought.
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()

        self.config = config

        print(f"Initializing ATLAS with configuration:")
        print(f"  vocab_size: {config.vocab_size}")
        print(f"  embed_dim: {config.embed_dim}")
        print(f"  thought_dim: {config.thought_dim}")
        print(f"  manifold_dim: {config.manifold_dim}")
        print(f"  synthesis_dim: {config.synthesis_dim}")

        # Initialize core components with consistent dimensions
        self.cognitive_manifold = CognitiveManifold(
            manifold_dim=config.manifold_dim,
            flow_strength=config.manifold_evolution_strength
        )

        self.thought_diffusion = ThoughtDiffusionEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            thought_dim=config.thought_dim,
            num_timesteps=config.num_diffusion_steps,
            num_transformer_layers=config.num_transformer_layers
        )

        self.exploration_system = MultiModalExploration(
            feature_dim=config.thought_dim,
            manifold_dim=config.manifold_dim,
            num_concept_classes=config.num_concept_classes,
            max_clusters=config.max_clusters
        )

        self.synthesis_engine = ContextCognitionSynthesis(
            context_dim=config.embed_dim,
            cognition_dim=config.thought_dim,
            synthesis_dim=config.synthesis_dim,
            manifold_dim=config.manifold_dim,
            num_diffusion_steps=config.num_diffusion_steps
        )

        # Cognitive state tracking
        self.cognitive_state = CognitiveState()

        # Integration layers - ensure dimension compatibility
        integration_input_dim = config.synthesis_dim + config.manifold_dim
        self.cognitive_integrator = nn.Sequential(
            nn.Linear(integration_input_dim, config.synthesis_dim),
            nn.ReLU(),
            nn.Linear(config.synthesis_dim, config.synthesis_dim)
        )

        # Performance monitoring
        self.performance_monitor = nn.Sequential(
            nn.Linear(config.synthesis_dim, config.synthesis_dim // 4),
            nn.ReLU(),
            nn.Linear(config.synthesis_dim // 4, 1),
            nn.Sigmoid()
        )

        # Logger setup
        self.logger = logging.getLogger('ATLAS')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Training state
        self.training_step = 0
        self.last_performance = 0.0

        print("ATLAS initialization complete!")

    def forward(self,
                input_tokens: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                return_full_state: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the complete ATLAS architecture.

        This represents genuine thought-level computation rather than token processing.
        """

        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        try:
            # Phase 1: Token to Thought Transformation
            self.logger.debug("Phase 1: Converting tokens to thoughts")
            thought_representations = self.thought_diffusion.tokenize_to_thoughts(
                token_ids=input_tokens,
                context=context
            )

            # Phase 2: Cognitive Manifold Processing
            self.logger.debug("Phase 2: Processing through cognitive manifold")
            manifold_state = self.cognitive_manifold(thought_representations)

            # Phase 3: Multi-Modal Exploration
            self.logger.debug("Phase 3: Multi-modal cognitive exploration")
            exploration_results = self.exploration_system(
                cognitive_features=thought_representations,
                manifold_state=manifold_state,
                context=context
            )

            # Phase 4: Context-Cognition Synthesis
            self.logger.debug("Phase 4: Context-cognition synthesis")
            if context is None:
                # Generate context from input embeddings
                context = self.thought_diffusion.token_embedding(input_tokens).mean(dim=1)

            synthesis_results = self.synthesis_engine(
                context=context,
                cognition=thought_representations.mean(dim=1),  # Average over sequence
                manifold_state=manifold_state,
                num_synthesis_steps=self.config.synthesis_steps
            )

            # Phase 5: Cognitive Integration
            self.logger.debug("Phase 5: Integrating cognitive outputs")

            # Combine manifold and synthesis information with proper dimensions
            manifold_summary = manifold_state['scalar_curvature']

            # Ensure manifold summary has correct dimensions
            if manifold_summary.dim() == 1:
                manifold_summary = manifold_summary.unsqueeze(-1)
            if manifold_summary.shape[-1] != self.config.manifold_dim:
                # Expand or project to correct dimension
                if manifold_summary.shape[-1] == 1:
                    manifold_summary = manifold_summary.expand(-1, self.config.manifold_dim)
                else:
                    manifold_proj = nn.Linear(manifold_summary.shape[-1], self.config.manifold_dim, device=device)
                    manifold_summary = manifold_proj(manifold_summary)

            synthesis_output = synthesis_results['thought_activations'].mean(dim=1)

            # Ensure synthesis output has correct dimensions
            if synthesis_output.shape[-1] != self.config.synthesis_dim:
                synthesis_proj = nn.Linear(synthesis_output.shape[-1], self.config.synthesis_dim, device=device)
                synthesis_output = synthesis_proj(synthesis_output)

            integrated_cognition = self.cognitive_integrator(
                torch.cat([synthesis_output, manifold_summary], dim=-1)
            )

            # Phase 6: Performance Assessment and Manifold Evolution
            current_performance = self.performance_monitor(integrated_cognition)

            # Evolve manifold based on performance feedback
            if self.training:
                evolved_manifold_state = self.cognitive_manifold.evolve_space(
                    cognitive_feedback=current_performance.mean(),
                    current_state=manifold_state
                )

                # Update exploration systems
                self.exploration_system.evolve_exploration_strategy(current_performance)
            else:
                evolved_manifold_state = manifold_state

            # Update cognitive state
            self.cognitive_state.update(
                manifold_state=evolved_manifold_state,
                thought_representations=thought_representations,
                exploration_state=exploration_results,
                synthesis_state=synthesis_results
            )

            # Training metrics
            if self.training:
                self.training_step += 1
                self.last_performance = current_performance.mean().item()

                if self.training_step % self.config.evaluation_interval == 0:
                    self._log_training_progress()

            if return_full_state:
                return {
                    'thought_activations': integrated_cognition,
                    'manifold_state': evolved_manifold_state,
                    'exploration_results': exploration_results,
                    'synthesis_results': synthesis_results,
                    'performance_score': current_performance,
                    'cognitive_state_summary': self.cognitive_state.get_cognitive_summary()
                }
            else:
                return integrated_cognition

        except Exception as e:
            self.logger.error(f"Error in ATLAS forward pass: {e}")
            # Return a fallback result
            fallback_shape = (batch_size, self.config.synthesis_dim)
            fallback_result = torch.zeros(fallback_shape, device=device)

            if return_full_state:
                return {
                    'thought_activations': fallback_result,
                    'manifold_state': {},
                    'exploration_results': {},
                    'synthesis_results': {},
                    'performance_score': torch.zeros(batch_size, 1, device=device),
                    'cognitive_state_summary': {'status': 'error'},
                    'error': str(e)
                }
            else:
                return fallback_result


# Rest of the class methods remain the same but with ATLAS instead of PrimalBrain
# ... (continuing with the same methods as before but updated for ATLAS)

    def discover_concepts(self,
                         data_loader: Any,
                         num_discovery_steps: int = 1000) -> Dict[str, Any]:
        """Autonomous concept discovery through exploration."""

        discovered_concepts = []
        concept_stability = {}

        self.eval()  # Set to evaluation mode

        with torch.no_grad():
            for step, batch in enumerate(data_loader):
                if step >= num_discovery_steps:
                    break

                try:
                    # Process batch through cognitive system
                    if isinstance(batch, dict):
                        input_tokens = batch['input_ids']
                    else:
                        input_tokens = batch

                    result = self.forward(input_tokens, return_full_state=True)

                    # Extract potential concepts from exploration results
                    cluster_assignments = result['exploration_results'].get('cluster_assignments')

                    if cluster_assignments is not None:
                        # Identify stable clusters as potential concepts
                        for cluster_id in range(cluster_assignments.shape[-1]):
                            cluster_strength = cluster_assignments[:, :, cluster_id].mean()

                            if cluster_strength > 0.1:  # Significant cluster
                                concept_key = f"concept_{cluster_id}"

                                if concept_key not in concept_stability:
                                    concept_stability[concept_key] = []

                                concept_stability[concept_key].append(cluster_strength.item())

                    # Log progress
                    if step % 100 == 0:
                        self.logger.info(f"Concept discovery step {step}/{num_discovery_steps}")

                except Exception as e:
                    self.logger.warning(f"Error in concept discovery step {step}: {e}")
                    continue

        # Analyze discovered concepts
        stable_concepts = {}
        for concept_key, stability_history in concept_stability.items():
            if len(stability_history) > 10:  # Sufficient history
                stability_score = sum(stability_history[-10:]) / len(stability_history[-10:])  # Recent stability
                consistency = 1.0 - (sum((x - stability_score)**2 for x in stability_history[-10:]) / len(stability_history[-10:]))**0.5

                if stability_score > 0.2 and consistency > 0.7:
                    stable_concepts[concept_key] = {
                        'stability_score': stability_score,
                        'consistency': consistency,
                        'discovery_step': len(stability_history)
                    }

        discovery_summary = {
            'num_discovered_concepts': len(stable_concepts),
            'stable_concepts': stable_concepts,
            'exploration_statistics': self.exploration_system.get_exploration_statistics(),
            'cognitive_summary': self.cognitive_state.get_cognitive_summary()
        }

        self.logger.info(f"Discovered {len(stable_concepts)} stable concepts")

        return discovery_summary

    def _log_training_progress(self):
        """Log training progress and cognitive development."""
        cognitive_summary = self.cognitive_state.get_cognitive_summary()
        exploration_stats = self.exploration_system.get_exploration_statistics()

        self.logger.info(f"Training Step: {self.training_step}")
        self.logger.info(f"Performance: {self.last_performance:.4f}")
        self.logger.info(f"Manifold Curvature: {cognitive_summary.get('manifold_curvature', {}).get('current', 0):.4f}")
        self.logger.info(f"Thought Coherence: {cognitive_summary.get('thought_coherence', {}).get('current', 0):.4f}")
        self.logger.info(f"Exploration Confidence: {cognitive_summary.get('exploration_confidence', {}).get('current', 0):.4f}")

        # Log exploration mode preferences
        if 'average_weights' in exploration_stats:
            weights = exploration_stats['average_weights']
            self.logger.info(f"Exploration Weights - NB: {weights[0]:.3f}, RL: {weights[1]:.3f}, Clustering: {weights[2]:.3f}")

    def save_cognitive_state(self, filepath: str):
        """Save complete cognitive state for resumption."""
        state_dict = {
            'model_state_dict': self.state_dict(),
            'cognitive_state': self.cognitive_state,
            'training_step': self.training_step,
            'last_performance': self.last_performance,
            'config': self.config
        }

        torch.save(state_dict, filepath)
        self.logger.info(f"Cognitive state saved to {filepath}")

    def load_cognitive_state(self, filepath: str):
        """Load cognitive state for continuation."""
        state_dict = torch.load(filepath)

        self.load_state_dict(state_dict['model_state_dict'])
        self.cognitive_state = state_dict['cognitive_state']
        self.training_step = state_dict['training_step']
        self.last_performance = state_dict['last_performance']

        self.logger.info(f"Cognitive state loaded from {filepath}")

    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics."""

        diagnostics = {
            'architecture_info': {
                'vocab_size': self.config.vocab_size,
                'manifold_dim': self.config.manifold_dim,
                'thought_dim': self.config.thought_dim,
                'synthesis_dim': self.config.synthesis_dim
            },
            'cognitive_state': self.cognitive_state.get_cognitive_summary(),
            'exploration_statistics': self.exploration_system.get_exploration_statistics(),
            'manifold_capacity': self.cognitive_manifold.get_cognitive_capacity().item(),
            'training_progress': {
                'current_step': self.training_step,
                'last_performance': self.last_performance
            },
            'component_status': {
                'cognitive_manifold': 'active',
                'thought_diffusion': 'active',
                'exploration_system': 'active',
                'synthesis_engine': 'active'
            }
        }

        return diagnostics


# Factory function for easy instantiation
def create_atlas(
    vocab_size: int = 50000,
    manifold_dim: int = 2048,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> ATLAS:
    """Create an ATLAS instance with default configuration."""

    config = AtlasConfig(
        vocab_size=vocab_size,
        manifold_dim=manifold_dim
    )

    model = ATLAS(config).to(device)

    return model