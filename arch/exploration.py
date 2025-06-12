"""
Multi-Modal Cognitive Exploration
=================================

Implements three complementary exploration modes: Geometric Naive Bayes,
Manifold Reinforcement Learning, and Dynamic Clustering for autonomous
cognitive space evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class GeometricNaiveBayes(nn.Module):
    """Naive Bayes classifier that uses manifold geometry for conditional independence."""

    def __init__(self,
                 feature_dim: int,
                 num_classes: int,
                 manifold_aware: bool = True):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.manifold_aware = manifold_aware

        # Prior probabilities (learnable)
        self.log_priors = nn.Parameter(torch.zeros(num_classes))

        # Feature statistics for each class
        self.class_means = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.class_log_vars = nn.Parameter(torch.zeros(num_classes, feature_dim))

        if manifold_aware:
            # Manifold-aware covariance structure
            self.manifold_interaction = nn.Parameter(torch.eye(feature_dim))

    def forward(self,
                x: torch.Tensor,
                metric_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute class probabilities using geometric structure."""
        batch_size, feature_dim = x.shape

        # Expand for broadcasting
        x_expanded = x.unsqueeze(1)  # [B, 1, D]
        means = self.class_means.unsqueeze(0)  # [1, C, D]
        log_vars = self.class_log_vars.unsqueeze(0)  # [1, C, D]

        # Compute differences
        diff = x_expanded - means  # [B, C, D]

        if self.manifold_aware and metric_tensor is not None:
            # Use manifold metric for distance computation
            # d² = (x-μ)ᵀ G (x-μ)
            metric = metric_tensor.unsqueeze(1)  # [B, 1, D, D]
            diff_expanded = diff.unsqueeze(-1)  # [B, C, D, 1]

            # Compute Mahalanobis distance using manifold metric
            mahalanobis_sq = torch.matmul(
                torch.matmul(diff.unsqueeze(-2), metric), diff_expanded
            ).squeeze(-1).squeeze(-1)  # [B, C]

            # Log-likelihood using manifold distance
            log_likelihood = -0.5 * (
                    mahalanobis_sq +
                    torch.sum(log_vars, dim=-1) +
                    feature_dim * np.log(2 * np.pi)
            )
        else:
            # Standard Naive Bayes assumption (feature independence)
            var = torch.exp(log_vars)
            log_likelihood = -0.5 * torch.sum(
                (diff ** 2) / var + log_vars + np.log(2 * np.pi), dim=-1
            )  # [B, C]

        # Add log priors
        log_posteriors = log_likelihood + self.log_priors.unsqueeze(0)

        # Normalize to get probabilities
        log_probs = F.log_softmax(log_posteriors, dim=-1)

        return log_probs

    def update_statistics(self, x: torch.Tensor, labels: torch.Tensor) -> None:
        """Update class statistics with new data."""
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    class_data = x[mask]

                    # Update means (exponential moving average)
                    alpha = 0.1
                    new_mean = class_data.mean(dim=0)
                    self.class_means[c] = (1 - alpha) * self.class_means[c] + alpha * new_mean

                    # Update variances
                    new_var = class_data.var(dim=0)
                    self.class_log_vars[c] = (1 - alpha) * self.class_log_vars[c] + alpha * torch.log(new_var + 1e-8)

                    # Update priors
                    class_freq = mask.float().mean()
                    self.log_priors[c] = (1 - alpha) * self.log_priors[c] + alpha * torch.log(class_freq + 1e-8)


class ManifoldRLAgent(nn.Module):
    """Reinforcement Learning agent that operates on manifold transformations."""

    def __init__(self,
                 manifold_dim: int,
                 action_dim: int = 64,
                 hidden_dim: int = 512):
        super().__init__()

        self.manifold_dim = manifold_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # State encoder (manifold geometry → state representation)
        self.state_encoder = nn.Sequential(
            nn.Linear(manifold_dim * manifold_dim + manifold_dim, hidden_dim),  # metric + curvature
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions bounded to [-1, 1]
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Action interpretation networks
        self.curvature_actions = nn.Linear(action_dim, manifold_dim)
        self.metric_actions = nn.Linear(action_dim, manifold_dim * manifold_dim)

    def encode_manifold_state(self,
                              metric_tensor: torch.Tensor,
                              curvature: torch.Tensor) -> torch.Tensor:
        """Encode manifold geometry into state representation."""
        batch_size = metric_tensor.shape[0]

        # Flatten metric tensor
        metric_flat = metric_tensor.view(batch_size, -1)

        # Combine metric and curvature information
        state_features = torch.cat([metric_flat, curvature], dim=-1)

        # Encode to state representation
        state = self.state_encoder(state_features)

        return state

    def forward(self,
                metric_tensor: torch.Tensor,
                curvature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action, log_prob, and value."""
        state = self.encode_manifold_state(metric_tensor, curvature)

        # Get action from policy
        action_logits = self.actor(state)

        # Sample action (for training) or use mean (for inference)
        if self.training:
            noise = torch.randn_like(action_logits) * 0.1
            action = torch.tanh(action_logits + noise)
        else:
            action = action_logits

        # Compute log probability (approximate for continuous actions)
        log_prob = -0.5 * torch.sum((action - action_logits) ** 2, dim=-1)

        # Get value estimate
        value = self.critic(state)

        return action, log_prob, value

    def interpret_actions(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert action vector to manifold transformations."""
        curvature_adjustments = self.curvature_actions(actions)
        metric_adjustments = self.metric_actions(actions)

        # Reshape metric adjustments
        batch_size = actions.shape[0]
        metric_adjustments = metric_adjustments.view(batch_size, self.manifold_dim, self.manifold_dim)

        # Ensure metric adjustments preserve positive definiteness
        metric_adjustments = torch.tril(metric_adjustments)  # Lower triangular

        return {
            'curvature_adjustment': curvature_adjustments,
            'metric_adjustment': metric_adjustments
        }

    def compute_reward(self,
                       old_state: Dict[str, torch.Tensor],
                       new_state: Dict[str, torch.Tensor],
                       cognitive_performance: torch.Tensor) -> torch.Tensor:
        """Compute reward based on cognitive improvement."""
        # Primary reward: cognitive performance improvement
        performance_reward = cognitive_performance

        # Secondary rewards: geometric stability
        old_curvature = old_state.get('scalar_curvature', torch.zeros_like(performance_reward))
        new_curvature = new_state.get('scalar_curvature', torch.zeros_like(performance_reward))

        # Penalize extreme curvature changes
        curvature_penalty = -0.1 * torch.abs(new_curvature - old_curvature)

        # Reward manifold smoothness
        smoothness_reward = -0.05 * torch.norm(new_curvature, dim=-1)

        total_reward = performance_reward + curvature_penalty + smoothness_reward

        return total_reward


class DynamicClustering(nn.Module):
    """Dynamic clustering system for unsupervised structure discovery."""

    def __init__(self,
                 feature_dim: int,
                 max_clusters: int = 50,
                 min_cluster_size: int = 5):
        super().__init__()

        self.feature_dim = feature_dim
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size

        # Learnable cluster parameters
        self.cluster_centers = nn.Parameter(torch.randn(max_clusters, feature_dim))
        self.cluster_scales = nn.Parameter(torch.ones(max_clusters))
        self.cluster_weights = nn.Parameter(torch.ones(max_clusters) / max_clusters)

        # Attention mechanism for adaptive clustering
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)

        # Cluster evolution network
        self.evolution_network = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.register_buffer('num_active_clusters', torch.tensor(max_clusters))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform dynamic clustering on input features."""
        batch_size, seq_len, feature_dim = x.shape

        # Compute attention-weighted features
        attended_features, attention_weights = self.attention(x, x, x)

        # Compute distances to cluster centers
        x_flat = attended_features.view(-1, feature_dim)  # [B*S, D]
        centers = self.cluster_centers[:self.num_active_clusters]  # [K, D]

        # Compute soft assignments using Gaussian mixture
        distances = torch.cdist(x_flat.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)  # [B*S, K]
        scaled_distances = distances / (self.cluster_scales[:self.num_active_clusters].unsqueeze(0) + 1e-8)

        # Soft assignments
        assignments = F.softmax(-scaled_distances, dim=-1)  # [B*S, K]

        # Weighted assignments
        weighted_assignments = assignments * self.cluster_weights[:self.num_active_clusters].unsqueeze(0)
        weighted_assignments = weighted_assignments / (weighted_assignments.sum(dim=-1, keepdim=True) + 1e-8)

        # Reshape back
        assignments = assignments.view(batch_size, seq_len, -1)
        weighted_assignments = weighted_assignments.view(batch_size, seq_len, -1)

        return {
            'assignments': assignments,
            'weighted_assignments': weighted_assignments,
            'attention_weights': attention_weights,
            'cluster_centers': centers,
            'num_clusters': self.num_active_clusters
        }

    def update_clusters(self, x: torch.Tensor, assignments: torch.Tensor) -> None:
        """Update cluster parameters based on new assignments."""
        with torch.no_grad():
            batch_size, seq_len, feature_dim = x.shape
            x_flat = x.view(-1, feature_dim)
            assignments_flat = assignments.view(-1, assignments.shape[-1])

            num_clusters = self.num_active_clusters.item()

            for k in range(num_clusters):
                # Get points assigned to this cluster
                cluster_weights = assignments_flat[:, k]
                total_weight = cluster_weights.sum()

                if total_weight > self.min_cluster_size:
                    # Update cluster center (weighted average)
                    weighted_sum = torch.sum(cluster_weights.unsqueeze(-1) * x_flat, dim=0)
                    new_center = weighted_sum / total_weight

                    # Exponential moving average update
                    alpha = 0.1
                    self.cluster_centers[k] = (1 - alpha) * self.cluster_centers[k] + alpha * new_center

                    # Update cluster scale
                    distances = torch.norm(x_flat - new_center.unsqueeze(0), dim=-1)
                    weighted_distances = cluster_weights * distances
                    new_scale = torch.sum(weighted_distances) / total_weight
                    self.cluster_scales[k] = (1 - alpha) * self.cluster_scales[k] + alpha * new_scale

                    # Update cluster weight
                    new_weight = total_weight / x_flat.shape[0]
                    self.cluster_weights[k] = (1 - alpha) * self.cluster_weights[k] + alpha * new_weight

    def split_merge_clusters(self, x: torch.Tensor, assignments: torch.Tensor) -> None:
        """Dynamically split or merge clusters based on data distribution."""
        with torch.no_grad():
            num_clusters = self.num_active_clusters.item()

            # Check for clusters to merge (too close centers)
            if num_clusters > 2:
                centers = self.cluster_centers[:num_clusters]
                distances = torch.cdist(centers, centers)

                # Find closest pair (excluding diagonal)
                distances.fill_diagonal_(float('inf'))
                min_dist, min_idx = torch.min(distances.view(-1), dim=0)

                if min_dist < 0.1:  # Merge threshold
                    i, j = min_idx // num_clusters, min_idx % num_clusters
                    if i != j:
                        # Merge clusters i and j
                        weight_i = self.cluster_weights[i]
                        weight_j = self.cluster_weights[j]
                        total_weight = weight_i + weight_j

                        # Weighted average of centers
                        new_center = (weight_i * self.cluster_centers[i] + weight_j * self.cluster_centers[j]) / total_weight

                        # Update cluster i with merged properties
                        self.cluster_centers[i] = new_center
                        self.cluster_weights[i] = total_weight
                        self.cluster_scales[i] = (self.cluster_scales[i] + self.cluster_scales[j]) / 2

                        # Remove cluster j by moving last cluster to position j
                        if j < num_clusters - 1:
                            self.cluster_centers[j] = self.cluster_centers[num_clusters - 1]
                            self.cluster_weights[j] = self.cluster_weights[num_clusters - 1]
                            self.cluster_scales[j] = self.cluster_scales[num_clusters - 1]

                        self.num_active_clusters -= 1

            # Check for clusters to split (high variance)
            x_flat = x.view(-1, x.shape[-1])
            assignments_flat = assignments.view(-1, assignments.shape[-1])

            for k in range(self.num_active_clusters.item()):
                cluster_weights = assignments_flat[:, k]
                if cluster_weights.sum() > 2 * self.min_cluster_size:
                    # Check cluster variance
                    center = self.cluster_centers[k]
                    weighted_points = cluster_weights.unsqueeze(-1) * x_flat
                    weighted_center = weighted_points.sum(dim=0) / cluster_weights.sum()

                    # Compute weighted variance
                    diff = x_flat - weighted_center.unsqueeze(0)
                    weighted_var = torch.sum(cluster_weights.unsqueeze(-1) * diff ** 2) / cluster_weights.sum()

                    if weighted_var > 1.0 and self.num_active_clusters < self.max_clusters:  # Split threshold
                        # Split cluster k
                        # Find direction of maximum variance
                        cov_matrix = torch.cov(diff.T, aweights=cluster_weights)
                        eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
                        max_var_dir = eigenvecs[:, -1]  # Eigenvector with largest eigenvalue

                        # Create two new centers
                        offset = 0.1 * torch.sqrt(eigenvals[-1]) * max_var_dir
                        new_center1 = center - offset
                        new_center2 = center + offset

                        # Update original cluster
                        self.cluster_centers[k] = new_center1
                        self.cluster_weights[k] = self.cluster_weights[k] / 2

                        # Add new cluster
                        new_idx = self.num_active_clusters.item()
                        self.cluster_centers[new_idx] = new_center2
                        self.cluster_weights[new_idx] = self.cluster_weights[k]
                        self.cluster_scales[new_idx] = self.cluster_scales[k]

                        self.num_active_clusters += 1
                        break  # Only split one cluster per update


class ExplorationCoordinator(nn.Module):
    """Meta-learning system that coordinates the three exploration modes."""

    def __init__(self,
                 feature_dim: int,
                 num_classes: int = 10,
                 manifold_dim: int = 2048):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.manifold_dim = manifold_dim

        # Exploration modules
        self.geometric_nb = GeometricNaiveBayes(feature_dim, num_classes)
        self.manifold_rl = ManifoldRLAgent(manifold_dim)
        self.dynamic_clustering = DynamicClustering(feature_dim)

        # Coordination weights (learnable)
        self.coordination_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        # Meta-learning network for coordination
        self.meta_network = nn.Sequential(
            nn.Linear(feature_dim + manifold_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Three coordination weights
            nn.Softmax(dim=-1)
        )

    def forward(self,
                features: torch.Tensor,
                manifold_state: Dict[str, torch.Tensor],
                context: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Coordinate all three exploration modes."""

        # Run individual exploration modes

        # 1. Geometric Naive Bayes
        metric_tensor = manifold_state.get('metric_tensor')
        nb_probs = self.geometric_nb(features.mean(dim=1), metric_tensor)

        # 2. Manifold RL
        curvature = manifold_state.get('scalar_curvature', torch.zeros(features.shape[0]))
        rl_action, rl_log_prob, rl_value = self.manifold_rl(metric_tensor, curvature)
        rl_transformations = self.manifold_rl.interpret_actions(rl_action)

        # 3. Dynamic Clustering
        clustering_results = self.dynamic_clustering(features)

        # Compute coordination weights
        meta_input = torch.cat([
            features.mean(dim=1),  # Average features
            curvature.unsqueeze(-1) if curvature.dim() == 1 else curvature,  # Curvature info
            nb_probs.mean(dim=1) if nb_probs.dim() > 1 else nb_probs.unsqueeze(0)  # Classification info
        ], dim=-1)

        dynamic_weights = self.meta_network(meta_input)

        # Combine static and dynamic weights
        final_weights = F.softmax(self.coordination_weights + dynamic_weights.mean(dim=0), dim=0)

        # Unified exploration decision
        exploration_signal = {
            'nb_influence': final_weights[0],
            'rl_influence': final_weights[1],
            'clustering_influence': final_weights[2],
            'nb_predictions': nb_probs,
            'rl_actions': rl_transformations,
            'clustering_assignments': clustering_results['weighted_assignments'],
            'coordination_weights': final_weights
        }

        return exploration_signal

    def update_exploration_systems(self,
                                   features: torch.Tensor,
                                   manifold_state: Dict[str, torch.Tensor],
                                   feedback: torch.Tensor,
                                   labels: Optional[torch.Tensor] = None) -> None:
        """Update all exploration systems based on feedback."""

        # Update Geometric Naive Bayes
        if labels is not None:
            self.geometric_nb.update_statistics(features.mean(dim=1), labels)

        # Update Dynamic Clustering
        clustering_results = self.dynamic_clustering(features)
        self.dynamic_clustering.update_clusters(features, clustering_results['assignments'])
        self.dynamic_clustering.split_merge_clusters(features, clustering_results['assignments'])

        # RL updates would typically be done with a separate optimizer
        # Here we just compute the reward for the RL agent
        old_curvature = manifold_state.get('scalar_curvature', torch.zeros_like(feedback))
        rl_reward = self.manifold_rl.compute_reward(
            {'scalar_curvature': old_curvature},
            {'scalar_curvature': old_curvature},  # Placeholder for new state
            feedback
        )

        return rl_reward


class MultiModalExploration(nn.Module):
    """
    Main multi-modal exploration system that orchestrates autonomous cognitive space evolution.

    Combines Geometric Naive Bayes, Manifold RL, and Dynamic Clustering through
    a meta-learning coordination system.
    """

    def __init__(self,
                 feature_dim: int = 2048,
                 manifold_dim: int = 2048,
                 num_concept_classes: int = 100,
                 max_clusters: int = 50):
        super().__init__()

        self.feature_dim = feature_dim
        self.manifold_dim = manifold_dim
        self.num_concept_classes = num_concept_classes

        # Main exploration coordinator
        self.coordinator = ExplorationCoordinator(
            feature_dim=feature_dim,
            num_classes=num_concept_classes,
            manifold_dim=manifold_dim
        )

        # Exploration history for meta-learning
        self.register_buffer('exploration_history', torch.zeros(1000, 3))  # Store coordination weights
        self.register_buffer('performance_history', torch.zeros(1000))
        self.register_buffer('history_step', torch.tensor(0))

    def forward(self,
                cognitive_features: torch.Tensor,
                manifold_state: Dict[str, torch.Tensor],
                context: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Perform multi-modal exploration of cognitive space."""

        # Get unified exploration signal
        exploration_results = self.coordinator(cognitive_features, manifold_state, context)

        # Store exploration history
        step = self.history_step.item()
        if step < 1000:
            self.exploration_history[step] = exploration_results['coordination_weights']
            self.history_step += 1

        # Generate exploration actions
        exploration_actions = self._generate_exploration_actions(exploration_results)

        # Compute exploration confidence
        exploration_confidence = self._compute_exploration_confidence(exploration_results)

        return {
            'exploration_actions': exploration_actions,
            'exploration_confidence': exploration_confidence,
            'coordination_weights': exploration_results['coordination_weights'],
            'nb_predictions': exploration_results['nb_predictions'],
            'rl_transformations': exploration_results['rl_actions'],
            'cluster_assignments': exploration_results['clustering_assignments'],
            'raw_exploration_data': exploration_results
        }

    def _generate_exploration_actions(self, exploration_results: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generate concrete exploration actions from all modes."""
        weights = exploration_results['coordination_weights']

        # Weighted combination of exploration directions
        exploration_actions = {
            'concept_discovery': weights[0] * self._nb_exploration_direction(exploration_results['nb_predictions']),
            'manifold_evolution': weights[1] * self._rl_exploration_direction(exploration_results['rl_actions']),
            'structure_discovery': weights[2] * self._clustering_exploration_direction(exploration_results['clustering_assignments'])
        }

        return exploration_actions

    def _nb_exploration_direction(self, nb_predictions: torch.Tensor) -> torch.Tensor:
        """Convert NB predictions to exploration direction."""
        # Explore towards uncertain classifications
        entropy = -torch.sum(torch.exp(nb_predictions) * nb_predictions, dim=-1)
        return F.normalize(entropy.unsqueeze(-1), dim=-1)

    def _rl_exploration_direction(self, rl_actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert RL actions to exploration direction."""
        # Use curvature adjustments as exploration signal
        curvature_adj = rl_actions['curvature_adjustment']
        return F.normalize(curvature_adj, dim=-1)

    def _clustering_exploration_direction(self, cluster_assignments: torch.Tensor) -> torch.Tensor:
        """Convert clustering to exploration direction."""
        # Explore towards cluster boundaries (high uncertainty)
        assignment_entropy = -torch.sum(cluster_assignments * torch.log(cluster_assignments + 1e-8), dim=-1)
        return F.normalize(assignment_entropy.mean(dim=1).unsqueeze(-1), dim=-1)

    def _compute_exploration_confidence(self, exploration_results: Dict[str, Any]) -> torch.Tensor:
        """Compute confidence in current exploration strategy."""
        # Base confidence on agreement between exploration modes
        weights = exploration_results['coordination_weights']

        # Higher confidence when one mode dominates (clear signal)
        # Lower confidence when modes are balanced (uncertain)
        weight_entropy = -torch.sum(weights * torch.log(weights + 1e-8))
        max_entropy = torch.log(torch.tensor(3.0))  # log(3) for 3 modes

        # Invert entropy - high entropy = low confidence
        confidence = 1.0 - (weight_entropy / max_entropy)

        return confidence

    def evolve_exploration_strategy(self, performance_feedback: torch.Tensor) -> None:
        """Evolve exploration strategy based on performance feedback."""
        step = self.history_step.item()
        if step > 0 and step <= 1000:
            self.performance_history[step - 1] = performance_feedback.mean()

        # Analyze exploration effectiveness
        if step > 10:  # Need some history
            recent_performance = self.performance_history[max(0, step-10):step]
            recent_weights = self.exploration_history[max(0, step-10):step]

            # Find best performing exploration strategy
            best_idx = torch.argmax(recent_performance)
            best_weights = recent_weights[best_idx]

            # Adjust coordination weights towards successful strategies
            with torch.no_grad():
                alpha = 0.1
                self.coordinator.coordination_weights.data = (
                        (1 - alpha) * self.coordinator.coordination_weights.data +
                        alpha * best_weights
                )

    def get_exploration_statistics(self) -> Dict[str, Any]:
        """Get statistics about exploration behavior."""
        step = min(self.history_step.item(), 1000)

        if step == 0:
            return {'exploration_stats': 'No exploration history yet'}

        weights_history = self.exploration_history[:step]
        performance_history = self.performance_history[:step]

        stats = {
            'average_weights': weights_history.mean(dim=0),
            'weight_std': weights_history.std(dim=0),
            'average_performance': performance_history.mean(),
            'performance_trend': performance_history[-10:].mean() - performance_history[:10].mean() if step > 20 else 0,
            'exploration_diversity': weights_history.std(dim=0).sum(),  # Higher = more diverse exploration
            'dominant_mode': torch.argmax(weights_history.mean(dim=0))
        }

        return stats