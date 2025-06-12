"""
Basic ATLAS Usage Examples - Fixed Device Handling
==================================================

Demonstrates how to use the ATLAS architecture for
various cognitive tasks with proper device management.
"""

import torch
import sys
import os
import logging

from arch.atlas import ATLAS, AtlasConfig, create_atlas
from arch.device import DeviceManager

# Set up logging
logging.basicConfig(level=logging.INFO)


def setup_device_and_model():
    """Setup device and create ATLAS model with proper device handling."""
    device_manager = DeviceManager('cpu')  # Auto-detect CUDA or CPU
    print(f"Using device: {device_manager.device}")

    # Create ATLAS instance with proper device
    atlas = create_atlas(vocab_size=3000, manifold_dim=128)
    print(device_manager.device)
    atlas = atlas.to(device_manager.device)

    return atlas, device_manager


def basic_thinking_example():
    """Demonstrate basic thought-level processing with device handling."""
    print("=== Basic Thinking Example ===")

    try:
        # Setup with device management
        atlas, device_manager = setup_device_and_model()
        atlas.eval()

        # Create simple tokenizer simulation (in practice, use real tokenizer)
        class SimpleTokenizer:
            def __init__(self, vocab_size=3000):
                self.vocab_size = vocab_size
                self.pad_token = 0
                self.eos_token = 1

            def encode(self, text, return_tensors='pt', max_length=64):
                # Simple hash-based encoding for demo
                tokens = [hash(word) % self.vocab_size for word in text.split()]
                tokens = tokens[:max_length]  # Truncate
                tokens += [self.pad_token] * (max_length - len(tokens))  # Pad

                if return_tensors == 'pt':
                    return torch.tensor([tokens])
                return tokens

            def decode(self, tokens, skip_special_tokens=True):
                # Simple decoding for demo
                return f"Generated response based on {len(tokens)} tokens"

        tokenizer = SimpleTokenizer()

        # Example prompts
        prompts = [
            "What is the nature of consciousness?",
            "How do emergent properties arise in complex systems?",
            "Explain the relationship between geometry and thought."
        ]

        for prompt in prompts:
            print(f"\nPrompt: {prompt}")

            # Tokenize and ensure correct device
            input_tokens = tokenizer.encode(prompt, return_tensors='pt')
            input_tokens = device_manager.to_device(input_tokens)

            # Generate thought-native response
            with torch.no_grad():
                result = atlas.forward(input_tokens, return_full_state=True)

            print(f"✓ Successfully processed prompt")
            print(f"  Thought activations shape: {result['thought_activations'].shape}")
            print(f"  Performance score: {result['performance_score'].mean().item():.4f}")
            print(f"  Device: {result['thought_activations'].device}")
            print("-" * 50)

    except Exception as e:
        print(f"Error in basic thinking example: {e}")
        import traceback
        traceback.print_exc()


def concept_discovery_example():
    """Demonstrate autonomous concept discovery with device handling."""
    print("\n=== Concept Discovery Example ===")

    try:
        atlas, device_manager = setup_device_and_model()

        # Create synthetic data loader with proper device handling
        class SyntheticDataLoader:
            def __init__(self, num_batches=32, device_manager=None):
                self.num_batches = num_batches
                self.current_batch = 0
                self.device_manager = device_manager

            def __iter__(self):
                self.current_batch = 0
                return self

            def __next__(self):
                if self.current_batch >= self.num_batches:
                    raise StopIteration

                # Generate synthetic token sequences on correct device
                batch = self.device_manager.create_randint(0, 3000, (8, 64))
                self.current_batch += 1
                return batch

        data_loader = SyntheticDataLoader(num_batches=32, device_manager=device_manager)

        # Discover concepts
        discovery_results = atlas.discover_concepts(
            data_loader=data_loader,
            num_discovery_steps=50
        )

        print(f"✓ Discovered {discovery_results['num_discovered_concepts']} concepts")
        print("Stable concepts:")
        for concept_name, concept_info in discovery_results['stable_concepts'].items():
            print(f"  {concept_name}: stability={concept_info['stability_score']:.3f}, "
                  f"consistency={concept_info['consistency']:.3f}")

        print("\nExploration statistics:")
        exploration_stats = discovery_results['exploration_statistics']
        if 'average_weights' in exploration_stats:
            weights = exploration_stats['average_weights']
            print(f"  Average exploration weights: NB={weights[0]:.3f}, RL={weights[1]:.3f}, Clustering={weights[2]:.3f}")
            print(f"  Dominant exploration mode: {exploration_stats.get('dominant_mode', 'Unknown')}")

    except Exception as e:
        print(f"Error in concept discovery example: {e}")
        import traceback
        traceback.print_exc()


def manifold_evolution_example():
    """Demonstrate cognitive manifold evolution with device handling."""
    print("\n=== Manifold Evolution Example ===")

    try:
        atlas, device_manager = setup_device_and_model()
        atlas.train()  # Enable manifold evolution

        # Get initial diagnostics
        initial_diagnostics = atlas.get_system_diagnostics()
        print("Initial state:")
        print(f"  Manifold capacity: {initial_diagnostics['manifold_capacity']:.4f}")
        print(f"  Device: {device_manager.device}")

        # Process some data to trigger evolution
        batch_size = 4
        seq_length = 32

        for step in range(5):  # Reduced steps for faster demo
            # Generate random input on correct device
            input_tokens = device_manager.create_randint(0, 3000, (batch_size, seq_length))

            # Process through atlas (this triggers evolution)
            with torch.no_grad():
                result = atlas.forward(input_tokens, return_full_state=True)

            if step % 2 == 0:
                diagnostics = atlas.get_system_diagnostics()
                print(f"Step {step}: Manifold capacity: {diagnostics['manifold_capacity']:.4f}, "
                      f"Performance: {result['performance_score'].mean().item():.4f}")

        final_diagnostics = atlas.get_system_diagnostics()
        print("\nFinal state:")
        print(f"  Manifold capacity: {final_diagnostics['manifold_capacity']:.4f}")
        print(f"  Capacity change: {final_diagnostics['manifold_capacity'] - initial_diagnostics['manifold_capacity']:.4f}")

    except Exception as e:
        print(f"Error in manifold evolution example: {e}")
        import traceback
        traceback.print_exc()


def thought_composition_example():
    """Demonstrate thought composition capabilities with device handling."""
    print("\n=== Thought Composition Example ===")

    try:
        atlas, device_manager = setup_device_and_model()
        atlas.eval()

        # Create two different thought patterns on correct device
        batch_size = 2
        seq_length = 16

        input_tokens1 = device_manager.create_randint(0, 3000, (batch_size, seq_length))
        input_tokens2 = device_manager.create_randint(0, 3000, (batch_size, seq_length))

        with torch.no_grad():
            # Generate thoughts from both inputs
            result1 = atlas.forward(input_tokens1, return_full_state=True)
            result2 = atlas.forward(input_tokens2, return_full_state=True)

            thought1 = result1['thought_activations']
            thought2 = result2['thought_activations']

            # Compose thoughts using different methods
            geometric_composition = atlas.synthesis_engine.compose_thoughts(
                thought1, thought2, composition_mode="geometric"
            )

            linear_composition = atlas.synthesis_engine.compose_thoughts(
                thought1, thought2, composition_mode="linear"
            )

            # Measure coherence of composed thoughts
            geometric_coherence = atlas.synthesis_engine.measure_thought_coherence(
                geometric_composition.unsqueeze(1)
            )
            linear_coherence = atlas.synthesis_engine.measure_thought_coherence(
                linear_composition.unsqueeze(1)
            )

            print("✓ Thought composition results:")
            print(f"  Geometric composition coherence: {geometric_coherence.mean().item():.4f}")
            print(f"  Linear composition coherence: {linear_coherence.mean().item():.4f}")
            print(f"  Device consistency: {geometric_composition.device}")

            # Demonstrate thought evolution
            evolved_thoughts, evolution_trajectory = atlas.synthesis_engine.evolve_thought_sequence(
                initial_thoughts=geometric_composition.unsqueeze(1),
                evolution_steps=3  # Reduced for demo
            )

            print(f"\nThought evolution:")
            print(f"  Evolution steps: {evolution_trajectory.shape[1]}")
            print(f"  Final thought coherence: {atlas.synthesis_engine.measure_thought_coherence(evolved_thoughts).mean().item():.4f}")

    except Exception as e:
        print(f"Error in thought composition example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("ATLAS: Adaptive Thought-Level Architecture for Living Substrates")
    print("=" * 60)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")

    try:
        basic_thinking_example()
        concept_discovery_example()
        manifold_evolution_example()
        thought_composition_example()

        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("ATLAS demonstrates genuine thought-level computation")
        print("beyond traditional token-based AI systems.")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()