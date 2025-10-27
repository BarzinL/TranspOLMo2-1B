#!/usr/bin/env python3
"""
Main orchestration script for full transparency analysis.
Runs all phases sequentially on OLMo2-1B.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import yaml
import gc

from src.config import Config
from src.models.loader import OLMo2Loader
from src.models.hooks import ActivationCapture
from src.extraction.dataset_builder import AnalysisDatasetBuilder
from src.analysis.geometry.manifold import ManifoldAnalyzer
from src.analysis.geometry.clustering import FeatureClusterer
from src.analysis.features.sparse_autoencoder import SAETrainer
from src.analysis.features.feature_analysis import FeatureAnalyzer
from src.analysis.circuits.discovery import CircuitDiscovery
from src.documentation.generator import DocumentationGenerator
from src.verification.intervention import ModelInterventions


def estimate_memory_requirements(model_params, num_samples, batch_size, hidden_size, seq_length=512):
    """
    Estimate memory requirements for activation extraction.

    Args:
        model_params: Number of model parameters
        num_samples: Number of samples to process
        batch_size: Batch size
        hidden_size: Hidden dimension size
        seq_length: Sequence length

    Returns:
        Dictionary with memory estimates in bytes
    """
    # Model weights in memory (assuming float16)
    model_memory = model_params * 2  # 2 bytes per param for float16

    # Activation memory per batch (float16)
    # batch_size * seq_length * hidden_size * 2 bytes
    activation_per_batch = batch_size * seq_length * hidden_size * 2

    # Gradient memory is negligible since we use no_grad()
    # But keep some buffer for intermediate computations
    buffer_memory = model_memory * 0.3  # 30% buffer

    # Peak memory estimate
    peak_memory = model_memory + activation_per_batch + buffer_memory

    return {
        'model_memory': model_memory,
        'activation_per_batch': activation_per_batch,
        'buffer_memory': buffer_memory,
        'peak_memory': peak_memory,
        'recommended_batch_size': int(batch_size)
    }


def save_checkpoint(checkpoint_path, layer_idx, completed_layers, all_texts):
    """Save progress checkpoint."""
    checkpoint = {
        'current_layer': layer_idx,
        'completed_layers': completed_layers,
        'num_texts': len(all_texts)
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(checkpoint_path):
    """Load progress checkpoint."""
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def diagnose_gpu_setup():
    """Diagnose GPU configuration for debugging."""
    print("=" * 80)
    print("GPU DIAGNOSTIC")
    print("=" * 80)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus == 0:
        print("⚠️  No GPUs detected - using CPU mode")
        print("=" * 80)
        return

    print(f"PyTorch detects: {num_gpus} GPU(s)\n")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_total = props.total_memory / 1e9

        # Try to get current memory usage
        try:
            mem_allocated = torch.cuda.memory_allocated(i) / 1e9
            mem_reserved = torch.cuda.memory_reserved(i) / 1e9
            mem_available = mem_total - mem_reserved
        except:
            mem_allocated = 0
            mem_reserved = 0
            mem_available = mem_total

        print(f"GPU {i}: {props.name}")
        print(f"  Total memory: {mem_total:.1f}GB")
        print(f"  Allocated: {mem_allocated:.2f}GB")
        print(f"  Reserved: {mem_reserved:.2f}GB")
        print(f"  Available: {mem_available:.1f}GB")
        print()

    if num_gpus > 1:
        total_memory = sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(num_gpus)
        )
        print(f"Total GPU memory across all devices: {total_memory/1e9:.1f}GB")
        print("Note: Multi-GPU detected - will use device_map='auto' for model distribution")

    print("=" * 80)


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def process_layer_with_oom_handling(model, dataloader, layer_name, layer_idx,
                                    device, initial_batch_size, max_batches=None):
    """
    Process a single layer with automatic OOM handling.

    Returns:
        Tuple of (activations, texts, final_batch_size)
    """
    batch_size = initial_batch_size
    min_batch_size = 4

    while batch_size >= min_batch_size:
        try:
            print(f"  Attempting with batch_size={batch_size}...")

            # Rebuild dataloader with current batch size
            from torch.utils.data import DataLoader
            current_dataloader = DataLoader(
                dataloader.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )

            batch_texts = []

            with ActivationCapture(model, device='cpu') as capturer:
                capturer.register_hooks([layer_name])

                with torch.no_grad():
                    for batch_idx, batch in enumerate(tqdm(current_dataloader, desc=f"Layer {layer_idx}")):
                        try:
                            inputs = {
                                'input_ids': batch['input_ids'].to(device),
                                'attention_mask': batch['attention_mask'].to(device)
                            }

                            # Forward pass
                            outputs = model(**inputs)

                            # Explicitly delete outputs to free memory
                            del outputs

                            # Store texts
                            batch_texts.extend(batch.get('text', [''] * len(batch['input_ids'])))

                            # Clear GPU memory periodically
                            if batch_idx % 5 == 0:
                                clear_gpu_memory()

                            # Limit batches if specified
                            if max_batches and batch_idx >= max_batches:
                                break

                        except torch.cuda.OutOfMemoryError:
                            print(f"  OOM during batch {batch_idx}, will retry with smaller batch size...")
                            clear_gpu_memory()
                            raise  # Re-raise to trigger outer catch

                # Get activations (already moved to CPU by ActivationCapture)
                acts = capturer.get_activations()

                if layer_name in acts:
                    activations = acts[layer_name]
                    print(f"  ✓ Captured activations: {activations.shape}")
                    return activations, batch_texts, batch_size
                else:
                    raise ValueError(f"No activations captured for {layer_name}")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"  ✗ OOM with batch_size={batch_size}")
                clear_gpu_memory()

                # Reduce batch size
                batch_size = batch_size // 2
                if batch_size < min_batch_size:
                    raise RuntimeError(
                        f"OOM even with minimum batch size {min_batch_size}. "
                        "Try using --dtype float16 or reducing --num-samples"
                    )
                print(f"  Retrying with reduced batch_size={batch_size}...")
            else:
                raise

    raise RuntimeError("Failed to process layer even with minimum batch size")


def main():
    """Run complete transparency analysis pipeline."""

    parser = argparse.ArgumentParser(description='Run TranspOLMo analysis pipeline')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (optional)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name or path (default: from config.py)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples for analysis (default: from config.py)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu) (default: from config.py)')
    parser.add_argument('--dtype', type=str, default=None,
                        help='Data type: float32, float16, bfloat16 (default: from config.py)')
    parser.add_argument('--skip-sae', action='store_true',
                        help='Skip SAE training (faster)')
    parser.add_argument('--layers', type=str, default=None,
                        help='Comma-separated layer indices to analyze (e.g., "0,6,11")')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for processing (default: from config.py)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: ./results)')

    args = parser.parse_args()

    # Initialize config from config.py (single source of truth)
    config = Config.default()

    # Load YAML config if provided
    yaml_config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            print(f"Loaded config from: {args.config}")
        else:
            print(f"Warning: Config file not found: {args.config}")

    # Apply YAML config values (if present)
    if 'model' in yaml_config:
        if 'name' in yaml_config['model']:
            config.model.model_name = yaml_config['model']['name']
        if 'revision' in yaml_config['model']:
            pass  # Can add revision support if needed

    if 'compute' in yaml_config:
        if 'device' in yaml_config['compute']:
            config.model.device = yaml_config['compute']['device']
        if 'dtype' in yaml_config['compute']:
            config.model.dtype = yaml_config['compute']['dtype']
        if 'batch_size' in yaml_config['compute']:
            config.extraction.batch_size = yaml_config['compute']['batch_size']

    if 'analysis' in yaml_config:
        if 'num_samples' in yaml_config['analysis']:
            config.extraction.num_samples = yaml_config['analysis']['num_samples']

    if 'paths' in yaml_config:
        if 'cache_dir' in yaml_config['paths']:
            config.model.cache_dir = Path(yaml_config['paths']['cache_dir'])
        if 'output_dir' in yaml_config['paths']:
            config.documentation.output_dir = Path(yaml_config['paths']['output_dir'])
        if 'hf_cache' in yaml_config['paths']:
            os.environ['HF_HOME'] = yaml_config['paths']['hf_cache']

    # CLI arguments override everything (highest priority)
    if args.model is not None:
        config.model.model_name = args.model
    if args.device is not None:
        config.model.device = args.device
    if args.dtype is not None:
        config.model.dtype = args.dtype
    if args.num_samples is not None:
        config.extraction.num_samples = args.num_samples
    if args.batch_size is not None:
        config.extraction.batch_size = args.batch_size
    if args.output_dir is not None:
        config.documentation.output_dir = Path(args.output_dir)

    # Parse skip_sae from YAML or CLI
    skip_sae = args.skip_sae
    if 'analysis' in yaml_config and 'skip_sae' in yaml_config['analysis']:
        skip_sae = skip_sae or yaml_config['analysis']['skip_sae']

    # Diagnose GPU setup
    diagnose_gpu_setup()
    print()

    print("=" * 80)
    print("TRANSPOLMO: First-Principles Neural Network Interpretability")
    print("=" * 80)
    print(f"\nModel: {config.model.model_name}")
    print(f"Device: {config.model.device}")
    print(f"Samples: {config.extraction.num_samples}")
    print()

    # Parse layer selection (CLI > YAML > default)
    if args.layers:
        layer_indices = [int(x) for x in args.layers.split(',')]
    elif 'analysis' in yaml_config and 'layers' in yaml_config['analysis']:
        layer_indices = yaml_config['analysis']['layers']
    else:
        layer_indices = None

    # Setup output directory early (needed for checkpoints)
    output_dir = Path(config.documentation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PHASE 1: Load Model
    print("\n" + "=" * 80)
    print("PHASE 1: Loading Model")
    print("=" * 80)

    loader = OLMo2Loader(
        model_name=config.model.model_name,
        cache_dir=config.model.cache_dir,
        device=config.model.device,
        dtype=config.model.dtype
    )

    model, tokenizer = loader.load()
    arch_info = loader.get_architecture_info(model)

    print(f"\nArchitecture:")
    print(f"  Layers: {arch_info['num_layers']}")
    print(f"  Hidden size: {arch_info['hidden_size']}")
    print(f"  Attention heads: {arch_info['num_attention_heads']}")
    print(f"  Vocabulary: {arch_info['vocab_size']}")

    # PHASE 2: Build Dataset
    print("\n" + "=" * 80)
    print("PHASE 2: Building Analysis Dataset")
    print("=" * 80)

    dataset_builder = AnalysisDatasetBuilder(
        dataset_name=config.extraction.dataset_name,
        tokenizer=tokenizer,
        num_samples=config.extraction.num_samples,
        max_seq_length=config.extraction.max_seq_length,
        subset=config.extraction.dataset_subset
    )

    dataloader = dataset_builder.build(batch_size=config.extraction.batch_size)
    print(f"Dataset ready with {len(dataloader.dataset)} samples")

    # PHASE 3: Extract Activations (Memory-Efficient)
    print("\n" + "=" * 80)
    print("PHASE 3: Extracting Activations (Memory-Efficient)")
    print("=" * 80)

    # Determine which layers to analyze
    if layer_indices is None:
        # Analyze a subset of layers (first, middle, last)
        total_layers = arch_info['num_layers']
        layer_indices = [0, total_layers // 2, total_layers - 1]

    print(f"Analyzing layers: {layer_indices}")

    # Memory estimation
    print("\n--- Memory Estimation ---")
    if torch.cuda.is_available():
        model_params = sum(p.numel() for p in model.parameters())
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            # Multi-GPU mode with device_map
            total_memory = sum(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(num_gpus)
            )
            print(f"Multi-GPU mode: {num_gpus} devices")
            print(f"Model parameters: {model_params/1e9:.2f}B")
            print(f"Total GPU memory: {total_memory/1e9:.2f} GB")
            print(f"Model distributed automatically by device_map='auto'")
            print(f"Note: Each GPU will receive a portion of the model based on layer sizes")

            # Show current usage per GPU
            print(f"\nCurrent memory usage per GPU:")
            for i in range(num_gpus):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {allocated:.2f}GB allocated / {total:.1f}GB total")

        else:
            # Single GPU mode
            mem_est = estimate_memory_requirements(
                model_params=model_params,
                num_samples=config.extraction.num_samples,
                batch_size=config.extraction.batch_size,
                hidden_size=arch_info['hidden_size'],
                seq_length=config.extraction.max_seq_length
            )

            available_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"Single GPU mode")
            print(f"Model parameters: {model_params/1e9:.2f}B")
            print(f"Estimated peak memory: {mem_est['peak_memory']/1e9:.2f} GB")
            print(f"Available GPU memory: {available_memory/1e9:.2f} GB")

            if mem_est['peak_memory'] > available_memory * 0.9:
                print("⚠️  WARNING: Estimated memory usage is close to GPU limit!")
                print("   OOM handling will automatically reduce batch size if needed.")

            # Show current GPU usage
            print(f"Current GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"Current GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    else:
        print("Running on CPU - no GPU memory estimation needed")

    # Setup for checkpointing
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "extraction_progress.json"

    # Check for existing checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    completed_layers = checkpoint['completed_layers'] if checkpoint else []

    if checkpoint:
        print(f"\n📁 Found checkpoint: {len(completed_layers)} layers already completed")
        print(f"   Resuming from layer {checkpoint['current_layer']}")

    # Process layers one at a time with memory management
    all_activations = {}
    all_texts = []
    current_batch_size = config.extraction.batch_size

    for layer_idx in layer_indices:
        # Skip if already completed
        if layer_idx in completed_layers:
            print(f"\n✓ Layer {layer_idx} already completed, loading from disk...")
            activation_file = output_dir / f"activations_layer_{layer_idx}.pt"
            if activation_file.exists():
                all_activations[layer_idx] = torch.load(activation_file)
                print(f"  Loaded: {all_activations[layer_idx].shape}")
                continue
            else:
                print(f"  Warning: Checkpoint exists but file not found, re-extracting...")

        print(f"\n{'='*60}")
        print(f"Processing Layer {layer_idx}/{max(layer_indices)}")
        print(f"{'='*60}")

        # Clear GPU memory before processing layer
        clear_gpu_memory()

        if config.model.device == "cuda" and torch.cuda.is_available():
            print(f"GPU memory before layer: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Hook the MLP output of this layer
        layer_name = f"model.layers.{layer_idx}.mlp"

        # Process with OOM handling
        activations, layer_texts, final_batch_size = process_layer_with_oom_handling(
            model=model,
            dataloader=dataloader,
            layer_name=layer_name,
            layer_idx=layer_idx,
            device=config.model.device,
            initial_batch_size=current_batch_size,
            max_batches=20  # Limit batches for speed (~320-640 samples)
        )

        # Update batch size for next layer if it was reduced
        if final_batch_size < current_batch_size:
            print(f"  Batch size reduced to {final_batch_size}, will use for remaining layers")
            current_batch_size = final_batch_size

        # Store texts from first layer only
        if layer_idx == layer_indices[0]:
            all_texts = layer_texts

        # Move to CPU (should already be on CPU from ActivationCapture)
        activations = activations.cpu()

        # Save to disk immediately
        activation_file = output_dir / f"activations_layer_{layer_idx}.pt"
        torch.save(activations, activation_file)
        print(f"  💾 Saved to: {activation_file}")

        # Keep in memory for immediate use (but we have disk backup)
        all_activations[layer_idx] = activations

        # Save checkpoint
        completed_layers.append(layer_idx)
        save_checkpoint(checkpoint_path, layer_idx, completed_layers, all_texts)
        print(f"  ✓ Checkpoint saved")

        # Aggressive memory cleanup
        clear_gpu_memory()

        if config.model.device == "cuda" and torch.cuda.is_available():
            print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"\n{'='*80}")
    print(f"✅ Captured activations for {len(all_activations)} layers")
    print(f"   Saved to: {output_dir}/activations_layer_*.pt")
    print(f"{'='*80}")

    # PHASE 4: Geometric Analysis
    print("\n" + "=" * 80)
    print("PHASE 4: Geometric Analysis")
    print("=" * 80)

    geometric_results = {}

    for layer_idx, activations in all_activations.items():
        print(f"\nAnalyzing layer {layer_idx}...")

        analyzer = ManifoldAnalyzer(activations)
        results = analyzer.analyze_all(n_components=50)

        geometric_results[layer_idx] = results

        print(f"  Intrinsic dimension (95%): {results['intrinsic_dimension']['intrinsic_dim_95pct_var']}")
        print(f"  Geometry type: {results['local_geometry']['geometry_type']}")

    # Save geometric analysis
    geom_output = output_dir / "geometric_analysis.json"

    with open(geom_output, 'w') as f:
        # Remove numpy arrays for JSON serialization
        save_results = {}
        for layer_idx, results in geometric_results.items():
            save_results[f"layer_{layer_idx}"] = {
                'activation_statistics': results['activation_statistics'],
                'intrinsic_dimension': results['intrinsic_dimension'],
                'local_geometry': results['local_geometry']
            }
        json.dump(save_results, f, indent=2)

    print(f"\nGeometric analysis saved to {geom_output}")

    # PHASE 5: Feature Extraction (Optional SAE Training)
    print("\n" + "=" * 80)
    print("PHASE 5: Feature Extraction")
    print("=" * 80)

    feature_results = {}

    if not skip_sae:
        print("Training Sparse Autoencoders...")

        for layer_idx, activations in all_activations.items():
            print(f"\nTraining SAE for layer {layer_idx}...")

            # Create dataloader from activations
            sae_dataloader = SAETrainer.create_dataloader_from_activations(
                activations,
                batch_size=config.analysis.sae_batch_size,
                shuffle=True
            )

            # Train SAE (small version for demo)
            trainer = SAETrainer(
                input_dim=arch_info['hidden_size'],
                hidden_dim=arch_info['hidden_size'] * 4,  # 4x expansion
                l1_coefficient=config.analysis.sae_l1_coefficient,
                learning_rate=config.analysis.sae_learning_rate,
                device=config.model.device
            )

            history = trainer.train(
                sae_dataloader,
                num_epochs=3,  # Quick training for demo
            )

            # Analyze features
            print(f"  Analyzing features...")
            feature_analyzer = FeatureAnalyzer(trainer.sae, tokenizer, model)

            features = feature_analyzer.analyze_all_features(
                activations,
                all_texts,
                max_features=20,  # Analyze top 20 features
                min_sparsity=0.001
            )

            feature_results[layer_idx] = features

            # Save SAE
            sae_path = output_dir / f"sae_layer_{layer_idx}.pt"
            trainer.save_model(str(sae_path))
            print(f"  Saved SAE to {sae_path}")

    else:
        print("Skipping SAE training (--skip-sae flag set)")

    # PHASE 6: Circuit Discovery
    print("\n" + "=" * 80)
    print("PHASE 6: Circuit Discovery")
    print("=" * 80)

    circuit_discoverer = CircuitDiscovery(model, tokenizer)

    test_inputs = [
        "The capital of France is",
        "Two plus two equals",
        "The quick brown fox",
    ]

    discovered_circuits = []

    for input_text in test_inputs:
        print(f"\nAnalyzing: '{input_text}'")
        circuit = circuit_discoverer.discover_circuit_structure(input_text)
        discovered_circuits.append(circuit)

    # Summarize circuits
    circuit_summary = circuit_discoverer.summarize_circuits(discovered_circuits)
    print(f"\nDiscovered {circuit_summary['num_circuits']} circuit patterns")

    # PHASE 7: Generate Documentation
    print("\n" + "=" * 80)
    print("PHASE 7: Generating Documentation")
    print("=" * 80)

    doc_generator = DocumentationGenerator(
        output_dir=config.documentation.output_dir
    )

    # Prepare layer info
    layer_info = []
    for layer_idx in layer_indices:
        geom = geometric_results.get(layer_idx, {})
        intrinsic = geom.get('intrinsic_dimension', {})

        layer_info.append({
            'layer_id': layer_idx,
            'layer_type': 'transformer',
            'hidden_dim': arch_info['hidden_size'],
            'num_heads': arch_info['num_attention_heads'],
            'intermediate_dim': arch_info['intermediate_size'],
            'num_features_discovered': len(feature_results.get(layer_idx, [])),
            'intrinsic_dimension': intrinsic.get('intrinsic_dim_95pct_var'),
            'geometry_type': geom.get('local_geometry', {}).get('geometry_type'),
            'compression_ratio': intrinsic.get('compression_ratio')
        })

    # Generate full documentation
    full_docs = doc_generator.generate_full_documentation(
        model_name=config.model.model_name,
        architecture=arch_info,
        features_by_layer=feature_results,
        circuits=discovered_circuits,
        layer_info=layer_info
    )

    # Save documentation
    doc_generator.save_documentation(full_docs, format='json')
    doc_generator.save_documentation(full_docs, format='markdown')
    doc_generator.save_summary(full_docs)

    # Save transparency scores in a separate file for easy access
    transparency_scores = {
        'transparency_score': full_docs.transparency_score,
        'total_features_discovered': full_docs.total_features_discovered,
        'total_circuits_discovered': full_docs.total_circuits_discovered,
        'layers_analyzed': len(layer_indices),
        'layer_indices': layer_indices,
        'model_name': config.model.model_name,
        'num_samples': config.extraction.num_samples,
    }

    scores_path = output_dir / "transparency_scores.json"
    with open(scores_path, 'w') as f:
        json.dump(transparency_scores, f, indent=2)

    print(f"\nDocumentation generated!")
    print(f"Transparency Score: {full_docs.transparency_score:.2%}")

    # PHASE 8: Verification (Optional)
    print("\n" + "=" * 80)
    print("PHASE 8: Verification Tests")
    print("=" * 80)

    interventions = ModelInterventions(model, tokenizer)

    # Test layer importance
    test_layer = layer_indices[0]
    print(f"\nTesting importance of layer {test_layer}...")

    importance_results = interventions.test_feature_importance(
        test_inputs,
        test_layer
    )

    print(f"  Mean impact: {importance_results['mean_impact']:.3f}")
    print(f"  Outputs changed: {importance_results['num_changed']}/{importance_results['num_tests']}")

    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  - Transparency Score: {full_docs.transparency_score:.2%}")
    print(f"  - Features Discovered: {full_docs.total_features_discovered}")
    print(f"  - Circuits Analyzed: {full_docs.total_circuits_discovered}")
    print(f"  - Layers Analyzed: {len(layer_indices)}")
    print(f"\nOutput files:")
    print(f"  - {geom_output}")
    print(f"  - {output_dir}/transparency_scores.json")
    print(f"  - {output_dir}/model_documentation.json")
    print(f"  - {output_dir}/model_documentation.md")
    print(f"  - {output_dir}/summary.json")
    print()


if __name__ == "__main__":
    main()
