#!/usr/bin/env python
"""Compare inference time across different model architectures."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import tensorflow as tf

from handflow.models.architectures import build_model, count_parameters
from handflow.utils import load_config


ARCHITECTURES = ["lstm", "gru", "cnn1d", "tcn", "transformer"]


def benchmark_keras_model(
    model: tf.keras.Model,
    input_data: np.ndarray,
    iterations: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Benchmark a Keras model's inference time.

    Args:
        model: Compiled Keras model
        input_data: Input tensor for inference
        iterations: Number of inference iterations
        warmup: Number of warmup iterations (not counted)

    Returns:
        Dictionary with timing statistics
    """
    # Warmup runs (not measured)
    for _ in range(warmup):
        _ = model.predict(input_data, verbose=0)

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = model.predict(input_data, verbose=0)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def benchmark_tflite_model(
    tflite_path: str | Path,
    input_data: np.ndarray,
    iterations: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Benchmark a TFLite model's inference time.

    Args:
        tflite_path: Path to .tflite file
        input_data: Input tensor for inference
        iterations: Number of inference iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with timing statistics
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure input matches expected shape
    input_shape = input_details[0]['shape']
    if input_data.shape != tuple(input_shape):
        input_data = input_data[:input_shape[0]]

    input_data = input_data.astype(np.float32)

    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def format_results(results: dict) -> str:
    """Format benchmark results as a table."""
    lines = []
    lines.append("")
    lines.append("=" * 85)
    lines.append(f"{'Model':<15} {'Mean (ms)':<12} {'Std (ms)':<10} {'Min (ms)':<10} {'P95 (ms)':<10} {'Params':<15}")
    lines.append("=" * 85)

    for name, data in results.items():
        stats = data["stats"]
        params = data.get("params", "-")
        if isinstance(params, int):
            params = f"{params:,}"
        lines.append(
            f"{name:<15} {stats['mean_ms']:<12.3f} {stats['std_ms']:<10.3f} "
            f"{stats['min_ms']:<10.3f} {stats['p95_ms']:<10.3f} {params:<15}"
        )

    lines.append("=" * 85)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference times")
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of inference iterations (default: 100)"
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for inference (default: 1)"
    )
    parser.add_argument(
        "--include-tflite", action="store_true",
        help="Include TFLite model benchmark if available"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--architectures", type=str, nargs="+", default=ARCHITECTURES,
        choices=ARCHITECTURES,
        help="Architectures to benchmark"
    )
    args = parser.parse_args()

    print(f"\nLoading config from {args.config}...")
    config = load_config(args.config)

    # Create dummy input data
    seq_len = config.data.sequence_length
    input_dim = config.model.input_dim
    batch_size = args.batch_size

    input_data = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)

    print(f"\nBenchmark Settings:")
    print(f"  Input shape: ({batch_size}, {seq_len}, {input_dim})")
    print(f"  Iterations: {args.iterations}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Architectures: {args.architectures}")

    results = {}

    # Benchmark each Keras architecture
    for arch in args.architectures:
        print(f"\nBenchmarking {arch.upper()}...")

        # Temporarily override architecture in config
        original_arch = config.model.architecture
        config.model.architecture = arch

        try:
            model = build_model(config)
            params = count_parameters(model)

            stats = benchmark_keras_model(
                model, input_data,
                iterations=args.iterations,
                warmup=args.warmup
            )

            results[arch] = {
                "stats": stats,
                "params": params["total"],
            }

            print(f"  Mean: {stats['mean_ms']:.3f} ms | Params: {params['total']:,}")

            # Clean up
            del model
            tf.keras.backend.clear_session()

        except Exception as e:
            print(f"  Error: {e}")
        finally:
            config.model.architecture = original_arch

    # Benchmark TFLite if requested
    if args.include_tflite:
        tflite_path = Path(config.model.model_path)
        if tflite_path.exists():
            print(f"\nBenchmarking TFLite ({tflite_path.name})...")
            try:
                stats = benchmark_tflite_model(
                    tflite_path, input_data,
                    iterations=args.iterations,
                    warmup=args.warmup
                )
                results["tflite"] = {
                    "stats": stats,
                    "params": "-",
                }
                print(f"  Mean: {stats['mean_ms']:.3f} ms")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"\nTFLite model not found at {tflite_path}")

    # Print results table
    print(format_results(results))

    # Find fastest
    if results:
        fastest = min(results.items(), key=lambda x: x[1]["stats"]["mean_ms"])
        print(f"\nFastest: {fastest[0].upper()} ({fastest[1]['stats']['mean_ms']:.3f} ms)")


if __name__ == "__main__":
    main()
