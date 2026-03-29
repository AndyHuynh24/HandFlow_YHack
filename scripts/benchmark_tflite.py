"""Benchmark the full detection pipeline end-to-end."""

import argparse
import numpy as np
import tensorflow as tf
import cv2
import time
import mediapipe as mp
from pathlib import Path


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def benchmark_stats(times: np.ndarray, name: str):
    """Calculate and print benchmark statistics."""
    avg = np.mean(times)
    std = np.std(times)
    min_t = np.min(times)
    max_t = np.max(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)

    print(f"\n{name}:")
    print(f"  Average: {avg:.3f} ms (std: {std:.3f} ms)")
    print(f"  Min: {min_t:.3f} ms, Max: {max_t:.3f} ms")
    print(f"  P50: {p50:.3f} ms, P95: {p95:.3f} ms, P99: {p99:.3f} ms")
    print(f"  Throughput: {1000/avg:.1f} ops/sec")

    return {'avg': avg, 'std': std, 'min': min_t, 'max': max_t, 'p50': p50, 'p95': p95, 'p99': p99}


def create_dummy_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """Create a dummy frame for benchmarking without camera."""
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return frame


def benchmark_mediapipe(frames: list, num_runs: int, warmup: int = 10):
    """Benchmark MediaPipe hand detection."""
    print_section("MediaPipe Hand Detection")

    # Load config
    from handflow.utils import load_config
    config = load_config("config/config.yaml")

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=config.mediapipe.min_detection_confidence,
        min_tracking_confidence=config.mediapipe.min_tracking_confidence,
        max_num_hands=config.mediapipe.max_num_hands,
        model_complexity=config.mediapipe.model_complexity
    )

    print(f"Model complexity: {config.mediapipe.model_complexity}")
    print(f"Max hands: {config.mediapipe.max_num_hands}")

    # Warmup
    print(f"Warming up ({warmup} runs)...")
    for i in range(warmup):
        frame = frames[i % len(frames)]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands.process(rgb)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    hands_detected = 0
    for i in range(num_runs):
        frame = frames[i % len(frames)]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()
        results = hands.process(rgb)
        end = time.perf_counter()

        times.append((end - start) * 1000)
        if results.multi_hand_landmarks:
            hands_detected += 1

    hands.close()

    stats = benchmark_stats(np.array(times), "MediaPipe Hand Detection")
    print(f"  Hands detected: {hands_detected}/{num_runs} frames")

    return stats


def benchmark_feature_engineering(num_runs: int, warmup: int = 10):
    """Benchmark feature engineering."""
    print_section("Feature Engineering")

    from handflow.features import FeatureEngineer

    feature_engineer = FeatureEngineer()

    # Create dummy sequence (12 frames of 21 landmarks * 3 coords = 63 values)
    # Plus 21 z-coordinates = 84 total keypoint values, but we use 63 (flattened xyz)
    seq_len = 12
    keypoint_dim = 63  # 21 landmarks * 3
    dummy_sequence = np.random.randn(seq_len, keypoint_dim).astype(np.float32)

    # Warmup
    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        feature_engineer.transform(dummy_sequence)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        features = feature_engineer.transform(dummy_sequence)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    stats = benchmark_stats(np.array(times), "Feature Engineering")
    print(f"  Input shape: {dummy_sequence.shape}")
    print(f"  Output shape: {features.shape}")

    return stats


def benchmark_tflite_model(model_path: str, num_runs: int, warmup: int = 10):
    """Benchmark TFLite gesture model inference."""
    print_section("TFLite Gesture Model")

    if not Path(model_path).exists():
        print(f"  Model not found: {model_path}")
        return None

    # Load model
    print(f"Loading model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"Input shape: {input_details['shape']}")
    print(f"Output shape: {output_details['shape']}")

    # Create dummy input
    dummy_input = np.random.randn(*input_details['shape']).astype(np.float32)

    # Warmup
    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        interpreter.set_tensor(input_details['index'], dummy_input)
        interpreter.invoke()

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])
        end = time.perf_counter()
        times.append((end - start) * 1000)

    stats = benchmark_stats(np.array(times), "TFLite Gesture Model")

    # Sample prediction
    output = interpreter.get_tensor(output_details['index'])[0]
    print(f"  Sample output: class={np.argmax(output)}, conf={output[np.argmax(output)]:.3f}")

    return stats


def benchmark_aruco_detection(frames: list, num_runs: int, warmup: int = 10):
    """Benchmark ArUco marker detection."""
    print_section("ArUco Marker Detection")

    from handflow.detector import ArUcoScreenDetector

    # Initialize detector
    detector = ArUcoScreenDetector(screen_width=1920, screen_height=1080)

    # Warmup
    print(f"Warming up ({warmup} runs)...")
    for i in range(warmup):
        frame = frames[i % len(frames)]
        detector.detect(frame)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    markers_detected = 0
    for i in range(num_runs):
        frame = frames[i % len(frames)]

        start = time.perf_counter()
        valid = detector.detect(frame)
        end = time.perf_counter()

        times.append((end - start) * 1000)
        if valid:
            markers_detected += 1

    stats = benchmark_stats(np.array(times), "ArUco Detection")
    print(f"  Valid detections: {markers_detected}/{num_runs} frames")

    return stats


def benchmark_macropad_detection(frames: list, num_runs: int, warmup: int = 10):
    """Benchmark MacroPad marker detection."""
    print_section("MacroPad Detection")

    from handflow.detector import MacroPadDetector

    # Initialize detector
    detector = MacroPadDetector()

    # Warmup
    print(f"Warming up ({warmup} runs)...")
    for i in range(warmup):
        frame = frames[i % len(frames)]
        detector.detect(frame)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    pads_detected = 0
    for i in range(num_runs):
        frame = frames[i % len(frames)]

        start = time.perf_counter()
        valid = detector.detect(frame)
        end = time.perf_counter()

        times.append((end - start) * 1000)
        if valid:
            pads_detected += 1

    stats = benchmark_stats(np.array(times), "MacroPad Detection")
    print(f"  Valid detections: {pads_detected}/{num_runs} frames")

    return stats


def benchmark_end_to_end(frames: list, model_path: str, num_runs: int, warmup: int = 10):
    """Benchmark the complete end-to-end pipeline."""
    print_section("End-to-End Pipeline")

    from handflow.utils import load_config
    from handflow.features import FeatureEngineer
    from handflow.detector import ArUcoScreenDetector

    config = load_config("config/config.yaml")

    # Initialize components
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=config.mediapipe.min_detection_confidence,
        min_tracking_confidence=config.mediapipe.min_tracking_confidence,
        max_num_hands=config.mediapipe.max_num_hands,
        model_complexity=config.mediapipe.model_complexity
    )

    feature_engineer = FeatureEngineer()
    aruco_detector = ArUcoScreenDetector(screen_width=1920, screen_height=1080)

    # Load TFLite model
    interpreter = None
    if Path(model_path).exists():
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

    seq_len = config.data.sequence_length
    keypoint_dim = 63

    # Warmup
    print(f"Warming up ({warmup} runs)...")
    for i in range(warmup):
        frame = frames[i % len(frames)]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands.process(rgb)
        aruco_detector.detect(frame)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times_total = []
    times_mediapipe = []
    times_features = []
    times_model = []
    times_aruco = []

    # Simulate sequence buffer
    dummy_sequence = np.random.randn(seq_len, keypoint_dim).astype(np.float32)

    for i in range(num_runs):
        frame = frames[i % len(frames)]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        total_start = time.perf_counter()

        # 1. MediaPipe
        mp_start = time.perf_counter()
        results = hands.process(rgb)
        mp_end = time.perf_counter()
        times_mediapipe.append((mp_end - mp_start) * 1000)

        # 2. ArUco
        aruco_start = time.perf_counter()
        aruco_detector.detect(frame)
        aruco_end = time.perf_counter()
        times_aruco.append((aruco_end - aruco_start) * 1000)

        # 3. Feature Engineering (using dummy sequence)
        feat_start = time.perf_counter()
        features = feature_engineer.transform(dummy_sequence)
        feat_end = time.perf_counter()
        times_features.append((feat_end - feat_start) * 1000)

        # 4. TFLite Model
        if interpreter:
            model_start = time.perf_counter()
            input_data = np.expand_dims(features, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details['index'])
            model_end = time.perf_counter()
            times_model.append((model_end - model_start) * 1000)

        total_end = time.perf_counter()
        times_total.append((total_end - total_start) * 1000)

    hands.close()

    # Print results
    print("\nComponent Breakdown:")
    benchmark_stats(np.array(times_mediapipe), "  MediaPipe")
    benchmark_stats(np.array(times_aruco), "  ArUco")
    benchmark_stats(np.array(times_features), "  Features")
    if times_model:
        benchmark_stats(np.array(times_model), "  TFLite Model")

    total_stats = benchmark_stats(np.array(times_total), "TOTAL End-to-End")

    # Summary
    print(f"\n{'='*60}")
    print(" SUMMARY")
    print('='*60)
    avg_total = np.mean(times_total)
    print(f"  Total pipeline: {avg_total:.2f} ms/frame")
    print(f"  Max sustainable FPS: {1000/avg_total:.1f} FPS")
    print(f"\n  Breakdown:")
    print(f"    MediaPipe:  {np.mean(times_mediapipe):6.2f} ms ({np.mean(times_mediapipe)/avg_total*100:5.1f}%)")
    print(f"    ArUco:      {np.mean(times_aruco):6.2f} ms ({np.mean(times_aruco)/avg_total*100:5.1f}%)")
    print(f"    Features:   {np.mean(times_features):6.2f} ms ({np.mean(times_features)/avg_total*100:5.1f}%)")
    if times_model:
        print(f"    TFLite:     {np.mean(times_model):6.2f} ms ({np.mean(times_model)/avg_total*100:5.1f}%)")

    return total_stats


def capture_frames(camera_idx: int, num_frames: int, width: int = 640, height: int = 360) -> list:
    """Capture frames from camera."""
    print(f"\nCapturing {num_frames} frames from camera {camera_idx}...")

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_idx}")
        return []

    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            # Resize to processing resolution
            frame_small = cv2.resize(frame, (width, height))
            frames.append(frame_small)
        if (i + 1) % 10 == 0:
            print(f"  Captured {i + 1}/{num_frames} frames")

    cap.release()
    print(f"Captured {len(frames)} frames")
    return frames


def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Benchmark")
    parser.add_argument("--model", type=str, default="models/hand_action.tflite",
                        help="Path to TFLite model")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup runs")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index")
    parser.add_argument("--use-dummy", action="store_true",
                        help="Use dummy frames instead of camera")
    parser.add_argument("--num-frames", type=int, default=30,
                        help="Number of frames to capture/generate")
    parser.add_argument("--component", type=str, default="all",
                        choices=["all", "mediapipe", "features", "tflite", "aruco", "macropad", "e2e"],
                        help="Which component to benchmark")
    args = parser.parse_args()

    print("="*60)
    print(" HandFlow Pipeline Benchmark")
    print("="*60)
    print(f"Runs: {args.runs}, Warmup: {args.warmup}")

    # Get frames
    if args.use_dummy:
        print(f"\nGenerating {args.num_frames} dummy frames...")
        frames = [create_dummy_frame() for _ in range(args.num_frames)]
    else:
        frames = capture_frames(args.camera, args.num_frames)
        if not frames:
            print("Falling back to dummy frames...")
            frames = [create_dummy_frame() for _ in range(args.num_frames)]

    # Run benchmarks
    results = {}

    if args.component in ["all", "mediapipe"]:
        results['mediapipe'] = benchmark_mediapipe(frames, args.runs, args.warmup)

    if args.component in ["all", "features"]:
        results['features'] = benchmark_feature_engineering(args.runs, args.warmup)

    if args.component in ["all", "tflite"]:
        results['tflite'] = benchmark_tflite_model(args.model, args.runs, args.warmup)

    if args.component in ["all", "aruco"]:
        results['aruco'] = benchmark_aruco_detection(frames, args.runs, args.warmup)

    if args.component in ["all", "macropad"]:
        results['macropad'] = benchmark_macropad_detection(frames, args.runs, args.warmup)

    if args.component in ["all", "e2e"]:
        results['e2e'] = benchmark_end_to_end(frames, args.model, args.runs, args.warmup)

    print("\n" + "="*60)
    print(" Benchmark Complete")
    print("="*60)


if __name__ == "__main__":
    main()
