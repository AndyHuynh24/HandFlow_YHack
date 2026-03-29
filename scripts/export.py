#!/usr/bin/env python
"""Export trained Keras models to TFLite for fast inference."""
#python scripts/export.py --input models/hand_action.keras
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from handflow.utils import setup_logging, get_logger
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export model to TFLite")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input Keras model (.h5 or .keras)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output TFLite model (default: same name with .tflite)",
    )
    
    return parser.parse_args()


def export_to_tflite(
    input_path: str,
    output_path: str | None = None,
) -> str:
    """
    Export Keras model to TFLite.
    
    Args:
        input_path: Path to Keras model
        output_path: Path for TFLite output
        
    Returns:
        Path to exported TFLite model
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix(".tflite")
    else:
        output_path = Path(output_path)
    
    print(f"📥 Loading model from {input_path}...")
    try:
        model = tf.keras.models.load_model(input_path)
    except TypeError as e:
        # Fallback: try loading without compilation
        print(f"   ⚠️ Standard load failed, trying with compile=False...")
        try:
            model = tf.keras.models.load_model(input_path, compile=False)
        except Exception as e2:
            # Second fallback: try with safe_mode=False for newer Keras
            print(f"   ⚠️ Still failing, trying with safe_mode=False...")
            model = tf.keras.models.load_model(input_path, compile=False, safe_mode=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # For Keras 3 compatibility, save as SavedModel first then convert
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = f"{tmpdir}/saved_model"
        print("   Saving as SavedModel...")
        model.export(saved_model_path)
        
        print("   Converting SavedModel to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        # Enable support for RNN models (GRU/LSTM) with dynamic operations
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # Required for RNN models
        ]
        converter._experimental_lower_tensor_list_ops = False
        converter.experimental_enable_resource_variables = True

        # Float16 quantization: halves model size, no accuracy loss
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    # Report sizes
    input_size = input_path.stat().st_size / 1024
    output_size = output_path.stat().st_size / 1024
    reduction = (1 - output_size / input_size) * 100
    
    print(f"\n✅ Exported to {output_path}")
    print(f"   Input size:  {input_size:.1f} KB")
    print(f"   Output size: {output_size:.1f} KB")
    print(f"   Size reduction: {reduction:.1f}%")
    
    # Verify
    print("\n🔍 Verifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape:  {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Input dtype:  {input_details[0]['dtype']}")
    
    return str(output_path)


def main() -> None:
    """Main export function."""
    args = parse_args()

    log_file = "logs/training.log"
    setup_logging(level="INFO", log_file=log_file)
    logger = get_logger("handflow.export")

    logger.info("🚀 HandFlow Model Export")
    logger.info("=" * 60)
    
    export_to_tflite(
        input_path=args.input,
        output_path=args.output,
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Export complete!")
    logger.info("\nNext steps:")
    logger.info("  Run main.py - it will auto-detect .tflite models")


if __name__ == "__main__":
    main()
