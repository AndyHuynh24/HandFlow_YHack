"""CLI interface for HandFlow training, export, and info commands."""

from __future__ import annotations

import click

from handflow import __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """HandFlow - Gesture-controlled computer interaction."""
    pass

@main.command()
@click.option("--hand", "-h", type=click.Choice(["right", "left"]), required=True)
@click.option("--architecture", "-a", default="gru", help="Model architecture")
@click.option("--epochs", "-e", default=100, help="Training epochs")
@click.option("--experiment", "-n", default=None, help="Experiment name")
@click.option("--config", "-f", default=None, help="Path to config file")
def train(
    hand: str, architecture: str, epochs: int, experiment: str | None, config: str | None
) -> None:
    """Train a gesture recognition model."""
    from pathlib import Path

    from handflow.models import Trainer, build_model, load_data
    from handflow.utils import load_config

    cfg = load_config(config)
    cfg.model.architecture = architecture
    cfg.training.epochs = epochs

    # Determine data path and actions
    if hand == "right":
        data_path = cfg.paths.raw_data_path
        actions = cfg.right_hand_gestures
        output_path = Path(cfg.paths.models_dir) / "right_action.h5"
    else:
        data_path = cfg.paths.left_data_path
        actions = cfg.left_hand_gestures
        output_path = Path(cfg.paths.models_dir) / "left_action.h5"

    click.echo(f"📊 Loading data from {data_path}")
    sequences, labels = load_data(data_path, actions, cfg.model.sequence_length)
    click.echo(f"   Found {len(sequences)} sequences")

    # Split data
    from sklearn.model_selection import train_test_split

    x_train, x_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=cfg.training.validation_split, random_state=42
    )

    click.echo(f"🏗️ Building {architecture.upper()} model")
    model = build_model(cfg)

    click.echo(f"🚀 Training for {epochs} epochs")
    trainer = Trainer(cfg, model, experiment_name=experiment or f"handflow-{hand}")
    trainer.train(x_train, y_train, x_val, y_val)

    click.echo(f"💾 Saving model to {output_path}")
    trainer.save(output_path)

    click.echo("✅ Training complete!")


@main.command()
@click.argument("model_path")
@click.option("--output", "-o", default=None, help="Output path")
def export(model_path: str, output: str | None) -> None:
    """Export model to TFLite format (quantized for faster inference)."""
    from pathlib import Path

    import tensorflow as tf

    model_path = Path(model_path)

    if output is None:
        output = model_path.with_suffix(".tflite")
    else:
        output = Path(output)

    click.echo(f"📦 Loading model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    click.echo("🔧 Converting to TFLite (quantized)")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output, "wb") as f:
        f.write(tflite_model)

    original_size = model_path.stat().st_size / 1024 / 1024
    new_size = output.stat().st_size / 1024 / 1024

    click.echo(f"✅ Exported to: {output}")
    click.echo(f"   Original: {original_size:.2f} MB → Exported: {new_size:.2f} MB")
    click.echo(f"   Size reduction: {(1 - new_size/original_size)*100:.0f}%")


@main.command()
@click.option("--config", "-f", default=None, help="Path to config file")
def info(config: str | None) -> None:
    """Show configuration and system information."""
    import sys

    from handflow.utils import load_config

    cfg = load_config(config)

    click.echo("🖐️ HandFlow Configuration")
    click.echo("=" * 40)
    click.echo(f"Version: {__version__}")
    click.echo(f"Python: {sys.version}")
    click.echo(f"Platform: {sys.platform}")
    click.echo()
    click.echo("Model Settings:")
    click.echo(f"  Architecture: {cfg.model.architecture}")
    click.echo(f"  Sequence Length: {cfg.model.sequence_length}")
    click.echo(f"  Input Dim: {cfg.get_input_dim()}")
    click.echo()
    click.echo("Feature Engineering:")
    click.echo(f"  Velocity: {cfg.features.velocity}")
    click.echo(f"  Acceleration: {cfg.features.acceleration}")
    click.echo(f"  Finger Angles: {cfg.features.finger_angles}")
    click.echo(f"  Hand BBox: {cfg.features.hand_bbox_size}")


if __name__ == "__main__":
    main()
