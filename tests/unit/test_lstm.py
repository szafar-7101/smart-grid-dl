# tests/unit/test_lstm.py
#
# Unit tests for LSTM model building and sequence creation.
# We test the architecture and data preparation logic
# without running actual training (which would be too slow for tests).

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.models.lstm import (
    build_lstm_model,
    create_sequences,
    LSTMModelError,
)
from src.models.config import (
    SEQUENCE_LENGTH,
    FORECAST_HORIZON,
    LSTM_UNITS_LAYER1,
    LSTM_UNITS_LAYER2,
)


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------

@pytest.fixture
def sample_feature_array() -> np.ndarray:
    """
    Creates a small random feature array for testing.
    500 rows × 10 features — large enough for sequences,
    small enough to run instantly.
    """
    rng = np.random.default_rng(seed=42)
    return rng.random(size=(500, 10)).astype(np.float32)
    # astype(np.float32) converts to 32-bit floats.
    # TensorFlow uses float32 by default — using float64 (Python default)
    # causes unnecessary memory usage and type conversion warnings.


@pytest.fixture
def small_model() -> keras.Model:
    """
    Builds a tiny model for testing — just 1 feature, short sequences.
    We override the sequence length and features for speed.
    """
    return build_lstm_model(n_features=10)


# ----------------------------------------------------------------
# Sequence creation tests
# ----------------------------------------------------------------

def test_create_sequences_output_shapes(sample_feature_array):
    """
    The output shapes must be exactly what the LSTM expects.
    X shape: (n_sequences, sequence_length, n_features)
    y shape: (n_sequences, forecast_horizon)
    """
    X_seq, y_seq = create_sequences(
        sample_feature_array,
        y_index=0,
        sequence_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
    )

    expected_n_sequences = (
        len(sample_feature_array) - SEQUENCE_LENGTH - FORECAST_HORIZON + 1
    )

    assert X_seq.shape == (expected_n_sequences, SEQUENCE_LENGTH, 10), (
        f"X_seq shape {X_seq.shape} does not match expected "
        f"({expected_n_sequences}, {SEQUENCE_LENGTH}, 10)"
    )
    assert y_seq.shape == (expected_n_sequences, FORECAST_HORIZON), (
        f"y_seq shape {y_seq.shape} does not match expected "
        f"({expected_n_sequences}, {FORECAST_HORIZON})"
    )


def test_create_sequences_window_is_correct(sample_feature_array):
    """
    The first sequence must contain exactly rows 0 to sequence_length-1.
    This verifies the window indexing is correct — no off-by-one errors.
    """
    X_seq, _ = create_sequences(
        sample_feature_array,
        y_index=0,
        sequence_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
    )
    # The first sequence should be rows 0 to SEQUENCE_LENGTH-1
    expected_first_seq = sample_feature_array[:SEQUENCE_LENGTH]
    np.testing.assert_array_almost_equal(
        X_seq[0],
        expected_first_seq,
        decimal=6,
        err_msg="First sequence does not match expected rows 0 to sequence_length-1"
    )


def test_create_sequences_target_is_correct(sample_feature_array):
    """
    The target for sequence 0 must be rows sequence_length to
    sequence_length + forecast_horizon - 1, column y_index.
    This verifies the target window starts exactly where the input window ends.
    """
    y_index = 0
    _, y_seq = create_sequences(
        sample_feature_array,
        y_index=y_index,
        sequence_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
    )
    expected_target = sample_feature_array[
        SEQUENCE_LENGTH : SEQUENCE_LENGTH + FORECAST_HORIZON,
        y_index
    ]
    np.testing.assert_array_almost_equal(
        y_seq[0],
        expected_target,
        decimal=6,
        err_msg="Target sequence does not match expected future values"
    )


def test_create_sequences_raises_for_short_data():
    """
    If data is shorter than sequence_length + forecast_horizon,
    create_sequences must raise LSTMModelError.
    """
    too_short = np.random.rand(10, 5)
    # 10 rows is much less than 168 + 24 = 192 minimum required

    with pytest.raises(LSTMModelError):
        create_sequences(too_short, y_index=0)


def test_consecutive_sequences_overlap(sample_feature_array):
    """
    Sequences should overlap by (sequence_length - 1) rows.
    Sequence 0 covers rows 0 to 167, sequence 1 covers rows 1 to 168.
    So rows 1 to 167 appear in both. This tests the sliding window logic.
    """
    X_seq, _ = create_sequences(
        sample_feature_array,
        y_index=0,
        sequence_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
    )
    # Row 1 of sequence 0 should equal row 0 of sequence 1
    np.testing.assert_array_almost_equal(
        X_seq[0, 1, :],
        # sequence 0, timestep index 1, all features
        X_seq[1, 0, :],
        # sequence 1, timestep index 0, all features
        decimal=6,
        err_msg="Consecutive sequences do not overlap correctly"
    )


# ----------------------------------------------------------------
# Model architecture tests
# ----------------------------------------------------------------

def test_model_is_keras_model(small_model):
    """The builder must return a compiled Keras model."""
    assert isinstance(small_model, keras.Model), (
        f"Expected keras.Model but got {type(small_model)}"
    )


def test_model_input_shape(small_model):
    """
    The model input must accept shape (batch, sequence_length, n_features).
    None in the shape means "any batch size" — which is correct.
    """
    expected_input_shape = (None, SEQUENCE_LENGTH, 10)
    actual_input_shape = tuple(small_model.input_shape)
    assert actual_input_shape == expected_input_shape, (
        f"Expected input shape {expected_input_shape} "
        f"but got {actual_input_shape}"
    )


def test_model_output_shape(small_model):
    """
    The model output must have shape (batch, forecast_horizon).
    For our project: (None, 24) — one prediction per forecast hour.
    """
    expected_output_shape = (None, FORECAST_HORIZON)
    actual_output_shape = tuple(small_model.output_shape)
    assert actual_output_shape == expected_output_shape, (
        f"Expected output shape {expected_output_shape} "
        f"but got {actual_output_shape}"
    )


def test_model_has_correct_number_of_layers(small_model):
    """
    The model must have the expected layer names.
    This verifies the architecture was assembled correctly.
    """
    layer_names = [layer.name for layer in small_model.layers]
    expected_layers = [
        "sequence_input",
        "bidirectional_lstm_1",
        "dropout_1",
        "lstm_2",
        "dropout_2",
        "forecast_output",
    ]
    for expected in expected_layers:
        assert expected in layer_names, (
            f"Expected layer '{expected}' not found in model. "
            f"Layers present: {layer_names}"
        )


def test_model_is_compiled(small_model):
    """
    The model must have a loss function attached (i.e. be compiled).
    An uncompiled model cannot be trained.
    """
    assert small_model.compiled_loss is not None, (
        "Model does not have a compiled loss function — "
        "call model.compile() before returning"
    )


def test_model_forward_pass_produces_correct_shape(small_model):
    """
    A forward pass through the model must produce the expected output shape.
    This is the most important integration test — it verifies that all
    layers connect correctly and data flows through without errors.
    """
    # Create a dummy batch of 4 sequences
    dummy_input = np.random.rand(4, SEQUENCE_LENGTH, 10).astype(np.float32)
    # shape: (4 sequences, 168 timesteps, 10 features)

    output = small_model.predict(dummy_input, verbose=0)
    # verbose=0 suppresses output during testing

    assert output.shape == (4, FORECAST_HORIZON), (
        f"Expected output shape (4, {FORECAST_HORIZON}) "
        f"but got {output.shape}"
    )