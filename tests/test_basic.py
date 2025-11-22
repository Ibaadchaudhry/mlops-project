"""Basic tests to validate CI/CD pipeline functionality."""

import pytest
import torch
import numpy as np


def test_basic_functionality():
    """Test that basic Python functionality works."""
    assert 1 + 1 == 2


def test_numpy_operations():
    """Test that numpy works correctly."""
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.sum() == 15


def test_torch_functionality():
    """Test that PyTorch works correctly."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = x * 2
    expected = torch.tensor([2.0, 4.0, 6.0])
    assert torch.allclose(y, expected)


def test_model_import():
    """Test that we can import the model."""
    try:
        from model import TabularMLP
        model = TabularMLP(input_dim=10)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    except ImportError:
        pytest.skip("Model module not available yet")


def test_drift_detector_import():
    """Test that we can import drift detector."""
    try:
        from drift_detector import psi, detect_drift_featurewise
        # Test PSI calculation
        baseline = np.random.normal(0, 1, 100)
        current = np.random.normal(0, 1, 100)
        psi_val = psi(baseline, current)
        assert isinstance(psi_val, float)
        assert psi_val >= 0
    except ImportError:
        pytest.skip("Drift detector module not available yet")


def test_data_types():
    """Test data type handling."""
    import pandas as pd
    
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    
    assert len(df) == 5
    assert list(df.columns) == ['col1', 'col2']
    assert df['col1'].dtype == 'int64'


@pytest.mark.parametrize("input_size", [10, 20, 50])
def test_different_input_sizes(input_size):
    """Test with different input sizes."""
    data = np.random.randn(100, input_size)
    assert data.shape == (100, input_size)
    assert data.dtype == np.float64


def test_performance_benchmark():
    """Simple performance test."""
    import time
    
    start_time = time.time()
    
    # Simulate some computation
    result = sum(i**2 for i in range(1000))
    
    end_time = time.time()
    duration = end_time - start_time
    
    assert result == 332833500  # Expected sum of squares
    assert duration < 1.0  # Should complete within 1 second