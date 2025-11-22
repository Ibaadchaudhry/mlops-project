"""Test configuration and fixtures."""

import pytest
import tempfile
import os
import shutil
import numpy as np
import pandas as pd


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    data = {
        'X': np.random.randn(100, 10).astype(np.float32),
        'y': np.random.randint(0, 2, 100).astype(np.float32)
    }
    return data


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame for testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    })
    return df


@pytest.fixture
def mock_client_data():
    """Mock client data structure."""
    np.random.seed(42)
    return {
        0: {
            'X_train_raw': pd.DataFrame(np.random.randn(100, 10)),
            'X_test_raw': pd.DataFrame(np.random.randn(20, 10)),
            'X_train_norm': np.random.randn(100, 10).astype(np.float32),
            'X_test_norm': np.random.randn(20, 10).astype(np.float32),
            'y_train': np.random.randint(0, 2, 100).astype(np.float32),
            'y_test': np.random.randint(0, 2, 20).astype(np.float32),
        },
        1: {
            'X_train_raw': pd.DataFrame(np.random.randn(80, 10)),
            'X_test_raw': pd.DataFrame(np.random.randn(15, 10)),
            'X_train_norm': np.random.randn(80, 10).astype(np.float32),
            'X_test_norm': np.random.randn(15, 10).astype(np.float32),
            'y_train': np.random.randint(0, 2, 80).astype(np.float32),
            'y_test': np.random.randint(0, 2, 15).astype(np.float32),
        }
    }