import os
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from src.data_preprocessing import preprocess_image
from app.main import preprocess_image as api_preprocess

# ── Mock Data for Testing ──
@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "test_img.jpg"
    # Create a random 300x300 image
    img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    img.save(path)
    return path

# ── Task 1: Data Pre-processing Unit Test ──
def test_data_preprocessing_resize(sample_image, tmp_path):
    """Verify that preprocessing correctly resizes an image to 224x224."""
    output_path = tmp_path / "processed_img.jpg"
    success = preprocess_image(sample_image, output_path)
    
    assert success is True
    assert output_path.exists()
    
    with Image.open(output_path) as img:
        assert img.size == (224, 224)
        assert img.mode == "RGB"

# ── Task 2: Model Utility/Inference Unit Test ──
def test_api_preprocessing_shape(sample_image):
    """Verify that API preprocessing converts image bytes to correct tensor shape."""
    with open(sample_image, "rb") as f:
        img_bytes = f.read()
    
    img_tensor = api_preprocess(img_bytes)
    
    # Check shape: (batch, height, width, channels)
    assert img_tensor.shape == (1, 224, 224, 3)
    # Check normalization
    assert img_tensor.max() <= 1.0
    assert img_tensor.min() >= 0.0
