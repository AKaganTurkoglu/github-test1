"""
Tests for the inference API.
Run before every deployment — if any fail, deploy is blocked.
"""

import pytest
from unittest.mock import patch, MagicMock
import torch
from PIL import Image
from io import BytesIO
import os

os.environ["API_KEYS"] = "test-key-123"
os.environ["DYNAMODB_TABLE"] = "test-table"
os.environ["S3_BUCKET"] = "test-bucket"


def test_model_output_shape():
    """ConvNeXt with 2 classes should output shape (1, 2)."""
    import timm
    model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=False, num_classes=2)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 384, 384))
    assert output.shape == (1, 2)


def test_transform_shape():
    """Transform should produce (3, 384, 384) tensor."""
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = t(Image.new('RGB', (100, 100), color='red'))
    assert tensor.shape == (3, 384, 384)


def test_softmax_sums_to_one():
    """Probabilities must sum to 1."""
    probs = torch.softmax(torch.tensor([[2.5, -1.3]]), dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_missing_api_key_returns_403():
    """No API key should get 403."""
    # Patch model loading since we don't have weights in CI
    with patch('app.download_model_from_s3'), \
         patch('app.load_model') as mock_load:
        mock_load.return_value = MagicMock()

        from app import app
        from fastapi.testclient import TestClient
        client = TestClient(app)

        img = Image.new('RGB', (50, 50))
        buf = BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)

        response = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert response.status_code == 403


def test_valid_api_key_returns_200():
    """Valid API key should work."""
    with patch('app.download_model_from_s3'), \
         patch('app.load_model') as mock_load, \
         patch('app.predict_image') as mock_predict, \
         patch('app.save_image_to_s3', return_value="uploads/test.jpg"), \
         patch('app.save_to_dynamodb'):

        mock_load.return_value = MagicMock()
        mock_predict.return_value = {
            "class": "Head", "confidence": 0.95,
            "probabilities": {"Head": 0.95, "Tail": 0.05}
        }

        from app import app
        from fastapi.testclient import TestClient
        client = TestClient(app)

        img = Image.new('RGB', (50, 50))
        buf = BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", buf, "image/jpeg")},
            headers={"X-API-Key": "test-key-123"}
        )
        assert response.status_code == 200
        assert response.json()["class"] == "Head"


def test_health_no_auth():
    """Health endpoint works without API key."""
    with patch('app.download_model_from_s3'), \
         patch('app.load_model', return_value=MagicMock()):

        from app import app
        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"