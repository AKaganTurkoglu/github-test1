"""
Sperm Morphology Splitter - Production Inference
=================================================
- Downloads model weights from S3 on startup
- API key authentication via X-API-Key header
- Config from environment variables
"""

import os
import torch
import timm
import uuid
import time
from datetime import datetime, timezone
from decimal import Decimal
from PIL import Image
from torchvision import transforms
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
import boto3
from boto3.dynamodb.conditions import Key
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIG
# ==========================================
MODEL_S3_BUCKET = os.getenv("MODEL_S3_BUCKET")
MODEL_S3_KEY = os.getenv("MODEL_S3_KEY")
MODEL_LOCAL_PATH = "/tmp/best_splitter_img_convnext.pth"

CLASSES = ["Head", "Tail"]
DEVICE = "cpu"

DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE")
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

API_KEYS = set(os.getenv("API_KEYS", "dev-test-key-change-me").split(","))
TTL_DAYS = 180

# ==========================================
# API KEY AUTH
# ==========================================
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key is None or api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key

# ==========================================
# MODEL DOWNLOAD & LOADING
# ==========================================
inference_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def download_model_from_s3():
    """Download model weights from S3 if not already cached locally."""
    if os.path.exists(MODEL_LOCAL_PATH):
        logger.info(f"Model already cached at {MODEL_LOCAL_PATH}")
        return

    logger.info(f"Downloading model from s3://{MODEL_S3_BUCKET}/{MODEL_S3_KEY}...")
    s3 = boto3.client('s3', region_name=AWS_REGION)
    s3.download_file(MODEL_S3_BUCKET, MODEL_S3_KEY, MODEL_LOCAL_PATH)
    logger.info(f"Model downloaded ({os.path.getsize(MODEL_LOCAL_PATH) / 1e6:.1f} MB)")

def load_model():
    download_model_from_s3()
    model = timm.create_model(
        'convnext_base.fb_in22k_ft_in1k',
        pretrained=False,
        num_classes=2
    )
    model.load_state_dict(torch.load(MODEL_LOCAL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    logger.info(f"Model loaded on {DEVICE}")
    return model

def predict_image(model, image_bytes: bytes) -> dict:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = inference_transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return {
        "class": CLASSES[pred_idx],
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASSES[i]: round(probs[0, i].item(), 4) for i in range(len(CLASSES))
        }
    }

# ==========================================
# AWS HELPERS
# ==========================================
def save_to_dynamodb(prediction_id, filename, prediction, s3_key):
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    table = dynamodb.Table(DYNAMODB_TABLE)

    now = datetime.now(timezone.utc)
    table.put_item(Item={
        'prediction_id': prediction_id,
        'predicted_class': prediction['class'],
        'created_date': now.strftime('%Y-%m-%d'),
        'created_at': now.isoformat(),
        'timestamp': int(now.timestamp()),
        'filename': filename,
        'confidence': Decimal(str(prediction['confidence'])),
        'prob_head': Decimal(str(prediction['probabilities']['Head'])),
        'prob_tail': Decimal(str(prediction['probabilities']['Tail'])),
        's3_image_key': s3_key,
        'environment': ENVIRONMENT,
        'ttl': int(now.timestamp()) + (TTL_DAYS * 86400)
    })

def save_image_to_s3(image_bytes, prediction_id, filename):
    s3 = boto3.client('s3', region_name=AWS_REGION)
    ext = filename.rsplit('.', 1)[-1] if '.' in filename else 'jpg'
    date_prefix = datetime.now(timezone.utc).strftime('%Y/%m/%d')
    s3_key = f"uploads/{date_prefix}/{prediction_id}.{ext}"
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=image_bytes, ContentType=f"image/{ext}")
    return s3_key

# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI(
    title="Sperm Morphology Classifier API",
    version="1.0.0",
    description="Upload a sperm microscopy image to classify as Head or Tail abnormality"
)
model = None

@app.on_event("startup")
def startup():
    global model
    model = load_model()

@app.get("/health")
def health():
    return {"status": "healthy", "model": "convnext_splitter", "environment": ENVIRONMENT}

@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    image_bytes = await file.read()
    prediction_id = str(uuid.uuid4())
    prediction = predict_image(model, image_bytes)

    try:
        s3_key = save_image_to_s3(image_bytes, prediction_id, file.filename)
    except Exception as e:
        logger.error(f"S3 failed: {e}")
        s3_key = "upload_failed"

    try:
        save_to_dynamodb(prediction_id, file.filename, prediction, s3_key)
    except Exception as e:
        logger.error(f"DynamoDB failed: {e}")

    return JSONResponse(content={
        "prediction_id": prediction_id,
        "filename": file.filename,
        **prediction
    })

@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: str, api_key: str = Depends(verify_api_key)):
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    table = dynamodb.Table(DYNAMODB_TABLE)

    response = table.get_item(Key={'prediction_id': prediction_id})
    if 'Item' not in response:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    item = {k: float(v) if isinstance(v, Decimal) else v for k, v in response['Item'].items()}
    return JSONResponse(content=item)

@app.get("/predictions/by-class/{class_name}")
def get_by_class(class_name: str, date: str = Query(None), api_key: str = Depends(verify_api_key)):
    if class_name not in CLASSES:
        return JSONResponse(status_code=400, content={"error": f"Must be one of {CLASSES}"})

    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    table = dynamodb.Table(DYNAMODB_TABLE)

    kwargs = {
        'IndexName': 'class-date-index',
        'KeyConditionExpression': Key('predicted_class').eq(class_name)
    }
    if date:
        kwargs['KeyConditionExpression'] &= Key('created_date').eq(date)

    items = table.query(**kwargs).get('Items', [])
    items = [{k: float(v) if isinstance(v, Decimal) else v for k, v in item.items()} for item in items]
    return JSONResponse(content={"class": class_name, "count": len(items), "predictions": items})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)