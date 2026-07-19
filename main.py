# ============================================
# House Price Predictor — FastAPI Backend
# Author: Sandun Godallage
# ============================================

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# කලින් සාදාගත් ML pipeline එක වෙනම file එකකින් import කරගන්නවා
from model_pipeline import load_model, predict_price

# Logging setup (Server එකේ වෙන දේ බලාගන්න)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# global variable එකක් ලෙස model එක තබා ගනී
ml_models = {}

# ─── Lifespan Events (Best Practice for ML Loading) ────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server එක start වෙද්දී සහ stop වෙද්දී ක්‍රියාත්මක වන හොඳම ක්‍රමවේදය"""
    logger.info("Loading Machine Learning Model...")
    # load_model() එක ඇතුළේ auto-train logic එක තියෙන නිසා මෙතනදී load විතරක් කරගන්නවා
    ml_models["house_predictor"] = load_model()
    logger.info("Model loaded successfully!")
    yield
    # Server එක shut down වෙද්දී clean up කරන්න ඕනේ නම් මෙතනට දාන්න පුළුවන්
    ml_models.clear()

# App + Templates setup
app = FastAPI(
    title="House Price Predictor API", 
    version="1.0.0",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ─── Routes ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Web UI Home Page"""
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.get("/predict")
async def predict(size: int, bedrooms: int, age: int):
    """House Price Prediction API"""

    # 1. Validation (නිවැරදි HTTP Status Codes සමඟ)
    if not (500 <= size <= 10000):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Size 500-10000 sq ft අතර විය යුතුය!"
        )
    if not (1 <= bedrooms <= 10):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Bedrooms 1-10 අතර විය යුතුය!"
        )
    if not (1 <= age <= 50):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Age 1-50 අතර විය යුතුය!"
        )

    # 2. Web inputs → California Housing features mapping
    medinc = round((size / 1000) * 1.5 + bedrooms * 0.4, 3)
    averooms = bedrooms + 2.0
    aveoccup = 3.0
    latitude = 34.0
    longitude = -118.0
    population = 1500.0
    avebedrms = 1.0

    try:
        # 3. Predict using cached model
        model = ml_models["house_predictor"]
        price = predict_price(
            model,
            medinc=medinc,
            houseage=float(age),
            averooms=averooms,
            averageoccupation=aveoccup,
            latitude=latitude,
            longitude=longitude,
            population=population,
            avebedrms=avebedrms,
        )

        return {
            "status": "success",
            "input": {
                "size": size,
                "bedrooms": bedrooms,
                "age": age
            },
            "predicted_price": f"${price:,.0f}"
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Prediction ක්‍රියාවලියේදී දෝෂයක් සිදු විය."
        )

@app.get("/health")
async def health():
    """API Health Check"""
    if "house_predictor" not in ml_models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is not loaded yet"
        )
    return {"status": "API Running", "model_loaded": True}