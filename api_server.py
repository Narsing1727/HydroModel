from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="HydroSphere Flood Prediction API 🌊")

# ✅ Load ML assets
model = joblib.load('flood_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('feature_names.pkl')

print("✅ Model Loaded")
print("🧩 Features:", features)

# ✅ Allow CORS (Frontend can call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FloodInput(BaseModel):
    rainfall: float = 0.0
    ndvi: float = 0.0
    distance_from_river: float = 0.0
    slope: float = 0.0


@app.post("/predict")
def predict(input: dict):
    # ✅ Convert user input to ordered feature vector
    x = np.array([input.get(f, 0) for f in features]).reshape(1, -1)

    # ✅ Scale input
    x_scaled = scaler.transform(x)

    # ✅ Model prediction
    prob = float(model.predict_proba(x_scaled)[0, 1])
    pred = int(prob > 0.5)

    print(f"\n📥 Input: {input}")
    print(f"🔮 Probability: {prob:.3f}")
    print(f"🧠 Prediction: {'FLOOD' if pred == 1 else 'SAFE'}")

    return {
        "prediction": "FLOOD" if pred == 1 else "SAFE",
        "probability": round(prob, 3)
    }


@app.get("/")
def home():
    return {
        "message": "HydroSphere ML API is running ✅",
        "features_count": len(features)
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
