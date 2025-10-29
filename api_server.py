from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="HydroSphere Flood Prediction API 🌊")

model = joblib.load('flood_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('feature_names.pkl')

print("✅ Model, Scaler, and Feature Names Loaded Successfully!")
print("📊 Feature Count:", len(features))
print("🧩 Features:", features)

class InputData(BaseModel):
    data: dict


@app.post("/predict")
def predict(input: InputData):
    data = input.data

    x = np.array([data.get(f, 0) for f in features]).reshape(1, -1)


    x_scaled = scaler.transform(x)

    prob = float(model.predict_proba(x_scaled)[0, 1])
    pred = int(prob > 0.5)


    z_scores = abs((x - scaler.mean_) / (scaler.scale_ + 1e-6))
    max_z = np.max(z_scores)
    high_deviation_features = [features[i] for i, z in enumerate(z_scores[0]) if z > 3]

    print("\n==============================")
    print("📥 Raw Data Received:", data)
    print("📏 Ordered Feature Vector:", dict(zip(features, x.flatten())))
    print(f"📉 Scaler Mean Range: {np.min(scaler.mean_):.2f} → {np.max(scaler.mean_):.2f}")
    print(f"📈 Max Input Z-Score: {max_z:.2f}")
    if high_deviation_features:
        print(f"⚠️ Features Out of Range (>3σ): {high_deviation_features}")
    else:
        print("✅ All input features within normal range.")
    print(f"🔮 Flood Probability: {prob:.3f}")
    print(f"🧠 Prediction: {'FLOOD' if pred == 1 else 'SAFE'}")
    print("==============================\n")

    
    return {
        "prediction": "FLOOD" if pred == 1 else "SAFE",
        "probability": round(prob, 3),
        "out_of_range_features": high_deviation_features,
        "max_deviation": round(max_z, 2)
    }


@app.get("/")
def home():
    return {
        "message": "HydroSphere Flood Prediction API running ✅",
        "status": "active",
        "model_features": len(features)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
