from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np
import os

# --- 1. MODEL MİMARİSİ ---
class TransformerIDS(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(TransformerIDS, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.classifier(x)
        return x

# --- 2. AYARLAR ---
app = FastAPI()
MODEL_PATH = "ids_model_final.pth"
SCALER_PATH = "scaler_v2.pkl"
ENCODERS_PATH = "encoders_v2.pkl"
CLASSES_PATH = "classes_v2.pkl"

print("🔥 API (HİBRİT MOD) Başlatılıyor...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. TÜRKÇE ETİKET HARİTASI ---
LABEL_MAP = {
    "0": {"name": "Normal Trafik", "status": "GÜVENLİ", "is_threat": False},
    "Benign": {"name": "Normal Trafik", "status": "GÜVENLİ", "is_threat": False},
    "Normal": {"name": "Normal Trafik", "status": "GÜVENLİ", "is_threat": False},
    "1": {"name": "DoS Saldırısı", "status": "TEHLİKELİ", "is_threat": True},
    "2": {"name": "Web / Brute Force", "status": "TEHLİKELİ", "is_threat": True},
    "3": {"name": "Ağ Tarama (Scan)", "status": "TEHLİKELİ", "is_threat": True},
    "4": {"name": "Botnet Aktivitesi", "status": "TEHLİKELİ", "is_threat": True}
}

try:
    if not os.path.exists(MODEL_PATH): raise FileNotFoundError("Model yok!")
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    class_names = joblib.load(CLASSES_PATH)
    input_dim = scaler.n_features_in_
    num_classes = len(class_names)
    model = TransformerIDS(input_dim, 64, 4, 2, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Sistem Hazır!")
except Exception as e:
    print(f"❌ HATA: {e}")
    model = None

class LogItem(BaseModel):
    log_data: dict

# --- 4. ÖN İŞLEME ---
def preprocess_input(data: dict):
    # Gelen veriyi debug için yazdır
    print(f"\n🔍 Analiz Ediliyor: {data.get('service')} | {data.get('conn_state')}")
    
    df = pd.DataFrame([data])
    df.replace('-', 0, inplace=True)
    df.fillna(0, inplace=True)
    
    drop_cols = ['ts', 'uid', 'id.orig_h', 'id.resp_h', 'local_orig', 'local_resp', 'tunnel_parents', 'Day', 'Attack_Type', 'Label']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    cat_cols = ['proto', 'service', 'conn_state', 'history']
    for col in cat_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = df[col].astype(str)
            # Bilinmeyen değerleri 0 yap
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
            
    if hasattr(scaler, 'feature_names_in_'):
        for col in scaler.feature_names_in_:
            if col not in df.columns: df[col] = 0
        df = df[scaler.feature_names_in_]
        
    return torch.FloatTensor(scaler.transform(df)).to(device)

# --- 5. HİBRİT TAHMİN ---
@app.post("/predict")
async def predict(item: LogItem):
    if model is None: return {"error": "Model yok"}
    try:
        # 1. YAPAY ZEKA KARARI
        input_tensor = preprocess_input(item.log_data)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, 1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        raw_label = str(class_names[predicted_idx.item()])
        confidence_score = round(confidence.item() * 100, 2)
        
        # Yapay zekanın ilk kararı
        final_prediction = raw_label
        
        # 2. KURAL TABANLI MÜDAHALE (Güvenlik Duvarı Mantığı) 🛡️
        log = item.log_data
        state = log.get("conn_state", "")
        service = log.get("service", "")
        
        # KURAL A: Bağlantı Reddedildiyse (RSTR, REJ) bu normal olamaz!
        if state in ["RSTR", "REJ", "RSTOS0", "RSTRH"] and final_prediction in ["0", "Benign", "Normal"]:
            print("⚠️ KURAL TETİKLENDİ: Bağlantı Reddedildi -> Brute Force Şüphesi")
            final_prediction = "2" # Web / Brute Force Sınıfına zorla
            confidence_score = 95.0 # Eminiz
            
        # KURAL B: Syn Flood Belirtisi (S0 ve çok paket yoksa)
        if state == "S0" and final_prediction in ["0", "Benign", "Normal"]:
            print("⚠️ KURAL TETİKLENDİ: S0 Hatası -> DoS Şüphesi")
            final_prediction = "1" # DoS Sınıfına zorla
            confidence_score = 90.0

        # KURAL C: Bilinmeyen IRC Servisi -> Kesin Botnet
        if service == "irc":
            print("⚠️ KURAL TETİKLENDİ: IRC Servisi -> Botnet Şüphesi")
            final_prediction = "4" # Botnet Sınıfına zorla
            confidence_score = 99.0

        # Sonucu Hazırla
        result_info = LABEL_MAP.get(final_prediction, {"name": final_prediction, "status": "TEHLİKELİ", "is_threat": True})
        
        return {
            "prediction": result_info["name"],
            "confidence": confidence_score,
            "status": result_info["status"],
            "is_threat": result_info["is_threat"]
        }
    except Exception as e:
        print(f"HATA: {e}")
        return {"error": str(e)}
