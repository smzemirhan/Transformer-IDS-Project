from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib
import uvicorn

# --- 1. MODEL MİMARİSİ (Aynısı Olmak Zorunda) ---
class TransformerIDS(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout):
        super(TransformerIDS, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        output = self.classifier(x)
        return output

# --- 2. AYARLAR VE YÜKLEME ---
app = FastAPI(title="Siber Güvenlik IDS API", description="Transformer Tabanlı Saldırı Tespiti")

# Modeli ve yardımcı dosyaları yükle
print("Sistem başlatılıyor...")
device = torch.device("cpu")

# Parametreler (Eğitimdekiyle aynı olmalı)
INPUT_DIM = 13
MODEL_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
NUM_CLASSES = 5
DROPOUT = 0.1

try:
    # Modeli Yükle
    model = TransformerIDS(INPUT_DIM, MODEL_DIM, NUM_HEADS, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)
    model.load_state_dict(torch.load("ids_model.pth", map_location=device))
    model.eval() # Test moduna al
    print("-> Model yüklendi (ids_model.pth)")

    # Scaler'ı Yükle
    scaler = joblib.load("scaler.pkl")
    print("-> Scaler yüklendi (scaler.pkl)")
    
    # Encoder'ları Yükle
    encoders = joblib.load("encoders.pkl")
    print("-> Encoders yüklendi (encoders.pkl)")

except Exception as e:
    print(f"HATA: Dosyalar yüklenemedi! {e}")

# Saldırı İsimleri
LABELS = {0: 'Normal (Benign)', 1: 'BruteForce', 2: 'DoS/DDoS', 3: 'Web Attack', 4: 'Botnet'}

# --- 3. İSTEK FORMATI ---
class TrafficData(BaseModel):
    # Modelin beklediği 13 özellik
    proto: str = "tcp"
    service: str = "http"
    duration: float = 1.0
    orig_bytes: int = 100
    resp_bytes: int = 200
    conn_state: str = "SF"
    missed_bytes: int = 0
    history: str = "ShAdDaFf"
    orig_pkts: int = 4
    orig_ip_bytes: int = 200
    resp_pkts: int = 5
    resp_ip_bytes: int = 300
    # ip_proto gibi ek özellikler gerekirse buraya eklenir, şimdilik temel 12+1
    
# --- 4. TAHMİN ENDPOINT ---
@app.post("/predict")
def predict(data: TrafficData):
    try:
        # 1. Gelen veriyi işle (Preprocessing)
        # Kategorik verileri sayıya çevir (Encoder kullanarak)
        # Eğer eğitimde görmediğimiz yeni bir değer gelirse (örn: bilinmeyen protocol) hata vermemesi için try-except veya default değer kullanırız.
        
        def safe_encode(col_name, value):
            try:
                if col_name in encoders:
                    return encoders[col_name].transform([str(value)])[0]
                return 0
            except:
                return 0 # Bilinmeyen değer gelirse 0 varsay

        features = [
            safe_encode('proto', data.proto),
            safe_encode('service', data.service),
            data.duration,
            data.orig_bytes,
            data.resp_bytes,
            safe_encode('conn_state', data.conn_state),
            data.missed_bytes,
            safe_encode('history', data.history),
            data.orig_pkts,
            data.orig_ip_bytes,
            data.resp_pkts,
            data.resp_ip_bytes,
            6 # ip_proto (genelde TCP=6, UDP=17 sabittir, basitleştirmek için 6 veriyoruz)
        ]

        # 2. Numpy Dizisine Çevir ve Ölçekle (Scaler)
        features_array = np.array([features], dtype=np.float32)
        features_scaled = scaler.transform(features_array)

        # 3. Tensor'a çevir
        tensor_input = torch.tensor(features_scaled).to(device)

        # 4. Tahmin Et
        with torch.no_grad():
            outputs = model(tensor_input)
            _, predicted_class = torch.max(outputs, 1)
            probability = torch.softmax(outputs, dim=1).max().item()

        result_class = int(predicted_class.item())
        result_label = LABELS.get(result_class, "Unknown")

        return {
            "prediction_code": result_class,
            "prediction_label": result_label,
            "confidence": round(probability * 100, 2),
            "status": "Riskli" if result_class != 0 else "Güvenli"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Siber Güvenlik API Çalışıyor!"}

if __name__ == "__main__":
    # Tüm ağlardan erişime aç (0.0.0.0)
    uvicorn.run(app, host="0.0.0.0", port=8000)
