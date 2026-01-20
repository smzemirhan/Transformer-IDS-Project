import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE 
import joblib
import time
import os

# --- 1. AYARLAR ---
# Senin doğru dosya yolun
CSV_PATH = "/home/emirhansmz/egitim_verisi.csv"
MODEL_SAVE_PREFIX = "ids_model_epoch" # Her epoch'ta bu isimle kaydedecek
EPOCHS = 20
BATCH_SIZE = 4096
LEARNING_RATE = 0.001

# Cihaz Seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eğitim {device} üzerinde yapılacak.")

# --- 2. VERİ YÜKLEME VE TEMİZLİK ---
print("Veri seti yükleniyor...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"HATA: Dosya bulunamadı! Lütfen yolu kontrol et: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print("Veri temizleniyor (Gereksiz karakterler ve boşluklar)...")
# 1. Tire (-) işaretlerini 0 yap
df.replace('-', 0, inplace=True)
# 2. Boşlukları (NaN) 0 ile doldur
df.fillna(0, inplace=True)

# --- 3. GEREKSİZ SÜTUNLARI ATMA ---
# Hem 'T' harfi içeren gereksizleri, hem de modelin görmemesi gerekenleri atıyoruz
print("Gereksiz sütunlar temizleniyor...")
drop_cols = [
    'ts', 'uid', 'id.orig_h', 'id.resp_h', 
    'local_orig', 'local_resp', 'tunnel_parents', 'Day', # Bunlar 'T' hatası veriyordu
    'Attack_Type', # Bunu X'ten atıyoruz çünkü hedefimiz Label
    'Label'        # Bunu y (hedef) yapacağız, o yüzden X'ten atıyoruz
]

# Sadece var olan sütunları düşür (Hata almamak için kontrol)
existing_drop_cols = [col for col in drop_cols if col in df.columns]
X = df.drop(columns=existing_drop_cols)

# Hedef Değişken (Büyük L ile Label)
y = df['Label']

# --- 4. ENCODING (Sayısal Dönüşüm) ---
print("Kategorik veriler sayısala çevriliyor...")
encoders = {}
# Encoding yapılacak sütunlar
cat_cols = ['proto', 'service', 'conn_state', 'history']

for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = X[col].astype(str) # Hata önlemek için string'e çevir
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

# Hedef Etiket Encoding (Label)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Sınıf isimlerini string olarak sakla (Rapor hatasını önlemek için)
classes = [str(c) for c in label_encoder.classes_]
print(f"Tespit edilecek sınıflar: {classes}")

# --- 5. SMOTE (VERİ DENGELEME) ---
print("Veri dengesizliği gideriliyor (SMOTE)... Bu işlem biraz sürebilir.")
print(f"SMOTE öncesi veri sayısı: {len(X)}")

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"SMOTE sonrası veri sayısı: {len(X_resampled)}")

# --- 6. SCALING (ÖLÇEKLEME) ---
print("Veriler 0-1 aralığına ölçekleniyor...")
scaler = MinMaxScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Scaler ve Encoder'ları kaydet (API için gerekli)
joblib.dump(scaler, "scaler_v2.pkl")
joblib.dump(encoders, "encoders_v2.pkl")
joblib.dump(classes, "classes_v2.pkl")
print("Scaler ve Encoder dosyaları kaydedildi.")

# --- 7. TENSOR DÖNÜŞÜMÜ ---
# Train/Test Ayrımı (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# NumPy dizilerini Tensor'a çevir (Burada .values hatasını fixledik)
# y_train zaten numpy array olarak geliyor SMOTE'tan sonra, ama garanti olsun diye direkt tensor yapıyoruz
train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- 8. MODEL MİMARİSİ (TRANSFORMER) ---
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

model = TransformerIDS(
    input_dim=X_train.shape[1], 
    model_dim=64, 
    num_heads=4, 
    num_layers=2, 
    num_classes=len(classes)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 9. EĞİTİM DÖNGÜSÜ ---
print(f"Eğitim başlıyor... Toplam {EPOCHS} Epoch.")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] -> Loss: {avg_loss:.4f} | Accuracy: %{epoch_acc:.2f}")
    
    # HER EPOCH SONUNDA KAYDET (Elektrik gitse bile elimizde olsun)
    torch.save(model.state_dict(), f"{MODEL_SAVE_PREFIX}_{epoch+1}.pth")
    print(f"-> Model yedeği alındı: {MODEL_SAVE_PREFIX}_{epoch+1}.pth")

print(f"Eğitim tamamlandı! Toplam Süre: {(time.time()-start_time)/60:.2f} dakika.")

# --- 10. TEST VE SONUÇ ---
print("Test ediliyor...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n--- SINIFLANDIRMA RAPORU ---")
# Hata vermemesi için target_names parametresini kaldırdık, varsayılanı kullanacak
print(classification_report(all_labels, all_preds))

# En son modeli de ana isimle kaydet
torch.save(model.state_dict(), "ids_model_final.pth")
print("Final model kaydedildi: ids_model_final.pth")
