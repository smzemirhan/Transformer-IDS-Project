import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# 1. Veriyi Oku
print("Veri seti yükleniyor...")
df = pd.read_csv("egitim_verisi.csv")

# Gereksiz sütunları at (Zaman damgası, IP adresleri ve UID model için genelde ezber yaratır, çıkarıyoruz)
# Modelin "davranışa" bakmasını istiyoruz, "kimin yaptığına" değil.
cols_to_drop = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'Day', 'Attack_Type', 'local_orig', 'local_resp', 'tunnel_parents']
# Mevcut sütunları kontrol edip varsa düşürelim
existing_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=existing_drop)

print(f"Kullanılacak Özellikler: {df.columns.tolist()}")

# 2. Eksik Verileri Temizle
# Zeek loglarında boş veriler '-' olarak görünür, bunları 0 yapalım veya NaN
df = df.replace('-', np.nan)
df = df.replace('(empty)', np.nan)

# Sayısal olması gereken sütunları zorla sayıya çevir (hata verirse NaN yapar)
numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# NaN olan yerleri 0 ile doldur
df = df.fillna(0)

# 3. Kategorik Verileri Sayıya Çevir (Encoding)
# 'proto', 'service', 'conn_state' gibi metin sütunlarını sayıya çevireceğiz
categorical_cols = ['proto', 'service', 'conn_state', 'history']
encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        # Veriyi string'e çevir ki hata almayalım
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Encoder'ları kaydedelim (İleride canlı sistemde lazım olacak)
joblib.dump(encoders, 'encoders.pkl')

# 4. Hedef (Label) ve Özellikleri (X) Ayır
y = df['Label'].values
X = df.drop(columns=['Label']).values

# 5. Veriyi Ölçekle (0-1 arasına sıkıştır)
# Deep Learning modelleri büyük sayılardan hoşlanmaz
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Scaler'ı da kaydet (Canlı sistem için)
joblib.dump(scaler, 'scaler.pkl')

# 6. Train / Test Ayrımı Yap (%80 Eğitim, %20 Test)
print("Veri Eğitim ve Test olarak ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Sonuçları Kaydet (Numpy formatında hızlı yükleme için)
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("------------------------------------------------")
print(f"İşlem Tamamlandı!")
print(f"Eğitim Verisi Boyutu: {X_train.shape}")
print(f"Test Verisi Boyutu:   {X_test.shape}")
print("Dosyalar kaydedildi: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print("Encoder ve Scaler kaydedildi: encoders.pkl, scaler.pkl")
