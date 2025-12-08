import pandas as pd
import numpy as np

# Veriyi oku
print("Veri okunuyor (bu biraz sürebilir)...")
df = pd.read_csv("tum_veriler.csv")

# Zaman damgasını datetime formatına çevir
df['ts'] = pd.to_datetime(df['ts'])

# Varsayılan etiket: 'Benign' (Normal) -> 0
df['Label'] = 0 
df['Attack_Type'] = 'Benign'

print("Etiketleme işlemi başlıyor...")

def apply_label(df, day, start_time, end_time, attack_name, label_code):
    """Belirli gün ve saat aralığındaki trafiği etiketler"""
    # CIC-IDS2017 yerel saati UTC-3'tür. Zeek logları UTC tutar.
    # Bu yüzden saatlere +3 ekleyerek UTC'ye çeviriyoruz.
    # Örn: Saldırı 09:20 ise UTC'de 12:20'dir.
    
    mask = (
        (df['Day'] == day) & 
        (df['ts'].dt.time >= pd.to_datetime(start_time).time()) & 
        (df['ts'].dt.time <= pd.to_datetime(end_time).time())
    )
    
    df.loc[mask, 'Label'] = label_code
    df.loc[mask, 'Attack_Type'] = attack_name
    
    count = mask.sum()
    if count > 0:
        print(f"  -> {attack_name} ({day}): {count} adet etiketlendi.")

# --- SALI (Tuesday) ---
# FTP-Patator (Brute Force) | 09:20 - 10:20 (Local) -> 12:20 - 13:20 (UTC)
apply_label(df, 'Tuesday', '12:20:00', '13:20:00', 'BruteForce-FTP', 1)
# SSH-Patator (Brute Force) | 14:00 - 15:00 (Local) -> 17:00 - 18:00 (UTC)
apply_label(df, 'Tuesday', '17:00:00', '18:00:00', 'BruteForce-SSH', 1)

# --- ÇARŞAMBA (Wednesday) ---
# DoS slowloris | 09:47 - 10:10 (Local) -> 12:47 - 13:10 (UTC)
apply_label(df, 'Wednesday', '12:47:00', '13:10:00', 'DoS-Slowloris', 2)
# DoS Slowhttptest | 10:14 - 10:35 (Local) -> 13:14 - 13:35 (UTC)
apply_label(df, 'Wednesday', '13:14:00', '13:35:00', 'DoS-Slowhttptest', 2)
# DoS Hulk | 10:43 - 11:00 (Local) -> 13:43 - 14:00 (UTC)
apply_label(df, 'Wednesday', '13:43:00', '14:00:00', 'DoS-Hulk', 2)
# DoS GoldenEye | 11:10 - 11:23 (Local) -> 14:10 - 14:23 (UTC)
apply_label(df, 'Wednesday', '14:10:00', '14:23:00', 'DoS-GoldenEye', 2)
# Heartbleed | 15:12 - 15:32 (Local) -> 18:12 - 18:32 (UTC)
apply_label(df, 'Wednesday', '18:12:00', '18:32:00', 'Heartbleed', 2)

# --- PERŞEMBE (Thursday) ---
# Web Attacks (Brute Force, XSS, SQLi)
# 09:20 - 10:42 (Local) -> 12:20 - 13:42 (UTC)
apply_label(df, 'Thursday', '12:20:00', '13:42:00', 'Web-Attack', 3)
# Infiltration (Sızma)
apply_label(df, 'Thursday', '17:19:00', '17:35:00', 'Infiltration', 3)

# --- CUMA (Friday) ---
# Botnet | 09:30 - 13:30 (Local) -> 12:30 - 16:30 (UTC)
apply_label(df, 'Friday', '12:30:00', '16:30:00', 'Botnet', 4)
# PortScan | 13:55 - 15:26 (Local) -> 16:55 - 18:26 (UTC)
apply_label(df, 'Friday', '16:55:00', '18:26:00', 'PortScan', 4)
# DDoS | 15:56 - 16:16 (Local) -> 18:56 - 19:16 (UTC)
apply_label(df, 'Friday', '18:56:00', '19:16:00', 'DDoS', 2) # DDoS'u sınıf 2'ye dahil edebiliriz veya 5 yapabiliriz. Şimdilik DoS ile aynı (2) olsun.

print("\n--- ÖZET İSTATİSTİKLER ---")
print(df['Attack_Type'].value_counts())
print("\nSınıf Dağılımı (Label):")
print(df['Label'].value_counts())

# Sonucu kaydet
output_file = "egitim_verisi.csv"
print(f"\nVeriler kaydediliyor: {output_file} ...")
df.to_csv(output_file, index=False)
print("BAŞARILI! Etiketleme tamamlandı.")
