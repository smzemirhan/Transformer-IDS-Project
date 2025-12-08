import pandas as pd
import os
import glob

# Klasör isimleri ve gün eşleştirmeleri
folders = {
    'Monday': 'Monday',
    'Tuesday': 'Tuesday',
    'Wednesday': 'Wednesday',
    'Thursday': 'Thursday',
    'Friday': 'Friday'
}

def parse_zeek_log(file_path):
    """Zeek loglarını okuyup DataFrame'e çevirir"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Zeek loglarında veri genelde 8. satırdan sonra başlar ama 
    # sütun isimleri #fields ile başlayan satırdır (genelde 7. satır)
    header_line = [x for x in lines if x.startswith("#fields")]
    if not header_line:
        return None
    
    # Sütun isimlerini temizle (başındaki #fields ve tab boşluklarını at)
    columns = header_line[0].strip().split('\t')[1:]
    
    # Veriyi oku (yorum satırlarını atlayarak)
    # Zeek logları TAB ile ayrılmıştır
    data = [x.strip().split('\t') for x in lines if not x.startswith("#")]
    
    df = pd.DataFrame(data, columns=columns)
    return df

all_data = []

print("Veriler okunuyor ve birleştiriliyor...")

for folder, day_name in folders.items():
    # Dosya yolunu oluştur (örn: ./Monday/conn.log)
    file_path = os.path.join(folder, 'conn.log')
    
    if os.path.exists(file_path):
        print(f"{day_name} verisi işleniyor: {file_path}")
        try:
            df = parse_zeek_log(file_path)
            if df is not None:
                # Hangi günden geldiğini bilelim diye sütun ekle
                df['Day'] = day_name
                all_data.append(df)
                print(f"  -> {len(df)} satır eklendi.")
            else:
                print(f"  -> HATA: {day_name} dosyası boş veya hatalı format.")
        except Exception as e:
            print(f"  -> HATA oluştu: {e}")
    else:
        print(f"UYARI: {folder} klasöründe conn.log bulunamadı!")

# Tüm günleri alt alta birleştir
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    
    # 'ts' (timestamp) sütununu okunabilir tarih formatına çevirelim
    # Bu, saldırı zamanlarını etiketlerken çok işimize yarayacak
    print("Zaman damgaları dönüştürülüyor...")
    final_df['ts'] = pd.to_datetime(final_df['ts'], unit='s', errors='coerce')
    
    # Sonucu kaydet
    output_file = "tum_veriler.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\nBAŞARILI! Tüm veriler '{output_file}' dosyasına kaydedildi.")
    print(f"Toplam Satır Sayısı: {len(final_df)}")
else:
    print("HATA: Hiçbir veri okunamadı.")
