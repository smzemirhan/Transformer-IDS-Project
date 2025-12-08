import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix

# --- AYARLAR (Hiperparametreler) ---
# CPU'da eğitim yapacağımız için batch_size'ı yüksek tutuyoruz (Hız artışı sağlar)
BATCH_SIZE = 2048  
EPOCHS = 3         # Vakit kaybetmemek için şimdilik 3 tur (Epoch) eğiteceğiz
LEARNING_RATE = 0.001

# Model Mimarisi Ayarları
INPUT_DIM = 13     # Sizin verinizdeki özellik sayısı (Resimden aldık)
MODEL_DIM = 64     # Transformer içindeki vektör boyutu
NUM_HEADS = 4      # Multi-Head Attention kafa sayısı
NUM_LAYERS = 2     # Kaç katmanlı Transformer olacak
NUM_CLASSES = 5    # 0:Benign, 1:BruteForce, 2:DoS, 3:WebAttack, 4:Botnet
DROPOUT = 0.1

print(f"Cihaz kontrol ediliyor...")
device = torch.device("cpu") # GPU olmadığı için CPU'ya sabitliyoruz
print(f"Eğitim Cihazı: {device}")

# --- 1. VERİ SETİ SINIFI ---
class ZeekDataset(Dataset):
    def __init__(self, x_path, y_path):
        print(f"Veri yükleniyor: {x_path}...")
        self.X = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Veri setlerini yükle
train_dataset = ZeekDataset('X_train.npy', 'y_train.npy')
test_dataset = ZeekDataset('X_test.npy', 'y_test.npy')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. TRANSFORMER MODEL MİMARİSİ ---
# Öneri formunda vaat edilen "Yenilikçi Mimari" burasıdır.
class TransformerIDS(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout):
        super(TransformerIDS, self).__init__()
        
        # 1. Embedding Katmanı: 13 özelliği 64'lük bir uzaya genişletir
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # 2. Transformer Encoder Bloğu
        # PyTorch'un hazır Transformer katmanlarını kullanıyoruz
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Sınıflandırıcı (Çıkış Katmanı)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch_Size, 13)
        
        # Transformer sıra tabanlı (sequence) çalışır. 
        # Bizim verimiz tek satırlık akışlar olduğu için boyutu (1, Batch, Model_Dim) yapacağız.
        x = self.embedding(x)  # (Batch, 64)
        x = x.unsqueeze(0)     # (1, Batch, 64) -> Sanki 1 kelimelik bir cümleymiş gibi
        
        # Transformer'dan geçir
        x = self.transformer_encoder(x)
        
        # Tekrar eski haline getir
        x = x.squeeze(0)       # (Batch, 64)
        
        # Sınıflandır
        output = self.classifier(x)
        return output

# Modeli başlat
model = TransformerIDS(INPUT_DIM, MODEL_DIM, NUM_HEADS, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss() # Çoklu sınıflandırma hatası hesaplayıcı
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nModel Mimarisi:")
print(model)
print(f"\nToplam Parametre Sayısı: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# --- 3. EĞİTİM DÖNGÜSÜ ---
print("\n--- EĞİTİM BAŞLIYOR ---")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Sıfırla, Hesapla, Geri Yay (Standart PyTorch döngüsü)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # İstatistikler
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Adım [{i}/{len(train_loader)}], Kayıp: {loss.item():.4f}")

    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Tamamlandı. Ort. Kayıp: {running_loss/len(train_loader):.4f}, Doğruluk: {epoch_acc:.2f}%")

total_time = time.time() - start_time
print(f"\nEğitim Tamamlandı! Toplam Süre: {total_time/60:.2f} dakika")

# --- 4. TEST VE RAPORLAMA ---
print("\n--- TEST SONUÇLARI ---")
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

# Raporu Yazdır
target_names = ['Benign(0)', 'BruteForce(1)', 'DoS(2)', 'WebAttack(3)', 'Botnet(4)']
print(classification_report(all_labels, all_preds, target_names=target_names))

# --- 5. MODELİ KAYDETME ---
# Bu dosyayı daha sonra Web Panelinde ve API'da kullanacağız
torch.save(model.state_dict(), "ids_model.pth")
print("Model kaydedildi: ids_model.pth")
