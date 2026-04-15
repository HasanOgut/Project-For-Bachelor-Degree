from ultralytics import RTDETR
import cv2
import torch

# 1. GPU Kontrolü
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Kullanılan Cihaz: {device}")

# 2. Modeli Yükle (RT-DETR Large)
# İlk çalışmada indirmişti, şimdi direkt cache'den alacak.
model = RTDETR('rtdetr-l.pt')

# 3. Video Kaynağı (0 = Webcam, veya 'video.mp4')
cap = cv2.VideoCapture(0)

# Döngü Başlangıcı
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 4. Tahmin Yap (Inference)
    # conf=0.45: Güven oranı %45'in altındakileri (o sol alttaki 0.29 gibileri) görmezden gel.
    # classes=[0]: Sadece insan sınıfına odaklan.
    results = model.predict(frame, conf=0.45, device=0, classes=[0], verbose=False)

    # 5. Sonuçları Çiz ve Veriyi Al
    annotated_frame = results[0].plot() # Çizilmiş kareyi al

    # Koordinatlara erişmek istersen:
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # Koordinatlar
        guven = box.conf[0].cpu().numpy() # Güven skoru
        print(f"Yaya Tespit Edildi -> Konum: {int(x1)},{int(y1)} Güven: {guven:.2f}")

    # 6. Ekrana Bas
    cv2.imshow("RT-DETR Yaya Tespiti", annotated_frame)

    # 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
