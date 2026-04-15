from ultralytics import YOLO
import cv2
import os

# ==========================================
# AYARLAR
VIDEO_DOSYASI = '/home/hasan/Masaüstü/video.mp4'  # Senin video ismin
MODEL_YOLU = '/home/hasan/rtdetr-l.engine'        # Dinamik motor
CIKIS_DOSYASI = '/home/hasan/Masaüstü/final_sonuc.avi'
# ==========================================

print(f"Dinamik Model Yükleniyor... ({MODEL_YOLU})")
# TensorRT motorunu yüklüyoruz
model = YOLO(MODEL_YOLU, task='detect')

# Video dosyasını kontrol et
if not os.path.exists(VIDEO_DOSYASI):
    print(f"HATA: Video dosyası bulunamadı -> {VIDEO_DOSYASI}")
    exit()

cap = cv2.VideoCapture(VIDEO_DOSYASI)
if not cap.isOpened():
    print("HATA: Video dosyası var ama açılamadı (Codec sorunu olabilir).")
    exit()

# Video Bilgileri
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30

# Kayıtçı
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(CIKIS_DOSYASI, fourcc, fps, (width, height))
font = cv2.FONT_HERSHEY_SIMPLEX

print("İşlem başlıyor... Kapatmak için videoya tıklayıp 'q' tuşuna bas.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ---------------------------------------------------------
    # KRİTİK NOKTA: 'imgsz' parametresini kaldırdık!
    # Artık model, videonun boyutuna göre dinamik karar verecek.
    # conf=0.25: İdeal güven oranı.
    # ---------------------------------------------------------
    results = model.track(frame, persist=True, classes=[0], conf=0.25, verbose=False)

    annotated_frame = results[0].plot()

    # FPS ve Hız Bilgisi
    if hasattr(results[0], 'speed'):
        hiz = results[0].speed
        toplam_ms = hiz.get('preprocess', 0) + hiz.get('inference', 0) + hiz.get('postprocess', 0)
        anlik_fps = 1000 / toplam_ms if toplam_ms > 0 else 0
    else:
        anlik_fps = 0

    kisi_sayisi = len(results[0].boxes)
    
    # Bilgileri Ekrana Yaz
    cv2.putText(annotated_frame, f"FPS: {anlik_fps:.1f}", (20, 50), font, 1, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Yaya: {kisi_sayisi}", (20, 90), font, 1, (0, 255, 0), 2)

    cv2.imshow("RT-DETR Dinamik & FP32", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Tamamlandı! Video kaydedildi: {CIKIS_DOSYASI}")
