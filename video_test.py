from ultralytics import YOLO
import cv2
import os

# ==========================================
# AYARLAR (Burayı kendi videona göre düzenle!)
VIDEO_DOSYASI = '/home/hasan/Masaüstü/video.mp4' 
MODEL_YOLU = '/home/hasan/rtdetr-l.engine'
CIKIS_DOSYASI = '/home/hasan/Masaüstü/hizli_sonuc.avi'
# ==========================================

print(f"Hızlandırılmış Model Yükleniyor... ({MODEL_YOLU})")
# TensorRT modelleri için task='detect' belirtiyoruz
model = YOLO(MODEL_YOLU, task='detect')

cap = cv2.VideoCapture(VIDEO_DOSYASI)
if not cap.isOpened():
    print(f"Hata: Video açılamadı! Lütfen dosya yolunu kontrol et: {VIDEO_DOSYASI}")
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

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # TAHMİN (INFERENCE) - FP32 Engine Testi
    # conf=0.25: Detayları kaçırmaması için güven eşiğini ayarladık
    results = model.track(frame, persist=True, classes=[0], conf=0.25, imgsz=640, verbose=False)

    annotated_frame = results[0].plot()

    # Hız ve Sayım Bilgileri
    if hasattr(results[0], 'speed'):
        hiz = results[0].speed
        # Toplam işlem süresi (milisaniye)
        toplam_ms = hiz.get('preprocess', 0) + hiz.get('inference', 0) + hiz.get('postprocess', 0)
        anlik_fps = 1000 / toplam_ms if toplam_ms > 0 else 0
    else:
        anlik_fps = 0

    kisi_sayisi = len(results[0].boxes)
    
    # Ekrana Yazdır
    cv2.putText(annotated_frame, f"FPS: {anlik_fps:.1f}", (20, 50), font, 1, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Yaya: {kisi_sayisi}", (20, 90), font, 1, (0, 255, 0), 2)

    cv2.imshow("RT-DETR FP32 Test", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("İşlem tamamlandı.")
