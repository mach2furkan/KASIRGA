import cv2
import sys

IP = "192.168.1.108"
USER = "Admin"
PASS = "Bisavunma8503"

# Yaygın kamera markaları için RTSP URL formatları
candidates = [
    f"rtsp://{USER}:{PASS}@{IP}:554/Streaming/Channels/101",      # Hikvision ana stream
    f"rtsp://{USER}:{PASS}@{IP}:554/Streaming/Channels/1",        # Hikvision alt.
    f"rtsp://{USER}:{PASS}@{IP}:554/cam/realmonitor?channel=1&subtype=0",  # Dahua
    f"rtsp://{USER}:{PASS}@{IP}:554/stream1",                     # Generic
    f"rtsp://{USER}:{PASS}@{IP}:554/live/ch00_0",                 # Reolink
    f"rtsp://{USER}:{PASS}@{IP}:554/h264Preview_01_main",         # Reolink alt.
    f"rtsp://{USER}:{PASS}@{IP}:554/",                            # Basit
    f"rtsp://{USER}:{PASS}@{IP}:8554/stream1",                    # Port 8554
]

print("Kamera bağlantısı test ediliyor...\n")

for url in candidates:
    masked = url.replace(PASS, "****")
    print(f"Deneniyor: {masked}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"\n BASARILI: {masked}")
            print(f" Cozunurluk: {int(cap.get(3))}x{int(cap.get(4))}")
            print(f" FPS: {cap.get(5):.1f}")
            # Bir kare kaydet
            cv2.imwrite("camera_snapshot.jpg", frame)
            print(" Goruntu camera_snapshot.jpg olarak kaydedildi")
            cap.release()
            # Çalışan URL'yi kaydet
            with open("working_rtsp.txt", "w") as f:
                f.write(url)
            sys.exit(0)
        cap.release()
    print(f"  Basarisiz")

print("\n Hicbir URL calismadi.")
print("Kameranin markasini veya web arayuzunden RTSP URL'sini kontrol edin.")
