import detect
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
OUTPUT_PIN = 37  # Sesuaikan dengan pin yang ingin digunakan
GPIO.setup(OUTPUT_PIN, GPIO.OUT)

try:
    while True:
        # Jalankan deteksi dan dapatkan jumlah objek
        total_objects = detect.run(
            weights='yolov5s.pt',
            source=0,  # atau source=0 untuk webcam
            conf_thres=0.5
        )
        
        print(f"Jumlah objek terdeteksi: {total_objects}")
        
        # Logic untuk output
        if total_objects > 10:
            print("Output: 1 (HIGH) - Lebih dari 5 objek terdeteksi")
            # Untuk Raspberry Pi:
            GPIO.output(OUTPUT_PIN, GPIO.HIGH)
        else:
            print("Output: 0 (LOW) - 5 atau kurang objek terdeteksi")
            # Untuk Raspberry Pi:
            GPIO.output(OUTPUT_PIN, GPIO.LOW)
            
        time.sleep(1)  # Delay 1 detik

except KeyboardInterrupt:
    print("Program dihentikan")
    # Untuk Raspberry Pi:
    GPIO.cleanup()