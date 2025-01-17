import RPi.GPIO as GPIO
import time

# Set mode GPIO (BCM atau BOARD)
GPIO.setmode(GPIO.BOARD)  # Menggunakan nomor pin fisik

# Setup pin 37 sebagai output
led_pin = 37
GPIO.setup(led_pin, GPIO.OUT)

try:
    # Nyalakan LED
    GPIO.output(led_pin, GPIO.HIGH)
    print("LED menyala")
    
    # Biarkan LED menyala selamanya
    # Anda bisa mengganti ini dengan loop atau kondisi lain
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    # Jika program dihentikan dengan Ctrl+C
    print("\nProgram dihentikan")

finally:
    # Bersihkan GPIO
    GPIO.cleanup()
    print("GPIO dibersihkan")