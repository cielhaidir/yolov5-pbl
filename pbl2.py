import detect
import time
import cv2
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import torch
import numpy as np
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BCM)
OUTPUT_PIN = 37  # Sesuaikan dengan pin yang ingin digunakan
GPIO.setup(OUTPUT_PIN, GPIO.OUT)

# Setup untuk virtual display
# import os
# os.environ['DISPLAY'] = ':99'
# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(800, 600))
# display.start()

def initialize_model():
    # Inisialisasi model sekali
    weights = 'yolov5n.pt'
    device = select_device('')  # Gunakan select_device untuk inisialisasi device
    imgsz = (500, 500)
    
    # Load model
    model = DetectMultiBackend(weights, device=device)
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)
    
    # Warmup
    model.warmup(imgsz=(1, 3, *imgsz))
    
    return model, imgsz, stride

def process_frame(frame, model, imgsz):
    # Preprocessing
    im = torch.from_numpy(frame).to(model.device)
    im = im.half() if model.fp16 else im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]
    
    # Inference
    pred = model(im, augment=False)
    
    # NMS
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
    
    # Count objects
    total_objects = sum(len(det) for det in pred)
    
    return total_objects

def main():
    # Initialize camera
    # device_path = "/dev/video0"
    # cap = cv2.VideoCapture(device_path)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Initialize model
    model, imgsz, stride = initialize_model()
    
    try:
        vehicle_count_history = []  # List untuk menyimpan riwayat jumlah kendaraan
        time_window = 5  # Window waktu dalam detik
        high_count_threshold = 7  # Threshold jumlah kendaraan
        last_output_state = 0  # Menyimpan state output terakhir
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Process frame
            frame = cv2.resize(frame, imgsz)
            frame = frame[..., ::-1].transpose((2, 0, 1))
            frame = np.ascontiguousarray(frame)
            
            # Get detections
            total_objects = process_frame(frame, model, imgsz)
            
            # Tambahkan deteksi saat ini ke history dengan timestamp
            current_time = time.time()
            vehicle_count_history.append((current_time, total_objects))
            
            # Hapus data yang lebih lama dari time window
            vehicle_count_history = [x for x in vehicle_count_history 
                                if current_time - x[0] <= time_window]
            
            # Hitung rata-rata kendaraan dalam time window
            avg_vehicles = sum(count for _, count in vehicle_count_history) / len(vehicle_count_history)
            
            print(f"Jumlah kendaraan saat ini: {total_objects}, "
                f"Rata-rata dalam {time_window} detik: {avg_vehicles:.1f}")
            
            # Logic untuk output
            if avg_vehicles > high_count_threshold and last_output_state == 0:
                print(f"Output: 1 (HIGH) - Rata-rata lebih dari {high_count_threshold} "
                    f"kendaraan dalam {time_window} detik")
                last_output_state = 1
                # Untuk Raspberry Pi:
                GPIO.output(OUTPUT_PIN, GPIO.HIGH)
            elif avg_vehicles <= high_count_threshold and last_output_state == 1:
                print(f"Output: 0 (LOW) - Rata-rata {high_count_threshold} atau kurang "
                    f"kendaraan dalam {time_window} detik")
                last_output_state = 0
                # Untuk Raspberry Pi:
                GPIO.output(OUTPUT_PIN, GPIO.LOW)
                
            time.sleep(0.1)  # Reduced delay for better responsiveness

    except KeyboardInterrupt:
        print("Program dihentikan")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        # display.stop()

if __name__ == "__main__":
    main()