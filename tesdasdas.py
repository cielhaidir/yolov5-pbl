from flask import Flask, Response
import threading
import cv2
import numpy as np
import detect
import time
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import torch
import platform

# GPIO Setup with platform check
USE_GPIO = False
try:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    # GPIO.setmode(GPIO.BCM)
    GPIO.setmode(GPIO.BOARD)
    OUTPUT_PIN = 37
    GPIO.setup(OUTPUT_PIN, GPIO.OUT)
    USE_GPIO = True
    print("GPIO initialized successfully")
except Exception as e:
    print(f"Failed to initialize GPIO: {e}")
    USE_GPIO = False

# try:
#     import RPi.GPIO as GPIO
#     GPIO.setmode(GPIO.BCM)
#     OUTPUT_PIN = 37  # Sesuaikan dengan pin yang ingin digunakan
#     GPIO.setup(OUTPUT_PIN, GPIO.OUT)
# except ImportError:
#     GPIO = None
#     print("RPi.GPIO module not found. GPIO functionality will be disabled.")


app = Flask(__name__)

# Variable global untuk frame yang akan ditampilkan di web
global_frame = None
frame_lock = threading.Lock()

def initialize_model():
    # Inisialisasi model sekali
    # weights = 'yolov5s.pt'
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


def generate_frames():
    """Generator untuk streaming frame ke web"""
    while True:
        with frame_lock:
            if global_frame is not None:
                # Encode frame ke jpg
                ret, buffer = cv2.imencode('.jpg', global_frame)
                if not ret:
                    continue
                # Convert ke bytes dan yield
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)

@app.route('/')
def index():
    """Halaman web sederhana dengan tampilan video"""
    return """
    <html>
    <head>
        <title>Vehicle Detection</title>
        <style>
            body { text-align: center; background: #f0f0f0; }
            img { max-width: 100%; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Vehicle Detection Stream</h1>
        <img src="/video_feed">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Route untuk stream video"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def process_frame(frame, model, imgsz):
    # Preprocessing
    im = torch.from_numpy(frame).to(model.device)
    im = im.half() if model.fp16 else im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]
    
    # Inference
    pred = model(im, augment=False)
    
    # NMS dengan filter kelas kendaraan
    pred = non_max_suppression(
        pred, 
        conf_thres=0.1,  # Confidence threshold
        iou_thres=0.1,   # IoU threshold
        # classes=[2, 3, 5, 7],  # Hanya kendaraan
        max_det=1000
    )
    
    total_objects = sum(len(det) for det in pred)
    return total_objects, pred[0]

# Dalam fungsi detection_loop, modifikasi bagian GPIO:
def detection_loop():
    """Function untuk menjalankan deteksi"""
    global global_frame
    
    device_path = "/dev/video0"
    cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    model, imgsz, stride = initialize_model()
    
    try:
        vehicle_count_history = []
        time_window = 5
        high_count_threshold = 7
        last_output_state = 0
        
        while True:
            # ... (kode lain tetap sama) ...
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Process frame
            frame_resized = cv2.resize(frame, imgsz)
            frame_rgb = frame_resized[..., ::-1].transpose((2, 0, 1))
            frame_rgb = np.ascontiguousarray(frame_rgb)
            
            # Get detections
            total_objects, detections = process_frame(frame_rgb, model, imgsz)
            
            # Draw detections
            display_frame = frame.copy()
            if len(detections):
                for *xyxy, conf, cls in detections:
                    x1, y1, x2, y2 = map(int, xyxy)
                    # Scale koordinat
                    h, w = frame.shape[:2]
                    x1 = int(x1 * w/imgsz[0])
                    x2 = int(x2 * w/imgsz[0])
                    y1 = int(y1 * h/imgsz[1])
                    y2 = int(y2 * h/imgsz[1])
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Tambah label
                    label = f'Object {conf:.2f}'
                    cv2.putText(display_frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add counter text
            cv2.putText(display_frame, f'Vehicles: {total_objects}', 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Update global frame
            with frame_lock:
                global_frame = display_frame
            
            # Process history dan output
            current_time = time.time()
            vehicle_count_history.append((current_time, total_objects))
            vehicle_count_history = [x for x in vehicle_count_history 
            if current_time - x[0] <= time_window]
            avg_vehicles = sum(count for _, count in vehicle_count_history) / len(vehicle_count_history)            
            if avg_vehicles > high_count_threshold and last_output_state == 0:
                print("Output: HIGH")
                last_output_state = 1
                if USE_GPIO:
                    try:
                        GPIO.output(37, GPIO.HIGH)
                    except Exception as e:
                        print(f"GPIO Error (HIGH): {e}")
                else:
                    print("GPIO not available")
            elif avg_vehicles <= high_count_threshold and last_output_state == 1:
                print("Output: LOW")
                last_output_state = 0
                if USE_GPIO:
                    try:
                        GPIO.output(OUTPUT_PIN, GPIO.LOW)
                    except Exception as e:
                        print(f"GPIO Error (LOW): {e}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Program dihentikan")
    finally:
        cap.release()
        if USE_GPIO:
            try:
                GPIO.cleanup()
            except Exception as e:
                print(f"GPIO Cleanup Error: {e}")

def main():
    # Start detection thread
    detection_thread = threading.Thread(target=detection_loop)
    detection_thread.daemon = True
    detection_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5050)

if __name__ == "__main__":
    main()