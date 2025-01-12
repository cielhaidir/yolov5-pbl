import detect
import time
import cv2
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import torch
import numpy as np

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
    
    # NMS dengan filter kelas kendaraan
    pred = non_max_suppression(
        pred, 
        conf_thres=0.25, 
        iou_thres=0.45, 
        max_det=1000
    )
    
    # Count objects
    total_objects = sum(len(det) for det in pred)
    
    return total_objects, pred[0]  # Mengembalikan total dan deteksi untuk visualisasi

def main():
    cap = cv2.VideoCapture(0)
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
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Simpan frame original untuk display
            original_frame = frame.copy()
            
            # Process frame
            frame_resized = cv2.resize(frame, imgsz)
            frame_rgb = frame_resized[..., ::-1].transpose((2, 0, 1))
            frame_rgb = np.ascontiguousarray(frame_rgb)
            
            # Get detections
            total_objects, detections = process_frame(frame_rgb, model, imgsz)
            
            # Draw detections pada frame
            if len(detections):
                for *xyxy, conf, cls in detections:
                    # Konversi koordinat ke int
                    x1, y1, x2, y2 = map(int, xyxy)
                    # Scaling koordinat ke frame original
                    h, w = original_frame.shape[:2]
                    x1 = int(x1 * w/imgsz[0])
                    x2 = int(x2 * w/imgsz[0])
                    y1 = int(y1 * h/imgsz[1])
                    y2 = int(y2 * h/imgsz[1])
                    
                    # Gambar box
                    cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Tambah label
                    label = f'Vehicle {conf:.2f}'
                    cv2.putText(original_frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Tambahkan text informasi
            cv2.putText(original_frame, f'Total Vehicles: {total_objects}', 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Vehicle Detection', original_frame)
            
            # Process history dan output seperti sebelumnya
            current_time = time.time()
            vehicle_count_history.append((current_time, total_objects))
            vehicle_count_history = [x for x in vehicle_count_history 
                                   if current_time - x[0] <= time_window]
            avg_vehicles = sum(count for _, count in vehicle_count_history) / len(vehicle_count_history)
            
            print(f"Jumlah kendaraan saat ini: {total_objects}, "
                  f"Rata-rata dalam {time_window} detik: {avg_vehicles:.1f}")
            
            if avg_vehicles > high_count_threshold and last_output_state == 0:
                print(f"Output: 1 (HIGH) - Rata-rata lebih dari {high_count_threshold} "
                      f"kendaraan dalam {time_window} detik")
                last_output_state = 1
                #GPIO.output(OUTPUT_PIN, GPIO.HIGH)
            elif avg_vehicles <= high_count_threshold and last_output_state == 1:
                print(f"Output: 0 (LOW) - Rata-rata {high_count_threshold} atau kurang "
                      f"kendaraan dalam {time_window} detik")
                last_output_state = 0
                #GPIO.output(OUTPUT_PIN, GPIO.LOW)
            
            # Break loop jika 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Program dihentikan")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()