from ultralytics import YOLO
import os 

# Load a model
model = YOLO("yolo11n.pt")  # Load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/ahmed/AIQ/Challenge1/yolo_format/data.yaml", 
                      epochs=100, 
                      imgsz=640, 
                      device='cuda')

# Export the trained model to ONNX format
onnx_path = "yolo11n.onnx"  # Specify the desired ONNX file name
model.export(format="onnx", dynamic=False, simplify=True, save_dir=os.getxwd())
