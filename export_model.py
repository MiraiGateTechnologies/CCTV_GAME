from ultralytics import YOLO

model = YOLO("yolo11x.pt").to("cuda")  # move model to GPU

model.export(
    format="engine",   # TensorRT engine (GPU-only)
    imgsz=1248,
    half=True,         # FP16 for GPU acceleration
    # batch=4,
    # dynamic=True,
    workspace=8,       # GB
    device=0           # explicitly GPU
)

# yolo export model=yolo11x.pt format=engine imgsz=1280 half=True device=0 workspace=4