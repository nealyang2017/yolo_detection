import cv2
import json
import os
import time
from ultralytics import YOLO

# ----------------------------------------
# 配置
# ----------------------------------------
MODEL_PATH = "./models/valve.pt"
CAMERA = "/dev/video0"
SHM_PATH = "/dev/shm/yolo_result"

# ----------------------------------------
# 加载模型
# ----------------------------------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA)

if not cap.isOpened():
    print("❌ Cannot open camera:", CAMERA)
    exit(1)

print("🚀 Fast YOLO shared memory inference started.")


# ----------------------------------------
# 写入共享内存
# ----------------------------------------
def write_shm(data_dict):
    try:
        tmp_path = SHM_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(json.dumps(data_dict))
        os.replace(tmp_path, SHM_PATH)
    except Exception as e:
        print("⚠️ Write SHM error:", e)


# ----------------------------------------
# 主循环
# ----------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed")
        time.sleep(0.05)
        continue

    t0 = time.time()
    results = model(frame, verbose=False)
    infer_time = time.time() - t0

    # 默认误差值
    x_err = y_err = 0.0

    boxes = results[0].boxes

    if len(boxes) > 0:
        # 只取第一个目标（或可选最中心的）
        box = boxes[0]
        x1, y1, x2, y2 = box.xyxy[0]

        # 图像中心
        h, w = frame.shape[:2]
        img_cx, img_cy = w // 2, h // 2

        # 框中心
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 百分比偏差
        x_err = (cx - img_cx) / (w / 2) * 100
        y_err = (cy - img_cy) / (h / 2) * 100

    # 输出到共享内存（控制程序去读取）
    shm_data = {
        "timestamp": time.time(),
        "x_error": float(x_err),
        "y_error": float(y_err),
        "z_error": 0.0,
        "roll_error": 0.0,
        "pitch_error": 0.0,
        "yaw_error": 0.0,
        "infer_ms": infer_time * 1000.0
    }

    write_shm(shm_data)

    # 打印日志（可关闭）
    print(f"[YOLO] x={x_err:+.1f}%  y={y_err:+.1f}%  infer={infer_time*1000:.1f}ms")

    # 可根据需要调节循环速度
    # time.sleep(0.005)
