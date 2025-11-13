import asyncio
import time
import cv2
import numpy as np
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import json
import os
import tempfile


# --- Model & camera setup ---
model = YOLO("./runs/pose/train_gpu_fast/weights/best.pt")  # 你的Pose模型
cap = cv2.VideoCapture("/dev/video0")
if not cap.isOpened():
    print("❌ Cannot open /dev/video0")

pcs = set()
latest_frame = None
stop_flag = False
executor = ThreadPoolExecutor(max_workers=1)
latest_offset = {"pos_dx": 0.0, "pos_dy": 0.0, "att_dx": 0.0, "att_dy": 0.0}

# --- Shared memory file path ---
SHM_PATH = "/dev/shm/yolo_result"
# --- Helper: write result to shared memory ---
def write_to_shm(result_dict):
    try:
        data = json.dumps(result_dict)
        tmp_path = SHM_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(data)
        os.replace(tmp_path, SHM_PATH)  # 原子替换，防止读写冲突
    except Exception as e:
        print(f"⚠️ Shared memory write failed: {e}")




# --- Shared frame capture loop ---
async def capture_loop():
    global latest_frame
    while not stop_flag:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame.copy()
        await asyncio.sleep(0.01)

# --- Async YOLO inference ---
async def run_yolo_async(frame):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: model(frame, verbose=False))

# --- 绘制检测框 + 关键点 + 偏差 ---
def draw_results(frame, results):
    boxes = results[0].boxes
    kpts = getattr(results[0], "keypoints", None)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # --- 只画检测框 ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # --- 只画关键点（红色圆点）---
        if kpts is not None and i < len(kpts.xy):
            for kp in kpts.xy[i]:
                px, py = int(kp[0]), int(kp[1])
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

    return frame



# --- Video track ---
class AdaptiveStreamTrack(VideoStreamTrack):
    def __init__(self, mode="raw"):
        super().__init__()
        self.mode = mode
        self.last = time.time()
        self.count = 0
        self.fps = 0.0
        self.last_yolo_result = None
        self.last_infer_time = 0.05

    async def recv(self):
        global latest_frame
        pts, time_base = await self.next_timestamp()
        if latest_frame is None:
            frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(frame, "No Camera Feed", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            frame = latest_frame.copy()
            if self.mode == "yolo":
                t0 = time.time()
                self.last_yolo_result = await run_yolo_async(frame)
                self.last_infer_time = time.time() - t0
                frame = draw_results(frame, self.last_yolo_result)
                cv2.putText(frame, "YOLOv8 Docking Monitor", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 3)

            frame = cv2.resize(frame, (1280, 720))
            self.count += 1
            now = time.time()
            if now - self.last >= 1.0:
                self.fps = self.count / (now - self.last)
                self.count = 0
                self.last = now
            cv2.putText(frame, f"{self.mode.upper()} FPS:{self.fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if self.mode == "raw" else (255, 128, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vf = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        vf.pts, vf.time_base = pts, time_base
        return vf


# --- Offer handlers ---
async def offer_generic(request, mode="raw"):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)
    pc.addTransceiver("video", direction="sendonly")
    track = AdaptiveStreamTrack(mode)
    pc.addTrack(track)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def offer_raw(request): return await offer_generic(request, "raw")
async def offer_yolo(request): return await offer_generic(request, "yolo")

# --- Offset data endpoint ---
async def get_offset(request):
    return web.json_response(latest_offset)

# --- Startup / Shutdown ---
async def on_startup(app):
    print("🚀 Starting capture loop...")
    app["cap_task"] = asyncio.create_task(capture_loop())

async def on_shutdown(app):
    global stop_flag
    stop_flag = True
    app["cap_task"].cancel()
    for pc in list(pcs):
        await pc.close()
        pcs.discard(pc)
    cap.release()
    print("✅ Shutdown complete")

# --- App setup ---
app = web.Application()
app.router.add_post("/offer_raw", offer_raw)
app.router.add_post("/offer_yolo", offer_yolo)
app.router.add_get("/offset", get_offset)
app.router.add_static("/", path=".", show_index=True)
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    print("🌐 YOLOv8 Docking Visualization Server running at http://0.0.0.0:8080")
    web.run_app(app, host="0.0.0.0", port=8080)
