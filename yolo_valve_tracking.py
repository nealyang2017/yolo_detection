import asyncio
import time
import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import json
import os
import argparse

# =========================================================
# Configuration
# =========================================================
YOLO_PERIOD = 0.1                 # <= 10 Hz
REF_H_RATIO = 0.28                # relative image height for z
CENTER_FUSE_ALPHA = 0.7
ATT_LPF_ALPHA = 0.25              # pitch/yaw low-pass
Z_LPF_ALPHA = 0.3                 # z low-pass
EPS = 1e-6

SHM_PATH = "/dev/shm/yolo_result"

# =========================================================
# Global states
# =========================================================
latest_frame = None
stop_flag = False
pcs = set()

last_pitch_error = 0.0
last_yaw_error = 0.0
last_z_error = 0.0

latest_offset = {
    "pos_dx": 0.0,
    "pos_dy": 0.0,
    "att_dx": 0.0,
    "att_dy": 0.0,
    "z_error": 0.0,
    "pitch_error": 0.0,
    "yaw_error": 0.0,
}

# =========================================================
# Model & Camera
# =========================================================
model = YOLO("./models/valve.pt")
cap = cv2.VideoCapture("/dev/video0")
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

executor = ThreadPoolExecutor(max_workers=1)

# =========================================================
# Utilities
# =========================================================
def write_to_shm(data):
    try:
        tmp = SHM_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, SHM_PATH)
    except Exception as e:
        print("SHM write failed:", e)

def lpf(prev, curr, alpha):
    return (1.0 - alpha) * prev + alpha * curr

# =========================================================
# Capture Loop
# =========================================================
async def capture_loop():
    global latest_frame
    while not stop_flag:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame
        await asyncio.sleep(0.01)

# =========================================================
# YOLO async
# =========================================================
async def run_yolo_async(frame):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: model(frame, verbose=False))

# =========================================================
# Keypoint role assignment (order-agnostic)
# =========================================================
def assign_5kpts_roles(kpts_xy, bbox_center):
    kpts = np.asarray(kpts_xy, dtype=np.float32)
    bc = np.asarray(bbox_center, dtype=np.float32)

    d2 = np.sum((kpts - bc) ** 2, axis=1)
    idx_c = int(np.argmin(d2))

    rem = [i for i in range(5) if i != idx_c]
    rem_pts = kpts[rem]

    idx_u = rem[int(np.argmin(rem_pts[:, 1]))]
    idx_d = rem[int(np.argmax(rem_pts[:, 1]))]
    idx_l = rem[int(np.argmin(rem_pts[:, 0]))]
    idx_r = rem[int(np.argmax(rem_pts[:, 0]))]

    return {
        "C": kpts[idx_c],
        "U": kpts[idx_u],
        "D": kpts[idx_d],
        "L": kpts[idx_l],
        "R": kpts[idx_r],
    }

# =========================================================
# Pitch/Yaw estimation
# =========================================================
def estimate_pitch_yaw(C, U, D, L, R, bbox_center):
    C = CENTER_FUSE_ALPHA * C + (1.0 - CENTER_FUSE_ALPHA) * bbox_center

    lu = np.linalg.norm(U - C)
    ld = np.linalg.norm(D - C)
    ll = np.linalg.norm(L - C)
    lr = np.linalg.norm(R - C)

    pitch = (lu - ld) / (lu + ld + EPS)
    yaw   = (lr - ll) / (lr + ll + EPS)
    return pitch, yaw

# =========================================================
# HUD
# =========================================================
def draw_hud(frame, x, y, z, pitch, yaw, fps):
    x0, y0, dy = 10, 30, 28
    lines = [
        "VISION STATUS",
        f"x: {x:+.2f} %",
        f"y: {y:+.2f} %",
        f"z: {z:+.3f}",
        f"pitch: {pitch:+.3f}",
        f"yaw: {yaw:+.3f}",
        f"YOLO: {fps:.1f} Hz",
    ]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (x0, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 if i else 0.8,
                    (0, 255, 0), 2)

# =========================================================
# Main processing
# =========================================================
def process_and_draw(frame, results):
    global last_pitch_error, last_yaw_error, last_z_error, latest_offset

    h, w = frame.shape[:2]
    img_cx, img_cy = w // 2, h // 2

    boxes = results[0].boxes
    kpts_obj = getattr(results[0], "keypoints", None)

    if boxes is None or len(boxes) == 0:
        return frame

    best_i = -1
    best_d2 = 1e18

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        d2 = (cx - img_cx) ** 2 + (cy - img_cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
            best_box = (x1, y1, x2, y2)

    x1, y1, x2, y2 = best_box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    pos_dx = (cx - img_cx) / (w / 2) * 100
    pos_dy = (cy - img_cy) / (h / 2) * 100

    h_box = max(1, y2 - y1)
    ref_h = REF_H_RATIO * h
    z_raw = (ref_h - h_box) / ref_h
    z_raw = np.clip(z_raw, -1.0, 1.0)
    z_error = lpf(last_z_error, z_raw, Z_LPF_ALPHA)
    last_z_error = z_error

    pitch_error = last_pitch_error
    yaw_error = last_yaw_error

    if kpts_obj is not None:
        kdata = results[0].keypoints.data[best_i].cpu().numpy()
        if kdata.shape[0] >= 5 and np.all(kdata[:5, 2] >= 2):
            kxy = kdata[:5, :2]
            bbox_center = np.array([cx, cy], dtype=np.float32)
            roles = assign_5kpts_roles(kxy, bbox_center)

            if roles["U"][1] < roles["D"][1] and roles["L"][0] < roles["R"][0]:
                raw_pitch, raw_yaw = estimate_pitch_yaw(
                    roles["C"], roles["U"], roles["D"],
                    roles["L"], roles["R"],
                    bbox_center
                )
                pitch_error = lpf(last_pitch_error, raw_pitch, ATT_LPF_ALPHA)
                yaw_error   = lpf(last_yaw_error, raw_yaw, ATT_LPF_ALPHA)
                last_pitch_error = pitch_error
                last_yaw_error   = yaw_error

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

    shm = {
        "timestamp": time.time(),
        "x_error": float(pos_dx),
        "y_error": float(pos_dy),
        "z_error": float(z_error),
        "roll_error": 0.0,
        "pitch_error": float(pitch_error),
        "yaw_error": float(yaw_error),
    }
    write_to_shm(shm)

    latest_offset.update({
        "pos_dx": pos_dx,
        "pos_dy": pos_dy,
        "z_error": z_error,
        "pitch_error": pitch_error,
        "yaw_error": yaw_error,
    })

    return frame

# =========================================================
# WebRTC Track
# =========================================================
class YoloStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.last_infer = 0.0
        self.last_results = None
        self.last_fps_time = time.time()
        self.frames = 0
        self.fps = 0.0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = latest_frame.copy() if latest_frame is not None else np.zeros((480, 640, 3), np.uint8)

        now = time.time()
        if self.last_results is None or (now - self.last_infer) >= YOLO_PERIOD:
            self.last_results = await run_yolo_async(frame)
            self.last_infer = now

        frame = process_and_draw(frame, self.last_results)

        self.frames += 1
        if now - self.last_fps_time > 1.0:
            self.fps = self.frames / (now - self.last_fps_time)
            self.frames = 0
            self.last_fps_time = now

        draw_hud(
            frame,
            latest_offset["pos_dx"],
            latest_offset["pos_dy"],
            latest_offset["z_error"],
            latest_offset["pitch_error"],
            latest_offset["yaw_error"],
            self.fps,
        )

        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        vf = VideoFrame.from_ndarray(frame, format="rgb24")
        vf.pts, vf.time_base = pts, time_base
        return vf

# =========================================================
# Web Server
# =========================================================
async def offer_yolo(request):
    params = await request.json()
    offer = RTCSessionDescription(**params)
    pc = RTCPeerConnection()
    pcs.add(pc)

    pc.addTransceiver("video", direction="sendonly")
    pc.addTrack(YoloStreamTrack())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def on_startup(app):
    app["cap_task"] = asyncio.create_task(capture_loop())

async def on_shutdown(app):
    global stop_flag
    stop_flag = True
    app["cap_task"].cancel()
    for pc in pcs:
        await pc.close()
    cap.release()
    executor.shutdown(wait=False)

# =========================================================
# YOLO-only Mode
# =========================================================
async def yolo_only_loop():
    global stop_flag
    stop_flag = False
    last_infer = 0.0
    last_results = None
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue
        now = time.time()
        if last_results is None or (now - last_infer) >= YOLO_PERIOD:
            last_results = await run_yolo_async(frame)
            last_infer = now
        process_and_draw(frame, last_results)
        await asyncio.sleep(0.005)

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-webrtc", action="store_true")
    args = parser.parse_args()

    if args.no_webrtc:
        asyncio.run(yolo_only_loop())
    else:
        app = web.Application()
        app.router.add_post("/offer_yolo", offer_yolo)
        app.on_startup.append(on_startup)
        app.on_shutdown.append(on_shutdown)
        print("YOLO docking server running on :8080 (/offer_yolo)")
        web.run_app(app, host="0.0.0.0", port=8080)
