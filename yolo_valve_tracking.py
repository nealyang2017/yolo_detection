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
import argparse  # ✅ 新增：命令行参数



# --- Model & camera setup ---
model = YOLO("./yolo_detection/models/valve_fivepoints.pt")  # 你的Pose模型
cap = cv2.VideoCapture("/dev/video0")
if not cap.isOpened():
    print("❌ Cannot open /dev/video0")

# ===============================
# 🔍 模型 & keypoints 自检（只跑一次）
# ===============================
print("MODEL PATH:", model.ckpt_path if hasattr(model, "ckpt_path") else "unknown")
print("MODEL NAMES:", model.names)

ret, _f = cap.read()
if ret:
    _r = model(_f, verbose=False)
    _k = getattr(_r[0], "keypoints", None)
    if _k is None:
        print("DEBUG: keypoints is None (not a pose model?)")
    else:
        print("DEBUG kpts.xy shape:", _k.xy.shape if hasattr(_k, "xy") else None)
        print("DEBUG kpts.data shape:", _k.data.shape if hasattr(_k, "data") else None)
else:
    print("⚠️ Could not read a frame for keypoint check")

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


# --- Shared frame capture loop (for WebRTC 模式用) ---
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
def draw_results(frame, results):
    """
    视觉检测 + 偏差计算：
      ✔ 绘制所有检测框（每个都显示 POS/ATT）
      ✔ 从所有框中选取“离图像中心最近”的目标
      ✔ 计算 pos_dx / pos_dy
      ✔ 计算 att_dx / att_dy（基于关键点）
      ✔ 用 bounding box 高度估计 z_error（距离误差）
      ✔ 写入 shared memory
    """
    global latest_offset

    boxes = results[0].boxes
    kpts = getattr(results[0], "keypoints", None)
    print("DEBUG keypoints:", kpts)

    h_img, w_img = frame.shape[:2]
    img_cx, img_cy = w_img // 2, h_img // 2

    # 用于记录最佳目标（最近）
    best_found = False
    best_dist2 = 1e18
    best_pos_dx = best_pos_dy = 0.0
    best_att_dx = best_att_dy = 0.0
    best_box = None

    # 如果没有检测结果
    if boxes is None or len(boxes) == 0:
        result_dict = {
            "timestamp": time.time(),
            "x_error": 0.0,
            "y_error": 0.0,
            "z_error": 0.0,
            "roll_error": 0.0,
            "pitch_error": 0.0,
            "yaw_error": 0.0,
        }
        latest_offset = {"pos_dx": 0, "pos_dy": 0, "att_dx": 0, "att_dy": 0}
        write_to_shm(result_dict)
        return frame

    # =============================
    #   扫描所有框，画图 + 找最近目标
    # =============================
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names.get(cls, str(cls))

        # 绘制框和中心
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        # --- Draw keypoints (for visualization)
        if kpts is not None and i < len(kpts.xy):
            for idx, (px, py) in enumerate(kpts.xy[i]):
                px, py = int(px), int(py)

                # Color-coded keypoints (for visualization)
                colors = [
                    (0, 0, 255),
                    (0, 255, 0),
                    (255, 0, 0),
                    (0, 255, 255),
                    (255, 0, 255),
                ]

                cv2.circle(frame, (px, py), 5, colors[idx % len(colors)], -1)

        # --- 位置偏差（百分比）
        pos_dx = (cx - img_cx) / (w_img / 2) * 100
        pos_dy = (cy - img_cy) / (h_img / 2) * 100

        # --- 姿态偏差（关键点）
        att_dx, att_dy = 0.0, 0.0
        if kpts is not None and i < len(kpts.xy):
            for kp in kpts.xy[i]:
                px, py = int(kp[0]), int(kp[1])
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
                cv2.line(frame, (cx, cy), (px, py), (200, 50, 255), 2)

                w_box = x2 - x1
                h_box = y2 - y1
                if w_box > 0 and h_box > 0:
                    att_dx = (px - cx) / (w_box / 2) * 100
                    att_dy = (py - cy) / (h_box / 2) * 100

        # 绘制文字
        text = (
            f"{label} POS({pos_dx:+.1f},{pos_dy:+.1f}) "
            f"ATT({att_dx:+.1f},{att_dy:+.1f})"
        )
        cv2.putText(
            frame,
            text,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

        # --- 计算与图像中心的距离
        dist2 = (cx - img_cx) ** 2 + (cy - img_cy) ** 2
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_found = True
            best_pos_dx = pos_dx
            best_pos_dy = pos_dy
            best_att_dx = att_dx
            best_att_dy = att_dy
            best_box = (x1, y1, x2, y2)

    # =============================
    #   最近目标 → 计算 z_error 并写入 shared memory
    # =============================
    if best_found and best_box is not None:
        x1, y1, x2, y2 = best_box
        h_box = y2 - y1

        # 🔵 你可以根据相机情况调整 ref_h（理想距离下的像素高度）
        ref_h = 200.0  # <== 想离近一点就调大，想远一点就调小

        # z_error = (期望 - 实际) / 期望
        z_error = (ref_h - h_box) / ref_h
        z_error = max(min(z_error, 1.0), -1.0)  # 限幅 [-1, 1]

        # 输出给共享内存
        latest_offset = {
            "pos_dx": best_pos_dx,
            "pos_dy": best_pos_dy,
            "att_dx": best_att_dx,
            "att_dy": best_att_dy,
            "z_error": z_error,
        }

        shm_dict = {
            "timestamp": time.time(),
            "x_error": best_pos_dx,
            "y_error": best_pos_dy,
            "z_error": z_error,
            "roll_error": best_att_dx,
            #"pitch_error": 0.0,
            "pitch_error": best_att_dy,
            #"yaw_error": 0.0,
            "yaw_error": 0.0,
        }
        write_to_shm(shm_dict)

        # 在图像上标亮最近目标
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"NEAREST z={z_error:+.2f}",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    return frame



# --- Video track (用于 WebRTC 模式) ---
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
            cv2.putText(
                frame,
                "No Camera Feed",
                (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        else:
            frame = latest_frame.copy()
            if self.mode == "yolo":
                t0 = time.time()
                self.last_yolo_result = await run_yolo_async(frame)
                self.last_infer_time = time.time() - t0
                frame = draw_results(frame, self.last_yolo_result)
                cv2.putText(
                    frame,
                    "YOLOv8 Docking Monitor",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 128, 0),
                    3,
                )

            frame = cv2.resize(frame, (1280, 720))
            self.count += 1
            now = time.time()
            if now - self.last >= 1.0:
                self.fps = self.count / (now - self.last)
                self.count = 0
                self.last = now
            cv2.putText(
                frame,
                f"{self.mode.upper()} FPS:{self.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if self.mode == "raw" else (255, 128, 0),
                2,
            )

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
    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


async def offer_raw(request):
    return await offer_generic(request, "raw")


async def offer_yolo(request):
    return await offer_generic(request, "yolo")


# --- Offset data endpoint ---
async def get_offset(request):
    return web.json_response(latest_offset)


# --- Startup / Shutdown (仅 WebRTC 模式用) ---
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


# --- App setup (WebRTC 模式用) ---
app = web.Application()
app.router.add_post("/offer_raw", offer_raw)
app.router.add_post("/offer_yolo", offer_yolo)
app.router.add_get("/offset", get_offset)
app.router.add_static("/", path=".", show_index=True)
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)


# --- YOLO-only 模式：只推理 + 写共享内存，不启动 WebRTC ---
async def yolo_only_loop():
    global stop_flag
    print("🚀 YOLO-only shared memory mode started (no WebRTC)")
    try:
        while not stop_flag:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            # 与 WebRTC 中的 YOLO 完全共用同一套逻辑
            results = await run_yolo_async(frame)
            _ = draw_results(frame, results)  # 不需要显示，只为了计算 + 写 shm

            # 适当睡一点，避免把 CPU 吃满
            await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        pass
    finally:
        print("🛑 YOLO-only loop stopped")


async def yolo_only_main():
    global stop_flag
    stop_flag = False
    task = asyncio.create_task(yolo_only_loop())
    try:
        await task
    finally:
        stop_flag = True
        cap.release()
        executor.shutdown(wait=False)
        print("✅ YOLO-only mode shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 + WebRTC docking monitor / shared memory server"
    )
    parser.add_argument(
        "--no-webrtc",
        action="store_true",
        help="Disable WebRTC server and only run YOLO inference + shared memory output",
    )
    args = parser.parse_args()

    if args.no_webrtc:
        # 纯推理 + 写共享内存模式
        try:
            asyncio.run(yolo_only_main())
        except KeyboardInterrupt:
            print("🧹 Interrupted by user (Ctrl+C)")
    else:
        # 原来的完整 WebRTC 服务器模式
        print(
            "🌐 YOLOv8 Docking Visualization Server running at http://0.0.0.0:8080"
        )
        web.run_app(app, host="0.0.0.0", port=8080)
