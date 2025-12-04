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
model = YOLO("./models/valve.pt")  # 你的Pose模型
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


# --- 绘制检测框 + 关键点 + 偏差 ---
def draw_results(frame, results):
    """
    注意：这里既负责画图，也负责更新 latest_offset + 写入 /dev/shm
    所有模式（WebRTC/纯推理）共用这一套逻辑，保证行为一致。
    """
    global latest_offset
    boxes = results[0].boxes
    kpts = getattr(results[0], "keypoints", None)

    h_img, w_img = frame.shape[:2]
    img_cx, img_cy = w_img // 2, h_img // 2  # 图像中心

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names.get(cls, str(cls))

        # 框中心
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        # --- 位置偏差（框中心 vs 图像中心）
        pos_dx = (cx - img_cx) / (w_img / 2) * 100
        pos_dy = (cy - img_cy) / (h_img / 2) * 100

        att_dx, att_dy = 0.0, 0.0

        # --- 姿态偏差（关键点 vs 框中心）
        if kpts is not None and i < len(kpts.xy):
            for kp in kpts.xy[i]:
                px, py = int(kp[0]), int(kp[1])
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
                cv2.line(frame, (cx, cy), (px, py), (100, 0, 255), 2)
                w_box, h_box = x2 - x1, y2 - y1
                if w_box > 0 and h_box > 0:
                    att_dx = (px - cx) / (w_box / 2) * 100
                    att_dy = (py - cy) / (h_box / 2) * 100

        latest_offset = {
            "pos_dx": pos_dx,
            "pos_dy": pos_dy,
            "att_dx": att_dx,
            "att_dy": att_dy,
        }

        text = (
            f"{label} POS(dx={pos_dx:+.1f}%, dy={pos_dy:+.1f}%) | "
            f"ATT(dx={att_dx:+.1f}%, dy={att_dy:+.1f}%)"
        )
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

        # --- 暂不计算姿态，默认 0
        z_err, roll_err, pitch_err, yaw_err = 0.0, 0.0, 0.0, 0.0

        # --- 统一写入共享内存（仅写一次）---
        result_dict = {
            "timestamp": time.time(),
            "x_error": pos_dx,
            "y_error": pos_dy,
            "z_error": z_err,
            "roll_error": roll_err,
            "pitch_error": pitch_err,
            "yaw_error": yaw_err,
        }
        write_to_shm(result_dict)

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
