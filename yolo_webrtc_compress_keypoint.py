import asyncio
import time
import cv2
import numpy as np
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCRtpCodecCapability
)
from av import VideoFrame
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# --- Model & camera setup ---
model = YOLO("best.pt")
cap = cv2.VideoCapture("/dev/video0")
if not cap.isOpened():
    print("❌ Cannot open /dev/video0")

pcs = set()
latest_frame = None
stop_flag = False
executor = ThreadPoolExecutor(max_workers=1)

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
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names.get(cls, str(cls))

        # 绘制检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)  # 白点中心

        # 如果存在关键点数据
        if kpts is not None and i < len(kpts.xy):
            for kp in kpts.xy[i]:
                px, py = int(kp[0]), int(kp[1])
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)  # 红点

                # 连线关键点与四角
                for (x, y) in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                    cv2.line(frame, (px, py), (x, y), (100, 0, 255), 1)

                # 偏差百分比计算
                w, h = x2 - x1, y2 - y1
                if w > 0 and h > 0:
                    dx = px - cx
                    dy = py - cy
                    dx_percent = dx / (w / 2) * 100
                    dy_percent = dy / (h / 2) * 100
                    text = f"{label} dx={dx_percent:+.1f}%, dy={dy_percent:+.1f}%"
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame



# --- Video track with adaptive bitrate ---
class AdaptiveStreamTrack(VideoStreamTrack):
    def __init__(self, mode="raw"):
        super().__init__()
        self.mode = mode
        self.last = time.time()
        self.count = 0
        self.fps = 0.0
        self.last_yolo_result = None
        self.last_infer_time = 0.05  # ~20 FPS baseline

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
                # Run YOLO every frame
                self.last_yolo_result = await run_yolo_async(frame)
                self.last_infer_time = time.time() - t0

                # ✅ 绘制检测框 + 关键点偏差（替代原来的 .plot()）
                frame = draw_results(frame, self.last_yolo_result)

                cv2.putText(frame, "YOLOv8 Keypoint Deviation", (10, 70),
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

        # Convert to VideoFrame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vf = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        vf.pts, vf.time_base = pts, time_base
        return vf


# --- Adaptive bitrate task ---
async def bitrate_controller(pc, track):
    """Dynamically adjust sender bitrate based on YOLO inference time."""
    print(f"🧠 Starting adaptive bitrate control for {track.mode} stream...")
    while True:
        try:
            senders = pc.getSenders()
            if not senders:
                print("⚠️ No active senders, stopping controller.")
                break

            sender = senders[0]
            params = getattr(sender, "parameters", None)
            if not params or not hasattr(params, "encodings"):
                await asyncio.sleep(2.0)
                continue

            fps_est = 1.0 / max(track.last_infer_time, 0.05)
            target_bitrate = int(min(2_000_000, max(300_000, fps_est * 100_000)))

            for enc in params.encodings:
                enc.maxBitrate = target_bitrate

            try:
                await sender.setParameters(params)
            except Exception:
                pass

            print(f"📶 Adaptive bitrate set: {target_bitrate/1000:.0f} kbps (fps≈{fps_est:.1f})")
            await asyncio.sleep(3.0)

        except asyncio.CancelledError:
            print("🛑 Bitrate control cancelled.")
            break
        except Exception as e:
            print(f"⚠️ Bitrate controller stopped safely: {e}")
            break


# --- Offer handler (shared) ---
async def offer_generic(request, mode="raw"):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    transceiver = pc.addTransceiver("video", direction="sendonly")
    track = AdaptiveStreamTrack(mode)
    pc.addTrack(track)

    # Prefer H.264
    try:
        for codec in ("H264", "VP8"):
            if codec in str(pc):
                print(f"🎥 Likely using {codec}")
                break
    except Exception:
        pass

    # Set remote / create answer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Start adaptive bitrate loop (only for YOLO)
    if mode == "yolo":
        asyncio.create_task(bitrate_controller(pc, track))

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def offer_raw(request): return await offer_generic(request, "raw")
async def offer_yolo(request): return await offer_generic(request, "yolo")


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
app.router.add_static("/", path=".", show_index=True)
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    print("🌐 YOLOv8 WebRTC (Dynamic Bitrate + Keypoint Deviation) http://0.0.0.0:8080")
    web.run_app(app, host="0.0.0.0", port=8080)
