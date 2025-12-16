import argparse
import asyncio
import time
import fractions

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from ultralytics import YOLO


# =========================================================
# YOLO model
# =========================================================
model = YOLO("./models/valve_fivepoints.pt")


# =========================================================
# HTML (极简)
# =========================================================
INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>YOLO Annotated WebRTC</title>
</head>
<body>
  <button id="start">Start</button><br><br>
  <video id="video" autoplay playsinline muted style="width:80vw;background:black"></video>
  <script>
    document.getElementById("start").onclick = async () => {
      const pc = new RTCPeerConnection();
      pc.ontrack = e => {
        const v = document.getElementById("video");
        v.srcObject = e.streams[0];
        v.play();
      };
      pc.addTransceiver("video", { direction: "recvonly" });
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const r = await fetch("/offer", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(pc.localDescription)
      });
      const ans = await r.json();
      await pc.setRemoteDescription(ans);
    };
  </script>
</body>
</html>
"""


# =========================================================
# Video Track: MP4 → YOLO → annotation → WebRTC
# =========================================================
class AnnotatedMP4Track(VideoStreamTrack):
    def __init__(self, mp4_path, fps=30):
        super().__init__()

        self.cap = cv2.VideoCapture(mp4_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {mp4_path}")

        self.fps = fps
        self.interval = 1.0 / fps
        self.time_base = fractions.Fraction(1, fps)
        self.pts = 0
        self.last_time = time.monotonic()

    async def recv(self):
        # ---------- 实时节拍 ----------
        now = time.monotonic()
        dt = now - self.last_time
        if dt < self.interval:
            await asyncio.sleep(self.interval - dt)
        self.last_time = time.monotonic()

        # ---------- 读一帧 ----------
        ret, frame = self.cap.read()
        if not ret:
            # 循环播放
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        # ---------- YOLO 推理 ----------
        results = model(frame, verbose=False)

        # ---------- 画标注 ----------
        annotated = results[0].plot()  # ultralytics 内置可视化

        # ---------- WebRTC 封装 ----------
        vf = VideoFrame.from_ndarray(annotated, format="bgr24")
        vf.pts = self.pts
        vf.time_base = self.time_base
        self.pts += 1
        return vf


# =========================================================
# WebRTC server
# =========================================================
pcs = set()

async def index(request):
    return web.Response(content_type="text/html", text=INDEX_HTML)

async def offer(request):
    params = await request.json()
    pc = RTCPeerConnection()
    pcs.add(pc)

    track = AnnotatedMP4Track(request.app["mp4_path"], fps=30)
    pc.addTrack(track)

    await pc.setRemoteDescription(
        RTCSessionDescription(params["sdp"], params["type"])
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp4", required=True, help="path to video file")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    app = web.Application()
    app["mp4_path"] = args.mp4
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    print(f"🌐 Open http://localhost:{args.port}")
    web.run_app(app, port=args.port)


if __name__ == "__main__":
    main()
