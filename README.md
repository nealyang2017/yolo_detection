# yolo_detection



## Run docker
sudo docker run -it --rm   --network=host   --device=/dev/video0:/dev/video0   -v ~/lab_data:/workspace   --gpus=all   --shm-size=8g -v /dev/shm:/dev/shm  usb_app_image   bash


## Unzip dataset

sudo unzip slider.v1i.yolov8.zip -d sliderv1i



# Train the model with given dataset
yolo pose train \
  model=yolov8n-pose.pt \
  data=/workspace/datasets/sliderv1i/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=8 \
  device=0 \
  workers=0 \
  val=False \
  name=train_gpu_fast \
  augment=True \
  mosaic=1.0 \
  mixup=0.1 \
  project=/workspace/runs/pose