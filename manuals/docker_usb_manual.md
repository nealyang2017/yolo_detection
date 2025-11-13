# 🐳 USB 上运行 Docker 的完整配置指南

本指南介绍如何将 Docker 的数据目录迁移到 USB、构建镜像并运行容器。  
适用于 Ubuntu 系统，可在支持 GPU 的环境（如 YOLO、ROS、Isaac Sim 等）中直接使用。

---

## 🧩 1️⃣ 准备 USB

1. 插入 USB，查看分区号：
   ```bash
   lsblk -f
   ```
   输出示例：
   ```
   sdd1  vfat  FAT32 FUGHA3
   ```
   → 确认 USB 分区号为 `/dev/sdd1`。

2. 格式化为 ext4（⚠️ 会清空数据）：
   ```bash
   sudo umount /dev/sdd1
   sudo mkfs.ext4 /dev/sdd1
   ```

3. 挂载 USB：
   ```bash
   sudo mkdir -p /media/usb
   sudo mount /dev/sdd1 /media/usb
   ```

---

## ⚙️ 2️⃣ 修改 Docker 数据路径到 USB

1. 创建 Docker 数据目录：
   ```bash
   sudo mkdir -p /media/usb/docker-data
   sudo chown -R root:root /media/usb/docker-data
   ```

2. 编辑 Docker 配置文件：
   ```bash
   sudo nano /etc/docker/daemon.json
   ```

3. 写入以下内容：
   ```json
   {
     "data-root": "/media/usb/docker-data"
   }
   ```

4. 重新加载并启动 Docker：
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart docker
   sudo systemctl status docker
   ```

5. 验证是否成功：
   ```bash
   sudo docker info | grep "Docker Root Dir"
   ```
   输出应为：
   ```
   Docker Root Dir: /media/usb/docker-data
   ```

---

## 🔁 3️⃣ 开机自动挂载 USB

1. 获取 USB UUID：
   ```bash
   sudo blkid /dev/sdd1
   ```
   示例输出：
   ```
   /dev/sdd1: UUID="b42e1aae-50a5-4369-90bd-e12b8e7b14c5" TYPE="ext4"
   ```

2. 编辑 `/etc/fstab`：
   ```bash
   sudo nano /etc/fstab
   ```

3. 在末尾添加：
   ```
   UUID=b42e1aae-50a5-4369-90bd-e12b8e7b14c5  /media/usb  ext4  defaults  0  2
   ```

4. 测试是否生效：
   ```bash
   sudo mount -a
   ```

无报错即配置成功。之后系统启动时会自动挂载 `/media/usb`，  
Docker 会自动加载 USB 上的镜像与容器数据。

---

## 🧱 4️⃣ 在 USB 上构建镜像

确保 `Dockerfile` 存在于 `/media/usb`：
```bash
cd /media/usb
sudo docker build -t usb_app_image .
```

查看是否构建成功：
```bash
sudo docker images
```

示例输出：
```
REPOSITORY        TAG       IMAGE ID       CREATED          SIZE
usb_app_image     latest    2d51a6bcabf3   1 minute ago     120MB
```

---

## 🚀 5️⃣ 运行容器（带 GPU 与摄像头）

执行以下命令运行容器：
```bash
cd /media/usb

sudo docker run -it --rm   --network=host   --device=/dev/video0:/dev/video0   -v /media/usb:/workspace   --gpus=all   --shm-size=8g   usb_app_image   bash
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--network=host` | 与主机共享网络，方便 ROS/YOLO 通信 |
| `--device=/dev/video0` | 将主机摄像头映射进容器 |
| `-v /media/usb:/workspace` | 挂载 USB 到容器内 `/workspace` |
| `--gpus=all` | 让容器访问所有 GPU |
| `--shm-size=8g` | 增大共享内存，避免 OpenCV/Torch 报错 |
| `usb_app_image` | 刚刚构建的镜像名 |
| `bash` | 启动交互终端 |

---

## 🔒 6️⃣ 安全移除 USB

拔出 USB 前务必先停止 Docker：
```bash
sudo systemctl stop docker
sudo umount /media/usb
```

> ⚠️ 不要在 Docker 运行时拔出 USB，否则可能导致 `/media/usb/docker-data` 损坏。

---

## ✅ 总结

| 步骤 | 操作 | 目的 |
|------|------|------|
| ① | 格式化 USB 为 ext4 | 支持 Docker 存储层 |
| ② | 修改 `/etc/docker/daemon.json` | 将 Docker 数据迁移至 USB |
| ③ | 配置 `/etc/fstab` | 实现开机自动挂载 |
| ④ | 构建镜像 | `sudo docker build -t usb_app_image .` |
| ⑤ | 运行容器 | 使用 GPU / 摄像头 / 本地挂载 |
| ⑥ | 安全移除 | 停止 Docker 并卸载 USB |

---

## 💡 附：验证命令速查

```bash
# 查看 Docker 数据路径
sudo docker info | grep "Docker Root Dir"

# 查看镜像
sudo docker images

# 查看运行中的容器
sudo docker ps -a

# 查看挂载状态
df -h | grep usb
```

---

🟢 **现在你的 Docker 已完全运行在 USB 上**，  
所有镜像、容器、日志和配置都会自动保存在 `/media/usb/docker-data` 中。
