1️⃣ 挂载新的 ext4 分区
sudo mkdir -p /media/usb
sudo mount /dev/sdd1 /media/usb

2️⃣ 确认挂载成功
df -h | grep /media/usb


你应该能看到类似：

/dev/sdd1  235G  60M  235G  1% /media/usb

3️⃣ 创建 Docker 数据目录
sudo mkdir -p /media/usb/docker-data
sudo chown -R root:root /media/usb/docker-data

4️⃣ 编辑 Docker 配置

打开：

sudo nano /etc/docker/daemon.json


写入：

{
  "data-root": "/media/usb/docker-data"
}


保存并退出（Ctrl+O, 回车, Ctrl+X）。

5️⃣ 重新加载并启动 Docker
sudo systemctl daemon-reload
sudo systemctl restart docker
sudo systemctl status docker


如果看到：

Active: active (running)





# 🧩 Docker USB 自动挂载与安全移除指南

## 1️⃣ 开机自动挂载

如果希望下次插上 USB 或重启后，Docker 能自动识别（无需手动挂载），可按以下步骤配置。

### （1）获取 USB 的 UUID

执行以下命令：
```bash
sudo blkid /dev/sdd1


示例输出：

/dev/sdd1: UUID="b42e1aae-50a5-4369-90bd-e12b8e7b14c5" TYPE="ext4"

（2）编辑 /etc/fstab

执行命令：

sudo nano /etc/fstab

（3）在文件末尾添加以下一行
UUID=b42e1aae-50a5-4369-90bd-e12b8e7b14c5  /media/usb  ext4  defaults  0  2


⚠️ 请将 b42e1aae-50a5-4369-90bd-e12b8e7b14c5 替换为你在上一步中获取的实际 UUID。

（4）测试是否能正确挂载

执行：

sudo mount -a


如果没有报错，说明配置成功。
此后系统启动时会自动挂载 /media/usb，Docker 也会自动加载 USB 上的数据。

⚙️ 2️⃣ 安全移除 USB

在拔出 USB 之前，务必先停止 Docker 服务并卸载挂载点，以防止文件系统损坏。

执行：

sudo systemctl stop docker
sudo umount /media/usb


🚨 注意：不要在 Docker 运行时直接拔出 USB，否则可能导致 /media/usb/docker-data 文件损坏或镜像丢失。

✅ 提示

完成上述配置后：

系统启动时，USB 会自动挂载；

Docker 会自动加载 /media/usb/docker-data；

安全拔出时，只需执行上面的两条命令即可。


---

你可以将上面内容保存为文件：  
```bash
nano docker_usb_mount_guide.md


粘贴进去后保存，即可直接用。
