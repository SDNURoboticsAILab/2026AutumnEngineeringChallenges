# CyberDog 四足机器人仿真环境搭建

- 项目：2026 秋季工程实践能力考核（项目二）
- 姓名：张鋆源
- 学号：202511700120
- 专业班级：物联网工程 物联本 2501

本目录包含 CyberDog 仿真实验使用的脚本、相机配置及配置补丁。约 11.8 GB 的 Docker 镜像文件不放入 Git 仓库，请从题目提供的网盘下载。

- 详细项目报告：[`reports/PROJECT_REPORT.md`](reports/PROJECT_REPORT.md)

## 环境要求

- Ubuntu 24.04
- Docker Engine，本项目实际使用 29.8.1；题目示例中的 20.10.21 已停止维护
- X11 图形桌面
- 约 30 GB 可用磁盘空间

镜像下载地址：

- [百度网盘](https://pan.baidu.com/s/1FHPks2QdmCywGyVa1Et5TQ?pwd=zxwg)，提取码：`zxwg`
- [123 网盘](https://www.123865.com/s/GoDdjv-E15UA?pwd=dWkW)，提取码：`dWkW`

## 目录结构

```text
.
├── README.md
├── config/
│   ├── gazebo.xacro.original
│   ├── gazebo.xacro.level4
│   └── gazebo.xacro.patch
├── media/
│   ├── 01_level1_ubuntu_docker.png
│   ├── 02_level2_docker_images.png
│   ├── 03_level2_container_terminal.png
│   ├── 04_level3_gazebo_rviz.png
│   ├── 04b_level3_gazebo_rviz_later.png
│   ├── 05_level4_gazebo_blue_cone.png
│   ├── 05b_level4_camera_frequency.png
│   ├── 06_level4_camera_topics.png
│   ├── 06b_level4_camera_messages.png
│   ├── 06c_level4_lidar_scan.png
│   ├── 07_level5_motion_manager_info.png
│   ├── 08_level6_motion_result_cmd.png
│   ├── 09_level6_motion_demo.gif
│   └── 09_level6_motion_demo.webm
├── reports/
│   └── PROJECT_REPORT.md
└── scripts/
    ├── load_image.sh
    ├── start_container.sh
    ├── apply_camera_config.sh
    ├── check_camera.sh
    ├── check_camera.py
    ├── check_lidar.sh
    ├── check_lidar.py
    ├── start_motion_manager.sh
    └── demo_motion.sh
```

## 使用步骤

以下命令均在当前目录执行。脚本使用 `sudo docker`，运行时可能要求输入管理员密码。

### 1. 导入镜像

下载 `cyberdog_race.tar` 或 `cyberdog_raceV2.tar` 后执行：

```bash
./scripts/load_image.sh /path/to/cyberdog_raceV2.tar
```

脚本会导入镜像，并确保存在标签 `cyberdog_sim:v2026`。

### 2. 创建并进入容器

```bash
./scripts/start_container.sh
```

容器名为 `cyberdog_sim_level2`。脚本授权 root 用户访问当前 X11 桌面，并以特权模式创建容器。

### 3. 启动 Gazebo 和 RViz2

进入容器后执行：

```bash
cd /home/cyberdog_sim
python3 src/cyberdog_simulator/cyberdog_gazebo/script/launchsim.py
```

正常启动后会出现 `cyberdog_gazebo`、`cyberdog_control` 和 `cyberdog_visual` 三个窗口。

### 4. 应用相机配置

从宿主机打开一个新终端执行：

```bash
./scripts/apply_camera_config.sh
```

该脚本会备份容器内的原始 `gazebo.xacro`，应用 `config/gazebo.xacro.level4`，并使用 `xacro` 校验配置。应用后需要重新运行仿真。

配置在 `RGB_camera_link` 上启用了名为 `race_rgb_camera` 的 RGB 相机，图像尺寸为 `640x480`，帧率为 `30 Hz`，并在 Gazebo 中显示相机视锥。

### 5. 检查相机和激光雷达

仿真重新启动后，在宿主机执行：

```bash
./scripts/check_camera.sh
./scripts/check_lidar.sh
```

相机检查会等待以下话题：

```text
/camera_raw/race_rgb_camera/image_raw
/camera_raw/race_rgb_camera/camera_info
```

激光雷达检查会等待 `/scan`。

### 6. 启动运动控制服务

在宿主机新终端执行：

```bash
./scripts/start_motion_manager.sh
```

节点启动后应持续输出 `[INFO] [MotionManager]` 日志。按 `Ctrl+C` 停止查看或停止节点。

### 7. 演示机器狗运动

在宿主机新终端执行：

```bash
./scripts/demo_motion.sh
```

默认前进 8 秒，也可以指定其他正整数秒数：

```bash
./scripts/demo_motion.sh 10
```

演示脚本依次激活状态机、执行站立动作 `motion_id=111`、发送慢速前进指令 `motion_id=303`、发送快速前进指令 `motion_id=305`，最后再次恢复站立。

## 配置补丁

`config/gazebo.xacro.patch` 记录了原始配置到 Level 4 相机配置的修改。需要复现补丁时，可以在 `config` 目录执行：

```bash
cp gazebo.xacro.original gazebo.xacro
patch -p1 < gazebo.xacro.patch
cmp gazebo.xacro gazebo.xacro.level4
```

## 注意事项

- 项目不提交 11.8 GB 的 Docker 镜像 tar 文件。
- 相机配置保存在容器中；如果删除并重新创建容器，需要重新执行 `apply_camera_config.sh`。
- `start_container.sh` 使用 `xhost +SI:localuser:root`，只授权本机 root 用户访问 X Server。
- 所有 ROS2 服务和话题均在同一容器内的仿真环境中运行。
