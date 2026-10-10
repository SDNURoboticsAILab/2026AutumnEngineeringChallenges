# 四足机器人仿真环境搭建


- **项目简介**： 本项目将带你跨入机器人仿真与 ROS2 全栈开发的大门，从零手把手搭建一套符合工业级标准的四足机器人（仿生狗）仿真系统。你将深入掌握如何使用 **URDF / Xacro** 进行机器人运动学与几何模型的精细化建模，并在**Gazebo 物理引擎** 中实现本体、关节动力学以及 IMU、激光雷达（LiDAR）、RGB 相机等关键传感器的仿真配置。

  通过本项目，你将深刻理解基于 **ROS2 Topic / Service** 的节点通信机制，实现机器人实时状态与传感器数据的采集，并结合 **RViz2** 完成三维数据流的可视化。这套仿真平台不仅能大幅降低算法测试的硬件成本与损坏风险，更为后续的运动控制（步态规划）、视觉感知、SLAM 实时建图与自主导航等高级算法验证提供了强大的工程基础设施。

- **核心技术栈**： Linux（Ubuntu）、ROS2 (Galactic/Humble)、容器化(Docker)、Gazabo物理仿真、URDF/Xacro机器人建模、RViz2数据可视化、机器人传感器与通信机制(LCM / ROS2 Topic)。  

- **考核及格线**： 至少完成 **Level 3**（成功运行 **Gazebo + RViz2 仿真程序**）  

- **发展方向**： 适合对 Linux学习、机器人开发 、具身智能、运动控制感兴趣的同学

## 一、考核目的与总提交要求

本考核旨在检验参赛者对机器人仿真环境的基本搭建能力以及对 ROS2 与 Gazebo 仿真系统的基础使用能力。

#### 1.参赛者需要完成以下任务：

1. 在本地电脑成功配置 **Ubuntu虚拟机/双系统/WSL2**；
2. 在本地电脑成功搭建 **CyberDog 仿真环境**；
3. 成功运行 **Gazebo + RViz2 仿真程序；**
4. 在机器狗模型上 **挂载相机和激光雷达；**
5. 启动 **运动控制管理程序；**
6. 在仿真环境中使 **机器狗正常运动**。


#### 2.总提交要求：

1. **环境配置项目报告**：详细记录环境搭建过程、遇到的问题及解决方案（包含各level的验证截图）；
2. **运行Gazabo仿真的视频**：包括仿真启动脚本、传感器画面展示以及运动控制程序效果等。

## 二、考核步骤

## Level 1： 准备 Ubuntu 环境与 Docker 基础

在开展机器人仿真前，需要先配置好标准的 Linux 系统环境，并安装容器化运行环境 Docker。

### 实践步骤：

#### 1. 操作系统环境搭建

**推荐系统：**

- Ubuntu **24.04**

**可选环境部署方式：**

- 双系统（Windows + Ubuntu）
- 虚拟机（推荐VMware / VirtualBox）
- WSL2（Windows Subsystem for Linux 2）

**配置提醒：**

在工程开发上，由于虚拟机的局限性，双系统在开发方面优于虚拟机，注意双系统安装有风险，请完整观看教程，谨慎安装

**系统安装参考教程：**

- [安装年轻人的第一个 Linux 虚拟机](https://analytics.hxcn.dev/q/first-vm)

- [Windows和Linux双系统的保姆级安装教程，新手小白跟着也能装_windows安装linux双系统-CSDN博客](https://blog.csdn.net/duduanwang/article/details/143393077)

- [Windows11 安装 Ubuntu 避坑指南](https://www.bilibili.com/video/BV1Cc41127B9/?share_source=copy_web&vd_source=20c309b591ec84d13d92a10bea5b2928)

#### 2. 软件环境搭建

**需要安装：**

- Docker **20.10.21**
- ROS2 Galactic（已包含在 Docker 镜像中）

**参考教程：**

- [Docker Engine 安装教程（Ubuntu）](https://docs.docker.com/engine/install/ubuntu/) — 官方 Docker Engine for Ubuntu 安装文档

### 验证标准与提交截图：

提交 `lsb_release -a` 与 `docker --version` 的终端运行截图。

## Level 2：导入 CyberDog 镜像与容器配置

`cyberdog_sim` 镜像提供了基于赛道的 CyberDog 四足机器人 Gazebo 仿真平台，能够使 Gazebo 仿真程序直接与 CyberDog 的控制程序 `cyberdog_control` 进行通信。仿真平台及依赖均已安装并编译完成。

### 实践步骤：

#### 1. 下载 Docker 镜像

下载文件：**cyberdog_race.tar**（大小约11.0GB）：

- [百度网盘（主下载线路）](https://pan.baidu.com/s/1FHPks2QdmCywGyVa1Et5TQ?pwd=zxwg) — 提取码：`zxwg`
- [123 网盘（备用下载线路）](https://www.123865.com/s/GoDdjv-E15UA?pwd=dWkW) — 提取码：`dWkW`

下载完成后，将文件放在 Ubuntu 系统目录中。

#### 2. 导入 Docker 镜像

打开终端，执行：

```bash
# 镜像文件较大（11GB），解压导入需要一定耗时，终端暂时无输出属于正常现象
sudo docker load -i cyberdog_race.tar

# 导入完成后检查镜像是否存在（查看列表中是否有 cyberdog_sim）
sudo docker images
```

#### 3. 授权X Server

在宿主机终端中允许 Docker 容器访问宿主机的显示服务：

```bash
xhost +
```

#### 4. 运行 Docker 容器

运行仿真环境容器：

```bash
sudo docker run -it --shm-size="1g" --privileged=true -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix cyberdog_sim:v2026
```

运行后终端提示符发生改变（如 `root@xxxxxxxxxxxx:/#`），代表已进入 Docker 内部终端。

### 验证标准与提交截图：

1. 执行 `sudo docker images` 成功显示 `cyberdog_sim:v2026` 镜像的截图；

2. 成功运行 `docker run` 进入容器内部终端的截图。

## Level 3：运行 Gazebo + RViz2 仿真程序

### 实践步骤：

#### 1. 一键启动仿真环境

在 Docker 终端中执行：

```bash
cd /home/cyberdog_sim
python3 src/cyberdog_simulator/cyberdog_gazebo/script/launchsim.py
```

该脚本将启动完整仿真系统。

启动后会出现三个窗口：

1. **cyberdog_gazebo**——Gazebo 仿真环境窗口
2. **cyberdog_control**——机器人控制程序窗口
3. **cyberdog_visual**——RViz2 可视化窗口

Gazebo 用于仿真机器人与环境交互，RViz 用于显示机器人状态与传感器数据。

![cyberdog_sim Gazebo 窗口](https://gastigado.cnies.org/d/public/image-20260312094943134.webp)

![cyberdog_sim RViz 窗口](https://gastigado.cnies.org/d/public/image-20260312094950057.webp)

#### 2.备用方案：分步启动（如一键启动失败时使用）

**1. 启动Gazebo仿真程序**

首先启动gazebo程序，于cyberdog_sim文件夹下进行如下操作

```bash
source /opt/ros/galactic/setup.bash
source install/setup.bash
ros2 launch cyberdog_gazebo race_gazebo.launch.py
```

**2. 启动cyberdog控制程序**

然后启动 cyberdog_locomotion 的控制程序。打开一个新的终端，在cyberdog_sim文件夹下运行：

```bash
source /opt/ros/galactic/setup.bash
source install/setup.bash
ros2 launch cyberdog_gazebo cyberdog_control_launch.py
```

**3. 启动rviz可视化界面**

最后打开可视化界面，打开一个新的终端，在cyberdog_sim文件夹下运行：

```bash
source /opt/ros/galactic/setup.bash
source install/setup.bash
ros2 launch cyberdog_visual cyberdog_visual.launch.py
```

### 验证标准与提交内容：

提交 Gazebo 和 RViz2 窗口成功弹出的全屏截图。

## Level 4：给机器狗挂载相机和激光雷达

### 前置知识：

在Docker容器内部找到机器人配置文件 **gazebo.xacro**  
**gazebo.xacro** 这个配置文件为机器人添加了

- 控制插件
- IMU
- 激光雷达
- 足端接触传感器
- 各个部件的摩擦/碰撞参数
- 深度相机
- 左右两侧鱼眼相机  

**.xacro** 可以理解成 “增强版 **URDF** 模板文件”  
ROS 里的机器人模型最终通常还是会变成 **.urdf** ，但直接写 **URDF** 很容易又长又重复，所以用 **Xacro** 来 **生成 URDF**  

### 实践步骤：

#### 1.添加相机配置

默认机器狗模型已经预留了相机的 **link挂载点位**，只是没有在这些 link 上加载 **Gazebo 相机传感器**。 
在 Docker 容器内部打开并编辑机器人配置文件**gazebo.xacro**，找到默认预留的**RGB_camera_link挂载点**，在 **gazabo.xacro** 中添加以下相机传感器配置:  

```xml
<gazebo reference="RGB_camera_link">

  <sensor name="race_rgb_camera" type="camera">

    <!-- 相机一直工作 -->
    <always_on>true</always_on>

    <!-- 30 FPS -->
    <update_rate>30</update_rate>

    <!-- 在 Gazebo 中显示相机视锥 -->
    <visualize>true</visualize>

    <!-- 相对于 RGB_camera_link 的位姿 -->
    <pose>0 0 0 0 0 0</pose>

    <!-- 相机本身参数 -->
    <camera>

      <!-- 水平视场角，1.047rad ≈ 60° -->
      <horizontal_fov>1.047</horizontal_fov>

      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>

      <clip>
        <near>0.05</near>
        <far>10.0</far>
      </clip>

    </camera>

    <!-- Gazebo -> ROS2 图像话题插件 -->
    <plugin
      name="race_rgb_camera_controller"
      filename="libgazebo_ros_camera.so">

      <ros>
        <namespace>/camera_raw</namespace>

        <remapping>
          image_raw:=rgb/image_raw
        </remapping>

        <remapping>
          camera_info:=rgb/camera_info
        </remapping>
      </ros>

      <camera_name>race_rgb_camera</camera_name>

      <frame_name>RGB_camera_link</frame_name>

    </plugin>

  </sensor>

</gazebo>
```

**注：这段配置加在最外层 `<robot>` 根节点内部，与 `<gazebo reference="...">` 同级放置，切勿把它写进 `<link>` 或 `<joint>` 里面。**

#### 2.验证相机添加

保存修改后，重新运行仿真程序，并在 Docker 新终端中验证相机话题是否发布，查看相机图像数据频率。

### 验证标准与提交内容：

运行仿真后，在 Gazebo 中成功显示相机蓝色视锥，且能够查看到相机话题输出的终端截图。

## Level 5：启动运动控制管理服务

### 实践步骤：

#### 1. 启动运动控制程序

打开 **新的终端（Docker 内）**，执行以下命令启动运动管理服务：

```bash
cd /home/cyberdog_ws
source /opt/ros/galactic/setup.bash
source install/setup.bash
ros2 run motion_manager motion_manager
```

该程序提供 ROS2 Topic 控制接口，可将 ROS2 控制指令转换为 LCM 指令发送给控制程序。

**运动控制接口参考文档：**

- [运动控制接口参考文档](https://analytics.hxcn.dev/q/cyberdog_loco_cn) — 小米机器人实验室公开的 CyberDog 运动控制接口说明

### 仿真通信结构简介：

仿真系统中主要包含两种通信方式：

#### 1. LCM 通信

用于机器人控制程序与仿真系统之间的通信。

#### 2. ROS2 Topic

用于：

- 机器人状态
- 传感器数据
- 控制命令

常见 topic 包括：

```text
/joint_states
/tf
/imu
/scan
```

Gazebo 会将仿真传感器数据发布为 ROS2 Topic。

### 验证标准与提交内容：

提交 `motion_manager` 节点成功启动并持续打印 `[INFO]` 日志的终端截图。

## Level ６：Motion 测试流程

仿真环境启动后，可通过 Motion 接口验证控制链路是否可用。以下流程假设 Gazebo、RViz、`motion_manager` 已经按上文步骤启动完成。

### 实践步骤：

#### 1. 激活状态机

新开一个终端进入运行中的 Docker 容器：

```bash
sudo docker ps -a
sudo docker exec -it 容器ID bash
cd /home/cyberdog_ws
```

激活 `motion_manager` 的状态机，切换机器狗状态至 `Active` 模式：

```bash
source /opt/ros/galactic/setup.bash
source install/setup.bash
ros2 service call /motion_managermachine_service protocol/srv/FsMachine "target_state: 'Active'"
```

#### 2. 执行 Motion 指令

指令的 `motion_id` 按需修改。

#### 指令 A：恢复站立姿态 (motion_id: 111)

```bash
ros2 service call /motion_result_cmd protocol/srv/MotionResultCmd "motion_id: 111"
```

#### 指令 B：慢速小跑前进 (motion_id: 303)

```bash
ros2 topic pub -r 20 /motion_servo_cmd protocol/msg/MotionServoCmd "{motion_id: 303, vel_des: [0.2, 0.0, 0.0]}"
```

#### 指令 C：快速前进 (motion_id: 305)

```bash
ros2 topic pub -r 10 /motion_servo_cmd protocol/msg/MotionServoCmd "{motion_id: 305,vel_des:[0.6,0,0]}"
```

**更多指令参考：**

[小米机器人实验室开发者指南](https://analytics.hxcn.dev/q/cyberdog_doc)

### 验证标准与提交内容：

1. 提交发送运动控制命令后，Gazebo 画面中机器狗成功站起并向前行进的视频记录（或连续动作截图）；

2. 提交终端成功调用 `/motion_result_cmd` 服务的返回值截图。

## 三、参考与推荐学习资源

如果你想进一步对四足机器人进行二次开发或深入学习 ROS2，推荐参考以下资料：

- [《ROS 2机器人开发从入门到实践》课程介绍](https://www.bilibili.com/video/BV1GW42197Ck/?share_source=copy_web&vd_source=20c309b591ec84d13d92a10bea5b2928)

- [ROS2 快速入门课程](https://www.bilibili.com/video/BV1sxMX6yEmb/?share_source=copy_web&vd_source=20c309b591ec84d13d92a10bea5b2928)