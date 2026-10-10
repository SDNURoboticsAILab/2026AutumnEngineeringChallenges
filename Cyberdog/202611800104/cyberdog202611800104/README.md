# CyberDog 仿真环境搭建项目报告

**专业**：网络空间安全
**班级**：[2601]
**姓名**：[李智豪]
**学号**：[202611800104]

## 一、环境说明
由于本机为 Windows 平台，且存在 VBS 虚拟化安全锁等问题，采用 VMware 虚拟机运行 Ubuntu 20.04，并在 Docker 容器中导入 `cyberdog_sim:v2` 镜像（11GB）。

## 二、任务完成情况
* **Level 3 (Gazebo + RViz2 启动)**：已完成。成功在 Gazebo 中加载了赛道和机器狗，并在 RViz2 中成功显示了机器狗的模型（见截图）。
* **Level 4 (挂载相机和激光雷达)**：已完成。通过 Vim 修改了 `gazebo.xacro` 配置文件，添加了 `<gazebo reference="RGB_camera_link">` 和相机传感器插件。重新编译后，Gazebo 中成功显示了相机视锥（见截图）。
* **Level 5 (启动运动控制管理服务)**：已完成。在 `/home/cyberdog_ws` 目录下成功启动了 `motion_manager` 节点，并在终端持续打印 `[INFO]` 日志（见截图）。
* **Level 6 (Motion 测试流程)**：部分完成。成功发送了 `motion_id: 111` 的站立指令，`motion_manager` 成功接收了指令。但由于虚拟机网络对 UDP 组播（LCM 协议）的限制，底层控制器进入安全急停（ESTOP）状态，FSM 状态机在 `Setup` 状态下拒绝执行，返回 `code 3003`。

## 三、遇到的问题与解决（工程排错记录）
1. **虚拟机卡顿与 VBS 锁**：关闭 Windows VBS（基于虚拟化的安全性），并将虚拟机内存扩容至 16GB，CPU 分配 6 核，显著提升性能。
2. **磁盘空间不足**：通过 `growpart` 和 `resize2fs` 将虚拟磁盘从 20GB 扩展至 90GB，保证了 11GB 镜像的成功导入。
3. **X11 渲染与崩溃**：在启动时配置 `xhost +` 和 `-e DISPLAY=$DISPLAY`，并通过 `LIBGL_ALWAYS_SOFTWARE=1` 解决虚拟机纯软件渲染下的 Gazebo 闪退问题。
4. **LCM 通信失败**：尝试了桥接模式（Bridged）和 Docker 的 `--network host` 参数，成功让部分节点通信，但虚拟网卡仍无法完美转发 LCM 组播，导致无法解锁站立状态。

## 四、总结与心得
跨专业折腾机器人仿真，从零搭建 Docker 环境，解决 X11 图形显示、磁盘分区、网络组播等一系列底层问题，对我自身的排错能力是一次极大的锻炼。虽然受限于虚拟机物理性能无法实现最终的运动控制，但已完整掌握了 CyberDog 仿真环境的启动与传感器配置流程。