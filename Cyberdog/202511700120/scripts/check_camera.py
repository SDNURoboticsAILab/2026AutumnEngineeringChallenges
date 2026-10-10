import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class CameraCheck(Node):
    def __init__(self):
        super().__init__('level4_camera_check')
        self.got_image = False
        self.got_info = False
        self.create_subscription(
            Image,
            '/camera_raw/race_rgb_camera/image_raw',
            self.on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            '/camera_raw/race_rgb_camera/camera_info',
            self.on_info,
            qos_profile_sensor_data,
        )

    def finish_if_ready(self):
        if self.got_image and self.got_info:
            print('CAMERA_TOPICS_OK')
            rclpy.shutdown()

    def on_image(self, msg):
        if not self.got_image:
            print(
                f'IMAGE_OK width={msg.width} height={msg.height} '
                f'encoding={msg.encoding} frame_id={msg.header.frame_id}'
            )
            self.got_image = True
            self.finish_if_ready()

    def on_info(self, msg):
        if not self.got_info:
            print(
                f'CAMERA_INFO_OK width={msg.width} height={msg.height} '
                f'frame_id={msg.header.frame_id}'
            )
            self.got_info = True
            self.finish_if_ready()


def main():
    rclpy.init()
    node = CameraCheck()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
