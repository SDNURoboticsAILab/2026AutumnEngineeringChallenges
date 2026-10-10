import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanCheck(Node):
    def __init__(self):
        super().__init__('level4_scan_check')
        self.done = False
        self.create_subscription(
            LaserScan,
            '/scan',
            self.on_scan,
            qos_profile_sensor_data,
        )

    def on_scan(self, msg):
        if not self.done:
            self.done = True
            print(
                f'SCAN_OK frame_id={msg.header.frame_id} '
                f'ranges={len(msg.ranges)} angle_min={msg.angle_min:.4f} '
                f'angle_max={msg.angle_max:.4f}'
            )
            rclpy.shutdown()


def main():
    rclpy.init()
    node = ScanCheck()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
