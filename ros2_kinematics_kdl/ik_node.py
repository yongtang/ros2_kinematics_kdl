import os

import PyKDL
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF

from .urdf import treeFromUrdfModel


class KDLInverseKinematicsNode(Node):
    def __init__(self):
        super().__init__("ik_node")

        # Declare and read parameters
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("base_link", "base_link")
        self.declare_parameter("ee_link", "ee_link")
        self.declare_parameter("pose_topic", "/pose")
        self.declare_parameter("joint_state_topic", "/joint_states")

        urdf_path = self.get_parameter("urdf_path").get_parameter_value().string_value
        base_link = self.get_parameter("base_link").get_parameter_value().string_value
        ee_link = self.get_parameter("ee_link").get_parameter_value().string_value
        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        joint_state_topic = (
            self.get_parameter("joint_state_topic").get_parameter_value().string_value
        )

        if not urdf_path or not os.path.exists(urdf_path):
            self.get_logger().error(f"Invalid or missing URDF path: {urdf_path}")
            raise RuntimeError("Valid URDF file required.")

        urdf_model = URDF.from_xml_file(urdf_path)
        if not treeFromUrdfModel:
            raise RuntimeError(
                "kdl_parser_py is not available. Please vendor it or build from source."
            )

        ok, tree = treeFromUrdfModel(urdf_model)
        if not ok:
            self.get_logger().error("Failed to parse URDF into KDL tree.")
            raise RuntimeError("Could not create KDL tree.")

        self.chain = tree.getChain(base_link, ee_link)
        self.joint_names = [
            joint.name for joint in urdf_model.joints if joint.type != "fixed"
        ][: self.chain.getNrOfJoints()]

        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.chain)
        self.ik_solver = PyKDL.ChainIkSolverPos_LMA(self.chain)

        # ROS interfaces
        self.pose_sub = self.create_subscription(
            PoseStamped, pose_topic, self.pose_callback, 10
        )
        self.joint_pub = self.create_publisher(JointState, joint_state_topic, 10)

        self.get_logger().info(
            f"KDL IK Node started. Subscribed to {pose_topic}, publishing to {joint_state_topic}."
        )

    def pose_callback(self, msg):
        pos = msg.pose.position
        ori = msg.pose.orientation

        target_frame = PyKDL.Frame(
            PyKDL.Rotation.Quaternion(ori.x, ori.y, ori.z, ori.w),
            PyKDL.Vector(pos.x, pos.y, pos.z),
        )

        q_init = PyKDL.JntArray(self.chain.getNrOfJoints())
        q_out = PyKDL.JntArray(self.chain.getNrOfJoints())

        result = self.ik_solver.CartToJnt(q_init, target_frame, q_out)
        if result >= 0:
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.name = self.joint_names
            joint_state_msg.position = [q_out[i] for i in range(q_out.rows())]
            self.joint_pub.publish(joint_state_msg)
        else:
            self.get_logger().warn("IK solution not found for given pose.")


def main(args=None):
    rclpy.init(args=args)
    node = KDLInverseKinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
