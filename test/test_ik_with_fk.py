import time

import launch
import launch_ros.actions
import numpy as np
import PyKDL
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from launch_testing.actions import ReadyToTest
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF

from ros2_kinematics_kdl.urdf import treeFromUrdfModel


# ----------------------------------------
# Launch test target (IK node)
# ----------------------------------------
@pytest.mark.launch_test
def generate_test_description():
    urdf_path = (
        "/opt/ros/humble/share/moveit_resources_panda_description/urdf/panda.urdf"
    )

    ik_node = launch_ros.actions.Node(
        package="ros2_kinematics_kdl",
        executable="ik_node",
        name="kdl_ik_node",
        output="screen",
        parameters=[
            {"urdf_path": urdf_path},
            {"base_link": "panda_link0"},
            {"ee_link": "ee_link"},
            {"pose_topic": "/pose"},
            {"joint_state_topic": "/joint_states"},
        ],
    )

    return launch.LaunchDescription([ik_node, ReadyToTest()]), {
        "test_context": {"urdf_path": urdf_path}
    }


# ----------------------------------------
# Actual test logic
# ----------------------------------------
def test_ik_with_fk(test_context):
    urdf_path = test_context["urdf_path"]
    base_link = "panda_link0"
    ee_link = "ee_link"

    rclpy.init()
    node = rclpy.create_node("test_ik_fk")

    # Load URDF and KDL chain
    urdf = URDF.from_xml_file(urdf_path)
    ok, tree = treeFromUrdfModel(urdf)
    assert ok
    chain = tree.getChain(base_link, ee_link)
    fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
    joint_names = [j.name for j in urdf.joints if j.type != "fixed"][
        : chain.getNrOfJoints()
    ]

    # Compose target pose
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = base_link
    pose_msg.pose.position.x = 0.3
    pose_msg.pose.position.y = 0.0
    pose_msg.pose.position.z = 0.5
    pose_msg.pose.orientation.x = 0.0
    pose_msg.pose.orientation.y = 0.0
    pose_msg.pose.orientation.z = 0.0
    pose_msg.pose.orientation.w = 1.0

    # Flags and helpers
    success = False
    received_joint_state = False

    def joint_callback(msg):
        nonlocal success, received_joint_state

        # Build KDL joint array
        q = PyKDL.JntArray(len(joint_names))
        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        for i, name in enumerate(joint_names):
            if name in name_to_index:
                q[i] = msg.position[name_to_index[name]]
            else:
                node.get_logger().warn(f"Missing joint: {name}")
                return

        # FK
        fk_frame = PyKDL.Frame()
        fk_solver.JntToCart(q, fk_frame)
        p_fk = np.array([fk_frame.p.x(), fk_frame.p.y(), fk_frame.p.z()])
        p_target = np.array(
            [
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z,
            ]
        )
        pos_error = np.linalg.norm(p_fk - p_target)
        print(f"[TEST] FK position: {p_fk}, target: {p_target}, error: {pos_error}")

        success = pos_error < 1e-3
        received_joint_state = True

    # ROS pub/sub
    pub = node.create_publisher(PoseStamped, "/pose", 10)
    sub = node.create_subscription(JointState, "/joint_states", joint_callback, 10)

    # Wait for the node to warm up and send
    time.sleep(1.0)
    pub.publish(pose_msg)

    # Spin for up to 5 seconds
    start = time.time()
    while rclpy.ok() and not received_joint_state and time.time() - start < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    assert received_joint_state, "No joint state received"
    assert success, "FK validation failed"
