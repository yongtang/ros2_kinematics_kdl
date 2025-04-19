import signal
import subprocess
import time

import numpy as np
import PyKDL
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF

from ros2_kinematics_kdl.urdf import treeFromUrdfModel


@pytest.fixture(scope="module")
def start_ik_node():
    """Start the IK node as a background subprocess."""
    cmd = [
        "ros2",
        "run",
        "ros2_kinematics_kdl",
        "ik_node",
        "--ros-args",
        "-p",
        "urdf_path:=/opt/ros/humble/share/moveit_resources_panda_description/urdf/panda.urdf",
        "-p",
        "base_link:=panda_link0",
        "-p",
        "ee_link:=panda_link8",
        "-p",
        "pose_topic:=/pose",
        "-p",
        "joint_state_topic:=/joint_states",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2.0)  # give it time to start
    yield
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_ik_with_fk(start_ik_node):
    urdf_path = (
        "/opt/ros/humble/share/moveit_resources_panda_description/urdf/panda.urdf"
    )
    base_link = "panda_link0"
    ee_link = "panda_link8"

    rclpy.init()
    node = rclpy.create_node("test_ik_fk")

    urdf = URDF.from_xml_file(urdf_path)
    ok, tree = treeFromUrdfModel(urdf)
    assert ok
    chain = tree.getChain(base_link, ee_link)
    fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
    joint_names = [j.name for j in urdf.joints if j.type != "fixed"][
        : chain.getNrOfJoints()
    ]

    pose_msg = PoseStamped()
    pose_msg.header.frame_id = base_link
    pose_msg.pose.position.x = 0.3
    pose_msg.pose.position.y = 0.0
    pose_msg.pose.position.z = 0.5
    pose_msg.pose.orientation.x = 0.0
    pose_msg.pose.orientation.y = 0.0
    pose_msg.pose.orientation.z = 0.0
    pose_msg.pose.orientation.w = 1.0

    received_joint_state = False
    fk_success = False

    def joint_callback(msg):
        nonlocal received_joint_state, fk_success

        q = PyKDL.JntArray(len(joint_names))
        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        for i, name in enumerate(joint_names):
            if name in name_to_index:
                q[i] = msg.position[name_to_index[name]]
            else:
                return

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
        print(f"[TEST] FK: {p_fk}, target: {p_target}, error: {pos_error:.6f}")

        received_joint_state = True
        fk_success = pos_error < 1e-3

    pub = node.create_publisher(PoseStamped, "/pose", 10)
    sub = node.create_subscription(JointState, "/joint_states", joint_callback, 10)

    # Wait for node to discover subscriber
    start = time.time()
    while pub.get_subscription_count() == 0 and time.time() - start < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    pub.publish(pose_msg)

    # Spin to receive joint state
    start = time.time()
    while not received_joint_state and time.time() - start < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    assert received_joint_state, "No joint state received"
    assert fk_success, "FK validation failed (position error too large)"
