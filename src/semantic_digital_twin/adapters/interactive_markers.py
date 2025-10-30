import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import rclpy.node

from .. import logger
from ..callbacks.callback import StateChangeCallback
from ..world import World
from ..world_description.connections import RevoluteConnection
from ..spatial_types.spatial_types import TransformationMatrix

try:
    from interactive_markers.interactive_marker_server import InteractiveMarkerServer
    from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
    from geometry_msgs.msg import Pose, Point, Quaternion
except ImportError as e:
    logger.warning(f"interactive_markers are not available: {e}")


def to_pose(transform: np.ndarray) -> Pose:
    """
    Converts a 4x4 transformation matrix to a Pose message.
    """
    from scipy.spatial.transform import Rotation

    pose = Pose()
    pose.position = Point(x=float(transform[0, 3]), y=float(transform[1, 3]), z=float(transform[2, 3]))
    quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
    pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
    return pose


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Returns the unit vector.
    """
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def quat_from_x_to_vec(target: np.ndarray) -> np.ndarray:
    """
    Builds a quaternion that rotates the X-axis onto the given unit vector.
    """
    from scipy.spatial.transform import Rotation

    x = np.array([1.0, 0.0, 0.0])
    t = normalize(target)
    if np.allclose(t, x):
        return np.array([0.0, 0.0, 0.0, 1.0])
    if np.allclose(t, -x):
        # 180 degrees around any axis perpendicular to X, choose Y
        return Rotation.from_rotvec(np.pi * np.array([0.0, 1.0, 0.0])).as_quat()
    axis = normalize(np.cross(x, t))
    angle = math.acos(np.clip(np.dot(x, t), -1.0, 1.0))
    return Rotation.from_rotvec(angle * axis).as_quat()


def roll_from_rotation(R: np.ndarray) -> float:
    """
    Extracts the roll angle for a rotation around the X-axis.
    """
    # Standard ZYX to roll extraction is: roll = atan2(R[2,1], R[1,1])
    return float(math.atan2(R[2, 1], R[1, 1]))


@dataclass
class RevoluteMarkerController:
    """
    Controls an interactive marker for one revolute connection.
    """

    world: World
    connection: RevoluteConnection
    server: InteractiveMarkerServer
    reference_frame: str

    marker_name: str = field(init=False)
    # State for feedback interpretation
    base_orientation_quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]), init=False)
    base_joint_position: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.marker_name = f"{self.connection.name}_revolute"
        self.insert_or_update_marker()

    def _global_connection_pose(self) -> np.ndarray:
        parent_T_conn = self.connection.parent_T_connection_expression.to_np()
        map_T_parent = self.world.compute_forward_kinematics(self.world.root, self.connection.parent).to_np()
        return map_T_parent @ parent_T_conn

    def _axis_in_connection_frame(self) -> np.ndarray:
        a_parent = self.connection.axis.to_np()[:3]
        R_pc = self.connection.parent_T_connection_expression.to_np()[:3, :3]
        # Express axis in the connection frame
        return normalize(R_pc.T @ a_parent)

    def insert_or_update_marker(self) -> None:
        # Base pose of the connection in the world (reference) frame
        T_world_conn = self._global_connection_pose()
        pose = to_pose(T_world_conn)

        # Set control orientation: X-axis aligned to revolute axis in connection frame
        a_conn = self._axis_in_connection_frame()
        q_axis = quat_from_x_to_vec(a_conn)

        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.reference_frame
        int_marker.name = self.marker_name
        int_marker.description = self.connection.name.name
        int_marker.scale = 0.3  # reasonable default; tune as needed
        int_marker.pose = pose

        # Create a visible marker control (always visible)
        visual_marker = Marker()
        visual_marker.type = Marker.ARROW
        visual_marker.scale.x = 0.25  # length
        visual_marker.scale.y = 0.01  # shaft diameter
        visual_marker.scale.z = 0.02  # head diameter
        visual_marker.color.r = 0.8
        visual_marker.color.g = 0.2
        visual_marker.color.b = 0.2
        visual_marker.color.a = 0.8
        visual_marker.pose.orientation = Quaternion(x=float(q_axis[0]), y=float(q_axis[1]), z=float(q_axis[2]), w=float(q_axis[3]))

        visual_control = InteractiveMarkerControl()
        visual_control.always_visible = True
        visual_control.markers.append(visual_marker)
        int_marker.controls.append(visual_control)

        # Create the interactive rotation control
        ctrl = InteractiveMarkerControl()
        ctrl.name = "rotate_dof"
        ctrl.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        ctrl.orientation = Quaternion(x=float(q_axis[0]), y=float(q_axis[1]), z=float(q_axis[2]), w=float(q_axis[3]))
        int_marker.controls.append(ctrl)

        # Store base orientation and joint position for delta calculation
        from scipy.spatial.transform import Rotation
        self.base_orientation_quat = Rotation.from_matrix(T_world_conn[:3, :3]).as_quat()
        self.base_joint_position = self.connection.position

        def _feedback(feedback):
            from scipy.spatial.transform import Rotation
            # Current orientation in world frame from feedback
            q_curr = np.array([
                feedback.pose.orientation.x,
                feedback.pose.orientation.y,
                feedback.pose.orientation.z,
                feedback.pose.orientation.w,
            ], dtype=float)

            # Relative rotation wrt base
            q0 = self.base_orientation_quat
            R_rel = (Rotation.from_quat(q0).inv() * Rotation.from_quat(q_curr)).as_matrix()

            # Extract the roll around X for the control frame
            delta = roll_from_rotation(R_rel)
            new_pos = self.base_joint_position + delta

            # Respect DOF limits if present
            lower = self.connection.dof.lower_limits.position
            upper = self.connection.dof.upper_limits.position
            if lower is not None:
                new_pos = max(lower, new_pos)
            if upper is not None:
                new_pos = min(upper, new_pos)

            # Update world; property setter notifies state change
            self.connection.position = new_pos

            # After world update, refresh base references so further drags are incremental
            T_world_conn_updated = self._global_connection_pose()
            self.base_orientation_quat = Rotation.from_matrix(T_world_conn_updated[:3, :3]).as_quat()
            self.base_joint_position = self.connection.position

            # Also snap marker pose to the exact new connection pose to avoid drift
            self.server.setPose(self.marker_name, to_pose(T_world_conn_updated), feedback.header)
            self.server.applyChanges()

        self.server.insert(int_marker, _feedback)
        self.server.setPose(self.marker_name, pose)
        self.server.applyChanges()

    def update_pose_from_world(self) -> None:
        T_world_conn = self._global_connection_pose()
        # Keep base references consistent when the world moves for reasons other than this marker
        from scipy.spatial.transform import Rotation
        self.base_orientation_quat = Rotation.from_matrix(T_world_conn[:3, :3]).as_quat()
        self.base_joint_position = self.connection.position
        self.server.setPose(self.marker_name, to_pose(T_world_conn))


@dataclass
class InteractiveConnectionMarkers(StateChangeCallback):
    """
    Publishes Interactive Markers for connections and updates the World on user input.
    """

    node: rclpy.node.Node
    reference_frame: str = "map"
    topic_ns: str = "/semworld/interactive_markers"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.server = InteractiveMarkerServer(self.node, self.topic_ns)
        self.controllers: Dict[str, RevoluteMarkerController] = {}
        self._rebuild()

    def _rebuild(self) -> None:
        # Clear old markers
        for name in list(self.controllers.keys()):
            self.server.erase(name)
        self.controllers.clear()

        # Create controllers for all revolute connections
        for conn in self.world.get_connections_by_type(RevoluteConnection):
            ctrl = RevoluteMarkerController(
                world=self.world,
                connection=conn,
                server=self.server,
                reference_frame=self.reference_frame,
            )
            self.controllers[ctrl.marker_name] = ctrl

        self.server.applyChanges()

    def _notify(self) -> None:
        """
        Reacts to world changes by updating marker poses.
        """
        # If connections are added/removed, rebuild; for simple motion, update poses
        # Simple heuristic: rebuild if counts differ; otherwise update.
        current_revolutes = list(self.world.get_connections_by_type(RevoluteConnection))
        if len(current_revolutes) != len(self.controllers):
            self._rebuild()
            return

        for ctrl in self.controllers.values():
            ctrl.update_pose_from_world()
        self.server.applyChanges()
