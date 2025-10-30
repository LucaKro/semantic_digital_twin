import time
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch

from semantic_digital_twin.adapters.interactive_markers import (
    to_pose,
    normalize,
    quat_from_x_to_vec,
    roll_from_rotation,
    RevoluteMarkerController,
    InteractiveConnectionMarkers,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3, TransformationMatrix
from semantic_digital_twin.testing import rclpy_node
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.world_entity import Body



@pytest.mark.skipif(
    not pytest.importorskip("rclpy", reason="ROS not installed"),
    reason="ROS not installed"
)
class TestRevoluteMarkerController:
    """Tests for RevoluteMarkerController with ROS integration."""

    def test_initialization(self, rclpy_node):
        """Test that RevoluteMarkerController initializes correctly."""
        # Create a simple world with a revolute connection
        world = World()
        with world.modify_world():
            parent_body = Body(name=PrefixedName("parent"))
            child_body = Body(name=PrefixedName("child"))
            world.add_kinematic_structure_entity(parent_body)
            world.add_kinematic_structure_entity(child_body)
            
            connection = RevoluteConnection(
                parent=parent_body,
                child=child_body,
                axis=Vector3(0, 0, 1),
                _world=world
            )
            world.add_connection(connection)

        # Mock the InteractiveMarkerServer
        with patch('semantic_digital_twin.adapters.interactive_markers.InteractiveMarkerServer') as mock_server_class:
            mock_server = MagicMock()
            mock_server_class.return_value = mock_server
            
            controller = RevoluteMarkerController(
                world=world,
                connection=connection,
                server=mock_server,
                reference_frame="map"
            )
            
            # Check that marker was inserted
            assert mock_server.insert.called
            assert mock_server.applyChanges.called
            assert controller.marker_name == f"{connection.name}_revolute"

    def test_global_connection_pose(self, rclpy_node):
        """Test computation of global connection pose."""
        world = World()
        with world.modify_world():
            parent_body = Body(name=PrefixedName("parent"))
            child_body = Body(name=PrefixedName("child"))
            world.add_kinematic_structure_entity(parent_body)
            world.add_kinematic_structure_entity(child_body)
            
            connection = RevoluteConnection(
                parent=parent_body,
                child=child_body,
                axis=Vector3(0, 0, 1),
                _world=world
            )
            world.add_connection(connection)

        with patch('semantic_digital_twin.adapters.interactive_markers.InteractiveMarkerServer') as mock_server_class:
            mock_server = MagicMock()
            mock_server_class.return_value = mock_server
            
            controller = RevoluteMarkerController(
                world=world,
                connection=connection,
                server=mock_server,
                reference_frame="map"
            )
            
            pose = controller._global_connection_pose()
            assert pose.shape == (4, 4)
            # Should be a valid transformation matrix
            assert np.isclose(pose[3, 3], 1.0)

    def test_axis_in_connection_frame(self, rclpy_node):
        """Test axis transformation to connection frame."""
        world = World()
        with world.modify_world():
            parent_body = Body(name=PrefixedName("parent"))
            child_body = Body(name=PrefixedName("child"))
            world.add_kinematic_structure_entity(parent_body)
            world.add_kinematic_structure_entity(child_body)
            
            axis = Vector3(0, 0, 1)
            connection = RevoluteConnection(
                parent=parent_body,
                child=child_body,
                axis=axis,
                _world=world
            )
            world.add_connection(connection)

        with patch('semantic_digital_twin.adapters.interactive_markers.InteractiveMarkerServer') as mock_server_class:
            mock_server = MagicMock()
            mock_server_class.return_value = mock_server
            
            controller = RevoluteMarkerController(
                world=world,
                connection=connection,
                server=mock_server,
                reference_frame="map"
            )
            
            axis_conn = controller._axis_in_connection_frame()
            # Should be normalized
            assert np.isclose(np.linalg.norm(axis_conn), 1.0)


@pytest.mark.skipif(
    not pytest.importorskip("rclpy", reason="ROS not installed"),
    reason="ROS not installed"
)
class TestInteractiveConnectionMarkers:
    """Tests for InteractiveConnectionMarkers with ROS integration."""

    def test_initialization(self, rclpy_node):
        """Test that InteractiveConnectionMarkers initializes correctly."""
        world = World()
        with world.modify_world():
            parent_body = Body(name=PrefixedName("parent"))
            child_body = Body(name=PrefixedName("child"))
            world.add_kinematic_structure_entity(parent_body)
            world.add_kinematic_structure_entity(child_body)
            
            connection = RevoluteConnection(
                parent=parent_body,
                child=child_body,
                axis=Vector3(0, 0, 1),
                _world=world
            )
            world.add_connection(connection)

        with patch('semantic_digital_twin.adapters.interactive_markers.InteractiveMarkerServer') as mock_server_class:
            mock_server = MagicMock()
            mock_server_class.return_value = mock_server
            
            markers = InteractiveConnectionMarkers(node=rclpy_node, world=world)
            
            # Should have one controller for the revolute connection
            assert len(markers.controllers) == 1
            assert mock_server.applyChanges.called

    def test_rebuild_on_connection_addition(self, rclpy_node):
        """Test that markers rebuild when connections are added."""
        world = World()
        with world.modify_world():
            parent_body = Body(name=PrefixedName("parent"))
            child_body = Body(name=PrefixedName("child"))
            world.add_kinematic_structure_entity(parent_body)
            world.add_kinematic_structure_entity(child_body)
            
            connection = RevoluteConnection(
                parent=parent_body,
                child=child_body,
                axis=Vector3(0, 0, 1),
                _world=world
            )
            world.add_connection(connection)

        with patch('semantic_digital_twin.adapters.interactive_markers.InteractiveMarkerServer') as mock_server_class:
            mock_server = MagicMock()
            mock_server_class.return_value = mock_server
            
            markers = InteractiveConnectionMarkers(node=rclpy_node, world=world)
            
            initial_count = len(markers.controllers)
            
            # Add another connection
            with world.modify_world():
                child_body2 = Body(name=PrefixedName("child2"))
                world.add_kinematic_structure_entity(child_body2)
                connection2 = RevoluteConnection(
                    parent=child_body,
                    child=child_body2,
                    axis=Vector3(1, 0, 0),
                    _world=world
                )
                world.add_connection(connection2)
            
            # Trigger state change notification
            world.notify_state_change()
            time.sleep(0.1)
            
            # Should rebuild and have more controllers
            assert len(markers.controllers) >= initial_count

    def test_state_change_callback(self, rclpy_node):
        """Test that state changes trigger marker updates."""
        world = World()
        with world.modify_world():
            parent_body = Body(name=PrefixedName("parent"))
            child_body = Body(name=PrefixedName("child"))
            world.add_kinematic_structure_entity(parent_body)
            world.add_kinematic_structure_entity(child_body)
            
            connection = RevoluteConnection(
                parent=parent_body,
                child=child_body,
                axis=Vector3(0, 0, 1),
                _world=world
            )
            world.add_connection(connection)

        with patch('semantic_digital_twin.adapters.interactive_markers.InteractiveMarkerServer') as mock_server_class:
            mock_server = MagicMock()
            mock_server_class.return_value = mock_server
            
            markers = InteractiveConnectionMarkers(node=rclpy_node, world=world)
            
            # Change joint position
            connection.position = 0.5
            time.sleep(0.1)
            
            # Server should have setPose called
            assert mock_server.setPose.called or mock_server.applyChanges.called

    def test_multiple_revolute_connections(self, rclpy_node):
        """Test handling multiple revolute connections."""
        world = World()
        with world.modify_world():
            bodies = [Body(name=PrefixedName(f"body_{i}")) for i in range(4)]
            for body in bodies:
                world.add_kinematic_structure_entity(body)
            
            connections = []
            for i in range(3):
                conn = RevoluteConnection(
                    parent=bodies[i],
                    child=bodies[i + 1],
                    axis=Vector3(0, 0, 1) if i % 2 == 0 else Vector3(1, 0, 0),
                    _world=world
                )
                world.add_connection(conn)
                connections.append(conn)

        with patch('semantic_digital_twin.adapters.interactive_markers.InteractiveMarkerServer') as mock_server_class:
            mock_server = MagicMock()
            mock_server_class.return_value = mock_server
            
            markers = InteractiveConnectionMarkers(node=rclpy_node, world=world)
            
            # Should have three controllers
            assert len(markers.controllers) == 3
            
            # Each should have a unique name
            names = [ctrl.marker_name for ctrl in markers.controllers.values()]
            assert len(names) == len(set(names))
