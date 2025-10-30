#!/usr/bin/env python3
"""
Interactive Markers Dresser Example

This example demonstrates how to use interactive markers with a dresser created using DresserFactory.
The dresser will be visualized in RViz2 with interactive markers that allow you to manipulate
the drawer positions by clicking and dragging.

Usage:
    1. Start this script: python examples/interactive_markers_dresser.py
    2. Open RViz2: rviz2
    3. Set Fixed Frame to "map"
    4. Add MarkerArray display with topic "/semworld/viz_marker"
    5. Add InteractiveMarkers display with Update Topic "/semworld/interactive_markers/update"
    6. You should see the dresser and be able to interact with the drawer using the red arrow markers

Press Ctrl+C to exit.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import threading
import time

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import TransformationMatrix
from semantic_digital_twin.semantic_annotations.factories import (
    DresserFactory,
    ContainerFactory,
    HandleFactory,
    DrawerFactory,
    Direction,
    SemanticPositionDescription,
    HorizontalSemanticDirection,
    VerticalSemanticDirection,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.interactive_markers import InteractiveConnectionMarkers


def create_dresser_world():
    """
    Creates a world with a dresser containing three drawers.
    
    :return: World object with a dresser
    """
    # Create drawer factories for three drawers at different vertical positions
    drawer_factories = []
    drawer_transforms = []
    
    for i, vertical_pos in enumerate([0.15, 0.0, -0.15]):
        drawer_factory = DrawerFactory(
            name=PrefixedName(f"drawer_{i}"),
            container_factory=ContainerFactory(
                name=PrefixedName(f"drawer_{i}_container"),
                direction=Direction.Z,
                scale=Scale(0.3, 0.3, 0.15),
            ),
            handle_factory=HandleFactory(name=PrefixedName(f"drawer_{i}_handle")),
            semantic_position=SemanticPositionDescription(
                horizontal_direction_chain=[HorizontalSemanticDirection.FULLY_CENTER],
                vertical_direction_chain=[VerticalSemanticDirection.FULLY_CENTER],
            ),
        )
        drawer_factories.append(drawer_factory)
        
        # Position the drawer at the specified vertical offset
        drawer_transform = TransformationMatrix.from_xyz_rpy(
            x=0.0, y=0.0, z=vertical_pos
        )
        drawer_transforms.append(drawer_transform)
    
    # Create the dresser container
    container_factory = ContainerFactory(
        name=PrefixedName("dresser_container"),
        scale=Scale(0.35, 0.35, 0.5)
    )
    
    # Create the dresser factory
    dresser_factory = DresserFactory(
        name=PrefixedName("dresser"),
        parent_T_drawers=drawer_transforms,
        drawers_factories=drawer_factories,
        container_factory=container_factory,
    )
    
    # Create the world
    world = dresser_factory.create()
    
    return world


def main():
    """
    Main function that sets up ROS2 node and interactive markers.
    """
    print("=" * 80)
    print("Interactive Markers Dresser Example")
    print("=" * 80)
    print()
    print("Setting up ROS2 node and creating dresser world...")
    
    # Initialize ROS2
    rclpy.init()
    
    # Create a ROS2 node
    node = Node("interactive_markers_dresser_example")
    
    # Create the world with dresser
    world = create_dresser_world()
    
    print(f"Created world with {len(world.bodies)} bodies")
    print(f"Found {len(world.connections)} connections")
    print(f"Found {len(world.degrees_of_freedom)} degrees of freedom")
    print()
    
    # Set up visualization marker publisher
    print("Setting up visualization markers...")
    viz_publisher = VizMarkerPublisher(
        world=world,
        node=node,
        topic_name="/semworld/viz_marker",
        reference_frame="map"
    )
    
    # Set up interactive markers
    print("Setting up interactive markers...")
    interactive_markers = InteractiveConnectionMarkers(
        world=world,
        node=node,
        reference_frame="map",
        topic_ns="/semworld/interactive_markers"
    )
    
    print()
    print("=" * 80)
    print("Setup Complete!")
    print("=" * 80)
    print()
    print("Instructions:")
    print("1. Open RViz2 in another terminal: rviz2")
    print("2. Set Fixed Frame to 'map' in RViz2")
    print("3. Add MarkerArray display:")
    print("   - Click 'Add' button")
    print("   - Select 'By topic' tab")
    print("   - Choose '/semworld/viz_marker' -> MarkerArray")
    print("4. Add InteractiveMarkers display:")
    print("   - Click 'Add' button")
    print("   - Select 'By display type' tab")
    print("   - Choose 'InteractiveMarkers'")
    print("   - Set 'Update Topic' to '/semworld/interactive_markers/update'")
    print()
    print("You should now see the dresser with three drawers.")
    print("Red arrow markers indicate the interactive controls.")
    print("Click and drag the arrows to open/close the drawers!")
    print()
    print("Press Ctrl+C to exit.")
    print("=" * 80)
    print()
    
    # Create executor and spin in a separate thread
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        # Keep the main thread alive
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean shutdown
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        print("Example terminated.")


if __name__ == "__main__":
    main()
