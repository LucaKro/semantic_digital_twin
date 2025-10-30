#!/usr/bin/env python3
"""
Tendon Connection Demo
======================

This demo showcases a tendon-driven robotic joint using the TendonConnection class.
The demo features:
- A simple robotic arm with a tendon-driven elbow joint
- Animation through various joint angles
- Variable stiffness control via pretension
- Visualization using ROS2 (if available) or console output

Run with: python examples/tendon_connection_demo.py
"""

import time
import numpy as np

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import TendonConnection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.geometry import Box, Cylinder, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3, TransformationMatrix

# Try to import ROS2 visualization
try:
    import rclpy
    from rclpy.node import Node
    from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 not available. Running in console-only mode.")


def create_tendon_arm_world():
    """
    Create a world with a simple robotic arm featuring a tendon-driven elbow joint.
    
    Structure:
        world (root) -> base_link -> upper_arm -> forearm (via TendonConnection)
    """
    world = World()
    
    # Create base link (fixed to ground)
    base_link = Body(
        name=PrefixedName("base_link"),
        collision=ShapeCollection([
            Cylinder(
                origin=TransformationMatrix.from_xyz_rpy(z=-0.05),
                height=0.1,
                width=0.08,
            )
        ])
    )
    
    # Create upper arm (shoulder to elbow)
    upper_arm = Body(
        name=PrefixedName("upper_arm"),
        collision=ShapeCollection([
            Box(
                origin=TransformationMatrix.from_xyz_rpy(z=0.15),
                scale=Scale(0.05, 0.05, 0.3),
            )
        ])
    )
    
    # Create forearm (elbow to end effector)
    forearm = Body(
        name=PrefixedName("forearm"),
        collision=ShapeCollection([
            Box(
                origin=TransformationMatrix.from_xyz_rpy(z=0.125),
                scale=Scale(0.04, 0.04, 0.25),
            ),
            # End effector sphere
            Cylinder(
                origin=TransformationMatrix.from_xyz_rpy(z=0.25),
                height=0.03,
                width=0.06,
            )
        ])
    )
    
    with world.modify_world():
        # Add bodies to world
        world.add_kinematic_structure_entity(base_link)
        world.add_kinematic_structure_entity(upper_arm)
        world.add_kinematic_structure_entity(forearm)
        
        # Fixed connection from base to upper arm (shoulder joint is fixed for simplicity)
        from semantic_digital_twin.world_description.connections import FixedConnection
        shoulder = FixedConnection(
            parent=base_link,
            child=upper_arm,
            parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(
                z=0.05,  # Raise above base
                reference_frame=base_link,
            )
        )
        world.add_connection(shoulder)
        
        # Tendon-driven elbow joint
        elbow = TendonConnection(
            parent=upper_arm,
            child=forearm,
            name=PrefixedName("elbow_tendon"),
            axis=Vector3.Y(reference_frame=upper_arm),  # Rotate around Z axis
            pulley_radius=0.025,  # 2.5cm moment arm
            parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(
                z=0.3,  # At top of upper arm
                reference_frame=upper_arm,
            )
        )
        world.add_connection(elbow)
    
    return world, elbow


def print_demo_header():
    """Print demo header with information."""
    print("\n" + "="*70)
    print("TENDON-DRIVEN ROBOTIC ARM DEMO")
    print("="*70)
    print("\nThis demo showcases a tendon-driven elbow joint using antagonistic")
    print("tendons to control position and stiffness.")
    print("\nKey features:")
    print("  - Joint angle controlled by differential tendon displacement")
    print("  - Variable stiffness via pretension control")
    print("  - Smooth animation through various configurations")
    print("="*70 + "\n")


def print_state(world, elbow, step, description):
    """Print current state of the tendon joint."""
    angle_deg = np.degrees(elbow.joint_angle)
    stiffness = elbow.stiffness
    t1_pos = world.state[elbow.tendon_1.name].position
    t2_pos = world.state[elbow.tendon_2.name].position
    
    print(f"\n[Step {step}] {description}")
    print(f"  Joint angle:    {angle_deg:6.1f}° ({elbow.joint_angle:.3f} rad)")
    print(f"  Stiffness:      {stiffness:.4f} (pretension)")
    print(f"  Tendon 1 pos:   {t1_pos:.4f} m")
    print(f"  Tendon 2 pos:   {t2_pos:.4f} m")
    print(f"  Differential:   {t1_pos - t2_pos:.4f} m")


def run_simulation_sequence(world, elbow):
    """
    Run a simulation sequence demonstrating various capabilities.
    """
    print_demo_header()
    
    step = 1
    
    # 1. Initial configuration - straight arm
    print_state(world, elbow, step, "Initial configuration (straight)")
    time.sleep(1.5)
    step += 1
    
    # 2. Bend to 45 degrees with low pretension
    print(f"\n{'─'*70}")
    print("Bending to 45° with LOW pretension (compliant joint)...")
    elbow.set_joint_angle(np.pi / 4, pretension=0.015)
    print_state(world, elbow, step, "45° bend, low stiffness")
    time.sleep(1.5)
    step += 1
    
    # 3. Increase stiffness while maintaining angle
    print(f"\n{'─'*70}")
    print("Increasing pretension (same angle, higher stiffness)...")
    elbow.set_joint_angle(np.pi / 4, pretension=0.04)
    print_state(world, elbow, step, "45° bend, high stiffness")
    time.sleep(1.5)
    step += 1
    
    # 4. Smooth motion through angles
    print(f"\n{'─'*70}")
    print("Smooth motion through various angles...")
    angles = np.linspace(np.pi / 4, np.pi / 2, 6)
    for i, angle in enumerate(angles):
        elbow.set_joint_angle(angle, pretension=0.025)
        if i % 2 == 0:  # Print every other step
            print_state(world, elbow, step, f"Moving to {np.degrees(angle):.0f}°")
        time.sleep(0.5)
        step += 1
    
    # 5. Maximum bend
    print(f"\n{'─'*70}")
    print("Maximum bend (100°)...")
    elbow.set_joint_angle(np.radians(100), pretension=0.03)
    print_state(world, elbow, step, "Maximum bend (100°)")
    time.sleep(1.5)
    step += 1
    
    # 6. Return to neutral
    print(f"\n{'─'*70}")
    print("Returning to neutral position...")
    elbow.set_joint_angle(0.0, pretension=0.02)
    print_state(world, elbow, step, "Neutral position")
    time.sleep(1.5)
    step += 1
    
    # 7. Negative angle (bend opposite direction)
    print(f"\n{'─'*70}")
    print("Bending in opposite direction...")
    elbow.set_joint_angle(-np.pi / 6, pretension=0.025)
    print_state(world, elbow, step, "Negative angle (-30°)")
    time.sleep(1.5)
    step += 1
    
    # 8. Final: demonstrate rapid stiffness change
    print(f"\n{'─'*70}")
    print("Demonstrating variable stiffness at same angle...")
    test_angle = np.pi / 4
    stiffness_levels = [0.01, 0.02, 0.03, 0.04, 0.05]
    for stiff in stiffness_levels:
        elbow.set_joint_angle(test_angle, pretension=stiff)
        print(f"  Pretension: {stiff:.3f} m → Stiffness: {elbow.stiffness:.4f}")
        time.sleep(0.3)
    step += 1
    
    # Return to rest
    print(f"\n{'─'*70}")
    print("Returning to rest position...")
    elbow.set_joint_angle(0.0, pretension=0.02)
    print_state(world, elbow, step, "Rest position")
    
    print(f"\n{'═'*70}")
    print("DEMO COMPLETE")
    print("="*70 + "\n")


def main():
    """Main demo function."""
    # Create the world with tendon arm
    world, elbow = create_tendon_arm_world()
    
    # Set up visualization if ROS2 is available
    if ROS2_AVAILABLE:
        rclpy.init()
        node = Node('tendon_demo')
        viz = VizMarkerPublisher(world=world, node=node, reference_frame="map")
        print("ROS2 visualization enabled. Use RViz2 to view the arm.")
        print("RViz2 config: Add MarkerArray on topic /semworld/viz_marker\n")
    else:
        viz = None
        print("ROS2 not available - running console-only demo.\n")
    
    try:
        # Run the simulation sequence
        run_simulation_sequence(world, elbow)
        
        # Keep alive for visualization if ROS2
        if ROS2_AVAILABLE:
            print("\nKeeping visualization alive for 5 seconds...")
            print("(You can view the final pose in RViz2)")
            time.sleep(5)
    
    finally:
        if ROS2_AVAILABLE:
            rclpy.shutdown()


if __name__ == "__main__":
    main()
