#!/usr/bin/env python3
"""
Multi-Joint Tendon-Driven Finger Demo
======================================

This demo showcases a tendon-driven robotic finger with MULTIPLE JOINTS controlled
by the SAME TENDONS, mimicking biological finger mechanics.

Key features:
- 3 phalanges (proximal, middle, distal) connected by 3 joints
- Shared flexor and extensor tendons routing through all joints
- Coordinated motion when pulling a single tendon
- Different pulley radii at each joint for realistic coupling
- Variable stiffness control
- Visualization using ROS2 (if available)

This demonstrates the power of tendon routing: one actuator (tendon) can control
multiple degrees of freedom through clever mechanical design.

Run with: python examples/finger_tendon_demo.py
"""

import time
import numpy as np

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import TendonConnection, FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.geometry import Box, Cylinder, Sphere, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3, TransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap

# Try to import ROS2 visualization
try:
    import rclpy
    from rclpy.node import Node
    from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 not available. Running in console-only mode.")


def create_finger_world():
    """
    Create a world with a robotic finger featuring 3 joints controlled by shared tendons.
    
    Structure:
        palm -> proximal_phalanx -> middle_phalanx -> distal_phalanx
        
    Tendons:
        - flexor_tendon: passes through all 3 joints, pulling causes flexion
        - extensor_tendon: passes through all 3 joints, pulling causes extension
        
    Each joint has a different pulley radius, creating natural coupling where
    distal joints bend more than proximal joints (like real fingers).
    """
    world = World()
    
    # Dimensions (realistic human finger proportions, scaled to decimeters)
    palm_height = 0.15
    proximal_length = 0.12
    middle_length = 0.08
    distal_length = 0.06
    phalanx_width = 0.03
    
    # Create palm (fixed base)
    palm = Body(
        name=PrefixedName("palm"),
        collision=ShapeCollection([
            Box(
                origin=TransformationMatrix.from_xyz_rpy(z=palm_height/2),
                scale=Scale(0.08, 0.06, palm_height),
            )
        ])
    )
    
    # Create proximal phalanx (first segment)
    proximal_phalanx = Body(
        name=PrefixedName("proximal_phalanx"),
        collision=ShapeCollection([
            Box(
                origin=TransformationMatrix.from_xyz_rpy(z=proximal_length/2),
                scale=Scale(phalanx_width, phalanx_width, proximal_length),
            )
        ])
    )
    
    # Create middle phalanx (second segment)
    middle_phalanx = Body(
        name=PrefixedName("middle_phalanx"),
        collision=ShapeCollection([
            Box(
                origin=TransformationMatrix.from_xyz_rpy(z=middle_length/2),
                scale=Scale(phalanx_width * 0.9, phalanx_width * 0.9, middle_length),
            )
        ])
    )
    
    # Create distal phalanx (fingertip segment)
    distal_phalanx = Body(
        name=PrefixedName("distal_phalanx"),
        collision=ShapeCollection([
            Box(
                origin=TransformationMatrix.from_xyz_rpy(z=distal_length/2),
                scale=Scale(phalanx_width * 0.8, phalanx_width * 0.8, distal_length),
            ),
            # Fingertip sphere
            Sphere(
                origin=TransformationMatrix.from_xyz_rpy(z=distal_length),
                radius=phalanx_width * 0.5,
            )
        ])
    )
    
    with world.modify_world():
        # Add bodies to world
        world.add_kinematic_structure_entity(palm)
        world.add_kinematic_structure_entity(proximal_phalanx)
        world.add_kinematic_structure_entity(middle_phalanx)
        world.add_kinematic_structure_entity(distal_phalanx)
        
        # Create SHARED tendon DOFs
        # These will be used by ALL three joints
        lower_limits = DerivativeMap()
        lower_limits.position = 0.0  # Tendons can't push
        lower_limits.velocity = -0.1  # Max relaxation rate (m/s)
        
        upper_limits = DerivativeMap()
        upper_limits.position = 0.15  # Max tendon displacement (m) - longer for finger
        upper_limits.velocity = 0.1  # Max contraction rate (m/s)
        
        # Flexor tendon (causes finger to curl inward)
        flexor_tendon = DegreeOfFreedom(
            name=PrefixedName("flexor_tendon"),
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )
        world.add_degree_of_freedom(flexor_tendon)
        
        # Extensor tendon (causes finger to straighten/extend)
        extensor_tendon = DegreeOfFreedom(
            name=PrefixedName("extensor_tendon"),
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )
        world.add_degree_of_freedom(extensor_tendon)
        
        # Pulley radii: smaller radius = more angular displacement per tendon displacement
        # Make distal joints more sensitive (smaller radius) for natural finger motion
        mcp_radius = 0.020  # Metacarpophalangeal joint (palm-proximal)
        pip_radius = 0.015  # Proximal interphalangeal joint (proximal-middle)
        dip_radius = 0.012  # Distal interphalangeal joint (middle-distal)
        
        # Joint 1: MCP (MetaCarpoPhalangeal) - palm to proximal phalanx
        mcp_joint = TendonConnection(
            parent=palm,
            child=proximal_phalanx,
            name=PrefixedName("mcp_joint"),
            axis=Vector3.Y(reference_frame=palm),  # Rotate around Y axis
            pulley_radius=mcp_radius,
            tendon_1_name=flexor_tendon.name,  # Shared flexor
            tendon_2_name=extensor_tendon.name,  # Shared extensor
            parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(
                z=palm_height,  # At top of palm
                reference_frame=palm,
            )
        )
        world.add_connection(mcp_joint)
        
        # Joint 2: PIP (Proximal InterPhalangeal) - proximal to middle phalanx
        pip_joint = TendonConnection(
            parent=proximal_phalanx,
            child=middle_phalanx,
            name=PrefixedName("pip_joint"),
            axis=Vector3.Y(reference_frame=proximal_phalanx),
            pulley_radius=pip_radius,
            tendon_1_name=flexor_tendon.name,  # SAME flexor tendon
            tendon_2_name=extensor_tendon.name,  # SAME extensor tendon
            parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(
                z=proximal_length,  # At top of proximal phalanx
                reference_frame=proximal_phalanx,
            )
        )
        world.add_connection(pip_joint)
        
        # Joint 3: DIP (Distal InterPhalangeal) - middle to distal phalanx
        dip_joint = TendonConnection(
            parent=middle_phalanx,
            child=distal_phalanx,
            name=PrefixedName("dip_joint"),
            axis=Vector3.Y(reference_frame=middle_phalanx),
            pulley_radius=dip_radius,
            tendon_1_name=flexor_tendon.name,  # SAME flexor tendon
            tendon_2_name=extensor_tendon.name,  # SAME extensor tendon
            parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(
                z=middle_length,  # At top of middle phalanx
                reference_frame=middle_phalanx,
            )
        )
        world.add_connection(dip_joint)
    
    return world, mcp_joint, pip_joint, dip_joint, flexor_tendon, extensor_tendon


def print_demo_header():
    """Print demo header with information."""
    print("\n" + "="*80)
    print("MULTI-JOINT TENDON-DRIVEN FINGER DEMO")
    print("="*80)
    print("\nThis demo showcases a robotic finger with 3 joints controlled by SHARED tendons.")
    print("\nBiomimetic design:")
    print("  • Single flexor tendon routes through all 3 joints")
    print("  • Single extensor tendon routes through all 3 joints")
    print("  • Pulling one tendon causes coordinated motion of ALL joints")
    print("  • Different pulley radii create natural coupling (distal bends more)")
    print("\nJoints:")
    print("  • MCP (MetaCarpoPhalangeal): palm → proximal phalanx")
    print("  • PIP (Proximal InterPhalangeal): proximal → middle phalanx")
    print("  • DIP (Distal InterPhalangeal): middle → distal phalanx")
    print("="*80 + "\n")


def print_state(world, mcp, pip, dip, flexor, extensor, step, description):
    """Print current state of all joints and tendons."""
    flexor_pos = world.state[flexor.name].position
    extensor_pos = world.state[extensor.name].position
    
    mcp_angle = np.degrees(mcp.joint_angle)
    pip_angle = np.degrees(pip.joint_angle)
    dip_angle = np.degrees(dip.joint_angle)
    
    print(f"\n[Step {step}] {description}")
    print(f"  Flexor tendon:   {flexor_pos:.4f} m")
    print(f"  Extensor tendon: {extensor_pos:.4f} m")
    print(f"  ┌─ MCP angle:  {mcp_angle:6.1f}° (r={mcp.pulley_radius*1000:.0f}mm)")
    print(f"  ├─ PIP angle:  {pip_angle:6.1f}° (r={pip.pulley_radius*1000:.0f}mm)")
    print(f"  └─ DIP angle:  {dip_angle:6.1f}° (r={dip.pulley_radius*1000:.0f}mm)")
    print(f"  Total flexion: {mcp_angle + pip_angle + dip_angle:6.1f}°")


def set_tendon_positions(world, flexor, extensor, flexor_pos, extensor_pos):
    """
    Set tendon positions directly.
    This simulates pulling on the tendons.
    """
    world.state[flexor.name].position = flexor_pos
    world.state[extensor.name].position = extensor_pos
    world.notify_state_change()


def run_simulation_sequence(world, mcp, pip, dip, flexor, extensor):
    """
    Run a simulation sequence demonstrating shared tendon control.
    """
    print_demo_header()
    
    step = 1
    
    # 1. Initial configuration - straight finger
    print_state(world, mcp, pip, dip, flexor, extensor, step, 
                "Initial configuration (finger extended)")
    time.sleep(2.0)
    step += 1
    
    # 2. Pull flexor tendon slightly - all joints flex together
    print(f"\n{'─'*80}")
    print("Pulling FLEXOR tendon by 2cm → All joints flex together...")
    set_tendon_positions(world, flexor, extensor, 0.02, 0.01)
    print_state(world, mcp, pip, dip, flexor, extensor, step,
                "Slight flexion - coordinated motion")
    time.sleep(2.0)
    step += 1
    
    # 3. Pull flexor more - observe different angles due to different radii
    print(f"\n{'─'*80}")
    print("Pulling flexor MORE (4cm) → Notice DIP bends most (smallest radius)...")
    set_tendon_positions(world, flexor, extensor, 0.04, 0.01)
    print_state(world, mcp, pip, dip, flexor, extensor, step,
                "Medium flexion - DIP bends more than MCP")
    time.sleep(2.0)
    step += 1
    
    # 4. Full flexion - make a fist
    print(f"\n{'─'*80}")
    print("FULL FLEXION - making a fist...")
    set_tendon_positions(world, flexor, extensor, 0.08, 0.01)
    print_state(world, mcp, pip, dip, flexor, extensor, step,
                "Full fist - maximum flexion")
    time.sleep(2.0)
    step += 1
    
    # 5. Release flexor, pull extensor - finger extends
    print(f"\n{'─'*80}")
    print("Releasing flexor, pulling EXTENSOR → Finger extends...")
    set_tendon_positions(world, flexor, extensor, 0.01, 0.03)
    print_state(world, mcp, pip, dip, flexor, extensor, step,
                "Extension - negative angles")
    time.sleep(2.0)
    step += 1
    
    # 6. Smooth flexion motion
    print(f"\n{'─'*80}")
    print("Smooth flexion motion - gradually pulling flexor tendon...")
    flexor_positions = np.linspace(0.01, 0.10, 8)
    for i, flex_pos in enumerate(flexor_positions):
        set_tendon_positions(world, flexor, extensor, flex_pos, 0.01)
        if i % 2 == 0:
            print_state(world, mcp, pip, dip, flexor, extensor, step,
                       f"Flexing... ({i+1}/8)")
        time.sleep(0.4)
        step += 1
    
    # 7. Smooth extension motion
    print(f"\n{'─'*80}")
    print("Smooth extension motion - gradually pulling extensor tendon...")
    extensor_positions = np.linspace(0.01, 0.06, 6)
    for i, ext_pos in enumerate(extensor_positions):
        set_tendon_positions(world, flexor, extensor, 0.01, ext_pos)
        if i % 2 == 1:
            print_state(world, mcp, pip, dip, flexor, extensor, step,
                       f"Extending... ({i+1}/6)")
        time.sleep(0.4)
        step += 1
    
    # 8. Demonstrate variable stiffness
    print(f"\n{'─'*80}")
    print("Demonstrating variable STIFFNESS (co-contraction)...")
    print("Both tendons pulled simultaneously → high stiffness, same angle")
    
    # Low stiffness (low pretension)
    set_tendon_positions(world, flexor, extensor, 0.03, 0.01)
    mcp_stiff_low = mcp.stiffness
    print_state(world, mcp, pip, dip, flexor, extensor, step,
                f"Low stiffness (pretension={mcp_stiff_low:.4f})")
    time.sleep(1.5)
    step += 1
    
    # High stiffness (high pretension) - same angle but more tension
    set_tendon_positions(world, flexor, extensor, 0.05, 0.03)
    mcp_stiff_high = mcp.stiffness
    print_state(world, mcp, pip, dip, flexor, extensor, step,
                f"High stiffness (pretension={mcp_stiff_high:.4f}) - same angle!")
    time.sleep(1.5)
    step += 1
    
    # 9. Return to neutral
    print(f"\n{'─'*80}")
    print("Returning to neutral position...")
    set_tendon_positions(world, flexor, extensor, 0.02, 0.02)
    print_state(world, mcp, pip, dip, flexor, extensor, step,
                "Neutral position - balanced tension")
    time.sleep(1.5)
    
    print(f"\n{'═'*80}")
    print("DEMO COMPLETE")
    print("\nKey observations:")
    print("  ✓ Single tendon controls multiple joints simultaneously")
    print("  ✓ Different pulley radii create natural coupling")
    print("  ✓ Distal joints bend more than proximal (realistic finger motion)")
    print("  ✓ Antagonistic control enables position and stiffness control")
    print("="*80 + "\n")


def main():
    """Main demo function."""
    # Create the world with finger
    world, mcp, pip, dip, flexor, extensor = create_finger_world()
    
    # Set up visualization if ROS2 is available
    if ROS2_AVAILABLE:
        rclpy.init()
        node = Node('finger_tendon_demo')
        viz = VizMarkerPublisher(world=world, node=node, reference_frame="map")
        print("ROS2 visualization enabled. Use RViz2 to view the finger.")
        print("RViz2 config: Add MarkerArray on topic /semworld/viz_marker\n")
    else:
        viz = None
        print("ROS2 not available - running console-only demo.\n")
    
    try:
        # Run the simulation sequence
        run_simulation_sequence(world, mcp, pip, dip, flexor, extensor)
        
        # Keep alive for visualization if ROS2
        if ROS2_AVAILABLE:
            print("\nKeeping visualization alive for 10 seconds...")
            print("(You can view the finger motion in RViz2)")
            time.sleep(10)
    
    finally:
        if ROS2_AVAILABLE:
            rclpy.shutdown()


if __name__ == "__main__":
    main()
