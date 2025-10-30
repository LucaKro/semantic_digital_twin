import numpy as np
import pytest

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import TendonConnection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3, TransformationMatrix


@pytest.fixture
def tendon_world():
    """
    Create a simple world with a tendon-driven joint.
    Structure: root -> base -> forearm (via TendonConnection)
    """
    world = World()
    
    base = Body(name=PrefixedName("base"))
    forearm = Body(name=PrefixedName("forearm"))
    
    with world.modify_world():
        world.add_kinematic_structure_entity(base)
        world.add_kinematic_structure_entity(forearm)
        
        tendon_joint = TendonConnection(
            parent=base,
            child=forearm,
            name=PrefixedName("elbow_tendon"),
            axis=Vector3.Z(reference_frame=base),  # Rotate around Z axis
            pulley_radius=0.03,  # 3cm moment arm
            parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(
                x=0.0, y=0.0, z=0.1,  # Joint 10cm above base
                reference_frame=base,
            )
        )
        
        world.add_connection(tendon_joint)
    
    return world, base, forearm, tendon_joint


def test_tendon_connection_instantiation(tendon_world):
    """Test that TendonConnection is properly instantiated."""
    world, base, forearm, tendon_joint = tendon_world
    
    # Check connection properties
    assert tendon_joint.parent == base
    assert tendon_joint.child == forearm
    assert tendon_joint.pulley_radius == 0.03
    
    # Check that two tendon DOFs were created
    assert tendon_joint.tendon_1 is not None
    assert tendon_joint.tendon_2 is not None
    assert tendon_joint.tendon_1.name == PrefixedName("tendon_1", "elbow_tendon")
    assert tendon_joint.tendon_2.name == PrefixedName("tendon_2", "elbow_tendon")
    
    # Check that DOFs are registered in the world
    assert world.get_degree_of_freedom_by_name(tendon_joint.tendon_1.name) is not None
    assert world.get_degree_of_freedom_by_name(tendon_joint.tendon_2.name) is not None


def test_tendon_connection_active_dofs(tendon_world):
    """Test that active_dofs returns both tendon DOFs."""
    world, base, forearm, tendon_joint = tendon_world
    
    active_dofs = tendon_joint.active_dofs
    assert len(active_dofs) == 2
    assert tendon_joint.tendon_1 in active_dofs
    assert tendon_joint.tendon_2 in active_dofs


def test_tendon_connection_initial_state(tendon_world):
    """Test that tendons start at zero position (no tension)."""
    world, base, forearm, tendon_joint = tendon_world
    
    # Initial tendon positions should be zero
    assert world.state[tendon_joint.tendon_1.name].position == 0.0
    assert world.state[tendon_joint.tendon_2.name].position == 0.0
    
    # Initial joint angle should be zero
    assert tendon_joint.joint_angle == 0.0


def test_tendon_connection_joint_angle_computation(tendon_world):
    """Test joint angle computation from tendon positions."""
    world, base, forearm, tendon_joint = tendon_world
    
    # Set tendon positions manually
    world.state[tendon_joint.tendon_1.name].position = 0.03  # 3cm pull
    world.state[tendon_joint.tendon_2.name].position = 0.0
    world.notify_state_change()
    
    # Expected angle: (0.03 - 0.0) / 0.03 = 1.0 radian
    expected_angle = 1.0
    assert np.isclose(tendon_joint.joint_angle, expected_angle, atol=1e-6)
    
    # Set opposite configuration
    world.state[tendon_joint.tendon_1.name].position = 0.0
    world.state[tendon_joint.tendon_2.name].position = 0.015  # 1.5cm pull
    world.notify_state_change()
    
    # Expected angle: (0.0 - 0.015) / 0.03 = -0.5 radian
    expected_angle = -0.5
    assert np.isclose(tendon_joint.joint_angle, expected_angle, atol=1e-6)


def test_tendon_connection_set_joint_angle(tendon_world):
    """Test setting joint angle with pretension."""
    world, base, forearm, tendon_joint = tendon_world
    
    # Set to 45 degrees (pi/4 radians) with 2cm pretension
    target_angle = np.pi / 4
    pretension = 0.02
    tendon_joint.set_joint_angle(target_angle, pretension=pretension)
    
    # Check that joint angle is correct
    assert np.isclose(tendon_joint.joint_angle, target_angle, atol=1e-6)
    
    # Check tendon positions
    # tendon_1 = pretension + (angle * radius) / 2
    # tendon_2 = pretension - (angle * radius) / 2
    displacement = target_angle * tendon_joint.pulley_radius / 2
    expected_tendon_1 = pretension + displacement
    expected_tendon_2 = pretension - displacement
    
    assert np.isclose(
        world.state[tendon_joint.tendon_1.name].position, 
        expected_tendon_1, 
        atol=1e-6
    )
    assert np.isclose(
        world.state[tendon_joint.tendon_2.name].position, 
        expected_tendon_2, 
        atol=1e-6
    )


def test_tendon_connection_set_joint_angle_zero(tendon_world):
    """Test setting joint angle to zero maintains equal tendon tension."""
    world, base, forearm, tendon_joint = tendon_world
    
    pretension = 0.03
    tendon_joint.set_joint_angle(0.0, pretension=pretension)
    
    # Both tendons should have equal tension (pretension)
    assert np.isclose(
        world.state[tendon_joint.tendon_1.name].position, 
        pretension, 
        atol=1e-6
    )
    assert np.isclose(
        world.state[tendon_joint.tendon_2.name].position, 
        pretension, 
        atol=1e-6
    )
    assert np.isclose(tendon_joint.joint_angle, 0.0, atol=1e-6)


def test_tendon_connection_stiffness(tendon_world):
    """Test stiffness computation from tendon pretension."""
    world, base, forearm, tendon_joint = tendon_world
    
    # Set low pretension
    tendon_joint.set_joint_angle(0.0, pretension=0.01)
    low_stiffness = tendon_joint.stiffness
    assert np.isclose(low_stiffness, 0.01, atol=1e-6)
    
    # Set high pretension
    tendon_joint.set_joint_angle(0.0, pretension=0.05)
    high_stiffness = tendon_joint.stiffness
    assert np.isclose(high_stiffness, 0.05, atol=1e-6)
    
    # Higher pretension should give higher stiffness
    assert high_stiffness > low_stiffness


def test_tendon_connection_stiffness_with_angle(tendon_world):
    """Test that stiffness is average of tendon tensions regardless of angle."""
    world, base, forearm, tendon_joint = tendon_world
    
    # Set angle with pretension
    tendon_joint.set_joint_angle(np.pi / 4, pretension=0.04)
    
    # Stiffness should be the average (pretension)
    expected_stiffness = 0.04
    assert np.isclose(tendon_joint.stiffness, expected_stiffness, atol=1e-6)


def test_tendon_connection_forward_kinematics(tendon_world):
    """Test that forward kinematics correctly computes pose from tendon positions."""
    world, base, forearm, tendon_joint = tendon_world
    
    # Set to 90 degrees
    target_angle = np.pi / 2
    tendon_joint.set_joint_angle(target_angle, pretension=0.02)
    
    # Compute forward kinematics
    base_T_forearm = world.compute_forward_kinematics(base, forearm)
    
    # The transformation should include rotation around Z axis by target_angle
    # and translation of 0.1m along Z (from parent_T_connection_expression)
    transform_np = base_T_forearm.to_np()
    
    # Check translation (should be at z=0.1)
    assert np.isclose(transform_np[2, 3], 0.1, atol=1e-6)
    
    # Check rotation (should be 90 degrees around Z)
    # For 90 degrees around Z: cos(90)=0, sin(90)=1
    # Rotation matrix should be approximately:
    # [0 -1  0]
    # [1  0  0]
    # [0  0  1]
    rotation_part = transform_np[:3, :3]
    expected_rotation = np.array([
        [np.cos(target_angle), -np.sin(target_angle), 0],
        [np.sin(target_angle), np.cos(target_angle), 0],
        [0, 0, 1]
    ])
    assert np.allclose(rotation_part, expected_rotation, atol=1e-6)


def test_tendon_connection_multiple_angles(tendon_world):
    """Test setting multiple different angles in sequence."""
    world, base, forearm, tendon_joint = tendon_world
    
    test_angles = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, -np.pi / 4]
    pretension = 0.025
    
    for target_angle in test_angles:
        tendon_joint.set_joint_angle(target_angle, pretension=pretension)
        
        # Verify angle is correct
        assert np.isclose(tendon_joint.joint_angle, target_angle, atol=1e-6)
        
        # Verify pretension is maintained (average of tendons)
        assert np.isclose(tendon_joint.stiffness, pretension, atol=1e-6)


def test_tendon_connection_dof_limits(tendon_world):
    """Test that tendon DOFs have appropriate limits."""
    world, base, forearm, tendon_joint = tendon_world
    
    # Check position limits
    assert tendon_joint.tendon_1.lower_limits.position == 0.0
    assert tendon_joint.tendon_1.upper_limits.position == 0.1
    assert tendon_joint.tendon_2.lower_limits.position == 0.0
    assert tendon_joint.tendon_2.upper_limits.position == 0.1
    
    # Check velocity limits
    assert tendon_joint.tendon_1.lower_limits.velocity == -0.05
    assert tendon_joint.tendon_1.upper_limits.velocity == 0.05
    assert tendon_joint.tendon_2.lower_limits.velocity == -0.05
    assert tendon_joint.tendon_2.upper_limits.velocity == 0.05


def test_tendon_connection_independent_tendon_control(tendon_world):
    """Test that tendons can be controlled independently."""
    world, base, forearm, tendon_joint = tendon_world
    
    # Set different velocities for each tendon
    world.state[tendon_joint.tendon_1.name].velocity = 0.02
    world.state[tendon_joint.tendon_2.name].velocity = -0.01
    
    # Verify independent control
    assert world.state[tendon_joint.tendon_1.name].velocity == 0.02
    assert world.state[tendon_joint.tendon_2.name].velocity == -0.01


def test_tendon_connection_hash(tendon_world):
    """Test that TendonConnection hash is based on parent and child."""
    world, base, forearm, tendon_joint = tendon_world
    
    # Hash should be based on parent and child
    expected_hash = hash((base, forearm))
    assert hash(tendon_joint) == expected_hash


def test_tendon_connection_without_world_fails():
    """Test that creating TendonConnection without world and without DOF names fails."""
    base = Body(name=PrefixedName("base"))
    forearm = Body(name=PrefixedName("forearm"))
    
    with pytest.raises(ValueError, match="cannot be created without a world"):
        tendon_joint = TendonConnection(
            parent=base,
            child=forearm,
            axis=Vector3.Z(reference_frame=base),
            pulley_radius=0.03,
        )
        tendon_joint._post_init_world_part()
