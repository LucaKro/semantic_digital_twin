from collections import defaultdict
from dataclasses import dataclass, field
from typing import Self

from .abstract_robot import (
    AbstractRobot,
    Arm,
    Neck,
    Finger,
    ParallelGripper,
    Camera,
    Torso,
    FieldOfView,
)
from .robot_mixins import HasNeck, HasArms
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World


@dataclass
class MMPDresden(AbstractRobot, HasArms, HasNeck):
    """
    Class that describes the Human Support Robot variant B (https://upmroboticclub.wordpress.com/robot/).
    """

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

    def load_srdf(self):
        """
        Loads the SRDF file for the PR2 robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates an mmp_dresden (Mobile Manipulation Platform) semantic annotation from a World that was parsed from
        resources/urdf/mmp_dresden.urdf. Assumes all URDF link names exist in the world.
        """
        with world.modify_world():
            mmp_dresden = cls(
                name=PrefixedName("mmp_dresden", prefix=world.name),
                root=world.get_body_by_name("base_link"),
                _world=world,
            )

            gripper_thumb = Finger(
                name=PrefixedName("thumb", prefix=mmp_dresden.name.name),
                root=world.get_body_by_name("arm_0_gripper_robotiq_85_left_knuckle_link"),
                tip=world.get_body_by_name("arm_0_gripper_robotiq_85_left_finger_tip_link"),
                _world=world,
            )

            gripper_finger = Finger(
                name=PrefixedName("finger", prefix=mmp_dresden.name.name),
                root=world.get_body_by_name("arm_0_gripper_robotiq_85_right_knuckle_link"),
                tip=world.get_body_by_name("arm_0_gripper_robotiq_85_right_finger_tip_link"),
                _world=world,
            )

            gripper = ParallelGripper(
                name=PrefixedName("gripper", prefix=mmp_dresden.name.name),
                root=world.get_body_by_name("arm_0_flange"),
                tool_frame=world.get_body_by_name("arm_0_end_effector_link"),
                thumb=gripper_thumb,
                finger=gripper_finger,
                front_facing_axis=Vector3(0, 0, 1), # doublecheck
                front_facing_orientation=Quaternion(-1, 0, -1, 0), # doublecheck
                _world=world,
            )

            arm = Arm(
                name=PrefixedName("arm", prefix=mmp_dresden.name.name),
                root=world.get_body_by_name("hub_holder_link"),
                tip=world.get_body_by_name("arm_0_flange"),
                manipulator=gripper,
                _world=world,
            )
            mmp_dresden.add_arm(arm)

            camera = Camera(
                name=PrefixedName("pan_and_tilt_camera_link", prefix=mmp_dresden.name.name),
                root=world.get_body_by_name("pan_and_tilt_camera_link"),
                forward_facing_axis=Vector3(0, 0, 1),
                field_of_view=FieldOfView(horizontal_angle=1.047, vertical_angle=0.785),
                minimal_height=0.8,
                maximal_height=1.7,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=mmp_dresden.name.name),
                sensors={
                    camera,
                },
                root=world.get_body_by_name("hub_holder_link"),
                tip=world.get_body_by_name("pan_and_tilt_camera_link"),
                _world=world,
            )
            mmp_dresden.add_neck(neck)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=mmp_dresden.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("hub_holder_link"),
                _world=world,
            )
            mmp_dresden.add_torso(torso)

            world.add_semantic_annotation(mmp_dresden, skip_duplicates=True)

            vel_limits = defaultdict(lambda: 1)
            mmp_dresden.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

        return mmp_dresden
