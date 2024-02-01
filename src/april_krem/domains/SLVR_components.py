from typing import Tuple
from enum import Enum

import rospy
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from april_msgs.srv import (
    GetGraspStrategy,
    ObjectsEstimatedPosesSrv,
    ObjectsEstimatedPosesSrvRequest,
    ObjectsEstimatedPosesSrvResponse,
)

from april_krem.plan_dispatcher import PlanDispatcher


class Item(Enum):
    nothing = "nothing"
    cable = "cable"
    cover = "cover"
    propeller = "propeller"


class ArmPose(Enum):
    unknown = "unknown"
    home = "home"
    arm_up = "arm_up"
    over_reject_box = "over_reject_box"
    over_cable_dispenser = "over_cable_dispenser"
    over_cable_station = "over_cable_station"
    soldering_pose = "soldering_pose"

    over_feeding_conveyor = "over_feeding_conveyor"

    cover_transition_pose = "cover_transition_pose"
    over_cover_station = "over_cover_station"

    propeller_transition_pose = "propeller_transition_pose"
    over_propeller_station = "over_propeller_station"


class Status(Enum):
    ok = "ok"
    nok = "nok"


class Color(Enum):
    red = "UC5_cable_soldering_red"
    blue = "UC5_cable_soldering_blue"
    brown = "UC5_cable_soldering_brown"
    white = "UC5_cable_soldering_white"


class Epic(Enum):
    epic2 = "epic2"
    epic3 = "epic3"
    epic4 = "epic4"


class Environment:
    def __init__(self, krem_logging, use_case: str = "uc5"):
        self._krem_logging = krem_logging

        self._use_case = use_case

        if self._use_case == "uc5_3":
            self.epic_done = {
                Epic.epic2: True,
                Epic.epic3: False,
                Epic.epic4: False,
            }
        elif self._use_case == "uc5_4":
            self.epic_done = {
                Epic.epic2: True,
                Epic.epic3: True,
                Epic.epic4: False,
            }
        else:
            self.epic_done = {
                Epic.epic2: False,
                Epic.epic3: False,
                Epic.epic4: False,
            }
        self.current_epic = None

        self.holding_item = Item.nothing
        self.arm_pose = ArmPose.unknown

        self.cables_available = {
            Color.red: False,
            Color.blue: False,
            Color.brown: False,
            Color.white: False,
        }
        self.cables_to_be_soldered = [Color.red, Color.blue, Color.brown, Color.white]
        self.current_color = Color.red

        self.item_perceived = {
            Item.cable: False,
            Item.cover: False,
            Item.propeller: False,
        }

        self.item_status = {
            Item.cable: None,
            Item.cover: None,
            Item.propeller: None,
        }

        self.cable_pose = False
        self.cover_pose = False
        self.cover_leveled = False
        self.cover_assembled = False
        self.propeller_assembled = False

        self.pallet_available = False
        self.cover_available = False
        self.propeller_available = False

        self.reject_box_status = {
            Item.cable: 0,
            Item.cover: 0,
            Item.propeller: 0,
        }

        self._perceived_objects = {}

        # Store objects with ID received from HICEM via Service
        rospy.Service(
            "/hicem/sfg/objects_estimated_poses",
            ObjectsEstimatedPosesSrv,
            self._object_poses_srv,
        )

        # Set inspection result service
        rospy.Service(
            "/hicem/sfg/quality_control/result",
            SetBool,
            self._inspection_result_srv,
        )

    def reset_env(self) -> None:
        self.holding_item = Item.nothing
        self.arm_pose = ArmPose.unknown

        self.cables_available = {
            Color.red: False,
            Color.blue: False,
            Color.brown: False,
            Color.white: False,
        }
        self.cables_to_be_soldered = [Color.red, Color.blue, Color.brown, Color.white]
        self.current_color = Color.red

        self.item_perceived = {
            Item.cable: False,
            Item.cover: False,
            Item.propeller: False,
        }

        self.item_status = {
            Item.cable: None,
            Item.cover: None,
            Item.propeller: None,
        }
        self.cable_pose = False
        self.cover_pose = False
        self.cover_leveled = False
        self.cover_assembled = False
        self.propeller_assembled = False

        self.pallet_available = False
        self.cover_available = False
        self.propeller_available = False

        self.reject_box_status = {
            Item.cable: 0,
            Item.cover: 0,
            Item.propeller: 0,
        }

        self.item_finished = {
            Item.cable: False,
            Item.cover: False,
            Item.propeller: False,
        }

        self._perceived_objects.clear()

    def reset_env_keep_counters(self) -> None:
        self.holding_item = Item.nothing
        self.arm_pose = ArmPose.unknown

        self.cables_available = {
            Color.red: False,
            Color.blue: False,
            Color.brown: False,
            Color.white: False,
        }

        self.item_perceived = {
            Item.cable: False,
            Item.cover: False,
            Item.propeller: False,
        }

        self.item_status = {
            Item.cable: None,
            Item.cover: None,
            Item.propeller: None,
        }
        self.cable_pose = False
        self.cover_pose = False
        self.cover_leveled = False
        self.cover_assembled = False
        self.propeller_assembled = False

        self.pallet_available = False
        self.cover_available = False
        self.propeller_available = False

        self.item_finished = {
            Item.cable: False,
            Item.cover: False,
            Item.propeller: False,
        }

        self._perceived_objects.clear()

    def _object_poses_srv(
        self, request: ObjectsEstimatedPosesSrvRequest
    ) -> ObjectsEstimatedPosesSrvResponse:
        for pose in request.objects_estimated_poses.objects_estimated_poses:
            self._perceived_objects[
                f"{pose.class_name}_{str(pose.tracked_id)}"
            ] = pose.bounding_box_3d

        success = True if len(self._perceived_objects) > 0 else False
        message = (
            "Success"
            if len(self._perceived_objects) > 0
            else "Received poses are empty!"
        )

        return ObjectsEstimatedPosesSrvResponse(success=success, message=message)

    def _get_item_type_and_id(self, item: str) -> Tuple[str, int]:
        # TODO return closest item
        smallest_class = None
        smallest_id = None
        items_to_receive = [item]
        for k in self._perceived_objects.keys():
            for i in items_to_receive:
                if i in k:
                    class_name, id = k.rsplit("_", 1)
                    if smallest_id is None or int(id) < smallest_id:
                        smallest_id = int(id)
                        smallest_class = class_name
        return smallest_class, smallest_id

    def _clear_item_type(self, item: str) -> None:
        items_to_remove = [item]
        for k in list(self._perceived_objects.keys()):
            for i in items_to_remove:
                if i in k:
                    del self._perceived_objects[k]
                    break

    def _inspection_result_srv(self, request: SetBoolRequest) -> SetBoolResponse:
        if not hasattr(request, "data"):
            return SetBoolResponse(success=False)
        if request.data:
            if self.holding_item is not None:
                self.item_status[self.holding_item] = Status.ok
            else:
                return SetBoolResponse(success=False)
        else:
            if self.holding_item is not None:
                self.item_status[self.holding_item] = Status.nok
            else:
                return SetBoolResponse(success=False)
        return SetBoolResponse(success=True)

    def epic_active(self, epic: Epic) -> bool:
        return self.current_epic == epic if self.current_epic is not None else False

    def epic_complete(self, epic: Epic) -> bool:
        return self.epic_done[epic]

    def holding(self, item: Item) -> bool:
        return self.holding_item == item

    def current_arm_pose(self, arm_pose: ArmPose) -> bool:
        return arm_pose == self.arm_pose

    def current_cable_color(self, color: Color) -> bool:
        return self.current_color == color

    def perceived_item(self, item: Item) -> bool:
        return self.item_perceived[item] if item != Item.nothing else False

    def item_status_known(self, item: Item) -> bool:
        return self.item_status[item] is not None if item != Item.nothing else False

    def cable_color_available(self, color: Color) -> bool:
        return self.cables_available[color]

    def cable_soldered(self, color: Color) -> bool:
        return color not in self.cables_to_be_soldered

    def status_of_item(self, item: Item, status: Status) -> bool:
        return (
            self.item_status[item] == status
            if item != Item.nothing and self.item_status[item] is not None
            else False
        )

    def space_in_reject_box(self) -> bool:
        return (
            self.reject_box_status[Item.cable] < 4
            and self.reject_box_status[Item.cover] < 1
            and self.reject_box_status[Item.propeller] < 1
        )

    def pallet_is_available(self) -> bool:
        return self.pallet_available

    def propeller_is_available(self) -> bool:
        return self.propeller_available

    def cover_is_available(self) -> bool:
        return self.cover_available

    def cable_pose_known(self) -> bool:
        return self.cable_pose

    def cover_pose_known(self) -> bool:
        return self.cover_pose

    def cover_is_leveled(self) -> bool:
        return self.cover_leveled

    def cover_is_assembled(self) -> bool:
        return self.cover_assembled

    def propeller_is_assembled(self) -> bool:
        return self.propeller_assembled


class Actions:
    def __init__(self, env: Environment):
        self._env = env

        # Grasp Library
        rospy.loginfo("Waiting for Grasp Library Service...")
        rospy.wait_for_service("/krem/grasp_library")
        self._grasp_library_srv = rospy.ServiceProxy(
            "/krem/grasp_library", GetGraspStrategy
        )
        rospy.loginfo("Grasp Library Service found!")

        self._non_robot_actions_timeout = rospy.get_param(
            "~non_robot_actions_timeout", default="20"
        )
        self._robot_actions_timeout = rospy.get_param(
            "~robot_actions_timeout", default="120"
        )

    def switch_to_epic(self, epic: Epic):
        if epic == Epic.epic2:
            result, msg = PlanDispatcher.run_symbolic_action(
                "switch_to_epic2",
                timeout=0.0,
            )
        elif epic == Epic.epic3:
            result, msg = PlanDispatcher.run_symbolic_action(
                "switch_to_epic3",
                timeout=0.0,
            )
        elif epic == Epic.epic4:
            result, msg = PlanDispatcher.run_symbolic_action(
                "switch_to_epic4",
                timeout=0.0,
            )
        else:
            result, msg = False, "failed"

        if result:
            self._env.current_epic = epic

        return result, msg

    def perceive_item(self, item: Item):
        result, msg = False, "failed"
        self._env._clear_item_type(item.value)
        if item == Item.cable:
            result, msg = PlanDispatcher.run_symbolic_action(
                "perceive_cable_soldering",
                timeout=self._non_robot_actions_timeout,
            )

            if result:
                for cable_color in Color:
                    type, _ = self._env._get_item_type_and_id(cable_color.value)
                    self._env.cables_available[cable_color] = type is not None
                if self._env.cables_available[self._env.current_color]:
                    self._env.item_perceived[Item.cable] = True

        elif item == Item.cover:
            result, msg = PlanDispatcher.run_symbolic_action(
                "perceive_external_cover",
                timeout=self._non_robot_actions_timeout,
            )

            if result:
                type, _ = self._env._get_item_type_and_id("cover")
                self._env.cover_available = type is not None

                if self._env.cover_available:
                    self._env.item_perceived[Item.cover] = True

        elif item == Item.propeller:
            result, msg = PlanDispatcher.run_symbolic_action(
                "perceive_propeller",
                timeout=self._non_robot_actions_timeout,
            )

            if result:
                type, _ = self._env._get_item_type_and_id("propeller")
                self._env.propeller_available = type is not None

                if self._env.propeller_available:
                    self._env.item_perceived[Item.propeller] = True

        return result, msg

    def get_next_pallet(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "get_next_pallet",
            ["Waiting for new engine."],
            timeout=0.0,
        )
        if result:
            self._env.pallet_available = True

        return result, msg

    def get_next_cables(self, color: Color):
        result, msg = PlanDispatcher.run_symbolic_action(
            "get_next_cables",
            ["Waiting for new cables."],
            timeout=0.0,
        )
        if result:
            self._env.item_perceived[Item.cable] = False
            self._env.cables_available = dict.fromkeys(self._env.cables_available, True)

        return result, msg

    def get_next_cover(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "get_next_external_cover",
            ["Waiting for new external cover."],
            timeout=0.0,
        )
        if result:
            self._env.item_perceived[Item.cover] = False
            self._env.cover_available = True

        return result, msg

    def get_next_propeller(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "get_next_propeller",
            ["Waiting for new propeller."],
            timeout=0.0,
        )
        if result:
            self._env.item_perceived[Item.propeller] = False
            self._env.propeller_available = True

        return result, msg

    def move_arm(self, arm_pose: ArmPose):
        result = False
        msg = "failed"
        if arm_pose == ArmPose.over_cable_dispenser:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_cable_soldering_dispenser",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.over_cable_station:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_cable_soldering_station",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.soldering_pose:
            if self._env.current_color is not None:
                result, msg = PlanDispatcher.run_symbolic_action(
                    "move_to_cable_soldering_pose",
                    [self._env.current_color.value],
                    timeout=self._robot_actions_timeout,
                )
            else:
                result, msg = False, "failed"
        elif arm_pose == ArmPose.over_feeding_conveyor:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_feeding_conveyor",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.cover_transition_pose:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_to_external_cover_transition_pose",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.over_cover_station:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_external_cover_assembly_station",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.propeller_transition_pose:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_to_propeller_transition_pose",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.over_propeller_station:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_propeller_assembly_station",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.over_reject_box:
            if self._env.holding_item == Item.cable:
                result, msg = PlanDispatcher.run_symbolic_action(
                    "move_over_cable_soldering_reject_box",
                    timeout=self._robot_actions_timeout,
                )
            elif self._env.holding_item == Item.cover:
                result, msg = PlanDispatcher.run_symbolic_action(
                    "move_over_external_cover_reject_box",
                    timeout=self._robot_actions_timeout,
                )
            elif self._env.holding_item == Item.propeller:
                result, msg = PlanDispatcher.run_symbolic_action(
                    "move_over_propeller_reject_box",
                    timeout=self._robot_actions_timeout,
                )
            else:
                result, msg = False, "failed"
        elif arm_pose == ArmPose.home:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_to_home_pose",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.arm_up:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_arm_up",
                timeout=self._robot_actions_timeout,
            )
        if result:
            self._env.arm_pose = arm_pose
        return result, msg

    def pick_cable(self, cable: Item, color: Color):
        # arguments: [ID of cable]
        class_name, id = self._env._get_item_type_and_id(self._env.current_color.value)
        if class_name is not None:
            class_name_without_color = class_name.rsplit("_", 1)[0]
            grasp_facts = self._grasp_library_srv(
                "mia", class_name_without_color, "soldering", False
            )
            result, msg = PlanDispatcher.run_symbolic_action(
                "pick_cable_soldering",
                [str(id)],
                grasp_facts.grasp_strategies,
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = Item.cable
                self._env.arm_pose = ArmPose.unknown
                self._env.item_perceived[Item.cable] = False
                self._env.cables_available[self._env.current_color] = False
            return result, msg
        return False, "failed"

    def pick_cover(self, cover: Item):
        # arguments: [ID of cover]
        class_name, id = self._env._get_item_type_and_id(cover.value)
        if class_name is not None:
            grasp_facts = self._grasp_library_srv("mia", class_name, "assembly", False)
            result, msg = PlanDispatcher.run_symbolic_action(
                "pick_external_cover",
                [str(id)],
                grasp_facts.grasp_strategies,
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = cover
                self._env.arm_pose = ArmPose.unknown
                self._env.item_perceived[cover] = False
                self._env.cover_available = False
            return result, msg
        return False, "failed"

    def pick_propeller(self, propeller: Item):
        # arguments: [ID of propeller]
        class_name, id = self._env._get_item_type_and_id(propeller.value)
        if class_name is not None:
            grasp_facts = self._grasp_library_srv("mia", class_name, "assembly", False)
            result, msg = PlanDispatcher.run_symbolic_action(
                "pick_propeller",
                [str(id)],
                grasp_facts.grasp_strategies,
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = propeller
                self._env.arm_pose = ArmPose.unknown
                self._env.item_perceived[propeller] = False
                self._env.propeller_available = False
            return result, msg
        return False, "failed"

    def get_cable_pose(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "get_cable_soldering_pose",
            [],
            timeout=self._robot_actions_timeout,
        )
        if result:
            self._env.cable_pose = True
            self._env.arm_pose = ArmPose.unknown

        return result, msg

    def get_cover_pose(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "get_external_cover_poses",
            [],
            timeout=self._robot_actions_timeout,
        )
        if result:
            self._env.cover_pose = True
            self._env.arm_pose = ArmPose.unknown

        return result, msg

    def level_cover(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "level_external_cover",
            [],
            timeout=self._robot_actions_timeout,
        )
        if result:
            self._env.cover_leveled = True
            self._env.arm_pose = ArmPose.unknown

        return result, msg

    def inspect(self, item: Item):
        if item == Item.cable:
            result, msg = PlanDispatcher.run_symbolic_action(
                "inspect_cable_soldering", timeout=0.0
            )
        elif item == Item.cover:
            result, msg = PlanDispatcher.run_symbolic_action(
                "inspect_external_cover", timeout=0.0
            )
        elif item == Item.propeller:
            result, msg = PlanDispatcher.run_symbolic_action(
                "inspect_propeller", timeout=0.0
            )

        if not self._env.item_status_known(item):
            result, msg = False, "failed"

        return result, msg

    def assemble_cover(self, cover: Item):
        result, msg = PlanDispatcher.run_symbolic_action(
            "assemble_external_cover", timeout=self._robot_actions_timeout
        )
        if result:
            self._env.holding_item = Item.nothing
            self._env.arm_pose = ArmPose.unknown
            self._env.cover_assembled = True

        return result, msg

    def assemble_propeller(self, propeller: Item):
        result, msg = PlanDispatcher.run_symbolic_action(
            "assemble_propeller", timeout=self._robot_actions_timeout
        )
        if result:
            self._env.holding_item = Item.nothing
            self._env.arm_pose = ArmPose.unknown
            self._env.propeller_assembled = True

        return result, msg

    def wait_for_soldering(self, color: Color):
        result, msg = PlanDispatcher.run_symbolic_action(
            "wait_until_cable_is_soldered",
            ["Waiting for cable to be soldered. Please confirm."],
            timeout=0.0,
        )
        if result:
            self._env.cables_to_be_soldered.remove(self._env.current_color)

        return result, msg

    def wait_for_fixed_cover(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "wait_until_external_cover_is_fixed",
            ["Waiting for cover to be fixed. Please confirm."],
            timeout=0.0,
        )
        if result:
            self._env.item_status[Item.cover] = None
            self._env.cover_assembled = False

            self._env.current_epic = None
            self._env._perceived_objects.clear()

            if self._env._use_case == "uc5":
                self._env.epic_done[Epic.epic3] = True
            else:
                self._env.pallet_available = False
                self._env._krem_logging.cycle_complete = True

        return result, msg

    def wait_for_fixed_propeller(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "wait_until_propeller_is_fixed",
            ["Waiting for propeller to be fixed. Please confirm."],
            timeout=0.0,
        )
        if result:
            self._env.item_status[Item.propeller] = None
            self._env.propeller_assembled = False

            self._env.current_epic = None
            self._env._perceived_objects.clear()

            self._env.pallet_available = False
            self._env._krem_logging.cycle_complete = True

            if self._env._use_case == "uc5":
                self._env.epic_done[Epic.epic2] = False
                self._env.epic_done[Epic.epic3] = False
                self._env.epic_done[Epic.epic4] = False

        return result, msg

    def release_cable(self, color: Color):
        result, msg = PlanDispatcher.run_symbolic_action(
            "release_cable_soldering_grasp",
            [],
            timeout=self._robot_actions_timeout,
        )

        if result:
            if self._env.cables_to_be_soldered:
                self._env.current_color = self._env.cables_to_be_soldered[0]
            self._env.holding_item = Item.nothing
            self._env.cable_pose = False

        return result, msg

    def move_arm_cable_end(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "move_over_cable_soldering_station",
            timeout=self._robot_actions_timeout,
        )
        if result:
            self._env.arm_pose = ArmPose.over_cable_station
            self._env.item_status[Item.cable] = None
            if not self._env.cables_to_be_soldered:
                self._env.current_color = Color.red
                self._env.cables_to_be_soldered = [
                    Color.red,
                    Color.blue,
                    Color.brown,
                    Color.white,
                ]
                self._env.current_epic = None
                self._env._perceived_objects.clear()
                if self._env._use_case == "uc5":
                    self._env.epic_done[Epic.epic2] = True
                else:
                    self._env.pallet_available = False
                    self._env._krem_logging.cycle_complete = True
        return result, msg

    def reject_item(self, item: Item):
        if self._env.holding_item == item:
            if item == Item.cable:
                result, msg = PlanDispatcher.run_symbolic_action(
                    "reject_cable_soldering",
                    [],
                    timeout=self._robot_actions_timeout,
                )
            elif item == Item.cover:
                result, msg = PlanDispatcher.run_symbolic_action(
                    "reject_external_cover",
                    [],
                    timeout=self._robot_actions_timeout,
                )
            elif item == Item.propeller:
                result, msg = PlanDispatcher.run_symbolic_action(
                    "reject_propeller",
                    [],
                    timeout=self._robot_actions_timeout,
                )
            if result:
                if item == Item.cable:
                    self._env.cable_pose = False
                elif item == Item.cover:
                    self._env.cover_pose = False
                    self._env.cover_leveled = False

                self._env.holding_item = Item.nothing
                self._env.arm_pose = ArmPose.unknown
                self._env.reject_box_status[item] += 1
                self._env.item_status[item] = None
                self._env._perceived_objects.clear()
            return result, msg
        return False, "failed"

    def empty_reject_box(self):
        self._env._krem_logging.wfhi_counter += 1
        result, msg = PlanDispatcher.run_symbolic_action(
            "wait_for_human_intervention",
            ["Reject box is full. Please empty box."],
            timeout=0.0,
        )
        if result:
            self._env.reject_box_status = dict.fromkeys(self._env.reject_box_status, 0)
        return result, msg
