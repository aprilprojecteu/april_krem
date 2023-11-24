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
    case = "case"
    insert_o = "insert_o"
    set = "set"


class ArmPose(Enum):
    unknown = "unknown"
    home = "home"
    arm_up = "arm_up"
    over_conveyor = "over_conveyor"
    over_fixture = "over_fixture"
    over_pallet = "over_pallet"
    over_boxes = "over_boxes"


class Size(Enum):
    small = "small"
    big = "big"


class Status(Enum):
    ok = "ok"
    nok = "nok"


class Environment:
    def __init__(self, krem_logging):
        self._krem_logging = krem_logging

        self.holding_item = Item.nothing
        self.item_in_hand = None
        self.arm_pose = ArmPose.unknown

        self.case_available = False
        self.placed_case = None
        self.inserted = False
        self.insert_perceived = False
        self.set_perceived = False

        self.item_size = None
        self.set_status = None
        self.box_status = {
            Status.ok: 1,
            Status.nok: 1,
        }
        self.small_insert_ids = [11, 12, 13, 14, 15, 16]
        self.big_insert_ids = [21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35, 36]

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
            self._set_inspection_result_srv,
        )

    def __str__(self) -> str:
        perceived_objects_str = "\n".join(
            [f'- "{k}"' for k in self._perceived_objects.keys()]
        )
        return (
            f"Holding: {self.holding_item.value}\n"
            f"Arm Pose: {self.arm_pose.value}\n"
            f"Perceived objects:\n {perceived_objects_str}"
        )

    def reset_env(self) -> None:
        self.holding_item = Item.nothing
        self.item_in_hand = None
        self.arm_pose = ArmPose.unknown

        self.case_available = False
        self.placed_case = None
        self.inserted = False
        self.insert_perceived = False
        self.set_perceived = False

        self.item_size = None
        self.set_status = None

        self.box_status = {
            Status.ok: 1,
            Status.nok: 1,
        }

        self.small_insert_ids = [11, 12, 13, 14, 15, 16]
        self.big_insert_ids = [21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35, 36]

        self._perceived_objects.clear()

    def reset_env_keep_counters(self) -> None:
        self.holding_item = Item.nothing
        self.item_in_hand = None
        self.arm_pose = ArmPose.unknown

        self.case_available = False
        self.placed_case = None
        self.inserted = False
        self.insert_perceived = False
        self.set_perceived = False

        self.item_size = None
        self.set_status = None

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

    def _set_inspection_result_srv(self, request: SetBoolRequest) -> SetBoolResponse:
        if not hasattr(request, "data"):
            return SetBoolResponse(success=False)
        if request.data:
            self.set_status = Status.ok
        else:
            self.set_status = Status.nok
        return SetBoolResponse(success=True)

    def holding(self, item: Item) -> bool:
        return self.holding_item == item

    def current_arm_pose(self, arm_pose: ArmPose) -> bool:
        return arm_pose == self.arm_pose

    def item_size_known(self) -> bool:
        return self.item_size is not None

    def current_item_size(self, size: Size) -> bool:
        return self.item_size == size

    def set_status_known(self) -> bool:
        return self.set_status is not None

    def status_of_set(self, status: Status) -> bool:
        return self.set_status == status if self.set_status is not None else False

    def item_in_fov(self) -> bool:
        return self.case_available

    def case_is_placed(self) -> bool:
        return self.placed_case is not None

    def perceived_insert(self) -> bool:
        return self.insert_perceived

    def perceived_set(self) -> bool:
        return self.set_perceived

    def inserted_insert(self) -> bool:
        return self.inserted

    def inserts_available(self, size: Size) -> bool:
        return (
            len(self.small_insert_ids) > 0
            if size == Size.small
            else len(self.big_insert_ids) > 0
        )

    def space_in_box(self, status: Status) -> bool:
        return self.box_status[status] < 3


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

    def get_next_case(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "get_next_case",
            timeout=self._non_robot_actions_timeout,
        )
        if result:
            self._env.case_available = True
        return result, msg

    def perceive_case(self):
        self._env._clear_item_type("case")
        result, msg = PlanDispatcher.run_symbolic_action(
            "perceive_case",
            timeout=self._non_robot_actions_timeout,
        )
        if result:
            type, _ = self._env._get_item_type_and_id("case")
            if type is not None and "big" in type:
                self._env.item_size = Size.big
            elif type is not None and "small" in type:
                self._env.item_size = Size.small
            else:
                self._env.item_size = None
                return False, "failed"
        return result, msg

    def perceive_insert(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "perceive_insert",
            timeout=self._non_robot_actions_timeout,
        )
        if result:
            self._env.insert_perceived = True
        return result, msg

    def perceive_set(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "perceive_set",
            timeout=self._non_robot_actions_timeout,
        )
        if result:
            self._env.set_perceived = True
        return result, msg

    def move_arm(self, arm_pose: ArmPose):
        result = False
        msg = "failed"
        if arm_pose == ArmPose.over_conveyor:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_conveyor_belt",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.over_fixture:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_fixture",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.over_pallet:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_pallet",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.over_boxes:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_boxes",
                timeout=self._robot_actions_timeout,
            )
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

    def pick_case(self, case: Item, size: Size):
        # arguments: [ID of case]
        class_name = None
        if self._env.item_size is not None:
            class_name, id = self._env._get_item_type_and_id(
                self._env.item_size.value + "_case"
            )
        if class_name is not None:
            grasp_facts = self._grasp_library_srv("mia", class_name, "placing", False)
            result, msg = PlanDispatcher.run_symbolic_action(
                "pick_case",
                [str(id)],
                grasp_facts.grasp_strategies,
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = Item.case
                self._env.item_in_hand = class_name + "_" + str(id)
                self._env.case_available = False
                self._env.arm_pose = ArmPose.unknown
            return result, msg
        return False, "failed"

    def pick_insert(self, insert: Item, size: Size):
        # arguments: [ID of insert]
        class_name = None
        if self._env.item_size is not None:
            class_name = "UC2_" + self._env.item_size.value + "_insert"
        if self._env.item_size == Size.small and self._env.small_insert_ids:
            id = self._env.small_insert_ids[0]
        elif self._env.item_size == Size.big and self._env.big_insert_ids:
            id = self._env.big_insert_ids[0]
        else:
            id = None
        if class_name is not None and id is not None:
            grasp_facts = self._grasp_library_srv("mia", class_name, "assembly", False)
            result, msg = PlanDispatcher.run_symbolic_action(
                "pick_insert",
                [str(id)],
                grasp_facts.grasp_strategies,
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = Item.insert_o
                self._env.item_in_hand = class_name + "_" + str(id)
                self._env.arm_pose = ArmPose.unknown
                if self._env.item_size == Size.small:
                    self._env.small_insert_ids.pop(0)
                elif self._env.item_size == Size.big:
                    self._env.big_insert_ids.pop(0)
                self._env.insert_perceived = False
            return result, msg
        return False, "failed"

    def pick_set(self, set: Item):
        # arguments: [ID of set]
        class_name = None
        if self._env.item_size is not None:
            class_name, id = self._env._get_item_type_and_id(
                self._env.item_size.value + "_set"
            )
        if class_name is not None:
            grasp_facts = self._grasp_library_srv("mia", class_name, "placing", False)
            result, msg = PlanDispatcher.run_symbolic_action(
                "pick_set",
                [str(id)],
                grasp_facts.grasp_strategies,
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = Item.set
                self._env.item_in_hand = (
                    "UC2_" + self._env.item_size.value + "_set" + "_" + str(id)
                )
                self._env.arm_pose = ArmPose.unknown
                self._env.set_perceived = False
                self._env.placed_case = None
            return result, msg
        return False, "failed"

    def place_case(self, case: Item, size: Size):
        # arguments: [Type of case]
        if self._env.item_in_hand is not None:
            class_name, _ = self._env.item_in_hand.rsplit("_", 1)
            result, msg = PlanDispatcher.run_symbolic_action(
                "place_case",
                [class_name],
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = Item.nothing
                self._env.placed_case = self._env.item_in_hand
                self._env.item_in_hand = None
                self._env.arm_pose = ArmPose.unknown
            return result, msg
        else:
            return False, "failed"

    def place_set_in_box(self, set: Item, status: Status):
        # arguments: [item class, which box(ok or nok)]
        if self._env.item_in_hand is not None:
            class_name, _ = self._env.item_in_hand.rsplit("_", 1)
            result, msg = PlanDispatcher.run_symbolic_action(
                "place_set_in_box",
                [
                    class_name,
                    self._env.set_status.value,
                    str(self._env.box_status[self._env.set_status]),
                ],
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = Item.nothing
                self._env.item_in_hand = None
                self._env.arm_pose = ArmPose.unknown
                self._env.box_status[self._env.set_status] += 1
            return result, msg
        else:
            return False, "failed"

    def insert(self, insert: Item, case: Item, size: Size):
        if self._env.item_in_hand is not None and self._env.placed_case is not None:
            insert_class, _ = self._env.item_in_hand.rsplit("_", 1)
            result, msg = PlanDispatcher.run_symbolic_action(
                "insert",
                [insert_class],
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = Item.nothing
                self._env.item_in_hand = None
                self._env.arm_pose = ArmPose.unknown
                self._env.inserted = True
            return result, msg
        else:
            return False, "failed"

    def inspect(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "inspect", timeout=self._non_robot_actions_timeout
        )
        return result, msg

    def move_arm_end(self):
        result = False
        msg = "failed"
        result, msg = PlanDispatcher.run_symbolic_action(
            "move_over_boxes",
            timeout=self._robot_actions_timeout,
        )
        if result:
            self._env.arm_pose = ArmPose.over_boxes
            self._env.set_status = None
            self._env.inserted = False
            self._env.item_size = None
            self._env._krem_logging.cycle_complete = True
            self._env._perceived_objects.clear()
        return result, msg

    def restock_inserts(self, size: Size):
        self._env._krem_logging.wfhi_counter += 1
        result, msg = PlanDispatcher.run_symbolic_action(
            "wait_for_human_intervention",
            [f"{size.value} inserts are empty. Restock {size.value} inserts."],
            timeout=0.0,
        )
        if result:
            if size == Size.small:
                self._env.small_insert_ids = [11, 12, 13, 14, 15, 16]
            elif size == Size.big:
                self._env.big_insert_ids = [
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                ]
        return result, msg

    def empty_box(self, status: Status):
        self._env._krem_logging.wfhi_counter += 1
        result, msg = PlanDispatcher.run_symbolic_action(
            "wait_for_human_intervention",
            [f"{status.value} box is full. Empty box."],
            timeout=0.0,
        )
        if result:
            self._env.box_status[status] = 1
        return result, msg
