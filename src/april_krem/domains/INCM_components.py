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
    passport = "passport"


class ArmPose(Enum):
    unknown = "unknown"
    home = "home"
    over_passport = "over_passport"
    over_mrz = "over_mrz"
    over_chip = "over_chip"
    over_boxes = "over_boxes"
    arm_up = "arm_up"


class Status(Enum):
    ok = "ok"
    nok = "nok"


class Environment:
    def __init__(self, krem_logging):
        self._krem_logging = krem_logging

        self.holding_item = Item.nothing
        self.item_in_hand = None
        self.arm_pose = ArmPose.unknown

        self.passport_perceived = False
        self.used_mrz_reader = False
        self.used_chip_reader = False
        self.passport_status = None
        self.detected_passport_corner = False
        self.passport_available = True

        self.box_status = {
            Status.ok: 1,
            Status.nok: 1,
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
            self._passport_inspection_result_srv,
        )

    def reset_env(self) -> None:
        self.holding_item = Item.nothing
        self.arm_pose = ArmPose.unknown
        self.item_in_hand = None

        self.passport_perceived = False
        self.used_mrz_reader = False
        self.used_chip_reader = False
        self.passport_status = None
        self.detected_passport_corner = False
        self.passport_available = True

        self.box_status = {
            Status.ok: 1,
            Status.nok: 1,
        }

        self._perceived_objects.clear()

    def reset_env_keep_counters(self) -> None:
        self.holding_item = Item.nothing
        self.arm_pose = ArmPose.unknown
        self.item_in_hand = None

        self.passport_perceived = False
        self.used_mrz_reader = False
        self.used_chip_reader = False
        self.passport_status = None
        self.detected_passport_corner = False
        self.passport_available = True

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

    def _passport_inspection_result_srv(
        self, request: SetBoolRequest
    ) -> SetBoolResponse:
        if not hasattr(request, "data"):
            return SetBoolResponse(success=False)
        if request.data:
            self.passport_status = Status.ok
        else:
            self.passport_status = Status.nok
        return SetBoolResponse(success=True)

    def holding(self, item: Item) -> bool:
        return self.holding_item == item

    def current_arm_pose(self, arm_pose: ArmPose) -> bool:
        return arm_pose == self.arm_pose

    def perceived_passport(self) -> bool:
        return self.passport_perceived

    def passport_status_known(self) -> bool:
        return self.passport_status is not None

    def status_of_passport(self, status: Status) -> bool:
        return (
            self.passport_status == status
            if self.passport_status is not None
            else False
        )

    def mrz_reader_used(self) -> bool:
        return self.used_mrz_reader

    def chip_reader_used(self) -> bool:
        return self.used_chip_reader

    def space_in_box(self, status: Status) -> bool:
        return self.box_status[status] < 7

    def passport_corner_detected(self) -> bool:
        return self.detected_passport_corner

    def passport_is_available(self) -> bool:
        return self.passport_available


class Actions:
    def __init__(self, env: Environment):
        self._env = env
        self._sliding_fails = 0

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

    def perceive_passport(self):
        self._env._clear_item_type("passport")
        result, msg = PlanDispatcher.run_symbolic_action(
            "perceive_passport",
            timeout=self._non_robot_actions_timeout,
        )

        self._env.passport_perceived = True
        type, _ = self._env._get_item_type_and_id("passport")
        self._env.passport_available = type is not None
        if not self._env.passport_available:
            return False, "failed"

        return result, msg

    def detect_passport_corner(self):
        _, passport_id = self._env.item_in_hand.rsplit("_", 1)
        result, msg = PlanDispatcher.run_symbolic_action(
            "detect_passport_corner",
            [str(passport_id)],
            timeout=self._non_robot_actions_timeout,
        )
        if result:
            self._env.detected_passport_corner = True
            self._env.arm_pose = ArmPose.unknown

        return result, msg

    def move_arm(self, arm_pose: ArmPose):
        result = False
        msg = "failed"
        if arm_pose == ArmPose.over_passport:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_passport_supports",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.over_mrz:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_mrz_reader",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.over_chip:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_chip_reader",
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

    def pick_passport(self, passport: Item):
        # arguments: [ID of passport]
        class_name, id = self._env._get_item_type_and_id("passport")
        if class_name is not None:
            grasp_facts = self._grasp_library_srv("mia", class_name, "sliding", False)
            result, msg = PlanDispatcher.run_symbolic_action(
                "pick_passport",
                [str(id)],
                grasp_facts.grasp_strategies,
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = Item.passport
                self._env.item_in_hand = class_name + "_" + str(id)
                self._env.arm_pose = ArmPose.unknown
                self._env.passport_perceived = False
            return result, msg
        return False, "failed"

    def read_mrz(self, passport: Item):
        if self._env.item_in_hand is not None:
            result, msg = PlanDispatcher.run_symbolic_action(
                "read_mrz", [], timeout=self._robot_actions_timeout, number_of_retries=0
            )
            if result:
                self._env.used_mrz_reader = True
                self._env.arm_pose = ArmPose.unknown
            else:
                if self._sliding_fails >= 2:
                    self._sliding_fails = 0
                    return False, "wait_for_human_intervention"
                self._sliding_fails += 1
                self._env.detected_passport_corner = False
            return result, msg
        return False, "failed"

    def read_chip(self, passport: Item):
        if self._env.item_in_hand is not None:
            result, msg = PlanDispatcher.run_symbolic_action(
                "read_chip",
                [],
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.used_chip_reader = True
                self._env.arm_pose = ArmPose.unknown
            return result, msg
        return False, "failed"

    def inspect(self, passport: Item):
        result, msg = PlanDispatcher.run_symbolic_action(
            "inspect", timeout=self._non_robot_actions_timeout
        )
        return result, msg

    def place_passport_in_box(self, passport: Item, status: Status):
        # arguments: ["ok" | "nok", 1 = empty | 2 = one in box]
        if self._env.item_in_hand is not None:
            result, msg = PlanDispatcher.run_symbolic_action(
                "place_passport_in_box",
                [
                    self._env.passport_status.value,
                    str(self._env.box_status[self._env.passport_status]),
                ],
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = Item.nothing
                self._env.item_in_hand = None
                self._env.arm_pose = ArmPose.unknown
                self._env.box_status[self._env.passport_status] += 1
                self._env.passport_status = None
                self._env.used_mrz_reader = False
                self._env.used_chip_reader = False
                self._env.detected_passport_corner = False
                self._env._krem_logging.cycle_complete = True
                self._env._perceived_objects.clear()
            return result, msg
        return False, "failed"

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

    def refill_passports(self):
        self._env._krem_logging.wfhi_counter += 1
        result, msg = PlanDispatcher.run_symbolic_action(
            "wait_for_human_intervention",
            ["Place new passports in passport supports."],
            timeout=0.0,
        )
        if result:
            self._env.passport_available = True
            self._env.passport_perceived = False
        return result, msg
