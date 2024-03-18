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
    pillow = "pillow"


class Location(Enum):
    table = "table"
    scale = "scale"
    box = "box"


class ArmPose(Enum):
    unknown = "unknown"
    home = "home"
    arm_up = "arm_up"
    over_table = "over_table"
    over_scale = "over_scale"
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

        self.pillow_perceived = {
            Location.table: False,
            Location.scale: False,
            Location.box: False,
        }
        self.pillow_status = None
        self.item_size = None
        self.weighted_pillow = False
        self.pillow_location = {
            Location.table: False,
            Location.scale: False,
            Location.box: False,
        }

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

        # Pillow inspection result service
        rospy.Service(
            "/hicem/sfg/quality_control/result",
            SetBool,
            self._pillow_inspection_result_srv,
        )

    def reset_env(self) -> None:
        self.holding_item = Item.nothing
        self.arm_pose = ArmPose.unknown
        self.item_in_hand = None

        self.pillow_perceived = {
            Location.table: False,
            Location.scale: False,
            Location.box: False,
        }
        self.pillow_status = None
        self.item_size = None
        self.weighted_pillow = False
        self.pillow_location = {
            Location.table: False,
            Location.scale: False,
            Location.box: False,
        }

        self.box_status = {
            Status.ok: 1,
            Status.nok: 1,
        }

        self._perceived_objects.clear()

    def reset_env_keep_counters(self) -> None:
        self.holding_item = Item.nothing
        self.arm_pose = ArmPose.unknown
        self.item_in_hand = None

        self.pillow_perceived = {
            Location.table: False,
            Location.scale: False,
            Location.box: False,
        }
        self.pillow_status = None
        self.item_size = None
        self.weighted_pillow = False
        self.pillow_location = {
            Location.table: False,
            Location.scale: False,
            Location.box: False,
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

    def _pillow_inspection_result_srv(self, request: SetBoolRequest) -> SetBoolResponse:
        if not hasattr(request, "data"):
            return SetBoolResponse(success=False)
        if request.data:
            self.pillow_status = Status.ok
        else:
            self.pillow_status = Status.nok
        return SetBoolResponse(success=True)

    def holding(self, item: Item) -> bool:
        return self.holding_item == item

    def current_arm_pose(self, arm_pose: ArmPose) -> bool:
        return arm_pose == self.arm_pose

    def item_size_known(self) -> bool:
        return self.item_size is not None

    def current_item_size(self, size: Size) -> bool:
        return self.item_size == size

    def pillow_status_known(self) -> bool:
        return self.pillow_status is not None

    def status_of_pillow(self, status: Status) -> bool:
        return self.pillow_status == status if self.pillow_status is not None else False

    def perceived_pillow(self, location: Location) -> bool:
        return self.pillow_perceived.get(location, False)

    def space_in_box(self, status: Status) -> bool:
        return self.box_status[status] < 3

    def pillow_weight_known(self) -> bool:
        return self.weighted_pillow

    def pillow_is_on(self, location: Location) -> bool:
        return self.pillow_location.get(location, False)


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

    def get_next_pillow(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "get_next_pillow",
            ["Waiting for new pillow on support."],
            timeout=0.0,
        )
        if result:
            self._env.pillow_location[Location.table] = True

        return result, msg

    def perceive_pillow(self, location: Location):
        self._env._clear_item_type("pillow")
        result, msg = False, "failed"
        if location == Location.table:
            result, msg = PlanDispatcher.run_symbolic_action(
                "perceive_pillow_on_table",
                timeout=self._non_robot_actions_timeout,
            )
        elif location == Location.scale:
            result, msg = PlanDispatcher.run_symbolic_action(
                "perceive_pillow_on_scale",
                timeout=self._non_robot_actions_timeout,
            )
        if result:
            type, _ = self._env._get_item_type_and_id("pillow")
            if type is not None and "big" in type:
                self._env.item_size = Size.big
                self._env.pillow_perceived[location] = True
                self._env.pillow_location[location] = True
            elif type is not None and "small" in type:
                self._env.item_size = Size.small
                self._env.pillow_perceived[location] = True
                self._env.pillow_location[location] = True
            else:
                self._env.pillow_perceived[location] = False
                if location == Location.table:
                    self._env.pillow_location[location] = False
                return False, "failed"
        else:
            self._env.pillow_perceived[location] = False
            if location == Location.table:
                self._env.pillow_location[location] = False
            return False, "failed"
        return result, msg

    def move_arm(self, arm_pose: ArmPose):
        result = False
        msg = "failed"
        if arm_pose == ArmPose.over_table:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_table",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.over_scale:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_scale",
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

    def pick_pillow(self, pillow: Item, size: Size, location: Location):
        # arguments: [ID of pillow]
        result, msg = False, "failed"
        class_name, id = self._env._get_item_type_and_id("pillow")
        if class_name is not None:
            grasp_facts = self._grasp_library_srv("mia", class_name, "placing", False)
            if location == Location.table:
                result, msg = PlanDispatcher.run_symbolic_action(
                    "pick_pillow_from_table",
                    [str(id)],
                    grasp_facts.grasp_strategies,
                    timeout=self._robot_actions_timeout,
                )
            elif location == Location.scale:
                result, msg = PlanDispatcher.run_symbolic_action(
                    "pick_pillow_from_scale",
                    [str(id)],
                    grasp_facts.grasp_strategies,
                    timeout=self._robot_actions_timeout,
                )
            if result:
                self._env.holding_item = Item.pillow
                self._env.item_in_hand = class_name + "_" + str(id)
                self._env.arm_pose = ArmPose.unknown
                self._env.pillow_perceived[location] = False
                self._env.pillow_location[location] = False
        return result, msg

    def place_pillow_on_scale(self, pillow: Item, size: Size):
        # arguments: [ID of pillow]
        if self._env.item_in_hand is not None:
            class_name, id = self._env.item_in_hand.rsplit("_", 1)
            result, msg = PlanDispatcher.run_symbolic_action(
                "place_pillow_on_scale",
                [str(id), class_name],
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = Item.nothing
                self._env.item_in_hand = None
                self._env.arm_pose = ArmPose.unknown
                self._env.pillow_location[Location.scale] = True
            return result, msg
        return False, "failed"

    def weigh_pillow(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "weigh_pillow",
            timeout=self._non_robot_actions_timeout,
        )
        if result:
            self._env.weighted_pillow = True
        return result, msg

    def place_pillow_in_box(self, pillow: Item, status: Status):
        # arguments: [item class, which box(ok or nok), box status]
        if self._env.item_in_hand is not None:
            class_name, _ = self._env.item_in_hand.rsplit("_", 1)
            result, msg = PlanDispatcher.run_symbolic_action(
                "place_pillow_in_box",
                [
                    class_name,
                    self._env.pillow_status.value,
                    str(self._env.box_status[self._env.pillow_status]),
                ],
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = Item.nothing
                self._env.item_in_hand = None
                self._env.arm_pose = ArmPose.unknown
                self._env.box_status[self._env.pillow_status] += 1
                self._env.pillow_location[Location.box] = True
            return result, msg
        return False, "failed"

    def inspect(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "inspect", timeout=self._non_robot_actions_timeout
        )
        return result, msg

    def update_pemu_server(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "update_pemu_server",
            timeout=self._non_robot_actions_timeout,
        )
        if result:
            self._env.pillow_location[Location.box] = False
            self._env.pillow_status = None
            self._env.item_size = None
            self._env.weighted_pillow = False
            self._env._krem_logging.cycle_complete = True
            self._env._perceived_objects.clear()
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
