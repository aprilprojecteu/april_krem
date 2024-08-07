from typing import Tuple
from enum import Enum

import rospy

from april_msgs.srv import (
    GetGraspStrategy,
    ObjectsEstimatedPosesSrv,
    ObjectsEstimatedPosesSrvRequest,
    ObjectsEstimatedPosesSrvResponse,
    ShelfLife,
    ShelfLifeRequest,
    ShelfLifeResponse,
)

from april_krem.plan_dispatcher import PlanDispatcher


class Item(Enum):
    nothing = "nothing"
    breast = "breast"
    drumstick = "drumstick"


class Tray(Enum):
    low_tray = 1
    med_tray = 2
    high_tray = 3
    discard_tray = 4


class ArmPose(Enum):
    unknown = "unknown"
    home = "home"
    over_conveyor = "over_conveyor"
    over_tray = "over_tray"


class Environment:
    def __init__(self, krem_logging):
        self._krem_logging = krem_logging

        self.num_chicken_in_tray = {
            Tray.high_tray: 0,
            Tray.med_tray: 0,
            Tray.low_tray: 0,
            Tray.discard_tray: 0,
        }

        self.type_chicken_in_tray = {
            Tray.high_tray: Item.nothing,
            Tray.med_tray: Item.nothing,
            Tray.low_tray: Item.nothing,
            Tray.discard_tray: Item.nothing,
        }

        self.chicken_in_fov = False
        self.chicken_type = None
        self.holding_item = Item.nothing
        self.tray_place = None
        self.tray_available = {
            Tray.high_tray: False,
            Tray.med_tray: False,
            Tray.low_tray: False,
            Tray.discard_tray: False,
        }
        self.perceived_trays = False
        self.arm_pose = ArmPose.unknown
        self.cb_moving = False

        self._perceived_objects = {}

        # Store objects with ID received from HICEM via Service
        rospy.Service(
            "/hicem/sfg/objects_estimated_poses",
            ObjectsEstimatedPosesSrv,
            self._object_poses_srv,
        )

        # Chicken shelf life service
        rospy.Service(
            "/hicem/sfg/asin/hsi_pc/shelf_life",
            ShelfLife,
            self._chicken_shelf_life_srv,
        )

    def reset_env(self) -> None:
        self.num_chicken_in_tray = {
            Tray.high_tray: 0,
            Tray.med_tray: 0,
            Tray.low_tray: 0,
            Tray.discard_tray: 0,
        }

        self.type_chicken_in_tray = {
            Tray.high_tray: Item.nothing,
            Tray.med_tray: Item.nothing,
            Tray.low_tray: Item.nothing,
            Tray.discard_tray: Item.nothing,
        }

        self.chicken_in_fov = False
        self.chicken_type = None
        self.holding_item = Item.nothing
        self.tray_place = None
        self.tray_available = {
            Tray.high_tray: False,
            Tray.med_tray: False,
            Tray.low_tray: False,
            Tray.discard_tray: False,
        }
        self.perceived_trays = False
        self.arm_pose = ArmPose.unknown
        self.cb_moving = False

        self._perceived_objects.clear()

    def reset_env_keep_counters(self) -> None:
        self.chicken_in_fov = False
        self.chicken_type = None
        self.holding_item = Item.nothing
        self.tray_place = None
        self.perceived_trays = False
        self.arm_pose = ArmPose.unknown
        self.cb_moving = False

        self._perceived_objects.clear()

    def _calc_space_in_tray(self, chicken_part: Item, tray: Tray) -> bool:
        counter = self.num_chicken_in_tray.get(tray, None)
        type_in_tray = self.type_chicken_in_tray.get(tray, None)
        if counter is not None and type_in_tray is not None:
            if chicken_part == Item.nothing:
                return True
            elif type_in_tray == Item.nothing:
                return True
            elif counter < 4 and type_in_tray == Item.breast:
                return True
            elif counter < 6 and type_in_tray == Item.drumstick:
                return True
            else:
                return False
        else:
            return True

    def _chicken_shelf_life_srv(self, request: ShelfLifeRequest) -> ShelfLifeResponse:
        success = True
        if request.shelf_life < 3:
            self.tray_place = Tray.high_tray
        elif 3 <= request.shelf_life < 5:
            self.tray_place = Tray.med_tray
        elif 5 <= request.shelf_life < 7:
            self.tray_place = Tray.low_tray
        elif 7 <= request.shelf_life:
            self.tray_place = Tray.discard_tray
        else:
            success = False
        return ShelfLifeResponse(confirm=success)

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
        if item == "chicken":
            items_to_receive.extend(["breast", "drumstick"])
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
        if item == "chicken":
            items_to_remove.extend(["breast", "drumstick"])
        for k in list(self._perceived_objects.keys()):
            for i in items_to_remove:
                if i in k:
                    del self._perceived_objects[k]
                    break

    def holding(self, item: Item) -> bool:
        return self.holding_item == item

    def item_type_is_known(self) -> bool:
        return self.chicken_type is not None

    def chicken_to_pick(self, item: Item) -> bool:
        return self.chicken_type == item if item != Item.nothing else False

    def item_in_fov(self) -> bool:
        return self.chicken_in_fov

    def tray_to_place(self, tray: Tray) -> bool:
        return self.tray_place == tray

    def tray_to_place_known(self) -> bool:
        return self.tray_place is not None

    def space_in_tray(self, chicken_part: Item, tray: Tray) -> bool:
        return self._calc_space_in_tray(chicken_part, tray)

    def type_in_tray(self, chicken_part: Item, tray: Tray) -> bool:
        chicken_type = self.type_chicken_in_tray.get(tray, None)
        return chicken_type is not None and chicken_type == chicken_part

    def tray_is_available(self, tray: Tray) -> bool:
        return self.tray_available.get(tray, False)

    def trays_perceived(self) -> bool:
        return self.perceived_trays

    def current_arm_pose(self, arm_pose: ArmPose) -> bool:
        return arm_pose == self.arm_pose

    def conveyor_is_moving(self) -> bool:
        return self.cb_moving


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
            "~non_robot_actions_timeout", default="120"
        )
        self._robot_actions_timeout = rospy.get_param(
            "~robot_actions_timeout", default="120"
        )

    def move_conveyor_belt(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "move_conveyor_belt",
            timeout=self._non_robot_actions_timeout,
        )
        if result:
            self._env.cb_moving = True
        return result, msg

    def get_next_chicken_part(self):
        result, msg = PlanDispatcher.run_symbolic_action(
            "get_next_chicken_part",
            timeout=self._non_robot_actions_timeout,
        )
        if result:
            self._env.chicken_in_fov = True
            self._env.cb_moving = False
        return result, msg

    def estimate_part_shelf_life(self):
        result, msg = PlanDispatcher.run_symbolic_action("estimate_part_shelf_life")
        if not result and self._env.tray_place is None:
            return False, "failed"
        return result, msg

    def perceive_chicken_part(self):
        self._env._clear_item_type("chicken")
        result, msg = PlanDispatcher.run_symbolic_action(
            "perceive_chicken_part",
            timeout=self._non_robot_actions_timeout,
        )
        if result:
            type, _ = self._env._get_item_type_and_id("chicken")
            if type is not None and "drumstick" in type:
                self._env.chicken_type = Item.drumstick
            elif type is not None and "breast" in type:
                self._env.chicken_type = Item.breast
            else:
                self._env.chicken_type = None
                return False
        return result, msg

    def move_arm(self, arm_pose: ArmPose):
        result = False
        msg = "failed"
        if arm_pose == ArmPose.over_conveyor:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_conveyor_belt",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.over_tray:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_over_tray_cart",
                timeout=self._robot_actions_timeout,
            )
        elif arm_pose == ArmPose.home:
            result, msg = PlanDispatcher.run_symbolic_action(
                "move_to_home_pose",
                timeout=self._robot_actions_timeout,
            )
        if result:
            self._env.arm_pose = arm_pose
        return result, msg

    def pick_chicken_part(self, chicken: Item):
        # arguments: [ID of chicken part]
        class_name, id = self._env._get_item_type_and_id("chicken")
        if class_name is not None:
            grasp_facts = self._grasp_library_srv("mia", class_name, "placing", False)
            result, msg = PlanDispatcher.run_symbolic_action(
                "pick_chicken_part",
                [str(id)],
                grasp_facts.grasp_strategies,
                timeout=self._robot_actions_timeout,
            )
            if result:
                self._env.holding_item = self._env.chicken_type
                self._env.chicken_in_fov = False
                self._env.arm_pose = ArmPose.unknown
            return result, msg
        return False, "failed"

    def perceive_trays(self):
        # self._env._clear_item_type("tray")
        result, msg = PlanDispatcher.run_symbolic_action(
            "perceive_trays", timeout=self._non_robot_actions_timeout
        )
        if result:
            for tray in [
                Tray.low_tray,
                Tray.med_tray,
                Tray.high_tray,
                Tray.discard_tray,
            ]:
                # HACK HARDCODED all trays available
                # class_name, _ = self._env._get_item_type_and_id(tray.name)
                # if class_name is not None:
                #     self._env.tray_available[tray] = True
                self._env.tray_available[tray] = True
            self._env.perceived_trays = True
        return result, msg

    def insert_part_in_container(self, chicken: Item, tray: Tray):
        # arguments: [ID of tray]
        if self._env.tray_place is not None:
            result, msg = PlanDispatcher.run_symbolic_action(
                "insert_part_in_container",
                [str(self._env.tray_place.value)],
                timeout=self._robot_actions_timeout,
            )
        else:
            return False, "failed"
        if result:
            self._env.holding_item = Item.nothing
            self._env.tray_place = None
            self._env.arm_pose = ArmPose.unknown
            if chicken in [Item.breast, Item.drumstick]:
                self._env.num_chicken_in_tray[tray] += 1
                self._env.type_chicken_in_tray[tray] = chicken
            else:
                return False, "failed"
            self._env._krem_logging.cycle_complete = True
            self._env._perceived_objects.clear()
        return result, msg

    def move_arm_end(self):
        result = False
        msg = "failed"
        result, msg = PlanDispatcher.run_symbolic_action(
            "move_over_tray_cart",
            timeout=self._robot_actions_timeout,
        )
        if result:
            self._env.arm_pose = ArmPose.over_tray
            self._env.perceived_trays = False
            self._env.chicken_type = None

            self._env._krem_logging.cycle_complete = True
            self._env._perceived_objects.clear()
        return result, msg

    def replace_filled_tray(self, chicken: Item, tray: Tray):
        self._env._krem_logging.wfhi_counter += 1
        result, msg = PlanDispatcher.run_symbolic_action(
            "wait_for_human_intervention",
            [f"{tray.name} is full. Replace tray."],
            timeout=0.0,
        )
        if result:
            self._env.num_chicken_in_tray[tray] = 0
            self._env.type_chicken_in_tray[tray] = Item.nothing
            self._env.perceived_trays = False
            self._env.tray_available[tray] = False
        return result, msg
