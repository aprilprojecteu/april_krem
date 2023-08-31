import os
from typing import Tuple
from enum import Enum

import rospy
import rospkg

from symbolic_fact_generation.fact_generation_with_config import (
    FactGenerationWithConfig,
)
from april_msgs.srv import (
    GetGraspStrategy,
    ObjectsEstimatedPosesSrv,
    ObjectsEstimatedPosesSrvRequest,
    ObjectsEstimatedPosesSrvResponse,
)

from april_krem.plan_dispatcher import PlanDispatcher


class Item(Enum):
    nothing = "nothing"
    insole = "insole"
    bag = "bag"
    set = "set"


class Location(Enum):
    conveyor_a = "conveyor_a"
    conveyor_b = "conveyor_b"
    dispenser = "dispenser"
    in_hand = "in_hand"
    in_bag = "in_bag"
    unknown = "unknown"


class Environment:
    def __init__(self):
        use_case = rospy.get_param("use_case", default="uc6")
        facts_config_file = use_case + "_facts_config.yaml"
        facts_config_path = os.path.join(
            rospkg.RosPack().get_path("symbolic_fact_generation_use_cases"),
            "config",
            facts_config_file,
        )

        self._fact_generator = FactGenerationWithConfig(facts_config_path)

        self.item_at_location = {
            Item.insole: Location.unknown,
            Item.bag: Location.unknown,
            Item.set: Location.unknown,
            Item.nothing: Location.in_hand,
        }
        self.types_match = False
        self.not_checked_types = True
        self.insole_in_fov = False
        self.set_released = False
        self.bag_probably_available = False
        self.bag_probably_open = False
        self.bag_open = False

        # Store objects with ID received from HICEM via Service
        rospy.Service(
            "/hicem/sfg/objects_estimated_poses",
            ObjectsEstimatedPosesSrv,
            self._object_poses_srv,
        )
        self._perceived_objects = {}

    def __str__(self) -> str:
        item_at_location_str = "\n".join(
            [f'"{k.value}" at "{v.value}"' for k, v in self.item_at_location.items()]
        )
        return (
            f"{item_at_location_str}\n"
            f"Types match: {self.types_match}\n"
            f"Item types not checked: {self.not_checked_item_types}\n"
            f"Insole in FOV: {self.insole_in_fov}\n"
            f"Set released: {self.set_released}\n"
            f"Bag probably available: {self.bag_probably_available}\n"
            f"Bag probably open: {self.bag_probably_open}\n"
            f"Bag open: {self.bag_open}"
        )

    def reset_env(self) -> None:
        self.item_at_location = {
            Item.insole: Location.unknown,
            Item.bag: Location.unknown,
            Item.set: Location.unknown,
            Item.nothing: Location.in_hand,
        }
        self.types_match = False
        self.not_checked_types = True
        self.insole_in_fov = False
        self.set_released = False
        self.bag_probably_available = False
        self.bag_probably_open = False
        self.bag_open = False

        self._perceived_objects = {}

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
        for k in self._perceived_objects.keys():
            if item in k:
                class_name, id = k.rsplit("_", 1)
                return class_name, id
        return None, None

    def _clear_item_type(self, item: str) -> None:
        for k in list(self._perceived_objects.keys()):
            if item in k:
                del self._perceived_objects[k]

    def stationary(self, conveyor: Location) -> bool:
        facts = self._fact_generator.generate_facts_with_name("stationary")
        for fact in facts:
            if fact.name == "stationary" and fact.values[0] == conveyor.name:
                return True
            elif fact.name == "moving" and fact.values[0] == conveyor.name:
                return False
        return True

    def moving(self, conveyor: Location) -> bool:
        facts = self._fact_generator.generate_facts_with_name("stationary")

        for fact in facts:
            if fact.name == "stationary" and fact.values[0] == conveyor.name:
                return False
            elif fact.name == "moving" and fact.values[0] == conveyor.name:
                return True
        return False

    def bag_dispenser_has_bags(self) -> bool:
        facts = self._fact_generator.generate_facts_with_name("bag_dispenser_has_bags")
        if facts[0].values[0] == "true":
            return True
        elif facts[0].values[0] == "false":
            return False
        else:
            return False

    def sealing_machine_ready(self) -> bool:
        facts = self._fact_generator.generate_facts_with_name("sealing_machine_ready")
        if facts[0].values[0] == "true":
            return True
        elif facts[0].values[0] == "false":
            return True
        else:
            return False

    def holding(self, item: Item) -> bool:
        return self.item_at_location.get(item, Location.unknown) == Location.in_hand

    def item_pose_is_known(self, item: Item) -> bool:
        item_loc = self.item_at_location.get(item, Location.unknown)
        return item_loc != Location.unknown and item_loc != Location.in_hand

    def item_type_is_known(self, item: Item) -> bool:
        item_loc = self.item_at_location.get(item, Location.unknown)
        return item_loc != Location.unknown and item_loc != Location.in_hand

    def item_types_match(self) -> bool:
        return self.types_match

    def not_checked_item_types(self) -> bool:
        return self.not_checked_types

    def item_in_fov(self) -> bool:
        return self.insole_in_fov

    def bag_set_released(self) -> bool:
        return self.set_released

    def bag_is_probably_available(self) -> bool:
        return self.bag_probably_available

    def bag_is_probably_open(self) -> bool:
        return self.bag_probably_open

    def bag_is_open(self) -> bool:
        return self.bag_open

    def insole_inside_bag(self, insole: Item) -> bool:
        return self.item_at_location.get(insole, Location.unknown) == Location.in_bag


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

    def reject_insole(self, insole: Item):
        # arguments: [ID of insole]
        _, id = self._env._get_item_type_and_id("insole")
        result = PlanDispatcher.run_symbolic_action(
            "reject_insole", action_arguments=[f"{str(id)}"], timeout=60.0
        )
        if result:
            self._env.item_at_location[insole] = Location.unknown
            self._env.insole_in_fov = False
            self._env.types_match = False
            self._env.not_checked_types = True
        return result

    def get_next_insole(self, conveyor: Location):
        result = PlanDispatcher.run_symbolic_action("get_next_insole", timeout=60.0)
        if result:
            self._env.insole_in_fov = True
        return result

    def preload_bag_bundle(self):
        result = PlanDispatcher.run_symbolic_action("preload_bag_bundle")
        return result

    def load_bag(self):
        result = PlanDispatcher.run_symbolic_action("load_bag", timeout=60.0)
        if result:
            self._env.bag_probably_available = True
        return result

    def open_bag(self):
        result = PlanDispatcher.run_symbolic_action("open_bag", timeout=60.0)
        if result:
            self._env.bag_probably_open = True
        return result

    def match_insole_bag(self, insole: Item, bag: Item):
        # arguments: [ID of insole, ID of bag]
        _, insole_id = self._env._get_item_type_and_id("insole")
        _, bag_id = self._env._get_item_type_and_id("bag")
        result = PlanDispatcher.run_symbolic_action(
            "match_insole_bag",
            action_arguments=[f"{str(insole_id)}", f"{str(bag_id)}"],
            timeout=60.0,
        )
        if result:
            self._env.types_match = True
            self._env.not_checked_types = False
        else:
            self._env.types_match = False
            self._env.not_checked_types = False
        return result

    def pick_insole(self, insole: Item):
        class_name, id = self._env._get_item_type_and_id("insole")
        grasp_facts = self._grasp_library_srv("mia", class_name, "insertion", False)
        # arguments: [ID of insole]
        result = PlanDispatcher.run_symbolic_action(
            "pick_insole", [f"{str(id)}"], grasp_facts.grasp_strategies, 180.0
        )
        if result:
            self._env.item_at_location[insole] = Location.in_hand
            self._env.item_at_location[Item.nothing] = Location.unknown
            self._env.insole_in_fov = False
        return result

    def pick_set(self, set: Item):
        class_name, id = self._env._get_item_type_and_id("set")
        grasp_facts = self._grasp_library_srv("mia", class_name, "sealing", False)
        # arguments: [ID of set]
        result = PlanDispatcher.run_symbolic_action(
            "pick_set", [f"{str(id)}"], grasp_facts.grasp_strategies, 180.0
        )
        if result:
            self._env.item_at_location[set] = Location.in_hand
            self._env.item_at_location[Item.nothing] = Location.unknown
        return result

    def insert(self, insole: Item, bag: Item):
        # arguments: [ID of bag (workaround: ID of set)​]
        _, id = self._env._get_item_type_and_id("set")
        result = PlanDispatcher.run_symbolic_action(
            "insert", [f"{str(id)}"], timeout=180.0
        )
        if result:
            self._env.item_at_location[insole] = Location.in_bag
            self._env.item_at_location[Item.nothing] = Location.in_hand
            self._env.bag_open = False
            self._env.types_match = False
        return result

    def perceive_insole(self, insole: Item):
        self._env._clear_item_type("insole")
        result = PlanDispatcher.run_symbolic_action("perceive_insole", timeout=60.0)
        if result:
            self._env.item_at_location[insole] = Location.conveyor_a
        return result

    def perceive_bag(self, bag: Item):
        self._env._clear_item_type("bag")
        result = PlanDispatcher.run_symbolic_action("perceive_bag", timeout=60.0)
        if result:
            self._env.item_at_location[bag] = Location.dispenser
            self._env.bag_probably_available = False
            self._env.bag_probably_open = False
            self._env.bag_open = True
        return result

    def perceive_set(self, insole: Item, bag: Item, set: Item):
        self._env._clear_item_type("set")
        result = PlanDispatcher.run_symbolic_action("perceive_set", timeout=60.0)
        if result:
            self._env.item_at_location[set] = Location.dispenser
        return result

    def release_set(self, set: Item):
        result = PlanDispatcher.run_symbolic_action("release_set", timeout=60.0)
        if result:
            self._env.set_released = True

        return result

    def seal_set(self, set: Item):
        # arguments: [ID of set​]
        _, id = self._env._get_item_type_and_id("set")
        result = PlanDispatcher.run_symbolic_action(
            "seal_set", [f"{str(id)}"], timeout=180.0
        )
        if result:
            self._env.item_at_location[Item.insole] = Location.unknown
            self._env.item_at_location[Item.bag] = Location.unknown
            self._env.item_at_location[set] = Location.unknown
            self._env.item_at_location[Item.nothing] = Location.in_hand
            self._env.set_released = False
            self._env.not_checked_types = True
        return result
