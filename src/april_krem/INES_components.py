import os
from enum import Enum

import rospy
import rospkg
import actionlib

from symbolic_fact_generation.fact_generation_with_config import (
    FactGenerationWithConfig,
)
from april_msgs.srv import GetGraspStrategy
from april_msgs.msg import (
    RunSymbolicActionAction,
    RunSymbolicActionGoal,
    GestureBackupButtonStates,
)


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

        self.item_at_location = {Item.insole: Location.unknown, Item.bag: Location.unknown, Item.set: Location.unknown, Item.nothing: Location.in_hand}
        self.types_match = False
        self.insole_in_fov = False
        self.set_released = False
        self.bag_probably_available = False
        self.bag_probably_open = False
        self.bag_open = False

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

    def human_available(self) -> bool:
        return True

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

    def item_in_fov(self) -> bool:
        return self.insole_in_fov

    def bag_set_released(self) -> bool:
        return self.set_released

    def bag_is_probably_available(self) -> bool:
        return self.bag_probably_available

    def bag_is_probably_open(self) -> bool:
        return self.bag_probably_open

    def bag_is_open(self, bag: Item) -> bool:
        return self.bag_open

    def insole_inside_bag(self, insole: Item, bag: Item) -> bool:
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
        rospy.loginfo("Waiting for HICEM Run Symbolic Action Server...")
        self._hicem_run_action_client = actionlib.SimpleActionClient(
            "/hicem/run/symbolic_action", RunSymbolicActionAction
        )
        self._hicem_run_action_client.wait_for_server()
        rospy.loginfo("HICEM Run Symbolic Action Server found!")

        # self._gesture_backup_buttons = rospy.Subscriber(
        #     "/isim/hmi/gesture_backup_buttons",
        #     GestureBackupButtonStates,
        #     self._gesture_backup_button_cb,
        # )
        # PENDING, ACTIVE, RECALLED, REJECTED, PREEMPTED, ABORTED, SUCCEEDED, LOST.
        self._current_action_status = "LOST"

    def run_symbolic_action(
        self, action_name: str, action_arguments = [], grasp_facts = [], timeout = 0.0
    ) -> bool:
        run_symbolic_action_goal_msg = RunSymbolicActionGoal(
            action_type=action_name, action_arguments=action_arguments, grasp_facts=grasp_facts
        )

        self._hicem_run_action_client.send_goal(
            run_symbolic_action_goal_msg, done_cb=self._symbolic_action_done_cb
        )
        self._current_action_status = "ACTIVE"
        start_time = rospy.get_rostime().to_sec()

        rate = rospy.Rate(10)

        while self._current_action_status == "ACTIVE" and not rospy.is_shutdown():
            if timeout > 0.0 and rospy.get_rostime().to_sec() - start_time > timeout:
                self._current_action_status = "TIMEOUT"
            rate.sleep()

        if self._current_action_status == "SUCCEEDED":
            return True
        elif self._current_action_status == "PREEMPTED":
            self._hicem_run_action_client.cancel_all_goals()
            return False
        elif self._current_action_status == "ERROR":
            self._hicem_run_action_client.cancel_all_goals()
            return False
        elif self._current_action_status == "LOST":
            self._hicem_run_action_client.cancel_all_goals()
            return False
        elif self._current_action_status == "TIMEOUT":
            rospy.logerr(f"{action_name} timed out after {timeout} seconds!")
            self._hicem_run_action_client.cancel_all_goals()
            return False

        return False

    def _symbolic_action_done_cb(self, state, result):
        if result is not None:
            # TODO process result message:
            # bool success
            # april_msgs/ActionError[] errors

            # ActionError message:
            # int16 WARNING = 4   # Warning level (Error code [100,199]
            # int16 ERROR = 8     # Error level (Error code [200,299]
            # int16 FATAL = 16    # Fatal level (Error code [300,399]
            # int16 ABORTED = 100 # Action server was terminated from action client
            # int16 DELAYED = 101 # Goal accomplished with time delay
            # int16 TIMEOUT = 200 # Timeout occured
            # int16 STUCKED = 201 # Robot IK solver stucked into local minima
            # int16 INVALID = 202 # Goal expression was invalid
            # string message
            # int16 exit_code
            # int16 level_error
            if result.success:
                self._current_action_status = "SUCCEEDED"
            else:
                self._current_action_status = "ERROR"

    def _gesture_backup_button_cb(self, msg):
        rospy.logwarn("Gesture Backup Button pressed! Canceling action execution.")
        if msg.gesture_backup_button_state_0:
            self._hicem_run_action_client.cancel_all_goals()
            self._current_action_status = "PREEMPTED"
        elif msg.gesture_backup_button_state_1:
            self._hicem_run_action_client.cancel_all_goals()
            self._current_action_status = "PREEMPTED"
        elif msg.gesture_backup_button_state_2:
            self._hicem_run_action_client.cancel_all_goals()
            self._current_action_status = "PREEMPTED"
        elif msg.gesture_backup_button_state_3:
            self._hicem_run_action_client.cancel_all_goals()
            self._current_action_status = "PREEMPTED"
        elif msg.gesture_backup_button_state_4:
            self._hicem_run_action_client.cancel_all_goals()
            self._current_action_status = "PREEMPTED"
        elif msg.gesture_backup_button_state_5:
            self._hicem_run_action_client.cancel_all_goals()
            self._current_action_status = "PREEMPTED"
        elif msg.gesture_backup_button_state_6:
            self._hicem_run_action_client.cancel_all_goals()
            self._current_action_status = "PREEMPTED"
        elif msg.gesture_backup_button_state_7:
            self._hicem_run_action_client.cancel_all_goals()
            self._current_action_status = "PREEMPTED"

    def wait_for_human_intervention(self):
        result = self.run_symbolic_action("wait_for_human_intervention")
        return result

    def reject_insole(self, conveyor: Location, insole: Item):
        # arguments: [ID of insole]
        result = self.run_symbolic_action("reject_insole", action_arguments=["1"], timeout=60.0)
        if result:
            self._env.item_at_location[insole] = Location.unknown
            self._env.insole_in_fov = False
            self._env.types_match = False
        return result

    def get_next_insole(self, conveyor: Location):
        result = self.run_symbolic_action("get_next_insole", timeout=60.0)
        if result:
            self._env.insole_in_fov = True
        return result

    def preload_bag_bundle(self):
        result = self.run_symbolic_action("preload_bag_bundle")
        return result

    def load_bag(self):
        result = self.run_symbolic_action("load_bag", timeout=60.0)
        if result:
            self._env.bag_probably_available = True
        return result

    def open_bag(self):
        result = self.run_symbolic_action("open_bag", timeout=60.0)
        if result:
            self._env.bag_probably_open = True
        return result

    def match_insole_bag(self, insole: Item, bag: Item):
        # arguments: [ID of insole, ID of bag]
        result = self.run_symbolic_action("match_insole_bag", action_arguments=["1", "1"], timeout=60.0)
        if result:
            self._env.types_match = True
        else:
            self._env.types_match = False
        return result

    def pick_insole(self, insole: Item):
        grasp_facts = self._grasp_library_srv("mia", "insole_model_1", "bagging", False)
        # TODO Object ID
        # arguments: [ID of insole]
        result = self.run_symbolic_action(
            "pick_insole", ["1"], grasp_facts.grasp_strategies, 180.0
        )
        if result:
            self._env.item_at_location[insole] = Location.in_hand
            self._env.item_at_location[Item.nothing] = Location.unknown
            self._env.insole_in_fov = False
        return result

    def pick_set(self, set: Item):
        grasp_facts = self._grasp_library_srv("mia", "set_1", "sealing", False)
        # TODO Object ID
        # arguments: [ID of set]
        result = self.run_symbolic_action(
            "pick_set", ["1"], grasp_facts.grasp_strategies, 180.0
        )
        if result:
            self._env.item_at_location[set] = Location.in_hand
            self._env.item_at_location[Item.nothing] = Location.unknown
        return result

    def insert(self, insole: Item, bag: Item):
        # TODO Object ID
        # arguments: [ID of bag (workaround: ID of set)​]
        result = self.run_symbolic_action("insert", ["1"], timeout=180.0)
        if result:
            self._env.item_at_location[insole] = Location.in_bag
            self._env.item_at_location[Item.nothing] = Location.in_hand
            self._env.bag_open = False
            self._env.types_match = False
            self._env.item_at_location[bag] = Location.unknown
        return result

    def perceive_insole(self, insole: Item):
        result = self.run_symbolic_action("perceive_insole", timeout=60.0)
        if result:
            self._env.item_at_location[insole] = Location.conveyor_a
        return result

    def perceive_bag(self, bag: Item):
        result = self.run_symbolic_action("perceive_bag", timeout=60.0)
        if result:
            self._env.item_at_location[bag] = Location.dispenser
            self._env.bag_probably_available = False
            self._env.bag_probably_open = False
            self._env.bag_open = True
        return result

    def perceive_set(self, insole: Item, bag: Item, set: Item):
        result = self.run_symbolic_action("perceive_set", timeout=60.0)
        if result:
            self._env.item_at_location[set] = Location.dispenser
            self._env.item_at_location[insole] = Location.unknown
        return result

    def release_set(self, set: Item):
        result = self.run_symbolic_action("release_set", timeout=60.0)
        if result:
            self._env.set_released = True

        return result

    def seal_set(self, set: Item):
        # TODO Object ID
        # arguments: [ID of set​]
        result = self.run_symbolic_action("seal_set", ["1"], timeout=180.0)
        if result:
            self._env.item_at_location[set] = Location.unknown
            self._env.item_at_location[Item.nothing] = Location.in_hand
            self._env.set_released = False
        return result
