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


class Location(Enum):
    conveyor_a = "conveyor_a"
    conveyor_b = "conveyor_b"
    in_hand = "in_hand"
    in_bag = "in_bag"


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

        self.item_in_hand = Item.nothing
        self.item_in_bag = Item.nothing

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
        # self._hicem_run_action_client.wait_for_server()
        rospy.loginfo("HICEM Run Symbolic Action Server found!")

        self._gesture_backup_buttons = rospy.Subscriber(
            "/isim/hmi/gesture_backup_buttons",
            GestureBackupButtonStates,
            self._gesture_backup_button_cb,
        )
        # PENDING, ACTIVE, RECALLED, REJECTED, PREEMPTED, ABORTED, SUCCEEDED, LOST.
        self._current_action_status = "LOST"

    def run_symbolic_action(
        self, action_name: str, target_id: str = "", grasp_facts=[], timeout=60.0
    ) -> bool:
        if target_id is None:
            target_id = ""
        run_symbolic_action_goal_msg = RunSymbolicActionGoal(
            action_type=action_name, target_id=target_id, grasp_facts=grasp_facts
        )

        self._hicem_run_action_client.send_goal(
            run_symbolic_action_goal_msg, done_cb=self._symbolic_action_done_cb
        )
        self._current_action_status = "ACTIVE"

        rate = rospy.Rate(10)

        while self._current_action_status == "ACTIVE" and not rospy.is_shutdown():
            rate.sleep()

        if self._current_action_status == "SUCCEEDED":
            return True
        elif self._current_action_status == "PREEMPTED":
            return False
        elif self._current_action_status == "ERROR":
            return False
        elif self._current_action_status == "LOST":
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

    def reject_insole(self, conveyor: Location):
        result = self.run_symbolic_action("reject_insole", timeout=20.0)
        return result

    def get_next_insole(self, conveyor: Location, insole: Item):
        result = self.run_symbolic_action("get_next_insole", timeout=20.0)
        return result

    def preload_bag_bundle(self):
        result = self.run_symbolic_action("preload_bag_bundle", timeout=20.0)
        return result

    def load_bag(self, bag: Item):
        result = self.run_symbolic_action("load_bag", timeout=20.0)
        return result

    def open_bag(self, bag: Item):
        result = self.run_symbolic_action("open_bag", timeout=20.0)
        return result

    def match_insole_bag(self, insole: Item, bag: Item):
        result = self.run_symbolic_action("match_insole_bag", timeout=20.0)
        return result

    def pick_insole(self, insole: Item):
        grasp_facts = self._grasp_library_srv("mia", "insole_model_1", "bagging", False)
        # TODO Object ID
        result = self.run_symbolic_action(
            "pick_insole", "1", grasp_facts.grasp_strategies, 120.0
        )
        if result:
            self._env.item_in_hand = insole
        return result

    def pick_set(self, insole: Item, bag: Item):
        grasp_facts = self._grasp_library_srv("mia", "set_1", "sealing", False)
        # TODO Object ID
        result = self.run_symbolic_action(
            "pick_set", "1", grasp_facts.grasp_strategies, 120.0
        )
        if result:
            self._env.item_in_hand = bag
        return result

    def insert(self, insole: Item, bag: Item):
        # TODO Object ID
        result = self.run_symbolic_action("insert", "1", timeout=120.0)
        if result:
            self._env.item_in_hand = Item.nothing
            self._env.item_in_bag = insole
        return result

    def perceive_insole(self, insole: Item):
        # TODO Object ID
        result = self.run_symbolic_action("perceive_insole", "1", timeout=20.0)
        return result

    def perceive_bag(self, bag: Item):
        # TODO Object ID
        result = self.run_symbolic_action("perceive_bag", "1", timeout=20.0)
        return result

    def perceive_set(self, insole: Item, bag: Item):
        # TODO Object ID
        result = self.run_symbolic_action("perceive_set", "1", timeout=20.0)
        return result

    def release_bag(self, insole: Item, bag: Item):
        result = self.run_symbolic_action("release_bag", timeout=20.0)
        return result

    def seal_set(self, bag: Item):
        # TODO Object ID
        result = self.run_symbolic_action("seal_set", "1", timeout=120.0)
        if result:
            self._env.item_in_hand = Item.nothing
            self._env.item_in_bag = Item.nothing
        return result
