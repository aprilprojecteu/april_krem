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

# simulated actions
from std_srvs.srv import Trigger, SetBool
from pbr_msgs.msg import PickObjectAction, PickObjectGoal, PlaceObjectAction
from pbr_msgs.msg import PlaceObjectGoal, InsertObjectAction, InsertObjectGoal
from pose_selector.srv import ClassQuery
from conveyor_belt_sim.srv import ConveyorBeltControl


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


class SimulationEnvironment:
    def __init__(self):
        use_case = rospy.get_param("use_case", default="uc6")
        facts_config_file = use_case + "_facts_config_sim.yaml"
        facts_config_path = os.path.join(
            rospkg.RosPack().get_path("symbolic_fact_generation_use_cases"),
            "config",
            facts_config_file,
        )

        self._fact_generator = FactGenerationWithConfig(facts_config_path)

        self.item_in_hand = Item.nothing
        self.item_in_bag = Item.nothing

    def holding(self, item: Item) -> bool:
        facts = self._fact_generator.generate_facts_with_name("holding")
        for fact in facts:
            if fact.name == "holding" and fact.values[0] == item.name:
                return True
            else:
                return False

        return False

    def item_pose_is_known(self, item: Item) -> bool:
        facts = self._fact_generator.generate_facts_with_name("item_pose_is_known")
        for fact in facts:
            if (
                fact.name == "item_pose_is_known" and item.name in fact.values[0]
            ):  # TODO IGNORING IDs
                return True

        return False

    def item_in_fov(self) -> bool:
        facts = self._fact_generator.generate_facts_with_name("item_in_fov")
        for fact in facts:
            if fact.name == "item_in_fov" and fact.values[0] == "true":
                return True

        return False

    def moving(self, conveyor: Location) -> bool:
        facts = self._fact_generator.generate_facts_with_name("moving_stationary")
        for fact in facts:
            if fact.name == "stationary" and fact.values[0] == conveyor.name:
                return False
            elif fact.name == "moving" and fact.values[0] == conveyor.name:
                return True
        return False

    def stationary(self, conveyor: Location) -> bool:
        facts = self._fact_generator.generate_facts_with_name("moving_stationary")
        for fact in facts:
            if fact.name == "stationary" and fact.values[0] == conveyor.name:
                return True
            elif fact.name == "moving" and fact.values[0] == conveyor.name:
                return False
        return True

    def bag_set_released(self) -> bool:
        facts = self._fact_generator.generate_facts_with_name("bag_set_released")
        for fact in facts:
            if fact.name == "bag_set_released" and fact.values[0] == "true":
                return True

        return False

    def set_available(self, insole: Item, bag: Item) -> bool:
        facts = self._fact_generator.generate_facts_with_name("april_on_facts")
        for fact in facts:
            # TODO IGNORING IDs
            if (
                fact.name == "set_available"
                and insole.name in fact.values[0]
                and bag.name in fact.values[1]
            ):
                return True
        return False

    def bag_is_available(self, bag: Item) -> bool:
        facts = self._fact_generator.generate_facts_with_name("april_on_facts")
        for fact in facts:
            if (
                fact.name == "bag_is_available" and bag.name in fact.values[0]
            ):  # TODO IGNORING IDs
                return True
        return False


class Actions:
    def __init__(self, env: Environment):
        self._env = env
        # Grasp Library
        rospy.wait_for_service("/krem/grasp_library")
        self._grasp_library_srv = rospy.ServiceProxy(
            "/krem/grasp_library", GetGraspStrategy
        )

        self._hicem_run_action_client = actionlib.SimpleActionClient(
            "/hicem/run/symbolic_action", RunSymbolicActionAction
        )
        self._hicem_run_action_client.wait_for_server()
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


class SimulatedActions:
    def __init__(self, env: SimulationEnvironment):
        self._timeout = 60.0
        self._env = env

        rospy.loginfo("Waiting for simulated action services.")

        # start_conveyor service
        conveyor_a_srv_name = "/machinery/conveyor_belt_a/control"
        conveyor_b_srv_name = "/machinery/conveyor_belt_b/control"
        rospy.wait_for_service(conveyor_a_srv_name, self._timeout)
        rospy.wait_for_service(conveyor_b_srv_name, self._timeout)
        self._conveyor_a_srv = rospy.ServiceProxy(
            conveyor_a_srv_name, ConveyorBeltControl
        )
        self._conveyor_b_srv = rospy.ServiceProxy(
            conveyor_b_srv_name, ConveyorBeltControl
        )

        # get_next_insole service
        next_insole_srv_name = "/get_next_insole"
        rospy.wait_for_service(next_insole_srv_name, self._timeout)
        self._get_next_insole_srv = rospy.ServiceProxy(next_insole_srv_name, Trigger)

        # spawn bag service
        spawn_bag_srv_name = "/load_bag"
        rospy.wait_for_service(spawn_bag_srv_name, self._timeout)
        self._spawn_bag_srv = rospy.ServiceProxy(spawn_bag_srv_name, Trigger)

        # release bag service
        release_bag_srv_name = "/release_bag"
        rospy.wait_for_service(release_bag_srv_name, self._timeout)
        self._release_bag_srv = rospy.ServiceProxy(release_bag_srv_name, Trigger)

        # reset bag dispenser service
        reset_bag_dispenser_srv_name = "/bag_dispenser_reset"
        rospy.wait_for_service(reset_bag_dispenser_srv_name, self._timeout)
        self._reset_bag_dispenser_srv = rospy.ServiceProxy(
            reset_bag_dispenser_srv_name, Trigger
        )

        # perception
        # pose selector activation/deactivation service
        pose_selector_activate_srv_name = rospy.get_param(
            "~pose_selector_activate_srv_name",
            "/pick_pose_selector_node/pose_selector_activate",
        )
        pose_selector_class_query_srv_name = rospy.get_param(
            "~pose_selector_class_query_srv_name",
            "/pick_pose_selector_node/pose_selector_class_query",
        )
        pose_selector_clear_srv_name = rospy.get_param(
            "~pose_selector_clear_srv_name",
            "/pick_pose_selector_node/pose_selector_clear",
        )

        rospy.wait_for_service(pose_selector_activate_srv_name, self._timeout)
        rospy.wait_for_service(pose_selector_class_query_srv_name, self._timeout)
        rospy.wait_for_service(pose_selector_clear_srv_name, self._timeout)

        self._activate_pose_selector_srv = rospy.ServiceProxy(
            pose_selector_activate_srv_name, SetBool
        )
        self._class_query_pose_selector_srv = rospy.ServiceProxy(
            pose_selector_class_query_srv_name, ClassQuery
        )
        self._clear_pose_selector_srv = rospy.ServiceProxy(
            pose_selector_clear_srv_name, Trigger
        )

        # pickup
        pick_object_server_name = rospy.get_param(
            "~pick_object_server_name", "/mia_hand_on_ur10e/pick_object"
        )
        self._pickup_action_client = actionlib.SimpleActionClient(
            pick_object_server_name, PickObjectAction
        )
        rospy.loginfo(f"waiting for {pick_object_server_name} action server")
        if self._pickup_action_client.wait_for_server(
            timeout=rospy.Duration.from_sec(self._timeout)
        ):
            rospy.loginfo(f"found {pick_object_server_name} action server")
        else:
            rospy.logerr(f"action server {pick_object_server_name} not available")

        # insert
        insert_object_server_name = rospy.get_param(
            "~insert_object_server_name", "/mia_hand_on_ur10e/insert_object"
        )
        self._insert_action_client = actionlib.SimpleActionClient(
            insert_object_server_name, InsertObjectAction
        )
        rospy.loginfo(f"waiting for {insert_object_server_name} action server")
        if self._insert_action_client.wait_for_server(
            timeout=rospy.Duration.from_sec(self._timeout)
        ):
            rospy.loginfo(f"found {insert_object_server_name} action server")
        else:
            rospy.logerr(f"action server {insert_object_server_name} not available")

        # place
        place_object_server_name = rospy.get_param(
            "~place_object_server_name", "/mia_hand_on_ur5/place_object"
        )
        self._place_action_client = actionlib.SimpleActionClient(
            place_object_server_name, PlaceObjectAction
        )
        rospy.loginfo(f"waiting for {place_object_server_name} action server")
        if self._place_action_client.wait_for_server(
            timeout=rospy.Duration.from_sec(self._timeout)
        ):
            rospy.loginfo(f"found {place_object_server_name} action server")
        else:
            rospy.logerr(f"action server {place_object_server_name} not available")

        rospy.loginfo("All simulated action services found!")

    def reject_insole(self, conveyor: Location):
        return True

    def get_next_insole(self, conveyor: Location, insole: Item):
        try:
            result = self._get_next_insole_srv()
            result = result.success
        except (rospy.ServiceException, AttributeError):
            result = False
        return result

    def preload_bag_bundle(self):
        # the worker inserts new bags, signal worker
        return True

    def load_bag(self, bag: Item):
        try:
            result = self._spawn_bag_srv()
            result = result.success
        except (rospy.ServiceException, AttributeError):
            result = False
        return result

    def open_bag(self, bag: Item):
        # Not possible in simulation
        return True

    def match_insole_bag(self, insole: Item, bag: Item):
        # Not possible in simulation
        return True

    def pick_insole(self, insole: Item):
        perceived_insoles = self.query_pose_selector_for_object_instances(insole.value)
        goal = PickObjectGoal()
        if perceived_insoles:
            goal.object_name = perceived_insoles[-1]  # take last insole
        else:
            goal.object_name = insole.value  # no insole perceived, take class name
        goal.support_surface_name = "conveyor_belt_a"  # TODO hardcoded surface name
        for i in range(3):
            rospy.loginfo(
                f"DispatcherROS: sending -> pick {goal.object_name}"
                f" from {goal.support_surface_name} <- goal to pick_object action server, try nr. {i+1}"
            )
            self._pickup_action_client.send_goal(goal)
            rospy.loginfo("Waiting for result from pick_object action server")
            if self._pickup_action_client.wait_for_result(
                rospy.Duration.from_sec(self._timeout)
            ):
                result = self._pickup_action_client.get_result()
                result = result.success
            else:
                rospy.logerr(f"Failed to pick {goal.object_name}, timeout?")
                result = False
            if result:
                self._env.item_in_hand = insole
                break
        return result

    def pick_set(self, insole: Item, bag: Item):
        result = False
        perceived_bags = self.query_pose_selector_for_object_instances(bag.value)
        perceived_insoles = self.query_pose_selector_for_object_instances(insole.value)
        goal = PickObjectGoal()
        if perceived_bags:
            goal.object_name = perceived_bags[-1]  # take last bag
        else:
            goal.object_name = bag.value  # no bag perceived, take class name
        if perceived_insoles:
            goal.ignore_object_list.append(perceived_insoles[-1])
        else:
            goal.ignore_object_list.append(insole.value)
        if len(goal.ignore_object_list) > 0:
            rospy.logwarn(
                f"the following objects: {goal.ignore_object_list} will not be added to the planning scene"
            )
        else:
            rospy.loginfo("all objects are taken into account in planning scene")
        goal.support_surface_name = "table"  # TODO hardcoded surface name
        for i in range(3):
            rospy.loginfo(
                f"DispatcherROS: sending -> pick {goal.object_name}"
                f" from {goal.support_surface_name} <- goal to pick_object action server, try nr. {i+1}"
            )
            self._pickup_action_client.send_goal(goal)
            rospy.loginfo("waiting for result from pick_object action server")
            if self._pickup_action_client.wait_for_result(
                rospy.Duration.from_sec(self._timeout)
            ):
                result = self._pickup_action_client.get_result()
                result = result.success
            else:
                rospy.logerr(f"Failed to pick {goal.object_name}, timeout?")
                result = False
            if result:
                self._env.item_in_hand = bag
                break
        return result

    def insert(self, insole: Item, bag: Item):
        result = False
        perceived_bags = self.query_pose_selector_for_object_instances(bag.value)
        goal = InsertObjectGoal()
        if perceived_bags:
            goal.support_surface_name = perceived_bags[-1]  # take last bag
        else:
            goal.support_surface_name = bag.value  # no bag perceived, take class name
        goal.observe_before_insert = False
        for i in range(3):
            rospy.loginfo(
                f"sending insert goal to insert_object action server, try nr. {i+1}"
            )
            self._insert_action_client.send_goal(goal)
            rospy.loginfo("waiting for result from insert_object action server")
            if self._insert_action_client.wait_for_result(
                rospy.Duration.from_sec(self._timeout)
            ):
                result = self._insert_action_client.get_result()
                result = result.success
            else:
                rospy.logerr("Failed to insert object")
                result = False
            if result:
                self._env.item_in_hand = Item.nothing
                self._env.item_in_bag = insole
                break
        return result

    def perceive_insole(self, insole: Item):
        try:
            # clear pose selector before perceiving
            self._clear_pose_selector_srv()
            # activate pose selector
            activation_result = self._activate_pose_selector_srv(True)
            # wait for some time
            rospy.loginfo(
                "DispatcherROS: pose selector activated, waiting 2 seconds before deactivation!"
            )
            rospy.sleep(2)
            # deactivate pose_selector
            deactivation_result = self._activate_pose_selector_srv(False)
            result = activation_result.success and deactivation_result.success
        except (rospy.ServiceException, AttributeError):
            result = False
        return result

    def perceive_bag(self, bag: Item):
        try:
            # clear pose selector before perceiving
            self._clear_pose_selector_srv()
            # activate pose selector
            activation_result = self._activate_pose_selector_srv(True)
            # wait for some time
            rospy.loginfo(
                "DispatcherROS: pose selector activated, waiting 2 seconds before deactivation!"
            )
            rospy.sleep(2)
            # deactivate pose_selector
            deactivation_result = self._activate_pose_selector_srv(False)
            result = activation_result.success and deactivation_result.success
        except (rospy.ServiceException, AttributeError):
            result = False
        return result

    def perceive_set(self, insole: Item, bag: Item):
        try:
            # clear pose selector before perceiving
            self._clear_pose_selector_srv()
            # activate pose selector
            activation_result = self._activate_pose_selector_srv(True)
            # wait for some time
            rospy.loginfo(
                "DispatcherROS: pose selector activated, waiting 2 seconds before deactivation!"
            )
            rospy.sleep(2)
            # deactivate pose_selector
            deactivation_result = self._activate_pose_selector_srv(False)
            result = activation_result.success and deactivation_result.success
        except (rospy.ServiceException, AttributeError):
            result = False
        return result

    def release_bag(self, insole: Item, bag: Item):
        try:
            result = self._release_bag_srv()
            result = result.success
        except (rospy.ServiceException, AttributeError):
            result = False
        return result

    def seal_set(self, bag: Item):
        result = False
        goal = PlaceObjectGoal()
        goal.support_surface_name = "conveyor_belt_b"  # TODO hardcoded surface name
        rospy.loginfo("sending place goal to place_object action server")
        self._place_action_client.send_goal(goal)
        rospy.loginfo("waiting for result from place_object action server")
        # timeout is 3 times longer because of 3 placing tries
        if self._place_action_client.wait_for_result(
            rospy.Duration.from_sec(self._timeout * 3)
        ):
            place_result = self._place_action_client.get_result()
            place_result = place_result.success
        else:
            rospy.logerr("Failed to place object")
            place_result = False

        if place_result:
            try:
                result = self._conveyor_b_srv(75.0)
                result = result.success
                # close bag dispenser fingers
                self._reset_bag_dispenser_srv()
                self._env.item_in_hand = Item.nothing
                self._env.item_in_bag = Item.nothing
            except (rospy.ServiceException, AttributeError):
                result = False

        return result

    def query_pose_selector_for_object_instances(self, object_class):
        object_instances = []
        resp = self._class_query_pose_selector_srv(object_class)
        for obj in resp.poses:
            object_instances.append(obj.class_id + "_" + str(obj.instance_id))
        return object_instances
