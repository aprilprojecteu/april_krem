import rospy
import actionlib
import networkx as nx
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple

from unified_planning.plans import (
    Plan,
    SequentialPlan,
    TimeTriggeredPlan,
    ActionInstance,
)

from april_krem.plan_monitor import PlanMonitor
from april_krem.plan_visualization import PlanVisualization

from april_msgs.msg import (
    RunSymbolicActionGoal,
    RunSymbolicActionAction,
)


class KREM_STATE(Enum):
    START = 0
    ACTIVE = 1
    PAUSED = 2
    CANCELED = 3
    TIMEOUT = 4
    ERROR = 5
    FINISHED = 6
    RESET = 7
    HOME = 8


class PlanDispatcher:
    STATE = KREM_STATE.START
    HICEM_ACTION_SERVER = actionlib.SimpleActionClient(
        "/hicem/run/symbolic_action", RunSymbolicActionAction
    )
    DOMAIN = None
    KREM_LOGGING = None

    def __init__(self, domain, krem_logging, enable_monitor: bool = False):
        PlanDispatcher.DOMAIN = domain
        PlanDispatcher.KREM_LOGGING = krem_logging
        self._plan = None
        self._graph = None
        self._executor = None
        self._monitor = None
        self._plan_viz = None
        self._enable_monitor = enable_monitor
        self._node_id_to_action_map: Dict[int, ActionInstance] = {}

        # HICEM Run Symbolic Action server
        rospy.loginfo("Waiting for HICEM Run Symbolic Action Server...")
        PlanDispatcher.HICEM_ACTION_SERVER.wait_for_server()
        rospy.loginfo("HICEM Run Symbolic Action Server found!")

        self._acb_display_text = rospy.get_param("~ACB_display_text", default="")

    def execute_plan(self, plan: Plan, graph: nx.DiGraph) -> bool:
        """Execute the plan."""

        execution_status = "executing"
        self._graph = graph
        self._plan = plan

        self.set_executor(plan, graph)
        if self._enable_monitor:
            self._monitor = PlanMonitor(plan, graph)

        self.create_plan_vizualization(plan)

        failed_actions = []
        executed_action_ids = []

        for node_id, node in self._graph.nodes(data=True):
            results = None
            if node["action"] == "end":
                continue

            successors = list(self._graph.successors(node_id))
            # remove start, end and already executed action nodes from successors
            successors = [
                id
                for id in successors
                if self._graph.nodes[id]["action"] not in ["start", "end"]
                and id not in executed_action_ids
            ]

            # check preconditions
            if self._enable_monitor:
                for succ_id in successors:
                    action_name = self._graph.nodes[succ_id]["action"]
                    if action_name not in ["start", "end"]:
                        precondition_result = self._monitor.check_preconditions(succ_id)
                        if not precondition_result:
                            execution_status = "failed"
                            failed_actions.append(action_name)
                            self._plan_viz.fail(self._node_id_to_action_map[succ_id])
                if execution_status != "executing":
                    break

            # execute action
            for succ_id in successors:
                action_name = self._graph.nodes[succ_id]["action"]
                if action_name not in ["start", "end"]:
                    self._plan_viz.execute(self._node_id_to_action_map[succ_id])
            if successors:
                results = self._executor.execute_action(successors)

            if results is not None:
                for succ_id, result in results.items():
                    action_name = self._graph.nodes[succ_id]["action"]
                    if action_name not in ["start", "end"]:
                        if not result[0]:
                            execution_status = result[1]
                            failed_actions.append(action_name)
                            self._plan_viz.fail(self._node_id_to_action_map[succ_id])
                if execution_status != "executing":
                    break

            # check postconditions
            if self._enable_monitor:
                for succ_id in successors:
                    action_name = self._graph.nodes[succ_id]["action"]
                    if action_name not in ["start", "end"]:
                        postcondition_result = self._monitor.check_postconditions(
                            succ_id
                        )
                        if not postcondition_result:
                            execution_status = "failed"
                            failed_actions.append(action_name)
                            self._plan_viz.fail(self._node_id_to_action_map[succ_id])
                if execution_status != "executing":
                    break

            for succ_id in successors:
                self._plan_viz.succeed(self._node_id_to_action_map[succ_id])
                executed_action_ids.append(succ_id)

        if execution_status == "reset":
            PlanDispatcher.change_state(KREM_STATE.CANCELED)
            rospy.sleep(1)
            if PlanDispatcher.DOMAIN is not None:
                PlanDispatcher.DOMAIN.specific_domain._env.reset_env_keep_counters()
            PlanDispatcher.wait_for_human_intervention_message(
                "Gesture or press continue to continue cycle at beginning."
            )

        if PlanDispatcher.STATE in [
            KREM_STATE.CANCELED,
            KREM_STATE.RESET,
            KREM_STATE.HOME,
        ]:
            return False

        if execution_status != "executing":
            if failed_actions:
                replanning = self._acb_display_text.get(failed_actions[0], {}).get(
                    "replanning", False
                )
                message = self._acb_display_text.get(failed_actions[0], {}).get(
                    "message", "UNKNOWN ERROR"
                )
                if execution_status == "wait_for_human_intervention":
                    replanning = False
                if replanning:
                    PlanDispatcher.KREM_LOGGING.error_replan_counter += 1
                    rospy.loginfo(message)
                else:
                    if not rospy.is_shutdown():
                        PlanDispatcher.KREM_LOGGING.wfhi_counter += 1
                    wait_for_human_intervention_result = (
                        self.wait_for_human_intervention_action(message)
                    )
                    if wait_for_human_intervention_result:
                        self.change_state(KREM_STATE.ACTIVE)
                    else:
                        self.change_state(KREM_STATE.ERROR)
            return False
        else:
            self.change_state(KREM_STATE.FINISHED)

        return True

    def create_plan_vizualization(self, plan: Plan) -> None:
        if isinstance(plan, SequentialPlan):
            self._plan_viz = PlanVisualization()
            self._plan_viz.set_actions(plan.actions)
            # create map between executed graph nodes and plan action instances for visualization
            id = 1
            for action in plan.actions:
                self._node_id_to_action_map[id] = action
                id += 1
        elif isinstance(plan, TimeTriggeredPlan):
            actions_viz = [a[1] for a in plan.timed_actions]
            self._plan_viz = PlanVisualization()
            self._plan_viz.set_actions(actions_viz)
            # create map between executed graph nodes and plan action instances for visualization
            id = 1
            for action in actions_viz:
                self._node_id_to_action_map[id] = action
                id += 1
        else:
            raise NotImplementedError("Plan type not supported")

    def set_executor(self, plan: Plan, graph: nx.DiGraph) -> None:
        """Get the executor for the given plan."""
        if isinstance(plan, SequentialPlan):
            self._executor = SequentialPlanExecutor(graph)
        elif isinstance(plan, TimeTriggeredPlan):
            self._executor = ParallelPlanExecutor(graph)
        else:
            raise NotImplementedError("Plan type not supported")

    def wait_for_human_intervention_action(self, display_message: str = "") -> bool:
        result, _ = self.run_symbolic_action(
            "wait_for_human_intervention", [display_message]
        )
        return result

    @classmethod
    def wait_for_human_intervention_message(cls, display_message: str = "") -> None:
        wait_for_human_intervention_goal_msg = RunSymbolicActionGoal(
            action_type="wait_for_human_intervention",
            action_arguments=[display_message],
            grasp_facts=[],
        )
        cls.HICEM_ACTION_SERVER.send_goal(wait_for_human_intervention_goal_msg)

    @classmethod
    def arm_to_home_pose_action(cls) -> bool:
        home_pose_action_msg = RunSymbolicActionGoal(
            action_type="move_to_home_pose",
            action_arguments=[],
            grasp_facts=[],
        )

        rospy.loginfo(
            '\033[92mDispatcherROS: Dispatching action "move_to_home_pose"\033[0m'
        )

        start_time = rospy.get_rostime().to_sec()

        cls.HICEM_ACTION_SERVER.send_goal(home_pose_action_msg)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if cls.HICEM_ACTION_SERVER.get_result():
                # result received
                home_result = cls.HICEM_ACTION_SERVER.get_result()
                if home_result.success:
                    return True
                else:
                    return False
            if rospy.get_rostime().to_sec() - start_time > 120.0:
                rospy.logwarn("Moving to home pose timed out after 120 seconds!")
                cls.HICEM_ACTION_SERVER.cancel_all_goals()
                break
            if cls.STATE in [
                KREM_STATE.ERROR,
                KREM_STATE.TIMEOUT,
                KREM_STATE.RESET,
                KREM_STATE.CANCELED,
                KREM_STATE.PAUSED,
            ]:
                break

            rate.sleep()
        return False

    @classmethod
    def change_state(cls, state):
        # only change state if in a different state
        if cls.STATE != state:
            if state == KREM_STATE.ACTIVE:
                cls.STATE = state
            elif state == KREM_STATE.PAUSED:
                cls.STATE = state
            elif state == KREM_STATE.CANCELED:
                cls.HICEM_ACTION_SERVER.cancel_all_goals()
                cls.STATE = state
            elif state == KREM_STATE.TIMEOUT:
                cls.STATE = state
            elif state == KREM_STATE.ERROR:
                cls.STATE = state
            elif state == KREM_STATE.FINISHED:
                cls.STATE = state
            elif state == KREM_STATE.RESET:
                cls.HICEM_ACTION_SERVER.cancel_all_goals()
                cls.STATE = state
            elif state == KREM_STATE.HOME:
                cls.HICEM_ACTION_SERVER.cancel_all_goals()
                cls.STATE = state
            else:
                raise RuntimeError(f"Tried to change to unknown state: {state}!")

    @classmethod
    def run_symbolic_action(
        cls,
        action_name: str,
        action_arguments=[],
        grasp_facts=[],
        timeout=0.0,
        number_of_retries=2,
    ) -> Tuple[bool, str]:
        rate = rospy.Rate(10)

        run_symbolic_action_goal_msg = RunSymbolicActionGoal(
            action_type=action_name,
            action_arguments=action_arguments,
            grasp_facts=grasp_facts,
        )

        rospy.loginfo(f"\033[92mDispatcherROS: Dispatching action {action_name}\033[0m")

        cls.HICEM_ACTION_SERVER.send_goal(run_symbolic_action_goal_msg)
        start_time = rospy.get_rostime().to_sec()
        _action_status = "EXECUTING"
        _action_result = None
        error_count = 0
        pause_start_time = 0.0
        initial_timeout = timeout

        # Loop till action is finished
        while not rospy.is_shutdown() and (
            _action_status == "EXECUTING" or _action_status == "PAUSED"
        ):
            # Check if KREM is paused
            while cls.STATE == KREM_STATE.PAUSED:
                # Remember start time of pause to extend timeout and cancel current execution
                if pause_start_time <= 0.0:
                    pause_start_time = rospy.get_rostime().to_sec()
                    cls.HICEM_ACTION_SERVER.cancel_all_goals()
                    # status is paused
                    _action_status = "PAUSED"
                    rate.sleep()
                    # send wait for human intervention action to HICEM
                    cls.wait_for_human_intervention_message(
                        "Press continue to restart action."
                    )

                # Wait for unpause
                rate.sleep()
            else:
                if cls.STATE in [
                    KREM_STATE.ERROR,
                    KREM_STATE.TIMEOUT,
                    KREM_STATE.RESET,
                    KREM_STATE.CANCELED,
                    KREM_STATE.HOME,
                ]:
                    break
                # Only execute after a pause
                if _action_status == "PAUSED":
                    # if timeout set, extend it by pause time
                    if timeout > 0.0:
                        timeout += rospy.get_rostime().to_sec() - pause_start_time
                        pause_start_time = 0.0
                    # resend action to HICEM
                    cls.HICEM_ACTION_SERVER.send_goal(run_symbolic_action_goal_msg)
                    rospy.loginfo(
                        f"\033[92mDispatcherROS: Dispatching action {action_name}\033[0m"
                    )
                    # status back to executing
                    _action_status = "EXECUTING"
            # check if result is available from HICEM
            if cls.HICEM_ACTION_SERVER.get_result():
                # result received
                _action_result = cls.HICEM_ACTION_SERVER.get_result()
                if _action_result.success:
                    rospy.loginfo(
                        f"\033[92mDispatcherROS: Action {action_name} successful!\033[0m"
                    )
                    cls.KREM_LOGGING.log_info(
                        f"Action {action_name} finished after {rospy.get_rostime().to_sec() - start_time} seconds."
                    )
                    break
                else:
                    for error in _action_result.errors:
                        # Check for FORCE_OVERLOAD = 181
                        if error.level_error == 181:
                            rospy.logwarn(
                                f"\033[93mDispatcherROS: FORCE_OVERLOAD during {action_name}"
                                " action! Waiting for human intervention!\033[0m"
                            )
                            return (False, "error")
                    if error_count < (number_of_retries):
                        error_count += 1
                        rospy.logwarn(
                            f"\033[93mDispatcherROS: Action {action_name} failed! Retrying...\033[0m"
                        )
                        cls.HICEM_ACTION_SERVER.send_goal(run_symbolic_action_goal_msg)
                    else:
                        rospy.logwarn(
                            f"\033[91mDispatcherROS: Action {action_name} failed!\033[0m"
                        )
                        return (False, "failed")
            # Check if action timed out, cancel action, return failure
            if timeout > 0.0 and rospy.get_rostime().to_sec() - start_time > timeout:
                rospy.logwarn(
                    f"{action_name} timed out after {initial_timeout} seconds!"
                )
                cls.HICEM_ACTION_SERVER.cancel_all_goals()
                return (False, "timeout")

            rate.sleep()

        return (
            (_action_result, "success")
            if _action_result is not None and _action_result
            else (False, "failed")
        )


class ParallelPlanExecutor:
    def __init__(self, graph: nx.DiGraph):
        self._graph = graph

    def execute_action(self, node_ids) -> Dict[int, bool]:
        """Execute the action."""
        results = {i: True for i in node_ids}
        # Fetch complete node data
        actions = [(id, self._graph.nodes[id]) for id in node_ids]

        with ThreadPoolExecutor(max_workers=len(node_ids)) as executor:
            futures = [
                executor.submit(self._execute_concurrent_action, action)
                for action in actions
            ]

        for future in as_completed(futures):
            id, result, msg = future.result()
            results[id] = (result, msg)

        return results

    def _execute_concurrent_action(self, action):
        # TODO: Better implementation
        # FIXME: Duplicate execution of actions
        parameters = action[1]["parameters"]
        executor = action[1]["context"][action[1]["action"]]
        result, msg = executor(**parameters)

        return action[0], result, msg


class SequentialPlanExecutor:
    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    def execute_action(self, node_ids) -> Dict[int, bool]:
        """Execute the action."""
        node = self._graph.nodes[node_ids[0]]
        parameters = node["parameters"]
        executor = node["context"][node["action"]]

        # Execute action
        result, msg = executor(**parameters)

        return {node_ids[0]: (result, msg)}
