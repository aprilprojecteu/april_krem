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
    WAITING = 7


class PlanDispatcher:

    STATE = KREM_STATE.START
    HICEM_ACTION_SERVER = actionlib.SimpleActionClient(
        "/hicem/run/symbolic_action", RunSymbolicActionAction
    )
    DOMAIN = None

    def __init__(self, domain, enable_monitor: bool = False):
        PlanDispatcher.DOMAIN = domain
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
                if execution_status == "failed":
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
                if execution_status == "failed" or execution_status == "timeout":
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
                if execution_status == "failed":
                    break

            for succ_id in successors:
                self._plan_viz.succeed(self._node_id_to_action_map[succ_id])
                executed_action_ids.append(succ_id)

        if PlanDispatcher.STATE == KREM_STATE.CANCELED:
            return False

        if execution_status == "failed" or execution_status == "timeout":
            if failed_actions:
                replanning = self._acb_display_text.get(failed_actions[0], {}).get(
                    "replanning", False
                )
                message = self._acb_display_text.get(failed_actions[0], {}).get(
                    "message", "UNKNOWN ERROR"
                )
                if replanning:
                    rospy.loginfo(message)
                else:
                    wait_for_human_intervention_result = (
                        self.wait_for_human_intervention(message)
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

    def wait_for_human_intervention(self, display_message: str = ""):
        PlanDispatcher.change_state(KREM_STATE.WAITING)
        result, _ = self.run_symbolic_action(
            "wait_for_human_intervention", [display_message]
        )
        return result

    @classmethod
    def change_state(cls, state):
        # only change state if in a different state
        if cls.STATE != state:
            if state == KREM_STATE.ACTIVE:
                cls._before_active()
                cls.STATE = state
            elif state == KREM_STATE.PAUSED:
                cls._before_paused()
                cls.STATE = state
            elif state == KREM_STATE.CANCELED:
                cls._before_canceled()
                cls.STATE = state
            elif state == KREM_STATE.TIMEOUT:
                cls._before_timeout()
                cls.STATE = state
            elif state == KREM_STATE.ERROR:
                cls._before_error()
                cls.STATE = state
            elif state == KREM_STATE.FINISHED:
                cls._before_finished()
                cls.STATE = state
            elif state == KREM_STATE.WAITING:
                cls.STATE = state
            else:
                raise RuntimeError(f"Tried to change to unknown state: {state}!")

    @classmethod
    def _before_active(cls) -> None:
        pass

    @classmethod
    def _before_paused(cls) -> None:
        pass

    @classmethod
    def _before_canceled(cls) -> None:
        cls.HICEM_ACTION_SERVER.cancel_all_goals()
        if cls.DOMAIN is not None:
            cls.DOMAIN.specific_domain._env.reset_env()

    @classmethod
    def _before_timeout(cls) -> None:
        pass

    @classmethod
    def _before_error(cls) -> None:
        pass

    @classmethod
    def _before_finished(cls) -> None:
        pass

    @classmethod
    def run_symbolic_action(
        cls, action_name: str, action_arguments=[], grasp_facts=[], timeout=0.0
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
                # Wait for unpause
                rate.sleep()
            else:
                if cls.STATE in [
                    KREM_STATE.CANCELED,
                    KREM_STATE.ERROR,
                    KREM_STATE.TIMEOUT,
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
                _action_result = cls.HICEM_ACTION_SERVER.get_result().success
                if _action_result:
                    rospy.loginfo(
                        f"\033[92mDispatcherROS: Action {action_name} successful!\033[0m"
                    )
                    break
                else:
                    if error_count < 2:
                        error_count += 1
                        rospy.logwarn(
                            f"\033[93mDispatcherROS: Action {action_name} failed! Retrying...\033[0m"
                        )
                        cls.HICEM_ACTION_SERVER.send_goal(run_symbolic_action_goal_msg)
                    else:
                        rospy.logerr(
                            f"\033[91mDispatcherROS: Action {action_name} failed 3 times!\033[0m"
                        )
                        return (False, "failed")
            # Check if action timed out, cancel action, return failure
            if timeout > 0.0 and rospy.get_rostime().to_sec() - start_time > timeout:
                rospy.logerr(
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
