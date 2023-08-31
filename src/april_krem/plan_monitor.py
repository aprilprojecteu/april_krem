import rospy
from threading import Thread
import networkx as nx

from unified_planning.plans import Plan, SequentialPlan, TimeTriggeredPlan
from unified_planning.shortcuts import StartTiming, EndTiming, TimeInterval


class PlanMonitor:
    def __init__(self, plan: Plan, graph: nx.DiGraph) -> None:
        self._plan = plan
        self._graph = graph
        self._monitor = self.get_monitor(plan)

    def get_monitor(self, plan: Plan):
        """Get the executor for the given plan."""
        if isinstance(plan, SequentialPlan):
            return SequentialPlanMonitor(self._graph)
        elif isinstance(plan, TimeTriggeredPlan):
            return ParallelPlanMonitor(self._graph)
        else:
            raise NotImplementedError("Plan type not supported")

    def check_preconditions(self, task_id) -> bool:
        return self._monitor.check_preconditions(task_id)

    def check_postconditions(self, task_id) -> bool:
        return self._monitor.check_postconditions(task_id)


class SequentialPlanMonitor:
    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    def check_preconditions(self, task_id):
        """Check preconditions of the given task."""
        result = True
        conditions = self._graph.nodes[task_id]["preconditions"]["start"]
        context = self._graph.nodes[task_id]["context"]
        node_name = self._graph.nodes[task_id]["node_name"]
        parameters = self._graph.nodes[task_id]["parameters"]

        for i, condition in enumerate(conditions):
            eval_result = eval(  # pylint: disable=eval-used
                compile(condition, filename="<ast>", mode="eval"), context
            )

            # Check if all preconditions return boolean True
            if not eval_result:
                rospy.logerr(
                    f"Precondition {i+1} for action {node_name}{tuple(parameters.values())} failed!"
                )
                result = False

        return result

    def check_postconditions(self, task_id):
        """Check postconditions of the given task."""
        result = True
        post_conditions = self._graph.nodes[task_id]["postconditions"]
        context = self._graph.nodes[task_id]["context"]
        node_name = self._graph.nodes[task_id]["node_name"]
        parameters = self._graph.nodes[task_id]["parameters"]

        for i, (_, conditions) in enumerate(post_conditions.items()):
            for condition, value in conditions:
                actual = eval(  # pylint: disable=eval-used
                    compile(condition, filename="<ast>", mode="eval"), context
                )
                expected = eval(  # pylint: disable=eval-used
                    compile(value, filename="<ast>", mode="eval"), context
                )

                if actual != expected:
                    rospy.logerr(
                        f"Postcondition {i+1} for action {node_name}{tuple(parameters.values())} failed!"
                    )
                    result = False

        return result


class ParallelPlanMonitor:
    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    def check_preconditions(self, task_id):
        """Check preconditions of the given task."""
        result = True
        conditions = self._graph.nodes[task_id]["preconditions"]
        context = self._graph.nodes[task_id]["context"]
        node_name = self._graph.nodes[task_id]["node_name"]
        parameters = self._graph.nodes[task_id]["parameters"]

        start, end = StartTiming(), EndTiming()
        start_interval, end_interval = TimeInterval(start, start), TimeInterval(
            end, end
        )
        overall_interval = TimeInterval(start, end)
        intervals = list(conditions.keys())
        start_conditions = (
            conditions[start_interval] if start_interval in intervals else []
        )
        overall_conditions = (
            conditions[overall_interval] if overall_interval in intervals else []
        )
        end_conditions = conditions[end_interval] if end_interval in intervals else []

        # Check start conditions
        for i, condition in enumerate(start_conditions):
            eval_result = self._check_precondition(condition, context)

            if not eval_result:
                rospy.logerr(
                    f"Precondition {i+1} for action {node_name}{tuple(parameters.values())} failed!"
                )
                result = False

        # Check overall conditions
        # Add threads for each overall condition
        # TODO: Add failure handling for threads
        for i, condition in enumerate(overall_conditions):
            thread = Thread(
                target=self._check_precondition,
                args=(condition, context),
                name=f"overall_condition_{i+1}",
                daemon=True,
            )
            thread.start()

        # Check end conditions
        for i, condition in enumerate(end_conditions):
            eval_result = self._check_precondition(condition, context)

            if not eval_result:
                rospy.logerr(
                    f"Precondition {i+1} for action {node_name}{tuple(parameters.values())} failed!"
                )
                result = False

        return result

    def check_postconditions(self, task_id):
        result = True
        conditions = self._graph.nodes[task_id]["postconditions"]
        context = self._graph.nodes[task_id]["context"]
        node_name = self._graph.nodes[task_id]["node_name"]
        parameters = self._graph.nodes[task_id]["parameters"]

        start, end = StartTiming(), EndTiming()
        start_interval, end_interval = TimeInterval(start, start), TimeInterval(
            end, end
        )
        overall_interval = TimeInterval(start, end)
        intervals = list(conditions.keys())
        start_conditions = (
            conditions[start_interval] if start_interval in intervals else []
        )
        overall_conditions = (
            conditions[overall_interval] if overall_interval in intervals else []
        )
        end_conditions = conditions[end_interval] if end_interval in intervals else []

        # Check start conditions
        for i, (condition, value) in enumerate(start_conditions):
            eval_result = self._check_postcondition(condition, value, context)

            if not eval_result:
                rospy.logerr(
                    f"Postcondition {i+1} for action {node_name}{tuple(parameters.values())} failed!"
                )
                result = False

        # Check overall conditions
        for i, (condition, value) in enumerate(overall_conditions):
            thread = Thread(
                target=self._check_postcondition,
                args=(condition, value, context),
                name=f"overall_condition_{i+1}",
                daemon=True,
            )
            thread.start()

        # Check end conditions
        for i, (condition, value) in enumerate(end_conditions):
            eval_result = self._check_postcondition(condition, value, context)

            if not eval_result:
                rospy.logerr(
                    f"Postcondition {i+1} for action {node_name}{tuple(parameters.values())} failed!"
                )
                result = False

        return result

    def _check_precondition(self, condition, context):
        result = eval(  # pylint: disable=eval-used
            compile(condition, filename="<ast>", mode="eval"), context
        )

        return result

    def _check_postcondition(self, condition, value, context):
        """Check postconditions of the given task."""
        actual = eval(  # pylint: disable=eval-used
            compile(condition, filename="<ast>", mode="eval"), context
        )
        expected = eval(  # pylint: disable=eval-used
            compile(value, filename="<ast>", mode="eval"), context
        )

        return actual == expected
