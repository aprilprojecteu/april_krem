import itertools
from typing import Dict, List, Optional, Union

from unified_planning.model import Object
from unified_planning.model.htn import HierarchicalProblem
from unified_planning.model.metrics import (
    MinimizeMakespan,
    MinimizeSequentialPlanLength,
)
from unified_planning.engines import OptimalityGuarantee
from unified_planning.plans import Plan
from unified_planning.shortcuts import OneshotPlanner


class Domain:
    def __init__(self, use_case: str = "uc6", temporal: bool = False):
        self._use_case = use_case

        if use_case == "uc1":
            pass
        elif use_case == "uc2":
            pass
        elif use_case == "uc3":
            pass
        elif use_case == "uc4":
            pass
        elif use_case == "uc5_1":
            pass
        elif use_case == "uc5_2":
            pass
        elif use_case == "uc5_3":
            pass
        elif use_case == "uc5_4":
            pass
        elif use_case == "uc5_5":
            pass
        elif use_case == "uc6":
            from april_krem.INES_domain import INESDomain

            self.specific_domain = INESDomain(temporal)

    def solve(
        self,
        planner_name: Optional[str] = None,
        metric: Optional[Union[MinimizeSequentialPlanLength, MinimizeMakespan]] = None,
    ) -> Optional[Plan]:
        """Solve planning problem and return list of UP actions."""

        if metric is not None:
            self.problem.add_quality_metric(metric)
            planner = OneshotPlanner(
                name=planner_name,
                problem_kind=self.problem.kind,
                optimality_guarantee=OptimalityGuarantee.SOLVED_OPTIMALLY,
            )
        else:
            planner = OneshotPlanner(name=planner_name, problem_kind=self.problem.kind)
        return planner.solve(self.problem).plan

    def define_problem(self) -> HierarchicalProblem:
        """Define UP problem by its (potential subsets of) fluents, actions, and objects."""
        # Note: Reset goals and initial values to reuse this problem.
        problem = HierarchicalProblem()
        for fluent in self.specific_domain._fluents.values():
            problem.add_fluent(fluent, default_initial_value=False)
        problem.add_actions(self.specific_domain._actions.values())
        problem.add_objects(self.specific_domain._objects.values())
        for task in self.specific_domain.tasks:
            problem.add_task(task)
        for method in self.specific_domain.methods:
            problem.add_method(method)
        return problem

    def set_initial_values(self) -> None:
        """
        Set all initial values using the functions corresponding to this problem's fluents, whenever possible.
        Fluents, which have no fluent generating function are not set.

        Note: This will update all values for all parameter combinations for each fluent, which has a fluent function.
         Its intended usage is to update the planning problem by the current system state
         with one single function call.
        """
        type_objects: Dict[type, List[Object]] = {}
        # Collect objects in problem for all parameters of all fluents with fluent functions.
        for fluent in self.problem.fluents:
            for parameter in fluent.signature:
                # Avoid redundancy.
                if parameter.type not in type_objects.keys():
                    type_objects[parameter.type] = list(
                        self.problem.objects(parameter.type)
                    )
        for fluent in self.problem.fluents:
            # Loop through all parameter value combinations.
            for parameters in itertools.product(
                *[type_objects[parameter.type] for parameter in fluent.signature]
            ):
                # Use the fluent function to calculate the initial values.
                # Only use fluent function if available
                if fluent.name in self.specific_domain._fluent_functions:
                    value = (
                        self.specific_domain.get_object(
                            self.specific_domain._fluent_functions[fluent.name](
                                *[
                                    self.specific_domain._api_objects[parameter.name]
                                    for parameter in parameters
                                ]
                            )
                        )
                        if fluent.name in self.specific_domain._api_function_names
                        else self.specific_domain._fluent_functions[fluent.name](
                            *parameters
                        )
                    )
                    self.problem.set_initial_value(fluent(*parameters), value)

    def set_goal(self, goal: str = None) -> None:
        self.problem = self.define_problem()
        self.set_initial_values()
        self.specific_domain.set_state_and_goal(self.problem, goal)
