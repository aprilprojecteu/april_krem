from rospy import logerr
from unified_planning.model.htn import Task, Method
from unified_planning.shortcuts import And, Or, Not, StartTiming, EndTiming
from april_krem.ASIN_components import (
    Item,
    Tray,
    ArmPose,
    Actions,
    Environment,
)
from up_esb.bridge import Bridge


class ASINDomain(Bridge):
    def __init__(self, temporal: bool = False) -> None:
        Bridge.__init__(self)

        self._env = Environment()

        # Create types for planning based on class types
        self.create_types([Item, Tray, ArmPose])
        type_item = self.get_type(Item)
        type_tray = self.get_type(Tray)
        type_arm_pose = self.get_type(ArmPose)

        # Create fluents for planning
        self.holding = self.create_fluent_from_function(self._env.holding)
        self.item_in_fov = self.create_fluent_from_function(self._env.item_in_fov)
        self.item_type_is_known = self.create_fluent_from_function(
            self._env.item_type_is_known
        )
        self.chicken_to_pick = self.create_fluent_from_function(
            self._env.chicken_to_pick
        )
        self.tray_to_place = self.create_fluent_from_function(self._env.tray_to_place)
        self.tray_to_place_known = self.create_fluent_from_function(
            self._env.tray_to_place_known
        )
        self.space_in_tray = self.create_fluent_from_function(self._env.space_in_tray)
        self.tray_is_available = self.create_fluent_from_function(
            self._env.tray_is_available
        )
        self.trays_perceived = self.create_fluent_from_function(
            self._env.trays_perceived
        )
        self.current_arm_pose = self.create_fluent_from_function(
            self._env.current_arm_pose
        )

        # Create objects for both planning and execution
        self.items = self.create_enum_objects(Item)
        self.nothing = self.objects[Item.nothing.name]
        self.chicken_part = self.objects[Item.chicken_part.name]
        self.breast = self.objects[Item.breast.name]
        self.drumstick = self.objects[Item.drumstick.name]

        self.trays = self.create_enum_objects(Tray)
        self.unknown_tray = self.objects[Tray.unknown_tray.name]
        self.high_tray = self.objects[Tray.high_tray.name]
        self.med_tray = self.objects[Tray.med_tray.name]
        self.low_tray = self.objects[Tray.low_tray.name]
        self.discard_tray = self.objects[Tray.discard_tray.name]

        self.arm_poses = self.create_enum_objects(ArmPose)
        self.unknown_pose = self.objects[ArmPose.unknown_pose.name]
        self.over_conveyor = self.objects[ArmPose.over_conveyor.name]
        self.over_tray = self.objects[ArmPose.over_tray.name]

        # Create actions for planning
        self._create_domain_actions(temporal)

        # Tasks
        self.get_chicken = Task("get_chicken", chicken=type_item, tray=type_tray)
        self.place_chicken = Task("place_chicken", chicken=type_item, tray=type_tray)
        self.pack_chicken = Task("pack_chicken")

        self.tasks = (self.get_chicken, self.place_chicken, self.pack_chicken)

        # Methods

        # get_chicken
        # chicken already there, all information gathered
        self.get_chicken_noop = Method(
            "get_chicken_noop", chicken=type_item, tray=type_tray
        )
        self.get_chicken_noop.set_task(
            self.get_chicken, self.get_chicken_noop.chicken, self.get_chicken_noop.tray
        )
        self.get_chicken_noop.add_precondition(self.item_type_is_known())
        self.get_chicken_noop.add_precondition(self.tray_to_place_known())
        self.get_chicken_noop.add_precondition(self.trays_perceived())
        self.get_chicken_noop.add_precondition(self.item_in_fov())

        # chicken already there and perceived, perceive tray
        self.get_chicken_ptray = Method(
            "get_chicken_ptray", chicken=type_item, tray=type_tray
        )
        self.get_chicken_ptray.set_task(
            self.get_chicken,
            self.get_chicken_ptray.chicken,
            self.get_chicken_ptray.tray,
        )
        self.get_chicken_ptray.add_precondition(self.item_type_is_known())
        self.get_chicken_ptray.add_precondition(self.item_in_fov())
        self.get_chicken_ptray.add_precondition(self.tray_to_place_known())
        self.get_chicken_ptray.add_subtask(self.perceive_trays)

        # chicken already there and tray is known, still need to perceive
        self.get_chicken_perceive = Method(
            "get_chicken_perceive", chicken=type_item, tray=type_tray
        )
        self.get_chicken_perceive.set_task(
            self.get_chicken,
            self.get_chicken_perceive.chicken,
            self.get_chicken_perceive.tray,
        )
        self.get_chicken_perceive.add_precondition(self.item_in_fov())
        self.get_chicken_perceive.add_precondition(self.tray_to_place_known())
        st1 = self.get_chicken_perceive.add_subtask(self.perceive_chicken_part)
        st2 = self.get_chicken_perceive.add_subtask(self.perceive_trays)
        self.get_chicken_perceive.set_ordered(st1, st2)

        # chicken already there, need to estimate and perceive
        self.get_chicken_estimate = Method(
            "get_chicken_estimate", chicken=type_item, tray=type_tray
        )
        self.get_chicken_estimate.set_task(
            self.get_chicken,
            self.get_chicken_estimate.chicken,
            self.get_chicken_estimate.tray,
        )
        self.get_chicken_estimate.add_precondition(self.item_in_fov())
        st1 = self.get_chicken_estimate.add_subtask(self.estimate_part_shelf_life)
        st2 = self.get_chicken_estimate.add_subtask(self.perceive_chicken_part)
        st3 = self.get_chicken_estimate.add_subtask(self.perceive_trays)
        self.get_chicken_estimate.set_ordered(st1, st2, st3)

        # get next chicken over conveyor and gather all necessary information
        self.get_chicken_full = Method(
            "get_chicken_full", chicken=type_item, tray=type_tray
        )
        self.get_chicken_full.set_task(
            self.get_chicken, self.get_chicken_full.chicken, self.get_chicken_full.tray
        )
        self.get_chicken_full.add_precondition(Not(self.item_in_fov()))
        st1 = self.get_chicken_full.add_subtask(self.get_next_chicken_part)
        st2 = self.get_chicken_full.add_subtask(self.estimate_part_shelf_life)
        st3 = self.get_chicken_full.add_subtask(self.perceive_chicken_part)
        st4 = self.get_chicken_full.add_subtask(self.perceive_trays)
        self.get_chicken_full.set_ordered(st1, st2, st3, st4)

        # place chicken
        # chicken tray is full
        self.place_chicken_tray_full = Method(
            "place_chicken_tray_full", chicken=type_item, tray=type_tray
        )
        self.place_chicken_tray_full.set_task(
            self.place_chicken,
            self.place_chicken_tray_full.chicken,
            self.place_chicken_tray_full.tray,
        )
        self.place_chicken_tray_full.add_precondition(
            Not(
                self.space_in_tray(
                    self.place_chicken_tray_full.chicken,
                    self.place_chicken_tray_full.tray,
                )
            )
        )
        self.place_chicken_tray_full.add_precondition(
            self.tray_to_place(self.place_chicken_tray_full.tray)
        )
        self.place_chicken_tray_full.add_subtask(
            self.replace_filled_tray,
            self.place_chicken_tray_full.chicken,
            self.place_chicken_tray_full.tray,
        )

        # already holding chicken, place chicken in tray
        self.place_chicken_in_tray = Method(
            "place_chicken_in_tray", chicken=type_item, tray=type_tray
        )
        self.place_chicken_in_tray.set_task(
            self.place_chicken,
            self.place_chicken_in_tray.chicken,
            self.place_chicken_in_tray.tray,
        )
        self.place_chicken_in_tray.add_precondition(self.tray_to_place_known())
        self.place_chicken_in_tray.add_precondition(
            self.tray_to_place(self.place_chicken_in_tray.tray)
        )
        self.place_chicken_in_tray.add_precondition(
            self.holding(self.place_chicken_in_tray.chicken)
        )
        self.place_chicken_in_tray.add_precondition(
            self.tray_is_available(self.place_chicken_in_tray.tray)
        )
        self.place_chicken_in_tray.add_precondition(
            self.space_in_tray(
                self.place_chicken_in_tray.chicken, self.place_chicken_in_tray.tray
            )
        )
        self.place_chicken_in_tray.add_precondition(
            self.current_arm_pose(self.over_tray)
        )
        self.place_chicken_in_tray.add_subtask(
            self.insert_part_in_container,
            self.place_chicken_in_tray.chicken,
            self.place_chicken_in_tray.tray,
        )

        # already holding chicken, move and place
        self.place_chicken_move = Method(
            "place_chicken_move", chicken=type_item, tray=type_tray
        )
        self.place_chicken_move.set_task(
            self.place_chicken,
            self.place_chicken_move.chicken,
            self.place_chicken_move.tray,
        )
        self.place_chicken_move.add_precondition(self.tray_to_place_known())
        self.place_chicken_move.add_precondition(
            self.tray_to_place(self.place_chicken_move.tray)
        )
        self.place_chicken_move.add_precondition(
            self.holding(self.place_chicken_move.chicken)
        )
        self.place_chicken_move.add_precondition(
            self.tray_is_available(self.place_chicken_move.tray)
        )
        self.place_chicken_move.add_precondition(
            self.space_in_tray(
                self.place_chicken_move.chicken, self.place_chicken_move.tray
            )
        )
        self.place_chicken_move.add_precondition(self.current_arm_pose(self.unknown_pose))
        st1 = self.place_chicken_move.add_subtask(
            self.move_over_tray_cart,
            self.place_chicken_move.chicken,
            self.place_chicken_move.tray,
        )
        st2 = self.place_chicken_move.add_subtask(
            self.insert_part_in_container,
            self.place_chicken_move.chicken,
            self.place_chicken_move.tray,
        )
        self.place_chicken_move.set_ordered(st1, st2)

        # already over conveyor, pick and place
        self.place_chicken_pick = Method(
            "place_chicken_pick", chicken=type_item, tray=type_tray
        )
        self.place_chicken_pick.set_task(
            self.place_chicken,
            self.place_chicken_pick.chicken,
            self.place_chicken_pick.tray,
        )
        self.place_chicken_pick.add_precondition(self.tray_to_place_known())
        self.place_chicken_pick.add_precondition(
            self.tray_to_place(self.place_chicken_pick.tray)
        )
        self.place_chicken_pick.add_precondition(self.item_type_is_known())
        self.place_chicken_pick.add_precondition(
            self.space_in_tray(
                self.place_chicken_pick.chicken, self.place_chicken_pick.tray
            )
        )
        self.place_chicken_pick.add_precondition(self.current_arm_pose(self.over_conveyor))
        st1 = self.place_chicken_pick.add_subtask(
            self.pick_chicken_part, self.place_chicken_pick.chicken
        )
        st2 = self.place_chicken_pick.add_subtask(
            self.move_over_tray_cart,
            self.place_chicken_pick.chicken,
            self.place_chicken_pick.tray,
        )
        st3 = self.place_chicken_pick.add_subtask(
            self.insert_part_in_container,
            self.place_chicken_pick.chicken,
            self.place_chicken_pick.tray,
        )
        self.place_chicken_pick.set_ordered(st1, st2, st3)

        # pick and place chicken
        self.place_chicken_full = Method(
            "place_chicken_full", chicken=type_item, tray=type_tray
        )
        self.place_chicken_full.set_task(
            self.place_chicken,
            self.place_chicken_full.chicken,
            self.place_chicken_full.tray,
        )
        self.place_chicken_full.add_precondition(self.tray_to_place_known())
        self.place_chicken_full.add_precondition(
            self.tray_to_place(self.place_chicken_full.tray)
        )
        self.place_chicken_full.add_precondition(self.item_type_is_known())
        self.place_chicken_full.add_precondition(
            self.space_in_tray(
                self.place_chicken_full.chicken, self.place_chicken_full.tray
            )
        )
        self.place_chicken_full.add_precondition(self.current_arm_pose(self.unknown_pose))
        st1 = self.place_chicken_full.add_subtask(
            self.move_over_conveyor_belt, self.place_chicken_full.chicken
        )
        st2 = self.place_chicken_full.add_subtask(
            self.pick_chicken_part, self.place_chicken_full.chicken
        )
        st3 = self.place_chicken_full.add_subtask(
            self.move_over_tray_cart, self.place_chicken_full.chicken, self.place_chicken_full.tray
        )
        st4 = self.place_chicken_full.add_subtask(
            self.insert_part_in_container,
            self.place_chicken_full.chicken,
            self.place_chicken_full.tray,
        )
        self.place_chicken_full.set_ordered(st1, st2, st3, st4)

        self.pack_chicken_get = Method(
            "pack_chicken_get", chicken=type_item, tray=type_tray
        )
        self.pack_chicken_get.set_task(self.pack_chicken)

        self.pack_chicken_get.add_precondition(
            Or(
                Not(self.item_type_is_known()),
                Not(self.tray_to_place_known()),
                Not(self.trays_perceived()),
                And(Not(self.item_in_fov()), self.holding(self.nothing)),
            )
        )
        self.pack_chicken_get.add_subtask(
            self.get_chicken, self.pack_chicken_get.chicken, self.pack_chicken_get.tray
        )

        self.pack_chicken_place = Method(
            "pack_chicken_place", chicken=type_item, tray=type_tray
        )
        self.pack_chicken_place.set_task(self.pack_chicken)
        self.pack_chicken_place.add_precondition(
            self.chicken_to_pick(self.pack_chicken_place.chicken)
        )
        self.pack_chicken_place.add_subtask(
            self.place_chicken,
            self.pack_chicken_place.chicken,
            self.pack_chicken_place.tray,
        )

        self.methods = (
            self.get_chicken_noop,
            self.get_chicken_ptray,
            self.get_chicken_estimate,
            self.get_chicken_perceive,
            self.get_chicken_full,
            self.place_chicken_tray_full,
            self.place_chicken_in_tray,
            self.place_chicken_move,
            self.place_chicken_pick,
            self.place_chicken_full,
            self.pack_chicken_get,
            self.pack_chicken_place,
        )

    def _create_domain_actions(self, temporal: bool = False) -> None:
        actions = Actions(self._env)

        if temporal:
            self.get_next_chicken_part, _ = self.create_action(
                "get_next_chicken_part",
                _callable=actions.get_next_insole,
                duration=10,
            )
            self.get_next_chicken_part.add_condition(
                StartTiming(), Not(self.item_in_fov())
            )
            self.get_next_chicken_part.add_effect(EndTiming(), self.item_in_fov(), True)

            # TODO TEMPORAL
        else:
            self.get_next_chicken_part, _ = self.create_action(
                "get_next_chicken_part",
                _callable=actions.get_next_chicken_part,
            )
            self.get_next_chicken_part.add_precondition(Not(self.item_in_fov()))
            self.get_next_chicken_part.add_effect(self.item_in_fov(), True)

            self.estimate_part_shelf_life, _ = self.create_action(
                "estimate_part_shelf_life", _callable=actions.estimate_part_shelf_life
            )
            self.estimate_part_shelf_life.add_precondition(
                Not(self.tray_to_place_known())
            )
            self.estimate_part_shelf_life.add_precondition(self.item_in_fov())
            self.estimate_part_shelf_life.add_effect(self.tray_to_place_known(), True)

            self.perceive_chicken_part, _ = self.create_action(
                "perceive_chicken_part",
                _callable=actions.perceive_chicken_part,
            )
            self.perceive_chicken_part.add_precondition(self.item_in_fov())
            self.perceive_chicken_part.add_precondition(Not(self.item_type_is_known()))
            self.perceive_chicken_part.add_effect(self.item_type_is_known(), True)

            self.move_over_conveyor_belt, [c] = self.create_action(
                "move_over_conveyor_belt", chicken=Item, _callable=actions.move_over_conveyor_belt
            )
            self.move_over_conveyor_belt.add_precondition(self.current_arm_pose(self.unknown_pose))
            self.move_over_conveyor_belt.add_precondition(self.item_in_fov())
            self.move_over_conveyor_belt.add_precondition(self.holding(self.nothing))
            self.move_over_conveyor_belt.add_precondition(self.item_type_is_known())
            self.move_over_conveyor_belt.add_precondition(self.chicken_to_pick(c))
            self.move_over_conveyor_belt.add_effect(self.current_arm_pose(self.over_conveyor), True)
            self.move_over_conveyor_belt.add_effect(self.current_arm_pose(self.unknown_pose), False)

            self.pick_chicken_part, [c] = self.create_action(
                "pick_chicken_part", chicken=Item, _callable=actions.pick_chicken_part
            )
            self.pick_chicken_part.add_precondition(self.item_in_fov())
            self.pick_chicken_part.add_precondition(self.holding(self.nothing))
            self.pick_chicken_part.add_precondition(self.item_type_is_known())
            self.pick_chicken_part.add_precondition(self.chicken_to_pick(c))
            self.pick_chicken_part.add_precondition(self.current_arm_pose(self.over_conveyor))
            self.pick_chicken_part.add_effect(self.holding(c), True)
            self.pick_chicken_part.add_effect(self.item_in_fov(), False)
            self.pick_chicken_part.add_effect(self.holding(self.nothing), False)
            self.pick_chicken_part.add_effect(self.current_arm_pose(self.over_conveyor), False)
            self.pick_chicken_part.add_effect(self.current_arm_pose(self.unknown_pose), True)

            self.perceive_trays, _ = self.create_action(
                "perceive_trays", _callable=actions.perceive_trays
            )
            self.perceive_trays.add_precondition(Not(self.trays_perceived()))
            self.perceive_trays.add_effect(self.trays_perceived(), True)

            self.move_over_tray_cart, [c, t] = self.create_action(
                "move_over_tray_cart", chicken=Item, tray=Tray, _callable=actions.move_over_tray_cart
            )
            self.move_over_tray_cart.add_precondition(self.current_arm_pose(self.unknown_pose))
            self.move_over_tray_cart.add_precondition(self.tray_to_place_known())
            self.move_over_tray_cart.add_precondition(self.tray_to_place(t))
            self.move_over_tray_cart.add_precondition(self.holding(c))
            self.move_over_tray_cart.add_precondition(self.space_in_tray(c, t))
            self.move_over_tray_cart.add_precondition(self.tray_is_available(t))
            self.move_over_tray_cart.add_precondition(self.item_type_is_known())
            self.move_over_tray_cart.add_effect(self.current_arm_pose(self.over_tray), True)
            self.move_over_tray_cart.add_effect(self.current_arm_pose(self.unknown_pose), False)

            self.insert_part_in_container, [c, t] = self.create_action(
                "insert_part_in_container",
                chicken=Item,
                tray=Tray,
                _callable=actions.insert_part_in_container,
            )
            self.insert_part_in_container.add_precondition(self.tray_to_place_known())
            self.insert_part_in_container.add_precondition(self.tray_to_place(t))
            self.insert_part_in_container.add_precondition(self.holding(c))
            self.insert_part_in_container.add_precondition(self.space_in_tray(c, t))
            self.insert_part_in_container.add_precondition(self.tray_is_available(t))
            self.insert_part_in_container.add_precondition(self.item_type_is_known())
            self.insert_part_in_container.add_precondition(self.current_arm_pose(self.over_tray))
            self.insert_part_in_container.add_effect(self.trays_perceived(), False)
            self.insert_part_in_container.add_effect(self.tray_to_place_known(), False)
            self.insert_part_in_container.add_effect(self.tray_to_place(t), False)
            self.insert_part_in_container.add_effect(self.holding(c), False)
            self.insert_part_in_container.add_effect(self.holding(self.nothing), True)
            self.insert_part_in_container.add_effect(self.item_type_is_known(), False)
            self.insert_part_in_container.add_effect(self.current_arm_pose(self.over_tray), False)
            self.insert_part_in_container.add_effect(self.current_arm_pose(self.unknown_pose), True)

            self.replace_filled_tray, [c, t] = self.create_action(
                "replace_filled_tray",
                chicken=Item,
                tray=Tray,
                _callable=actions.replace_filled_tray,
            )
            self.replace_filled_tray.add_precondition(self.tray_is_available(t))
            self.replace_filled_tray.add_precondition(Not(self.space_in_tray(c, t)))
            self.replace_filled_tray.add_effect(self.space_in_tray(c, t), True)
            self.replace_filled_tray.add_effect(self.tray_is_available(t), False)

    def set_state_and_goal(self, problem, goal=None) -> None:
        success = True
        if goal is None:
            problem.task_network.add_subtask(self.pack_chicken())
        elif goal == "get_next_insole":
            problem.set_initial_value(self.holding(self.nothing), True)

            problem.task_network.add_subtask(self.get_next_insole, self.conveyor_a)
        elif goal == "preload_bag_bundle":
            problem.set_initial_value(self.holding(self.nothing), True)

            problem.task_network.add_subtask(self.preload_bag_bundle)
        elif goal == "load_bag":
            problem.set_initial_value(self.holding(self.nothing), True)
            problem.set_initial_value(self.bag_dispenser_has_bags(), True)

            problem.task_network.add_subtask(self.load_bag)
        elif goal == "pick_insole":
            problem.set_initial_value(self.holding(self.nothing), True)
            problem.set_initial_value(self.item_pose_is_known(self.insole), True)
            problem.set_initial_value(self.item_type_is_known(self.insole), True)
            problem.set_initial_value(self.item_type_is_known(self.bag), True)
            problem.set_initial_value(self.item_pose_is_known(self.bag), True)
            problem.set_initial_value(self.item_types_match(), True)
            problem.set_initial_value(self.bag_is_open(), True)
            problem.set_initial_value(self.bag_dispenser_has_bags(), True)
            problem.set_initial_value(self.item_in_fov(), True)

            problem.task_network.add_subtask(self.pick_insole, self.insole)
        elif goal == "pick_set":
            problem.set_initial_value(self.item_pose_is_known(self.set), True)

            problem.task_network.add_subtask(self.pick_set, self.set)
        elif goal == "open_bag":
            problem.set_initial_value(self.holding(self.nothing), True)
            problem.set_initial_value(self.bag_is_probably_available(), True)

            problem.task_network.add_subtask(self.open_bag)
        elif goal == "insert":
            problem.set_initial_value(self.holding(self.insole), True)
            problem.set_initial_value(self.holding(self.nothing), False)
            problem.set_initial_value(self.item_types_match(), True)
            problem.set_initial_value(self.item_pose_is_known(self.bag), True)
            problem.set_initial_value(self.bag_is_open(), True)

            problem.task_network.add_subtask(self.insert, self.insole, self.bag)
        elif goal == "release_set":
            problem.task_network.add_subtask(self.release_set, self.set)
        elif goal == "seal_set":
            problem.set_initial_value(self.holding(self.set), True)
            problem.set_initial_value(self.bag_set_released(), True)

            problem.task_network.add_subtask(self.seal_set, self.set)
        elif goal == "perceive_insole":
            problem.set_initial_value(self.item_in_fov(), True)

            problem.task_network.add_subtask(self.perceive_insole, self.insole)
        elif goal == "perceive_bag":
            problem.set_initial_value(self.bag_is_probably_available(), True)

            problem.task_network.add_subtask(self.perceive_bag, self.bag)
        elif goal == "perceive_set":
            problem.set_initial_value(self.insole_inside_bag(self.insole), True)

            problem.task_network.add_subtask(
                self.perceive_set, self.insole, self.bag, self.set
            )
        elif goal == "reject_insole":
            problem.set_initial_value(self.item_in_fov(), True)
            problem.set_initial_value(self.not_checked_item_types(), False)

            problem.task_network.add_subtask(self.reject_insole, self.insole)
        elif goal == "match_insole_bag":
            problem.set_initial_value(self.item_type_is_known(self.insole), True)
            problem.set_initial_value(self.item_type_is_known(self.bag), True)

            problem.task_network.add_subtask(
                self.match_insole_bag, self.insole, self.bag
            )
        else:
            logerr(
                (
                    f"Task ({goal}) is unknown! Please use a task from this list: "
                    "get_next_insole, preload_bag_bundle, load_bag, pick_insole, "
                    "open_bag, release_set, seal_set"
                )
            )
            success = False
        return success
