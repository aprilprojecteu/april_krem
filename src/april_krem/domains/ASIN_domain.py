from rospy import logerr
from unified_planning.model.htn import Task, Method
from unified_planning.shortcuts import And, Or, Not, StartTiming, EndTiming
from april_krem.domains.ASIN_components import (
    Item,
    Tray,
    ArmPose,
    Actions,
    Environment,
)
from up_esb.bridge import Bridge


class ASINDomain(Bridge):
    def __init__(self, krem_logging, temporal: bool = False) -> None:
        Bridge.__init__(self)

        self._env = Environment(krem_logging)

        # Create types for planning based on class types
        self.create_types([Item, Tray, ArmPose])
        type_item = self.get_type(Item)
        type_tray = self.get_type(Tray)

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
        self.type_in_tray = self.create_fluent_from_function(self._env.type_in_tray)
        self.tray_is_available = self.create_fluent_from_function(
            self._env.tray_is_available
        )
        self.trays_perceived = self.create_fluent_from_function(
            self._env.trays_perceived
        )
        self.current_arm_pose = self.create_fluent_from_function(
            self._env.current_arm_pose
        )
        self.conveyor_is_moving = self.create_fluent_from_function(
            self._env.conveyor_is_moving
        )

        # Create objects for both planning and execution
        self.items = self.create_enum_objects(Item)
        self.nothing = self.objects[Item.nothing.name]
        self.chicken_part = self.objects[Item.chicken_part.name]
        self.breast = self.objects[Item.breast.name]
        self.drumstick = self.objects[Item.drumstick.name]

        self.trays = self.create_enum_objects(Tray)
        self.high_tray = self.objects[Tray.high_tray.name]
        self.med_tray = self.objects[Tray.med_tray.name]
        self.low_tray = self.objects[Tray.low_tray.name]
        self.discard_tray = self.objects[Tray.discard_tray.name]

        self.arm_poses = self.create_enum_objects(ArmPose)
        self.unknown_pose = self.objects[ArmPose.unknown.name]
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

        # GET CHICKEN
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

        # get next chicken and gather all necessary information
        self.get_chicken_get = Method(
            "get_chicken_get", chicken=type_item, tray=type_tray
        )
        self.get_chicken_get.set_task(
            self.get_chicken, self.get_chicken_get.chicken, self.get_chicken_get.tray
        )
        self.get_chicken_get.add_precondition(Not(self.item_in_fov()))
        self.get_chicken_get.add_precondition(self.conveyor_is_moving())
        self.get_chicken_get.add_precondition(self.current_arm_pose(self.over_conveyor))
        st1 = self.get_chicken_get.add_subtask(self.get_next_chicken_part)
        st2 = self.get_chicken_get.add_subtask(self.estimate_part_shelf_life)
        st3 = self.get_chicken_get.add_subtask(self.perceive_chicken_part)
        st4 = self.get_chicken_get.add_subtask(self.perceive_trays)
        self.get_chicken_get.set_ordered(st1, st2, st3, st4)

        # get next chicken, move arm over conveyor and gather all necessary information
        self.get_chicken_move = Method(
            "get_chicken_move", chicken=type_item, tray=type_tray
        )
        self.get_chicken_move.set_task(
            self.get_chicken, self.get_chicken_move.chicken, self.get_chicken_move.tray
        )
        self.get_chicken_move.add_precondition(Not(self.item_in_fov()))
        self.get_chicken_move.add_precondition(self.conveyor_is_moving())
        st1 = self.get_chicken_move.add_subtask(self.move_arm, self.over_conveyor)
        st2 = self.get_chicken_move.add_subtask(self.get_next_chicken_part)
        st3 = self.get_chicken_move.add_subtask(self.estimate_part_shelf_life)
        st4 = self.get_chicken_move.add_subtask(self.perceive_chicken_part)
        st5 = self.get_chicken_move.add_subtask(self.perceive_trays)
        self.get_chicken_move.set_ordered(st1, st2, st3, st4, st5)

        # get next chicken over conveyor and gather all necessary information
        self.get_chicken_full = Method(
            "get_chicken_full", chicken=type_item, tray=type_tray
        )
        self.get_chicken_full.set_task(
            self.get_chicken, self.get_chicken_full.chicken, self.get_chicken_full.tray
        )
        self.get_chicken_full.add_precondition(Not(self.item_in_fov()))
        self.get_chicken_full.add_precondition(Not(self.conveyor_is_moving()))
        st1 = self.get_chicken_full.add_subtask(self.move_conveyor_belt)
        st2 = self.get_chicken_full.add_subtask(self.move_arm, self.over_conveyor)
        st3 = self.get_chicken_full.add_subtask(self.get_next_chicken_part)
        st4 = self.get_chicken_full.add_subtask(self.estimate_part_shelf_life)
        st5 = self.get_chicken_full.add_subtask(self.perceive_chicken_part)
        st6 = self.get_chicken_full.add_subtask(self.perceive_trays)
        self.get_chicken_full.set_ordered(st1, st2, st3, st4, st5, st6)

        # PLACE CHICKEN
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
        self.place_chicken_move_t = Method(
            "place_chicken_move_t", chicken=type_item, tray=type_tray
        )
        self.place_chicken_move_t.set_task(
            self.place_chicken,
            self.place_chicken_move_t.chicken,
            self.place_chicken_move_t.tray,
        )
        self.place_chicken_move_t.add_precondition(self.tray_to_place_known())
        self.place_chicken_move_t.add_precondition(
            self.tray_to_place(self.place_chicken_move_t.tray)
        )
        self.place_chicken_move_t.add_precondition(
            self.holding(self.place_chicken_move_t.chicken)
        )
        self.place_chicken_move_t.add_precondition(
            self.tray_is_available(self.place_chicken_move_t.tray)
        )
        self.place_chicken_move_t.add_precondition(
            self.space_in_tray(
                self.place_chicken_move_t.chicken, self.place_chicken_move_t.tray
            )
        )
        self.place_chicken_move_t.add_precondition(
            Not(self.current_arm_pose(self.over_tray))
        )
        self.place_chicken_move_t.add_precondition(self.conveyor_is_moving())
        st1 = self.place_chicken_move_t.add_subtask(
            self.move_arm,
            self.over_tray,
        )
        st2 = self.place_chicken_move_t.add_subtask(
            self.insert_part_in_container,
            self.place_chicken_move_t.chicken,
            self.place_chicken_move_t.tray,
        )
        self.place_chicken_move_t.set_ordered(st1, st2)

        # already holding chicken, start conveyor, move arm and place
        self.place_chicken_start_cb = Method(
            "place_chicken_start_cb", chicken=type_item, tray=type_tray
        )
        self.place_chicken_start_cb.set_task(
            self.place_chicken,
            self.place_chicken_start_cb.chicken,
            self.place_chicken_start_cb.tray,
        )
        self.place_chicken_start_cb.add_precondition(self.tray_to_place_known())
        self.place_chicken_start_cb.add_precondition(
            self.tray_to_place(self.place_chicken_move_t.tray)
        )
        self.place_chicken_start_cb.add_precondition(
            self.holding(self.place_chicken_move_t.chicken)
        )
        self.place_chicken_start_cb.add_precondition(
            self.tray_is_available(self.place_chicken_move_t.tray)
        )
        self.place_chicken_start_cb.add_precondition(
            self.space_in_tray(
                self.place_chicken_move_t.chicken, self.place_chicken_move_t.tray
            )
        )
        self.place_chicken_start_cb.add_precondition(
            Not(self.current_arm_pose(self.over_tray))
        )
        self.place_chicken_start_cb.add_precondition(Not(self.conveyor_is_moving()))
        st1 = self.place_chicken_start_cb.add_subtask(
            self.move_conveyor_belt,
        )
        st2 = self.place_chicken_start_cb.add_subtask(
            self.move_arm,
            self.over_tray,
        )
        st3 = self.place_chicken_start_cb.add_subtask(
            self.insert_part_in_container,
            self.place_chicken_start_cb.chicken,
            self.place_chicken_start_cb.tray,
        )
        self.place_chicken_start_cb.set_ordered(st1, st2, st3)

        # already holding chicken, move and place
        self.place_chicken_move_cb = Method(
            "place_chicken_move_cb", chicken=type_item, tray=type_tray
        )
        self.place_chicken_move_cb.set_task(
            self.place_chicken,
            self.place_chicken_move_cb.chicken,
            self.place_chicken_move_cb.tray,
        )
        self.place_chicken_move_cb.add_precondition(self.tray_to_place_known())
        self.place_chicken_move_cb.add_precondition(
            self.tray_to_place(self.place_chicken_move_cb.tray)
        )
        self.place_chicken_move_cb.add_precondition(
            self.holding(self.place_chicken_move_cb.chicken)
        )
        self.place_chicken_move_cb.add_precondition(
            self.tray_is_available(self.place_chicken_move_cb.tray)
        )
        self.place_chicken_move_cb.add_precondition(
            self.space_in_tray(
                self.place_chicken_move_cb.chicken, self.place_chicken_move_cb.tray
            )
        )
        self.place_chicken_move_cb.add_precondition(
            Not(self.current_arm_pose(self.over_conveyor))
        )
        st1 = self.place_chicken_move_cb.add_subtask(self.move_arm, self.over_conveyor)
        st2 = self.place_chicken_move_cb.add_subtask(
            self.move_conveyor_belt,
        )
        st3 = self.place_chicken_move_cb.add_subtask(
            self.move_arm,
            self.over_tray,
        )
        st4 = self.place_chicken_move_cb.add_subtask(
            self.insert_part_in_container,
            self.place_chicken_move_cb.chicken,
            self.place_chicken_move_cb.tray,
        )
        self.place_chicken_move_cb.set_ordered(st1, st2, st3, st4)

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
        self.place_chicken_pick.add_precondition(
            self.current_arm_pose(self.over_conveyor)
        )
        st1 = self.place_chicken_pick.add_subtask(
            self.pick_chicken_part, self.place_chicken_pick.chicken
        )
        st2 = self.place_chicken_pick.add_subtask(self.move_arm, self.over_conveyor)
        st3 = self.place_chicken_pick.add_subtask(
            self.move_conveyor_belt,
        )
        st4 = self.place_chicken_pick.add_subtask(
            self.move_arm,
            self.over_tray,
        )
        st5 = self.place_chicken_pick.add_subtask(
            self.insert_part_in_container,
            self.place_chicken_pick.chicken,
            self.place_chicken_pick.tray,
        )
        self.place_chicken_pick.set_ordered(st1, st2, st3, st4, st5)

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
        self.place_chicken_full.add_precondition(
            Not(self.current_arm_pose(self.over_conveyor))
        )
        st1 = self.place_chicken_full.add_subtask(self.move_arm, self.over_conveyor)
        st2 = self.place_chicken_full.add_subtask(
            self.pick_chicken_part, self.place_chicken_full.chicken
        )
        st3 = self.place_chicken_full.add_subtask(self.move_arm, self.over_conveyor)
        st4 = self.place_chicken_full.add_subtask(
            self.move_conveyor_belt,
        )
        st5 = self.place_chicken_full.add_subtask(
            self.move_arm,
            self.over_tray,
        )
        st6 = self.place_chicken_full.add_subtask(
            self.insert_part_in_container,
            self.place_chicken_full.chicken,
            self.place_chicken_full.tray,
        )
        self.place_chicken_full.set_ordered(st1, st2, st3, st4, st5, st6)

        # PACK CHICKEN
        # get chicken, estimate, perceive
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

        # pick chicken, place in corresponding tray
        self.pack_chicken_place = Method(
            "pack_chicken_place", chicken=type_item, tray=type_tray
        )
        self.pack_chicken_place.set_task(self.pack_chicken)
        self.pack_chicken_place.add_precondition(
            self.chicken_to_pick(self.pack_chicken_place.chicken)
        )
        st1 = self.pack_chicken_place.add_subtask(
            self.place_chicken,
            self.pack_chicken_place.chicken,
            self.pack_chicken_place.tray,
        )
        st2 = self.pack_chicken_place.add_subtask(self.move_arm, self.over_tray)
        self.pack_chicken_place.set_ordered(st1, st2)

        # chicken tray is full
        self.pack_chicken_replace = Method(
            "pack_chicken_replace", chicken=type_item, tray=type_tray
        )
        self.pack_chicken_replace.set_task(self.pack_chicken)
        self.pack_chicken_replace.add_precondition(
            Not(
                self.space_in_tray(
                    self.pack_chicken_replace.chicken,
                    self.pack_chicken_replace.tray,
                )
            )
        )
        self.pack_chicken_replace.add_precondition(
            self.type_in_tray(
                self.pack_chicken_replace.chicken, self.pack_chicken_replace.tray
            )
        )
        self.pack_chicken_replace.add_subtask(
            self.replace_filled_tray,
            self.pack_chicken_replace.chicken,
            self.pack_chicken_replace.tray,
        )

        self.methods = (
            self.get_chicken_noop,
            self.get_chicken_ptray,
            self.get_chicken_estimate,
            self.get_chicken_perceive,
            self.get_chicken_get,
            self.get_chicken_move,
            self.get_chicken_full,
            self.place_chicken_in_tray,
            self.place_chicken_move_t,
            self.place_chicken_start_cb,
            self.place_chicken_move_cb,
            self.place_chicken_pick,
            self.place_chicken_full,
            self.pack_chicken_get,
            self.pack_chicken_place,
            self.pack_chicken_replace,
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
            self.move_conveyor_belt, _ = self.create_action(
                "move_conveyor_belt",
                _callable=actions.move_conveyor_belt,
            )
            self.move_conveyor_belt.add_precondition(Not(self.conveyor_is_moving()))
            self.move_conveyor_belt.add_effect(self.conveyor_is_moving(), True)

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

            self.move_arm, [a] = self.create_action(
                "move_arm",
                arm_pose=ArmPose,
                _callable=actions.move_arm,
            )
            self.move_arm.add_effect(self.current_arm_pose(a), True)

            self.pick_chicken_part, [c] = self.create_action(
                "pick_chicken_part", chicken=Item, _callable=actions.pick_chicken_part
            )
            self.pick_chicken_part.add_precondition(self.item_in_fov())
            self.pick_chicken_part.add_precondition(self.holding(self.nothing))
            self.pick_chicken_part.add_precondition(self.item_type_is_known())
            self.pick_chicken_part.add_precondition(self.chicken_to_pick(c))
            self.pick_chicken_part.add_precondition(
                self.current_arm_pose(self.over_conveyor)
            )
            self.pick_chicken_part.add_effect(self.holding(c), True)
            self.pick_chicken_part.add_effect(self.item_in_fov(), False)
            self.pick_chicken_part.add_effect(self.holding(self.nothing), False)
            self.pick_chicken_part.add_effect(
                self.current_arm_pose(self.over_conveyor), False
            )
            self.pick_chicken_part.add_effect(
                self.current_arm_pose(self.unknown_pose), True
            )

            self.perceive_trays, _ = self.create_action(
                "perceive_trays", _callable=actions.perceive_trays
            )
            self.perceive_trays.add_precondition(Not(self.trays_perceived()))
            self.perceive_trays.add_effect(self.trays_perceived(), True)

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
            self.insert_part_in_container.add_precondition(
                self.current_arm_pose(self.over_tray)
            )
            self.insert_part_in_container.add_effect(self.trays_perceived(), False)
            self.insert_part_in_container.add_effect(self.tray_to_place_known(), False)
            self.insert_part_in_container.add_effect(self.tray_to_place(t), False)
            self.insert_part_in_container.add_effect(self.holding(c), False)
            self.insert_part_in_container.add_effect(self.holding(self.nothing), True)
            self.insert_part_in_container.add_effect(self.item_type_is_known(), False)
            self.insert_part_in_container.add_effect(
                self.current_arm_pose(self.over_tray), False
            )
            self.insert_part_in_container.add_effect(
                self.current_arm_pose(self.unknown_pose), True
            )

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
        else:
            logerr((f"Task ({goal}) is unknown! Please leave the goal empty."))
            success = False
        return success
