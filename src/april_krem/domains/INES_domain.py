from rospy import logerr
from unified_planning.model.htn import Task, Method
from unified_planning.shortcuts import Not, And, Or, StartTiming, EndTiming
from april_krem.domains.INES_components import (
    Item,
    Location,
    Actions,
    Environment,
)
from up_esb.bridge import Bridge


class INESDomain(Bridge):
    def __init__(self, krem_logging, temporal: bool = False) -> None:
        Bridge.__init__(self)

        self._env = Environment(krem_logging)

        # Create types for planning based on class types
        self.create_types([Item, Location])
        type_item = self.get_type(Item)
        type_location = self.get_type(Location)

        # Create fluents for planning
        self.holding = self.create_fluent_from_function(self._env.holding)
        self.item_pose_is_known = self.create_fluent_from_function(
            self._env.item_pose_is_known
        )
        self.item_in_fov = self.create_fluent_from_function(self._env.item_in_fov)
        self.bag_set_released = self.create_fluent_from_function(
            self._env.bag_set_released
        )
        self.bag_dispenser_has_bags = self.create_fluent_from_function(
            self._env.bag_dispenser_has_bags
        )
        self.sealing_machine_ready = self.create_fluent_from_function(
            self._env.sealing_machine_ready
        )
        self.item_type_is_known = self.create_fluent_from_function(
            self._env.item_type_is_known
        )
        self.item_types_match = self.create_fluent_from_function(
            self._env.item_types_match
        )
        self.not_checked_item_types = self.create_fluent_from_function(
            self._env.not_checked_item_types
        )
        self.insole_inside_bag = self.create_fluent_from_function(
            self._env.insole_inside_bag
        )
        self.bag_is_probably_available = self.create_fluent_from_function(
            self._env.bag_is_probably_available
        )
        self.bag_is_probably_open = self.create_fluent_from_function(
            self._env.bag_is_probably_open
        )
        self.bag_is_open = self.create_fluent_from_function(self._env.bag_is_open)

        self.moving = self.create_fluent_from_function(self._env.moving)
        self.stationary = self.create_fluent_from_function(self._env.stationary)

        # Create objects for both planning and execution
        self.items = self.create_enum_objects(Item)
        self.nothing = self.objects[Item.nothing.name]
        self.insole = self.objects[Item.insole.name]
        self.bag = self.objects[Item.bag.name]
        self.set = self.objects[Item.set.name]

        self.locations = self.create_enum_objects(Location)
        self.conveyor_a = self.objects[Location.conveyor_a.name]
        self.conveyor_b = self.objects[Location.conveyor_b.name]
        self.dispenser = self.objects[Location.dispenser.name]
        self.in_hand = self.objects[Location.in_hand.name]
        self.in_bag = self.objects[Location.in_bag.name]
        self.unknown = self.objects[Location.unknown.name]

        # Create actions for planning
        self._create_domain_actions(temporal)

        # Tasks
        self.get_insole = Task("get_insole", conveyor=type_location, insole=type_item)
        self.prepare_bag = Task("prepare_bag", bag=type_item)
        self.insert_insole = Task("insert_insole", insole=type_item, bag=type_item)
        self.finish_set = Task("finish_set", set=type_item)
        self.bag_insole = Task(
            "bag_insole", insole=type_item, bag=type_item, set=type_item
        )
        self.tasks = (
            self.get_insole,
            self.prepare_bag,
            self.insert_insole,
            self.finish_set,
            self.bag_insole,
        )

        # Methods

        # get_insole
        # insole already there and perceived
        self.get_insole_noop = Method(
            "get_insole_noop",
            conveyor=type_location,
            insole=type_item,
        )
        self.get_insole_noop.set_task(
            self.get_insole, self.get_insole_noop.conveyor, self.get_insole_noop.insole
        )
        self.get_insole_noop.add_precondition(
            Or(
                And(
                    self.item_type_is_known(self.get_insole_noop.insole),
                    self.item_pose_is_known(self.get_insole_noop.insole),
                ),
                self.holding(self.get_insole_noop.insole),
                self.insole_inside_bag(self.get_insole_noop.insole),
            )
        )

        # insole already there, still need to perceive
        self.get_insole_perceive = Method(
            "get_insole_perceive",
            conveyor=type_location,
            insole=type_item,
        )
        self.get_insole_perceive.set_task(
            self.get_insole,
            self.get_insole_perceive.conveyor,
            self.get_insole_perceive.insole,
        )
        self.get_insole_perceive.add_precondition(self.item_in_fov())
        self.get_insole_perceive.add_subtask(
            self.perceive_insole, self.get_insole_perceive.insole
        )

        # get next insole by activating conveyor and perceiving it
        self.get_insole_full = Method(
            "get_insole_full",
            conveyor=type_location,
            insole=type_item,
        )
        self.get_insole_full.set_task(
            self.get_insole, self.get_insole_full.conveyor, self.get_insole_full.insole
        )
        self.get_insole_full.add_precondition(
            self.stationary(self.get_insole_full.conveyor)
        )
        self.get_insole_full.add_precondition(
            Not(self.item_pose_is_known(self.get_insole_full.insole))
        )
        st1 = self.get_insole_full.add_subtask(
            self.get_next_insole, self.get_insole_full.conveyor
        )
        st2 = self.get_insole_full.add_subtask(
            self.perceive_insole, self.get_insole_full.insole
        )
        self.get_insole_full.set_ordered(st1, st2)

        # prepare_bag
        # bag dispenser has bags, bag already there and perceived
        self.prepare_bag_noop = Method("prepare_bag_noop", bag=type_item)
        self.prepare_bag_noop.set_task(self.prepare_bag, self.prepare_bag_noop.bag)
        self.prepare_bag_noop.add_precondition(
            self.item_pose_is_known(self.prepare_bag_noop.bag)
        )

        # bag dispenser has bags, bag already there, bag opened, not perceived yet
        self.prepare_bag_perceive = Method("prepare_bag_perceive", bag=type_item)
        self.prepare_bag_perceive.set_task(
            self.prepare_bag, self.prepare_bag_perceive.bag
        )
        self.prepare_bag_perceive.add_precondition(self.bag_is_probably_available())
        self.prepare_bag_perceive.add_precondition(self.bag_is_probably_open())
        self.prepare_bag_perceive.add_subtask(
            self.perceive_bag, self.prepare_bag_perceive.bag
        )

        # bag dispenser has bags, bag already there, not opened yet
        self.prepare_bag_open = Method("prepare_bag_open", bag=type_item)
        self.prepare_bag_open.set_task(self.prepare_bag, self.prepare_bag_open.bag)
        self.prepare_bag_open.add_precondition(self.bag_is_probably_available())
        st1 = self.prepare_bag_open.add_subtask(self.open_bag)
        st2 = self.prepare_bag_open.add_subtask(
            self.perceive_bag, self.prepare_bag_open.bag
        )
        self.prepare_bag_open.set_ordered(st1, st2)

        # bag dispenser has bags, load and inspect bag
        self.prepare_bag_full = Method("prepare_bag_full", bag=type_item)
        self.prepare_bag_full.set_task(self.prepare_bag, self.prepare_bag_full.bag)
        self.prepare_bag_full.add_precondition(self.bag_dispenser_has_bags())
        st1 = self.prepare_bag_full.add_subtask(self.load_bag)
        st2 = self.prepare_bag_full.add_subtask(self.open_bag)
        st3 = self.prepare_bag_full.add_subtask(
            self.perceive_bag, self.prepare_bag_full.bag
        )
        self.prepare_bag_full.set_ordered(st1, st2, st3)

        # bag dispenser empty, refill dispenser, then load and inspect bag
        self.prepare_bag_refill = Method("prepare_bag_refill", bag=type_item)
        self.prepare_bag_refill.set_task(self.prepare_bag, self.prepare_bag_refill.bag)
        st1 = self.prepare_bag_refill.add_subtask(self.preload_bag_bundle)
        st2 = self.prepare_bag_refill.add_subtask(self.load_bag)
        st3 = self.prepare_bag_refill.add_subtask(self.open_bag)
        st4 = self.prepare_bag_refill.add_subtask(
            self.perceive_bag, self.prepare_bag_refill.bag
        )
        self.prepare_bag_refill.set_ordered(st1, st2, st3, st4)

        # insert_insole
        # already inserted insole
        self.insert_insole_noop = Method(
            "insert_insole_noop", insole=type_item, bag=type_item, set=type_item
        )
        self.insert_insole_noop.set_task(
            self.insert_insole,
            self.insert_insole_noop.insole,
            self.insert_insole_noop.bag,
        )
        self.insert_insole_noop.add_precondition(
            self.insole_inside_bag(self.insert_insole_noop.insole)
        )
        self.insert_insole_noop.add_precondition(self.holding(self.nothing))
        self.insert_insole_noop.add_subtask(
            self.perceive_set,
            self.insert_insole_noop.insole,
            self.insert_insole_noop.bag,
            self.insert_insole_noop.set,
        )

        # already holding insole
        self.insert_insole_insert = Method(
            "insert_insole_insert", insole=type_item, bag=type_item, set=type_item
        )
        self.insert_insole_insert.set_task(
            self.insert_insole,
            self.insert_insole_insert.insole,
            self.insert_insole_insert.bag,
        )
        self.insert_insole_insert.add_precondition(
            self.holding(self.insert_insole_insert.insole)
        )
        st1 = self.insert_insole_insert.add_subtask(
            self.insert, self.insert_insole_insert.insole, self.insert_insole_insert.bag
        )
        st2 = self.insert_insole_insert.add_subtask(
            self.perceive_set,
            self.insert_insole_insert.insole,
            self.insert_insole_insert.bag,
            self.insert_insole_insert.set,
        )
        self.insert_insole_insert.set_ordered(st1, st2)

        # bag and insole are matching, pick and insert
        self.insert_insole_match = Method(
            "insert_insole_match", insole=type_item, bag=type_item, set=type_item
        )
        self.insert_insole_match.set_task(
            self.insert_insole,
            self.insert_insole_match.insole,
            self.insert_insole_match.bag,
        )
        self.insert_insole_match.add_precondition(self.item_types_match())
        self.insert_insole_match.add_precondition(Not(self.not_checked_item_types()))
        st1 = self.insert_insole_match.add_subtask(
            self.pick_insole, self.insert_insole_match.insole
        )
        st2 = self.insert_insole_match.add_subtask(
            self.insert, self.insert_insole_match.insole, self.insert_insole_match.bag
        )
        st3 = self.insert_insole_match.add_subtask(
            self.perceive_set,
            self.insert_insole_match.insole,
            self.insert_insole_match.bag,
            self.insert_insole_match.set,
        )
        self.insert_insole_match.set_ordered(st1, st2, st3)

        # check if bag and insole are matching, pick and insert
        self.insert_insole_full = Method(
            "insert_insole_full", insole=type_item, bag=type_item, set=type_item
        )
        self.insert_insole_full.set_task(
            self.insert_insole,
            self.insert_insole_full.insole,
            self.insert_insole_full.bag,
        )
        self.insert_insole_full.add_precondition(self.not_checked_item_types())
        self.insert_insole_full.add_precondition(
            self.item_type_is_known(self.insert_insole_full.insole)
        )
        self.insert_insole_full.add_precondition(
            self.item_type_is_known(self.insert_insole_full.bag)
        )
        st1 = self.insert_insole_full.add_subtask(
            self.match_insole_bag,
            self.insert_insole_full.insole,
            self.insert_insole_full.bag,
        )
        st2 = self.insert_insole_full.add_subtask(
            self.pick_insole, self.insert_insole_full.insole
        )
        st3 = self.insert_insole_full.add_subtask(
            self.insert, self.insert_insole_full.insole, self.insert_insole_full.bag
        )
        st4 = self.insert_insole_full.add_subtask(
            self.perceive_set,
            self.insert_insole_full.insole,
            self.insert_insole_full.bag,
            self.insert_insole_full.set,
        )
        self.insert_insole_full.set_ordered(st1, st2, st3, st4)

        # finish_set
        # already released, seal set
        self.finish_set_seal = Method("finish_set_seal", set=type_item)
        self.finish_set_seal.set_task(self.finish_set, self.finish_set_seal.set)
        self.finish_set_seal.add_precondition(self.bag_set_released())
        self.finish_set_seal.add_precondition(self.holding(self.finish_set_seal.set))
        self.finish_set_seal.add_subtask(self.seal_set, self.finish_set_seal.set)

        # set already in hand, release set and seal set
        self.finish_set_release = Method("finish_set_release", set=type_item)
        self.finish_set_release.set_task(self.finish_set, self.finish_set_release.set)
        self.finish_set_release.add_precondition(
            self.holding(self.finish_set_release.set)
        )
        st1 = self.finish_set_release.add_subtask(
            self.release_set, self.finish_set_release.set
        )
        st2 = self.finish_set_release.add_subtask(
            self.seal_set, self.finish_set_release.set
        )
        self.finish_set_release.set_ordered(st1, st2)

        # pick set, release set, seal set
        self.finish_set_full = Method("finish_set_full", set=type_item)
        self.finish_set_full.set_task(self.finish_set, self.finish_set_full.set)
        self.finish_set_full.add_precondition(
            self.item_pose_is_known(self.finish_set_full.set)
        )

        st1 = self.finish_set_full.add_subtask(self.pick_set, self.finish_set_full.set)
        st2 = self.finish_set_full.add_subtask(
            self.release_set, self.finish_set_full.set
        )
        st3 = self.finish_set_full.add_subtask(self.seal_set, self.finish_set_full.set)

        self.finish_set_full.set_ordered(st1, st2, st3)

        # bag_insole
        # insole already in bag
        self.bag_insole_finish = Method(
            "bag_insole_finish", insole=type_item, bag=type_item, set=type_item
        )
        self.bag_insole_finish.set_task(
            self.bag_insole,
            self.bag_insole_finish.insole,
            self.bag_insole_finish.bag,
            self.bag_insole_finish.set,
        )
        self.bag_insole_finish.add_precondition(
            Or(
                self.item_pose_is_known(self.bag_insole_finish.set),
                self.holding(self.bag_insole_finish.set),
            )
        )
        self.bag_insole_finish.add_subtask(self.finish_set, self.bag_insole_finish.set)

        # bag and insole are NOT matching, reject insole
        self.bag_insole_no_match = Method(
            "bag_insole_no_match", insole=type_item, bag=type_item, set=type_item
        )
        self.bag_insole_no_match.set_task(
            self.bag_insole,
            self.bag_insole_no_match.insole,
            self.bag_insole_no_match.bag,
            self.bag_insole_no_match.set,
        )

        self.bag_insole_no_match.add_precondition(Not(self.item_types_match()))
        self.bag_insole_no_match.add_precondition(Not(self.not_checked_item_types()))
        self.bag_insole_no_match.add_precondition(
            self.item_type_is_known(self.bag_insole_no_match.insole)
        )
        self.bag_insole_no_match.add_precondition(
            self.item_type_is_known(self.bag_insole_no_match.bag)
        )
        self.bag_insole_no_match.add_precondition(self.item_in_fov())
        self.bag_insole_no_match.add_precondition(
            Not(self.item_pose_is_known(self.bag_insole_no_match.set))
        )
        self.bag_insole_no_match.add_subtask(
            self.reject_insole, self.bag_insole_no_match.insole
        )

        # insert insole and finish set
        self.bag_insole_full = Method(
            "bag_insole_full", insole=type_item, bag=type_item, set=type_item
        )
        self.bag_insole_full.set_task(
            self.bag_insole,
            self.bag_insole_full.insole,
            self.bag_insole_full.bag,
            self.bag_insole_full.set,
        )
        st1 = self.bag_insole_full.add_subtask(
            self.insert_insole, self.bag_insole_full.insole, self.bag_insole_full.bag
        )
        st2 = self.bag_insole_full.add_subtask(
            self.finish_set, self.bag_insole_full.set
        )
        self.bag_insole_full.set_ordered(st1, st2)

        self.methods = (
            self.get_insole_noop,
            self.get_insole_perceive,
            self.get_insole_full,
            self.prepare_bag_noop,
            self.prepare_bag_perceive,
            self.prepare_bag_open,
            self.prepare_bag_full,
            self.prepare_bag_refill,
            self.insert_insole_noop,
            self.insert_insole_match,
            self.insert_insole_insert,
            self.insert_insole_full,
            self.finish_set_seal,
            self.finish_set_release,
            self.finish_set_full,
            self.bag_insole_finish,
            self.bag_insole_no_match,
            self.bag_insole_full,
        )

    def _create_domain_actions(self, temporal: bool = False) -> None:
        actions = Actions(self._env)

        if temporal:
            self.match_insole_bag, [i, b] = self.create_action(
                "match_insole_bag",
                insole=Item,
                bag=Item,
                _callable=actions.match_insole_bag,
                duration=10,
            )
            self.match_insole_bag.add_condition(
                StartTiming(), self.item_type_is_known(i)
            )
            self.match_insole_bag.add_condition(
                StartTiming(), self.item_type_is_known(b)
            )
            self.match_insole_bag.add_condition(
                StartTiming(), self.not_checked_item_types()
            )
            self.match_insole_bag.add_effect(EndTiming(), self.item_types_match(), True)
            self.match_insole_bag.add_effect(
                EndTiming(), self.not_checked_item_types(), False
            )

            self.reject_insole, [i] = self.create_action(
                "reject_insole",
                insole=Item,
                _callable=actions.reject_insole,
                duration=10,
            )
            self.reject_insole.add_condition(StartTiming(), self.item_in_fov())
            self.reject_insole.add_condition(
                StartTiming(), Not(self.item_types_match())
            )
            self.reject_insole.add_condition(
                StartTiming(), Not(self.not_checked_item_types())
            )
            self.reject_insole.add_effect(EndTiming(), self.item_in_fov(), False)
            self.reject_insole.add_effect(
                EndTiming(), self.item_type_is_known(i), False
            )
            self.reject_insole.add_effect(
                EndTiming(), self.item_pose_is_known(i), False
            )
            self.reject_insole.add_effect(
                EndTiming(), self.not_checked_item_types(), True
            )

            self.get_next_insole, [c] = self.create_action(
                "get_next_insole",
                conveyor=Location,
                _callable=actions.get_next_insole,
                duration=10,
            )
            self.get_next_insole.add_condition(StartTiming(), self.stationary(c))
            self.get_next_insole.add_effect(EndTiming(), self.item_in_fov(), True)

            self.preload_bag_bundle, _ = self.create_action(
                "preload_bag_bundle", _callable=actions.preload_bag_bundle, duration=10
            )
            self.preload_bag_bundle.add_effect(
                EndTiming(), self.bag_dispenser_has_bags(), True
            )

            self.load_bag, _ = self.create_action(
                "load_bag", _callable=actions.load_bag, duration=10
            )
            self.load_bag.add_condition(StartTiming(), self.bag_dispenser_has_bags())
            self.load_bag.add_effect(
                EndTiming(), self.bag_is_probably_available(), True
            )

            self.open_bag, _ = self.create_action(
                "open_bag", _callable=actions.open_bag, duration=10
            )
            self.open_bag.add_condition(StartTiming(), self.bag_is_probably_available())
            self.open_bag.add_effect(EndTiming(), self.bag_is_probably_open(), True)

            self.pick_insole, [i] = self.create_action(
                "pick_insole", insole=Item, _callable=actions.pick_insole, duration=30
            )
            self.pick_insole.add_condition(StartTiming(), self.item_pose_is_known(i))
            self.pick_insole.add_condition(StartTiming(), self.item_in_fov())
            self.pick_insole.add_condition(StartTiming(), self.holding(self.nothing))
            self.pick_insole.add_condition(StartTiming(), self.item_types_match())
            self.pick_insole.add_effect(EndTiming(), self.holding(i), True)
            self.pick_insole.add_effect(EndTiming(), self.item_pose_is_known(i), False)
            self.pick_insole.add_effect(EndTiming(), self.item_in_fov(), False)
            self.pick_insole.add_effect(EndTiming(), self.holding(self.nothing), False)

            self.pick_set, [s] = self.create_action(
                "pick_set",
                set=Item,
                _callable=actions.pick_set,
                duration=30,
            )
            self.pick_set.add_condition(StartTiming(), self.item_pose_is_known(s))
            self.pick_set.add_condition(StartTiming(), self.holding(self.nothing))
            self.pick_set.add_effect(EndTiming(), self.holding(self.nothing), False)
            self.pick_set.add_effect(EndTiming(), self.holding(s), True)
            self.pick_set.add_effect(EndTiming(), self.item_pose_is_known(s), False)
            self.pick_set.add_effect(EndTiming(), self.bag_is_probably_available, False)

            self.insert, [i, b] = self.create_action(
                "insert", insole=Item, bag=Item, _callable=actions.insert, duration=30
            )
            self.insert.add_condition(StartTiming(), self.item_pose_is_known(b))
            self.insert.add_condition(StartTiming(), self.bag_is_open())
            self.insert.add_condition(StartTiming(), self.holding(i))
            self.insert.add_condition(StartTiming(), self.item_types_match())
            self.insert.add_effect(EndTiming(), self.insole_inside_bag(i), True)
            self.insert.add_effect(EndTiming(), self.bag_is_open(), False)
            self.insert.add_effect(EndTiming(), self.item_pose_is_known(b), False)
            self.insert.add_effect(EndTiming(), self.holding(i), False)
            self.insert.add_effect(EndTiming(), self.holding(self.nothing), True)

            self.perceive_insole, [i] = self.create_action(
                "perceive_insole",
                insole=Item,
                _callable=actions.perceive_insole,
                duration=10,
            )
            self.perceive_insole.add_condition(StartTiming(), self.item_in_fov())
            self.perceive_insole.add_effect(
                EndTiming(), self.item_pose_is_known(i), True
            )
            self.perceive_insole.add_effect(
                EndTiming(), self.item_type_is_known(i), True
            )

            self.perceive_bag, [b] = self.create_action(
                "perceive_bag", bag=Item, _callable=actions.perceive_bag, duration=10
            )
            self.perceive_bag.add_condition(
                StartTiming(), self.bag_is_probably_available()
            )
            self.perceive_bag.add_effect(
                EndTiming(), self.bag_is_probably_open(), False
            )
            self.perceive_bag.add_effect(EndTiming(), self.item_pose_is_known(b), True)
            self.perceive_bag.add_effect(EndTiming(), self.bag_is_open(), True)
            self.perceive_bag.add_effect(EndTiming(), self.item_type_is_known(b), True)

            self.perceive_set, [i, b, s] = self.create_action(
                "perceive_set",
                insole=Item,
                bag=Item,
                set=Item,
                _callable=actions.perceive_set,
                duration=10,
            )
            self.perceive_set.add_condition(StartTiming(), self.insole_inside_bag(i))
            self.perceive_set.add_effect(EndTiming(), self.insole_inside_bag(i), False)
            self.perceive_set.add_effect(EndTiming(), self.item_pose_is_known(s), True)

            self.release_set, [s] = self.create_action(
                "release_set",
                set=Item,
                _callable=actions.release_set,
                duration=10,
            )
            self.release_set.add_effect(EndTiming(), self.bag_set_released(), True)

            self.seal_set, [s] = self.create_action(
                "seal_set", set=Item, _callable=actions.seal_set, duration=60
            )
            self.seal_set.add_condition(StartTiming(), self.holding(s))
            self.seal_set.add_condition(StartTiming(), self.bag_set_released())
            self.seal_set.add_condition(StartTiming(), self.sealing_machine_ready())
            self.seal_set.add_effect(EndTiming(), self.holding(s), False)
            self.seal_set.add_effect(EndTiming(), self.bag_set_released(), False)
            self.seal_set.add_effect(EndTiming(), self.holding(self.nothing), True)
        else:
            self.reject_insole, [i] = self.create_action(
                "reject_insole",
                insole=Item,
                _callable=actions.reject_insole,
            )
            self.reject_insole.add_precondition(self.item_in_fov())
            self.reject_insole.add_precondition(Not(self.item_types_match()))
            self.reject_insole.add_precondition(Not(self.not_checked_item_types()))
            self.reject_insole.add_effect(self.item_in_fov(), False)
            self.reject_insole.add_effect(self.item_type_is_known(i), False)
            self.reject_insole.add_effect(self.item_pose_is_known(i), False)
            self.reject_insole.add_effect(self.not_checked_item_types(), True)

            self.match_insole_bag, [i, b] = self.create_action(
                "match_insole_bag",
                insole=Item,
                bag=Item,
                _callable=actions.match_insole_bag,
            )
            self.match_insole_bag.add_precondition(self.item_type_is_known(i))
            self.match_insole_bag.add_precondition(self.item_type_is_known(b))
            self.match_insole_bag.add_precondition(self.not_checked_item_types())
            self.match_insole_bag.add_effect(self.item_types_match(), True)
            self.match_insole_bag.add_effect(self.not_checked_item_types(), False)

            self.get_next_insole, [c] = self.create_action(
                "get_next_insole",
                conveyor=Location,
                _callable=actions.get_next_insole,
            )
            self.get_next_insole.add_precondition(self.stationary(c))
            self.get_next_insole.add_effect(self.item_in_fov(), True)

            self.preload_bag_bundle, _ = self.create_action(
                "preload_bag_bundle", _callable=actions.preload_bag_bundle
            )
            self.preload_bag_bundle.add_effect(self.bag_dispenser_has_bags(), True)

            self.load_bag, _ = self.create_action(
                "load_bag", _callable=actions.load_bag
            )
            self.load_bag.add_precondition(self.bag_dispenser_has_bags())
            self.load_bag.add_effect(self.bag_is_probably_available(), True)

            self.open_bag, _ = self.create_action(
                "open_bag", _callable=actions.open_bag
            )
            self.open_bag.add_precondition(self.bag_is_probably_available())
            self.open_bag.add_effect(self.bag_is_probably_open(), True)

            self.pick_insole, [i] = self.create_action(
                "pick_insole", insole=Item, _callable=actions.pick_insole
            )
            self.pick_insole.add_precondition(self.item_pose_is_known(i))
            self.pick_insole.add_precondition(self.item_in_fov())
            self.pick_insole.add_precondition(self.holding(self.nothing))
            self.pick_insole.add_precondition(self.item_types_match())
            self.pick_insole.add_effect(self.holding(i), True)
            self.pick_insole.add_effect(self.item_pose_is_known(i), False)
            self.pick_insole.add_effect(self.item_in_fov(), False)
            self.pick_insole.add_effect(self.holding(self.nothing), False)

            self.pick_set, [s] = self.create_action(
                "pick_set", set=Item, _callable=actions.pick_set
            )
            self.pick_set.add_precondition(self.item_pose_is_known(s))
            self.pick_set.add_precondition(self.holding(self.nothing))
            self.pick_set.add_effect(self.holding(self.nothing), False)
            self.pick_set.add_effect(self.holding(s), True)
            self.pick_set.add_effect(self.item_pose_is_known(s), False)
            self.pick_set.add_effect(self.bag_is_probably_available, False)

            self.insert, [i, b] = self.create_action(
                "insert", insole=Item, bag=Item, _callable=actions.insert
            )
            self.insert.add_precondition(self.item_pose_is_known(b))
            self.insert.add_precondition(self.bag_is_open())
            self.insert.add_precondition(self.holding(i))
            self.insert.add_precondition(self.item_types_match())
            self.insert.add_effect(self.insole_inside_bag(i), True)
            self.insert.add_effect(self.bag_is_open(), False)
            self.insert.add_effect(self.item_pose_is_known(b), False)
            self.insert.add_effect(self.holding(i), False)
            self.insert.add_effect(self.holding(self.nothing), True)
            self.insert.add_effect(self.item_types_match(), False)

            self.perceive_insole, [i] = self.create_action(
                "perceive_insole", insole=Item, _callable=actions.perceive_insole
            )
            self.perceive_insole.add_precondition(self.item_in_fov())
            self.perceive_insole.add_effect(self.item_pose_is_known(i), True)
            self.perceive_insole.add_effect(self.item_type_is_known(i), True)

            self.perceive_bag, [b] = self.create_action(
                "perceive_bag", bag=Item, _callable=actions.perceive_bag
            )
            self.perceive_bag.add_precondition(self.bag_is_probably_available())
            self.perceive_bag.add_effect(self.bag_is_probably_open(), False)
            self.perceive_bag.add_effect(self.item_pose_is_known(b), True)
            self.perceive_bag.add_effect(self.bag_is_open(), True)
            self.perceive_bag.add_effect(self.item_type_is_known(b), True)

            self.perceive_set, [i, b, s] = self.create_action(
                "perceive_set",
                insole=Item,
                bag=Item,
                set=Item,
                _callable=actions.perceive_set,
            )
            self.perceive_set.add_precondition(self.insole_inside_bag(i))
            self.perceive_set.add_effect(self.insole_inside_bag(i), False)
            self.perceive_set.add_effect(self.item_pose_is_known(s), True)

            self.release_set, [s] = self.create_action(
                "release_set", set=Item, _callable=actions.release_set
            )
            self.release_set.add_effect(self.bag_set_released(), True)

            self.seal_set, [s] = self.create_action(
                "seal_set", set=Item, _callable=actions.seal_set
            )
            self.seal_set.add_precondition(self.holding(s))
            self.seal_set.add_precondition(self.bag_set_released())
            self.seal_set.add_precondition(self.sealing_machine_ready())
            self.seal_set.add_effect(self.holding(s), False)
            self.seal_set.add_effect(self.bag_set_released(), False)
            self.seal_set.add_effect(self.holding(self.nothing), True)

    def set_state_and_goal(self, problem, goal=None) -> bool:
        success = True
        if goal is None:
            subtask_get_insole = problem.task_network.add_subtask(
                self.get_insole(self.conveyor_a, self.insole)
            )
            subtask_prepare_bag = problem.task_network.add_subtask(
                self.prepare_bag(self.bag)
            )
            subtask_bag_insole = problem.task_network.add_subtask(
                self.bag_insole(self.insole, self.bag, self.set)
            )

            problem.task_network.set_ordered(subtask_get_insole, subtask_bag_insole)
            problem.task_network.set_ordered(subtask_prepare_bag, subtask_bag_insole)
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
                    f"Task ({goal}) is unknown! Please use no goal for the complete scenario or a task from this list: "
                    "get_next_insole, preload_bag_bundle, load_bag, pick_insole, pick_set, "
                    "open_bag, insert, release_set, seal_set, perceive_insole, perceive_bag, "
                    "perceive_set, reject_insole, match_insole_bag."
                )
            )
            success = False
        return success
