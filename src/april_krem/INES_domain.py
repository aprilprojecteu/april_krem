from unified_planning.model.htn import Task, Method
from unified_planning.shortcuts import Or, StartTiming, EndTiming
from april_krem.INES_components import (
    Item,
    Location,
    Actions,
    Environment,
)
from up_esb.bridge import Bridge


class INESDomain(Bridge):
    def __init__(self, temporal: bool = False) -> None:
        Bridge.__init__(self)

        env = Environment()

        # Create types for planning based on class types
        self.create_types([Item, Location])
        type_item = self.get_type(Item)
        type_location = self.get_type(Location)

        # Create fluents for planning
        self.holding = self.create_fluent("holding", item=Item)
        self.item_pose_is_known = self.create_fluent("item_pose_is_known", item=Item)
        self.item_in_fov = self.create_fluent("item_in_fov")
        self.bag_set_released = self.create_fluent("bag_set_released")
        self.set_available = self.create_fluent("set_available", insole=Item, bag=Item)
        self.bag_is_available = self.create_fluent("bag_is_available", bag=Item)
        self.bag_dispenser_has_bags = self.create_fluent_from_function(
            env.bag_dispenser_has_bags
        )
        self.sealing_machine_ready = self.create_fluent_from_function(
            env.sealing_machine_ready
        )

        self.human_available = self.create_fluent("human_available")
        self.item_type_is_known = self.create_fluent("item_type_is_known", item=Item)
        self.insole_inside_bag = self.create_fluent(
            "insole_inside_bag", insole=Item, bag=Item
        )
        self.bag_is_probably_available = self.create_fluent("bag_is_probably_available")
        self.bag_is_probably_open = self.create_fluent("bag_is_probably_open")
        self.bag_is_open = self.create_fluent("bag_is_open", bag=Item)

        self.moving = self.create_fluent_from_function(env.moving)
        self.stationary = self.create_fluent_from_function(env.stationary)

        # Create objects for both planning and execution
        self.items = self.create_enum_objects(Item)
        self.nothing = self.objects[Item.nothing.name]
        self.insole = self.objects[Item.insole.name]
        self.bag = self.objects[Item.bag.name]

        self.locations = self.create_enum_objects(Location)
        self.conveyor_a = self.objects[Location.conveyor_a.name]
        self.conveyor_b = self.objects[Location.conveyor_b.name]
        self.in_hand = self.objects[Location.in_hand.name]
        self.in_bag = self.objects[Location.in_bag.name]

        # Create actions for planning
        self._create_domain_actions(env, temporal)

        self._monitored_fluents = [
            self.moving,
            self.stationary,
            self.bag_dispenser_has_bags,
            self.sealing_machine_ready,
        ]

        # Tasks
        self.get_insole = Task("get_insole", conveyor=type_location, insole=type_item)
        self.prepare_bag = Task("prepare_bag", bag=type_item)
        self.insert_insole = Task("insert_insole", insole=type_item, bag=type_item)
        self.finish_set = Task("finish_set", insole=type_item, bag=type_item)
        self.bag_insole = Task("bag_insole", insole=type_item, bag=type_item)
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
            self.item_pose_is_known(self.get_insole_noop.insole)
        )
        self.get_insole_noop.add_precondition(
            self.item_type_is_known(self.get_insole_noop.insole)
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
        st1 = self.get_insole_full.add_subtask(
            self.get_next_insole,
            self.get_insole_full.conveyor,
            self.get_insole_full.insole,
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
            self.bag_is_available(self.prepare_bag_noop.bag)
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
        st1 = self.prepare_bag_open.add_subtask(
            self.open_bag, self.prepare_bag_open.bag
        )
        st2 = self.prepare_bag_open.add_subtask(
            self.perceive_bag, self.prepare_bag_open.bag
        )
        self.prepare_bag_open.set_ordered(st1, st2)

        # bag dispenser has bags, load and inspect bag
        self.prepare_bag_full = Method("prepare_bag_full", bag=type_item)
        self.prepare_bag_full.set_task(self.prepare_bag, self.prepare_bag_full.bag)
        self.prepare_bag_full.add_precondition(self.bag_dispenser_has_bags())
        st1 = self.prepare_bag_full.add_subtask(
            self.load_bag, self.prepare_bag_full.bag
        )
        st2 = self.prepare_bag_full.add_subtask(
            self.open_bag, self.prepare_bag_full.bag
        )
        st3 = self.prepare_bag_full.add_subtask(
            self.perceive_bag, self.prepare_bag_full.bag
        )
        self.prepare_bag_full.set_ordered(st1, st2, st3)

        # bag dispenser empty, refill dispenser, then load and inspect bag
        self.prepare_bag_refill = Method("prepare_bag_refill", bag=type_item)
        self.prepare_bag_refill.set_task(self.prepare_bag, self.prepare_bag_refill.bag)
        self.prepare_bag_refill.add_precondition(self.human_available())
        st1 = self.prepare_bag_refill.add_subtask(self.preload_bag_bundle)
        st2 = self.prepare_bag_refill.add_subtask(
            self.load_bag, self.prepare_bag_refill.bag
        )
        st3 = self.prepare_bag_refill.add_subtask(
            self.open_bag, self.prepare_bag_refill.bag
        )
        st4 = self.prepare_bag_refill.add_subtask(
            self.perceive_bag, self.prepare_bag_refill.bag
        )
        self.prepare_bag_refill.set_ordered(st1, st2, st3, st4)

        # insert_insole
        # already inserted insole
        self.insert_insole_noop = Method(
            "insert_insole_noop", insole=type_item, bag=type_item
        )
        self.insert_insole_noop.set_task(
            self.insert_insole,
            self.insert_insole_noop.insole,
            self.insert_insole_noop.bag,
        )
        self.insert_insole_noop.add_precondition(
            self.insole_inside_bag(
                self.insert_insole_noop.insole, self.insert_insole_noop.bag
            )
        )
        self.insert_insole_noop.add_precondition(self.holding(self.nothing))
        self.insert_insole_noop.add_subtask(
            self.perceive_set,
            self.insert_insole_noop.insole,
            self.insert_insole_noop.bag,
        )

        # already holding insole
        self.insert_insole_insert = Method(
            "insert_insole_insert", insole=type_item, bag=type_item
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
        )
        self.insert_insole_insert.set_ordered(st1, st2)

        # bag and insole are matching, pick and insert
        self.insert_insole_full = Method(
            "insert_insole_full", insole=type_item, bag=type_item
        )
        self.insert_insole_full.set_task(
            self.insert_insole,
            self.insert_insole_full.insole,
            self.insert_insole_full.bag,
        )
        self.insert_insole_full.add_precondition(
            self.item_type_is_known(self.insert_insole_full.insole)
        )
        self.insert_insole_full.add_precondition(
            self.item_type_is_known(self.insert_insole_full.bag)
        )
        st1 = self.insert_insole_full.add_subtask(
            self.pick_insole, self.insert_insole_full.insole
        )
        st2 = self.insert_insole_full.add_subtask(
            self.insert, self.insert_insole_full.insole, self.insert_insole_full.bag
        )
        st3 = self.insert_insole_full.add_subtask(
            self.perceive_set,
            self.insert_insole_full.insole,
            self.insert_insole_full.bag,
        )
        self.insert_insole_full.set_ordered(st1, st2, st3)

        # finish_set
        # already released, seal set
        self.finish_set_seal = Method(
            "finish_set_seal", insole=type_item, bag=type_item
        )
        self.finish_set_seal.set_task(
            self.finish_set, self.finish_set_seal.insole, self.finish_set_seal.bag
        )
        self.finish_set_seal.add_precondition(self.bag_set_released())
        self.finish_set_seal.add_precondition(self.holding(self.finish_set_seal.bag))
        self.finish_set_seal.add_subtask(self.seal_set, self.finish_set_seal.bag)

        # set already in hand, release set and seal set
        self.finish_set_release = Method(
            "finish_set_release", insole=type_item, bag=type_item
        )
        self.finish_set_release.set_task(
            self.finish_set, self.finish_set_release.insole, self.finish_set_release.bag
        )
        self.finish_set_release.add_precondition(
            self.holding(self.finish_set_release.bag)
        )
        st1 = self.finish_set_release.add_subtask(
            self.release_bag,
            self.finish_set_release.insole,
            self.finish_set_release.bag,
        )
        st2 = self.finish_set_release.add_subtask(
            self.seal_set, self.finish_set_release.bag
        )
        self.finish_set_release.set_ordered(st1, st2)

        # pick set, release set, seal set
        self.finish_set_full = Method(
            "finish_set_full", insole=type_item, bag=type_item
        )
        self.finish_set_full.set_task(
            self.finish_set, self.finish_set_full.insole, self.finish_set_full.bag
        )
        self.finish_set_full.add_precondition(
            self.set_available(self.finish_set_full.insole, self.finish_set_full.bag)
        )

        st1 = self.finish_set_full.add_subtask(
            self.pick_set, self.finish_set_full.insole, self.finish_set_full.bag
        )
        st2 = self.finish_set_full.add_subtask(
            self.release_bag, self.finish_set_full.insole, self.finish_set_full.bag
        )
        st3 = self.finish_set_full.add_subtask(self.seal_set, self.finish_set_full.bag)

        self.finish_set_full.set_ordered(st1, st2, st3)

        # bag_insole
        # insole already in bag
        self.bag_insole_finish = Method(
            "bag_insole_finish", insole=type_item, bag=type_item
        )
        self.bag_insole_finish.set_task(
            self.bag_insole, self.bag_insole_finish.insole, self.bag_insole_finish.bag
        )
        self.bag_insole_finish.add_precondition(
            Or(
                self.set_available(
                    self.bag_insole_finish.insole, self.bag_insole_finish.bag
                ),
                self.holding(self.bag_insole_finish.bag),
            )
        )
        self.bag_insole_finish.add_subtask(
            self.finish_set, self.bag_insole_finish.insole, self.bag_insole_finish.bag
        )

        # insert insole and finish set
        self.bag_insole_full = Method(
            "bag_insole_full", insole=type_item, bag=type_item
        )
        self.bag_insole_full.set_task(
            self.bag_insole, self.bag_insole_full.insole, self.bag_insole_full.bag
        )
        st1 = self.bag_insole_full.add_subtask(
            self.insert_insole, self.bag_insole_full.insole, self.bag_insole_full.bag
        )
        st2 = self.bag_insole_full.add_subtask(
            self.finish_set, self.bag_insole_full.insole, self.bag_insole_full.bag
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
            self.insert_insole_insert,
            self.insert_insole_full,
            self.finish_set_seal,
            self.finish_set_release,
            self.finish_set_full,
            self.bag_insole_finish,
            self.bag_insole_full,
        )

    def _create_domain_actions(self, env, temporal: bool = False) -> None:
        actions = Actions(env)

        if temporal:
            self.reject_insole, [x] = self.create_action(
                "reject_insole",
                conveyor=Location,
                _callable=actions.reject_insole,
                duration=10,
            )
            self.reject_insole.add_condition(StartTiming(), self.stationary(x))
            self.reject_insole.add_effect(StartTiming(), self.moving(x), True)
            self.reject_insole.add_effect(EndTiming(), self.moving(x), False)
            self.reject_insole.add_effect(EndTiming(), self.stationary(x), True)

            self.get_next_insole, [x, _] = self.create_action(
                "get_next_insole",
                conveyor=Location,
                insole=Item,
                _callable=actions.get_next_insole,
                duration=10,
            )
            self.get_next_insole.add_condition(StartTiming(), self.stationary(x))
            self.get_next_insole.add_effect(EndTiming(), self.item_in_fov(), True)

            self.preload_bag_bundle, _ = self.create_action(
                "preload_bag_bundle", _callable=actions.preload_bag_bundle, duration=10
            )
            self.preload_bag_bundle.add_condition(StartTiming(), self.human_available())
            self.preload_bag_bundle.add_effect(
                EndTiming(), self.bag_dispenser_has_bags(), True
            )

            self.load_bag, _ = self.create_action(
                "load_bag", bag=Item, _callable=actions.load_bag, duration=10
            )
            self.load_bag.add_condition(StartTiming(), self.bag_dispenser_has_bags())
            self.load_bag.add_effect(
                EndTiming(), self.bag_is_probably_available(), True
            )

            self.open_bag, _ = self.create_action(
                "open_bag", bag=Item, _callable=actions.open_bag, duration=10
            )
            self.open_bag.add_condition(StartTiming(), self.bag_is_probably_available())
            self.open_bag.add_effect(EndTiming(), self.bag_is_probably_open(), True)

            self.pick_insole, [x] = self.create_action(
                "pick_insole", insole=Item, _callable=actions.pick_insole, duration=30
            )
            self.pick_insole.add_condition(StartTiming(), self.item_pose_is_known(x))
            self.pick_insole.add_condition(StartTiming(), self.item_in_fov())
            self.pick_insole.add_condition(StartTiming(), self.holding(self.nothing))
            self.pick_insole.add_effect(EndTiming(), self.holding(x), True)
            self.pick_insole.add_effect(EndTiming(), self.item_pose_is_known(x), False)
            self.pick_insole.add_effect(EndTiming(), self.item_in_fov(), False)
            self.pick_insole.add_effect(EndTiming(), self.holding(self.nothing), False)

            self.pick_set, [x, y] = self.create_action(
                "pick_set",
                insole=Item,
                bag=Item,
                _callable=actions.pick_set,
                duration=30,
            )
            self.pick_set.add_condition(StartTiming(), self.set_available(x, y))
            self.pick_set.add_condition(StartTiming(), self.holding(self.nothing))
            self.pick_set.add_effect(EndTiming(), self.holding(self.nothing), False)
            self.pick_set.add_effect(EndTiming(), self.holding(y), True)

            self.insert, [x, y] = self.create_action(
                "insert", insole=Item, bag=Item, _callable=actions.insert, duration=30
            )
            self.insert.add_condition(StartTiming(), self.bag_is_available(y))
            self.insert.add_condition(StartTiming(), self.bag_is_open(y))
            self.insert.add_condition(StartTiming(), self.holding(x))
            self.insert.add_effect(EndTiming(), self.insole_inside_bag(x, y), True)
            self.insert.add_effect(EndTiming(), self.bag_is_open(y), False)
            self.insert.add_effect(EndTiming(), self.bag_is_available(y), False)
            self.insert.add_effect(EndTiming(), self.holding(x), False)
            self.insert.add_effect(EndTiming(), self.holding(self.nothing), True)

            self.perceive_insole, [x] = self.create_action(
                "perceive_insole",
                insole=Item,
                _callable=actions.perceive_insole,
                duration=10,
            )
            self.perceive_insole.add_condition(StartTiming(), self.item_in_fov())
            self.perceive_insole.add_effect(
                EndTiming(), self.item_pose_is_known(x), True
            )
            self.perceive_insole.add_effect(
                EndTiming(), self.item_type_is_known(x), True
            )

            self.perceive_bag, [x] = self.create_action(
                "perceive_bag", bag=Item, _callable=actions.perceive_bag, duration=10
            )
            self.perceive_bag.add_condition(
                StartTiming(), self.bag_is_probably_available()
            )
            self.perceive_bag.add_effect(
                EndTiming(), self.bag_is_probably_available(), False
            )
            self.perceive_bag.add_effect(EndTiming(), self.bag_is_available(x), True)
            self.perceive_bag.add_effect(EndTiming(), self.bag_is_open(x), True)
            self.perceive_bag.add_effect(EndTiming(), self.item_type_is_known(x), True)

            self.perceive_set, [x, y] = self.create_action(
                "perceive_set",
                insole=Item,
                bag=Item,
                _callable=actions.perceive_set,
                duration=10,
            )
            self.perceive_set.add_condition(StartTiming(), self.insole_inside_bag(x, y))
            self.perceive_set.add_effect(
                EndTiming(), self.insole_inside_bag(x, y), False
            )
            self.perceive_set.add_effect(EndTiming(), self.set_available(x, y), True)

            self.release_bag, [x, y] = self.create_action(
                "release_bag",
                insole=Item,
                bag=Item,
                _callable=actions.release_bag,
                duration=10,
            )
            self.release_bag.add_condition(StartTiming(), self.set_available(x, y))
            self.release_bag.add_effect(EndTiming(), self.set_available(x, y), False)
            self.release_bag.add_effect(EndTiming(), self.bag_set_released(), True)

            self.seal_set, [x] = self.create_action(
                "seal_set", bag=Item, _callable=actions.seal_set, duration=60
            )
            self.seal_set.add_condition(StartTiming(), self.holding(x))
            self.seal_set.add_condition(StartTiming(), self.bag_set_released())
            self.seal_set.add_condition(StartTiming(), self.sealing_machine_ready())
            self.seal_set.add_effect(EndTiming(), self.holding(x), False)
            self.seal_set.add_effect(EndTiming(), self.bag_set_released(), False)
            self.seal_set.add_effect(EndTiming(), self.holding(self.nothing), True)
        else:
            self.reject_insole, [x] = self.create_action(
                "reject_insole", conveyor=Location, _callable=actions.reject_insole
            )
            self.reject_insole.add_precondition(self.stationary(x))

            self.get_next_insole, [x, _] = self.create_action(
                "get_next_insole",
                conveyor=Location,
                insole=Item,
                _callable=actions.get_next_insole,
            )
            self.get_next_insole.add_precondition(self.stationary(x))
            self.get_next_insole.add_effect(self.item_in_fov(), True)

            self.preload_bag_bundle, _ = self.create_action(
                "preload_bag_bundle", _callable=actions.preload_bag_bundle
            )
            self.preload_bag_bundle.add_precondition(self.human_available())
            self.preload_bag_bundle.add_effect(self.bag_dispenser_has_bags(), True)

            self.load_bag, _ = self.create_action(
                "load_bag", bag=Item, _callable=actions.load_bag
            )
            self.load_bag.add_precondition(self.bag_dispenser_has_bags())
            self.load_bag.add_effect(self.bag_is_probably_available(), True)

            self.open_bag, _ = self.create_action(
                "open_bag", bag=Item, _callable=actions.open_bag
            )
            self.open_bag.add_precondition(self.bag_is_probably_available())
            self.open_bag.add_effect(self.bag_is_probably_open(), True)

            self.pick_insole, [x] = self.create_action(
                "pick_insole", insole=Item, _callable=actions.pick_insole
            )
            self.pick_insole.add_precondition(self.item_pose_is_known(x))
            self.pick_insole.add_precondition(self.item_in_fov())
            self.pick_insole.add_precondition(self.holding(self.nothing))
            self.pick_insole.add_effect(self.holding(x), True)
            self.pick_insole.add_effect(self.item_pose_is_known(x), False)
            self.pick_insole.add_effect(self.item_in_fov(), False)
            self.pick_insole.add_effect(self.holding(self.nothing), False)

            self.pick_set, [x, y] = self.create_action(
                "pick_set", insole=Item, bag=Item, _callable=actions.pick_set
            )
            self.pick_set.add_precondition(self.set_available(x, y))
            self.pick_set.add_precondition(self.holding(self.nothing))
            self.pick_set.add_effect(self.holding(self.nothing), False)
            self.pick_set.add_effect(self.holding(y), True)

            self.insert, [x, y] = self.create_action(
                "insert", insole=Item, bag=Item, _callable=actions.insert
            )
            self.insert.add_precondition(self.bag_is_available(y))
            self.insert.add_precondition(self.bag_is_open(y))
            self.insert.add_precondition(self.holding(x))
            self.insert.add_effect(self.insole_inside_bag(x, y), True)
            self.insert.add_effect(self.bag_is_open(y), False)
            self.insert.add_effect(self.bag_is_available(y), False)
            self.insert.add_effect(self.holding(x), False)
            self.insert.add_effect(self.holding(self.nothing), True)

            self.perceive_insole, [x] = self.create_action(
                "perceive_insole", insole=Item, _callable=actions.perceive_insole
            )
            self.perceive_insole.add_precondition(self.item_in_fov())
            self.perceive_insole.add_effect(self.item_pose_is_known(x), True)
            self.perceive_insole.add_effect(self.item_type_is_known(x), True)

            self.perceive_bag, [x] = self.create_action(
                "perceive_bag", bag=Item, _callable=actions.perceive_bag
            )
            self.perceive_bag.add_precondition(self.bag_is_probably_available())
            self.perceive_bag.add_effect(self.bag_is_probably_available(), False)
            self.perceive_bag.add_effect(self.bag_is_available(x), True)
            self.perceive_bag.add_effect(self.bag_is_open(x), True)
            self.perceive_bag.add_effect(self.item_type_is_known(x), True)

            self.perceive_set, [x, y] = self.create_action(
                "perceive_set", insole=Item, bag=Item, _callable=actions.perceive_set
            )
            self.perceive_set.add_precondition(self.insole_inside_bag(x, y))
            self.perceive_set.add_effect(self.insole_inside_bag(x, y), False)
            self.perceive_set.add_effect(self.set_available(x, y), True)

            self.release_bag, [x, y] = self.create_action(
                "release_bag", insole=Item, bag=Item, _callable=actions.release_bag
            )
            self.release_bag.add_precondition(self.set_available(x, y))
            self.release_bag.add_effect(self.set_available(x, y), False)
            self.release_bag.add_effect(self.bag_set_released(), True)

            self.seal_set, [x] = self.create_action(
                "seal_set", bag=Item, _callable=actions.seal_set
            )
            self.seal_set.add_precondition(self.holding(x))
            self.seal_set.add_precondition(self.bag_set_released())
            self.seal_set.add_precondition(self.sealing_machine_ready())
            self.seal_set.add_effect(self.holding(x), False)
            self.seal_set.add_effect(self.bag_set_released(), False)
            self.seal_set.add_effect(self.holding(self.nothing), True)

    def set_state_and_goal(self, problem, goal=None) -> None:
        problem.set_initial_value(self.human_available(), True)

        if goal is None:
            problem.set_initial_value(self.holding(self.nothing), True)

            subtask_get_insole = problem.task_network.add_subtask(
                self.get_insole(self.conveyor_a, self.insole)
            )
            subtask_prepare_bag = problem.task_network.add_subtask(
                self.prepare_bag(self.bag)
            )
            subtask_bag_insole = problem.task_network.add_subtask(
                self.bag_insole(self.insole, self.bag)
            )

            problem.task_network.set_ordered(subtask_get_insole, subtask_bag_insole)
            problem.task_network.set_ordered(subtask_prepare_bag, subtask_bag_insole)
        elif goal == "get_next_insole":
            problem.set_initial_value(self.holding(self.nothing), True)

            subtask_get_insole = problem.task_network.add_subtask(
                self.get_insole(self.conveyor_a, self.insole)
            )
        elif goal == "preload_bag_bundle":
            problem.set_initial_value(self.holding(self.nothing), True)

            subtask_prepare_bag = problem.task_network.add_subtask(
                self.prepare_bag(self.bag)
            )
        elif goal == "load_bag":
            problem.set_initial_value(self.holding(self.nothing), True)
            problem.set_initial_value(self.bag_dispenser_has_bags(), True)

            subtask_prepare_bag = problem.task_network.add_subtask(
                self.prepare_bag(self.bag)
            )
        elif goal == "pick_insole":
            problem.set_initial_value(self.holding(self.nothing), True)
            problem.set_initial_value(self.item_pose_is_known(self.insole), True)
            problem.set_initial_value(self.item_type_is_known(self.insole), True)
            problem.set_initial_value(self.item_type_is_known(self.bag), True)
            problem.set_initial_value(self.bag_is_available(self.bag), True)
            problem.set_initial_value(self.bag_is_open(self.bag), True)
            problem.set_initial_value(self.bag_dispenser_has_bags(), True)
            problem.set_initial_value(self.item_in_fov(), True)

            subtask_bag_insole = problem.task_network.add_subtask(
                self.bag_insole(self.insole, self.bag)
            )
        elif goal == "open_bag":
            problem.set_initial_value(self.holding(self.nothing), True)
            problem.set_initial_value(self.bag_is_probably_available(), True)

            subtask_prepare_bag = problem.task_network.add_subtask(
                self.prepare_bag(self.bag)
            )
        elif goal == "release_bag":
            problem.set_initial_value(self.holding(self.bag), True)
            problem.set_initial_value(self.set_available(self.insole, self.bag), True)

            subtask_bag_insole = problem.task_network.add_subtask(
                self.bag_insole(self.insole, self.bag)
            )
        elif goal == "seal_set":
            problem.set_initial_value(self.holding(self.bag), True)
            problem.set_initial_value(self.bag_set_released(), True)

            subtask_bag_insole = problem.task_network.add_subtask(
                self.bag_insole(self.insole, self.bag)
            )
        else:
            print(
                (
                    f"Task ({goal}) is unknown! Please use a task from this list: "
                    "get_next_insole, preload_bag_bundle, load_bag, pick_insole, "
                    "open_bag, release_bag, seal_set"
                )
            )
