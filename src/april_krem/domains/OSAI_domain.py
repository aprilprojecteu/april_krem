from rospy import logerr
from unified_planning.model.htn import Task, Method
from unified_planning.shortcuts import Not
from april_krem.domains.OSAI_components import (
    Item,
    Size,
    Status,
    ArmPose,
    Actions,
    Environment,
)
from up_esb.bridge import Bridge


class OSAIDomain(Bridge):
    def __init__(self, krem_logging, temporal: bool = False) -> None:
        Bridge.__init__(self)

        self._env = Environment(krem_logging)

        # Create types for planning based on class types
        self.create_types([Item, ArmPose, Size, Status])
        type_item = self.get_type(Item)
        type_size = self.get_type(Size)
        type_status = self.get_type(Status)

        # Create fluents for planning
        self.holding = self.create_fluent_from_function(self._env.holding)

        self.current_arm_pose = self.create_fluent_from_function(
            self._env.current_arm_pose
        )

        self.item_size_known = self.create_fluent_from_function(
            self._env.item_size_known
        )
        self.current_item_size = self.create_fluent_from_function(
            self._env.current_item_size
        )
        self.case_is_placed = self.create_fluent_from_function(self._env.case_is_placed)
        self.perceived_insert = self.create_fluent_from_function(
            self._env.perceived_insert
        )
        self.perceived_set = self.create_fluent_from_function(self._env.perceived_set)
        self.item_in_fov = self.create_fluent_from_function(self._env.item_in_fov)
        self.set_status_known = self.create_fluent_from_function(
            self._env.set_status_known
        )
        self.status_of_set = self.create_fluent_from_function(self._env.status_of_set)
        self.inserted_insert = self.create_fluent_from_function(
            self._env.inserted_insert
        )
        self.inserts_available = self.create_fluent_from_function(
            self._env.inserts_available
        )
        self.space_in_box = self.create_fluent_from_function(self._env.space_in_box)

        # Create objects for both planning and execution
        self.items = self.create_enum_objects(Item)
        self.nothing = self.objects[Item.nothing.name]
        self.case = self.objects[Item.case.name]
        self.insert_o = self.objects[Item.insert_o.name]
        self.set = self.objects[Item.set.name]

        self.sizes = self.create_enum_objects(Size)
        self.small = self.objects[Size.small.name]
        self.big = self.objects[Size.big.name]

        self.status = self.create_enum_objects(Status)
        self.ok = self.objects[Status.ok.name]
        self.nok = self.objects[Status.nok.name]

        self.arm_poses = self.create_enum_objects(ArmPose)
        self.unknown_pose = self.objects[ArmPose.unknown.name]
        self.home = self.objects[ArmPose.home.name]
        self.arm_up = self.objects[ArmPose.arm_up.name]
        self.over_conveyor = self.objects[ArmPose.over_conveyor.name]
        self.over_fixture = self.objects[ArmPose.over_fixture.name]
        self.over_pallet = self.objects[ArmPose.over_pallet.name]
        self.over_boxes = self.objects[ArmPose.over_boxes.name]

        # Create actions for planning
        self._create_domain_actions(temporal)

        # Tasks
        self.t_get_case = Task("t_get_case", case=type_item)
        self.t_insertion = Task("t_insertion", case=type_item, insert=type_item)
        self.t_place_set = Task("t_place_set", set=type_item)
        self.t_assemble_set = Task(
            "t_assemble_set", case=type_item, insert=type_item, set=type_item
        )

        self.tasks = (
            self.t_get_case,
            self.t_insertion,
            self.t_place_set,
            self.t_assemble_set,
        )

        # Methods

        # GET CASE
        # case perceived and ready to be picked up
        self.get_case_noop = Method("get_case_noop", case=type_item)
        self.get_case_noop.set_task(self.t_get_case, self.get_case_noop.case)
        self.get_case_noop.add_precondition(self.item_size_known())

        # case on conveyor, need to perceive
        self.get_case_perceive = Method("get_case_perceive", case=type_item)
        self.get_case_perceive.set_task(self.t_get_case, self.get_case_perceive.case)
        self.get_case_perceive.add_precondition(self.item_in_fov())
        self.get_case_perceive.add_precondition(
            self.current_arm_pose(self.over_conveyor)
        )
        self.get_case_perceive.add_subtask(self.perceive_case)

        # arm in position, wait for case and perceive
        self.get_case_wait = Method("get_case_wait", case=type_item)
        self.get_case_wait.set_task(self.t_get_case, self.get_case_wait.case)
        self.get_case_wait.add_precondition(self.current_arm_pose(self.over_conveyor))
        s1 = self.get_case_wait.add_subtask(self.get_next_case)
        s2 = self.get_case_wait.add_subtask(self.perceive_case)
        self.get_case_wait.set_ordered(s1, s2)

        # move arm to conveyor, wait for case, perceive case
        self.get_case_full = Method("get_case_full", case=type_item)
        self.get_case_full.set_task(self.t_get_case, self.get_case_full.case)
        self.get_case_full.add_precondition(Not(self.item_in_fov()))
        self.get_case_full.add_precondition(Not(self.item_size_known()))
        s1 = self.get_case_full.add_subtask(self.move_arm, self.over_conveyor)
        s2 = self.get_case_full.add_subtask(self.get_next_case)
        s3 = self.get_case_full.add_subtask(self.perceive_case)
        self.get_case_full.set_ordered(s1, s2, s3)

        # INSERTION
        # insertion and inspection done, nothing to do
        self.insertion_noop = Method("insertion_noop", case=type_item, insert=type_item)
        self.insertion_noop.set_task(
            self.t_insertion, self.insertion_noop.case, self.insertion_noop.insert
        )
        self.insertion_noop.add_precondition(self.set_status_known())
        self.insertion_noop.add_precondition(self.inserted_insert())
        self.insertion_noop.add_precondition(self.perceived_set())

        # inserted and inspected, still need to perceive set
        self.insertion_perceive_set = Method(
            "insertion_perceive_set", case=type_item, insert=type_item
        )
        self.insertion_perceive_set.set_task(
            self.t_insertion,
            self.insertion_perceive_set.case,
            self.insertion_perceive_set.insert,
        )
        self.insertion_perceive_set.add_precondition(Not(self.perceived_set()))
        self.insertion_perceive_set.add_precondition(self.set_status_known())
        self.insertion_perceive_set.add_precondition(self.case_is_placed())
        self.insertion_perceive_set.add_precondition(self.inserted_insert())
        self.insertion_perceive_set.add_precondition(
            self.current_arm_pose(self.over_fixture)
        )
        self.insertion_perceive_set.add_precondition(self.holding(self.nothing))
        self.insertion_perceive_set.add_subtask(self.perceive_set)

        # inserted, need to inspect and perceive
        self.insertion_inspect = Method(
            "insertion_inspect", case=type_item, insert=type_item
        )
        self.insertion_inspect.set_task(
            self.t_insertion, self.insertion_inspect.case, self.insertion_inspect.insert
        )
        self.insertion_inspect.add_precondition(Not(self.set_status_known()))
        self.insertion_inspect.add_precondition(self.case_is_placed())
        self.insertion_inspect.add_precondition(self.inserted_insert())
        self.insertion_inspect.add_precondition(
            self.current_arm_pose(self.over_fixture)
        )
        self.insertion_inspect.add_precondition(self.holding(self.nothing))
        s1 = self.insertion_inspect.add_subtask(self.inspect)
        s2 = self.insertion_inspect.add_subtask(self.perceive_set)
        self.insertion_inspect.set_ordered(s1, s2)

        # inserted, arm moved up, move arm over fixture, inspect, perceive
        self.insertion_move_arm_8 = Method(
            "insertion_move_arm_8", case=type_item, insert=type_item
        )
        self.insertion_move_arm_8.set_task(
            self.t_insertion,
            self.insertion_move_arm_8.case,
            self.insertion_move_arm_8.insert,
        )
        self.insertion_move_arm_8.add_precondition(self.case_is_placed())
        self.insertion_move_arm_8.add_precondition(self.inserted_insert())
        self.insertion_move_arm_8.add_precondition(self.current_arm_pose(self.arm_up))
        self.insertion_move_arm_8.add_precondition(self.holding(self.nothing))
        s1 = self.insertion_move_arm_8.add_subtask(self.move_arm, self.over_fixture)
        s2 = self.insertion_move_arm_8.add_subtask(self.inspect)
        s3 = self.insertion_move_arm_8.add_subtask(self.perceive_set)
        self.insertion_move_arm_8.set_ordered(s1, s2, s3)

        # already inserted, move arm out of way, inspect, perceive
        self.insertion_move_arm_1 = Method(
            "insertion_move_arm_1", case=type_item, insert=type_item
        )
        self.insertion_move_arm_1.set_task(
            self.t_insertion,
            self.insertion_move_arm_1.case,
            self.insertion_move_arm_1.insert,
        )
        self.insertion_move_arm_1.add_precondition(self.case_is_placed())
        self.insertion_move_arm_1.add_precondition(self.inserted_insert())
        self.insertion_move_arm_1.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.insertion_move_arm_1.add_precondition(self.holding(self.nothing))
        s1 = self.insertion_move_arm_1.add_subtask(self.move_arm, self.arm_up)
        s2 = self.insertion_move_arm_1.add_subtask(self.move_arm, self.over_fixture)
        s3 = self.insertion_move_arm_1.add_subtask(self.inspect)
        s4 = self.insertion_move_arm_1.add_subtask(self.perceive_set)
        self.insertion_move_arm_1.set_ordered(s1, s2, s3, s4)

        # insert in hand, insert, inspect, perceive
        self.insertion_insert = Method(
            "insertion_insert", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_insert.set_task(
            self.t_insertion, self.insertion_insert.case, self.insertion_insert.insert
        )
        self.insertion_insert.add_precondition(self.case_is_placed())
        self.insertion_insert.add_precondition(Not(self.inserted_insert()))
        self.insertion_insert.add_precondition(self.current_arm_pose(self.over_fixture))
        self.insertion_insert.add_precondition(
            self.holding(self.insertion_insert.insert)
        )
        self.insertion_insert.add_precondition(
            self.current_item_size(self.insertion_insert.size)
        )
        s1 = self.insertion_insert.add_subtask(
            self.insert,
            self.insertion_insert.case,
            self.insertion_insert.insert,
            self.insertion_insert.size,
        )
        s2 = self.insertion_insert.add_subtask(self.move_arm, self.arm_up)
        s3 = self.insertion_insert.add_subtask(self.move_arm, self.over_fixture)
        s4 = self.insertion_insert.add_subtask(self.inspect)
        s5 = self.insertion_insert.add_subtask(self.perceive_set)
        self.insertion_insert.set_ordered(s1, s2, s3, s4, s5)

        # insert in hand, move to fixture, insert, inspect, perceive
        self.insertion_move_arm_2 = Method(
            "insertion_move_arm_2", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_move_arm_2.set_task(
            self.t_insertion,
            self.insertion_move_arm_2.case,
            self.insertion_move_arm_2.insert,
        )
        self.insertion_move_arm_2.add_precondition(self.case_is_placed())
        self.insertion_move_arm_2.add_precondition(Not(self.inserted_insert()))
        self.insertion_move_arm_2.add_precondition(
            self.current_arm_pose(self.over_pallet)
        )
        self.insertion_move_arm_2.add_precondition(
            self.holding(self.insertion_move_arm_2.insert)
        )
        self.insertion_move_arm_2.add_precondition(
            self.current_item_size(self.insertion_move_arm_2.size)
        )
        s1 = self.insertion_move_arm_2.add_subtask(self.move_arm, self.over_fixture)
        s2 = self.insertion_move_arm_2.add_subtask(
            self.insert,
            self.insertion_move_arm_2.case,
            self.insertion_move_arm_2.insert,
            self.insertion_move_arm_2.size,
        )
        s3 = self.insertion_move_arm_2.add_subtask(self.move_arm, self.arm_up)
        s4 = self.insertion_move_arm_2.add_subtask(self.move_arm, self.over_fixture)
        s5 = self.insertion_move_arm_2.add_subtask(self.inspect)
        s6 = self.insertion_move_arm_2.add_subtask(self.perceive_set)
        self.insertion_move_arm_2.set_ordered(s1, s2, s3, s4, s5, s6)

        # insert in hand, move over palle, then to fixture, insert, inspect, perceive
        self.insertion_move_arm_3 = Method(
            "insertion_move_arm_3", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_move_arm_3.set_task(
            self.t_insertion,
            self.insertion_move_arm_3.case,
            self.insertion_move_arm_3.insert,
        )
        self.insertion_move_arm_3.add_precondition(self.case_is_placed())
        self.insertion_move_arm_3.add_precondition(Not(self.inserted_insert()))
        self.insertion_move_arm_3.add_precondition(self.current_arm_pose(self.arm_up))
        self.insertion_move_arm_3.add_precondition(
            self.holding(self.insertion_move_arm_3.insert)
        )
        self.insertion_move_arm_3.add_precondition(
            self.current_item_size(self.insertion_move_arm_3.size)
        )
        s1 = self.insertion_move_arm_3.add_subtask(self.move_arm, self.over_pallet)
        s2 = self.insertion_move_arm_3.add_subtask(self.move_arm, self.over_fixture)
        s3 = self.insertion_move_arm_3.add_subtask(
            self.insert,
            self.insertion_move_arm_3.case,
            self.insertion_move_arm_3.insert,
            self.insertion_move_arm_3.size,
        )
        s4 = self.insertion_move_arm_3.add_subtask(self.move_arm, self.arm_up)
        s5 = self.insertion_move_arm_3.add_subtask(self.move_arm, self.over_fixture)
        s6 = self.insertion_move_arm_3.add_subtask(self.inspect)
        s7 = self.insertion_move_arm_3.add_subtask(self.perceive_set)
        self.insertion_move_arm_3.set_ordered(s1, s2, s3, s4, s5, s6, s7)

        # insert picked, move arm up, move to fixture, insert, inspect, perceive
        self.insertion_move_arm_9 = Method(
            "insertion_move_arm_9", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_move_arm_9.set_task(
            self.t_insertion,
            self.insertion_move_arm_9.case,
            self.insertion_move_arm_9.insert,
        )
        self.insertion_move_arm_9.add_precondition(self.case_is_placed())
        self.insertion_move_arm_9.add_precondition(Not(self.inserted_insert()))
        self.insertion_move_arm_9.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.insertion_move_arm_9.add_precondition(
            self.holding(self.insertion_move_arm_9.insert)
        )
        self.insertion_move_arm_9.add_precondition(
            self.current_item_size(self.insertion_move_arm_9.size)
        )
        s1 = self.insertion_move_arm_9.add_subtask(self.move_arm, self.arm_up)
        s2 = self.insertion_move_arm_9.add_subtask(self.move_arm, self.over_pallet)
        s3 = self.insertion_move_arm_9.add_subtask(self.move_arm, self.over_fixture)
        s4 = self.insertion_move_arm_9.add_subtask(
            self.insert,
            self.insertion_move_arm_9.case,
            self.insertion_move_arm_9.insert,
            self.insertion_move_arm_9.size,
        )
        s5 = self.insertion_move_arm_9.add_subtask(self.move_arm, self.arm_up)
        s6 = self.insertion_move_arm_9.add_subtask(self.move_arm, self.over_fixture)
        s7 = self.insertion_move_arm_9.add_subtask(self.inspect)
        s8 = self.insertion_move_arm_9.add_subtask(self.perceive_set)
        self.insertion_move_arm_9.set_ordered(s1, s2, s3, s4, s5, s6, s7, s8)

        # hand over pallet, pick insert, move to fixture, insert, inspect, perceive
        self.insertion_pick_insert = Method(
            "insertion_pick_insert", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_pick_insert.set_task(
            self.t_insertion,
            self.insertion_pick_insert.case,
            self.insertion_pick_insert.insert,
        )
        self.insertion_pick_insert.add_precondition(self.case_is_placed())
        self.insertion_pick_insert.add_precondition(Not(self.inserted_insert()))
        self.insertion_pick_insert.add_precondition(
            self.current_arm_pose(self.over_pallet)
        )
        self.insertion_pick_insert.add_precondition(self.holding(self.nothing))
        self.insertion_pick_insert.add_precondition(
            self.current_item_size(self.insertion_pick_insert.size)
        )
        self.insertion_pick_insert.add_precondition(
            self.inserts_available(self.insertion_pick_insert.size)
        )
        self.insertion_pick_insert.add_precondition(self.perceived_insert())
        s1 = self.insertion_pick_insert.add_subtask(
            self.pick_insert,
            self.insertion_pick_insert.insert,
            self.insertion_pick_insert.size,
        )
        s2 = self.insertion_pick_insert.add_subtask(self.move_arm, self.arm_up)
        s3 = self.insertion_pick_insert.add_subtask(self.move_arm, self.over_pallet)
        s4 = self.insertion_pick_insert.add_subtask(self.move_arm, self.over_fixture)
        s5 = self.insertion_pick_insert.add_subtask(
            self.insert,
            self.insertion_pick_insert.case,
            self.insertion_pick_insert.insert,
            self.insertion_pick_insert.size,
        )
        s6 = self.insertion_pick_insert.add_subtask(self.move_arm, self.arm_up)
        s7 = self.insertion_pick_insert.add_subtask(self.move_arm, self.over_fixture)
        s8 = self.insertion_pick_insert.add_subtask(self.inspect)
        s9 = self.insertion_pick_insert.add_subtask(self.perceive_set)
        self.insertion_pick_insert.set_ordered(s1, s2, s3, s4, s5, s6, s7, s8, s9)

        # placed_case, move over pallet, perceive insert, pick insert, move to fixture,
        # insert, inspect, perceive set
        self.insertion_perceive_insert = Method(
            "insertion_perceive_insert",
            case=type_item,
            insert=type_item,
            size=type_size,
        )
        self.insertion_perceive_insert.set_task(
            self.t_insertion,
            self.insertion_perceive_insert.case,
            self.insertion_perceive_insert.insert,
        )
        self.insertion_perceive_insert.add_precondition(self.case_is_placed())
        self.insertion_perceive_insert.add_precondition(Not(self.inserted_insert()))
        self.insertion_perceive_insert.add_precondition(
            self.current_arm_pose(self.over_pallet)
        )
        self.insertion_perceive_insert.add_precondition(self.holding(self.nothing))
        self.insertion_perceive_insert.add_precondition(
            self.current_item_size(self.insertion_perceive_insert.size)
        )
        self.insertion_perceive_insert.add_precondition(
            self.inserts_available(self.insertion_perceive_insert.size)
        )
        self.insertion_perceive_insert.add_precondition(Not(self.perceived_insert()))
        s1 = self.insertion_perceive_insert.add_subtask(self.perceive_insert)
        s2 = self.insertion_perceive_insert.add_subtask(
            self.pick_insert,
            self.insertion_perceive_insert.insert,
            self.insertion_perceive_insert.size,
        )
        s3 = self.insertion_perceive_insert.add_subtask(self.move_arm, self.arm_up)
        s4 = self.insertion_perceive_insert.add_subtask(self.move_arm, self.over_pallet)
        s5 = self.insertion_perceive_insert.add_subtask(
            self.move_arm, self.over_fixture
        )
        s6 = self.insertion_perceive_insert.add_subtask(
            self.insert,
            self.insertion_perceive_insert.case,
            self.insertion_perceive_insert.insert,
            self.insertion_perceive_insert.size,
        )
        s7 = self.insertion_perceive_insert.add_subtask(self.move_arm, self.arm_up)
        s8 = self.insertion_perceive_insert.add_subtask(
            self.move_arm, self.over_fixture
        )
        s9 = self.insertion_perceive_insert.add_subtask(self.inspect)
        s10 = self.insertion_perceive_insert.add_subtask(self.perceive_set)
        self.insertion_perceive_insert.set_ordered(
            s1, s2, s3, s4, s5, s6, s7, s8, s9, s10
        )

        # placed_case, move over pallet, perceive insert, pick insert, move to fixture,
        # insert, inspect, perceive set
        self.insertion_move_arm_4 = Method(
            "insertion_move_arm_4", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_move_arm_4.set_task(
            self.t_insertion,
            self.insertion_move_arm_4.case,
            self.insertion_move_arm_4.insert,
        )
        self.insertion_move_arm_4.add_precondition(self.case_is_placed())
        self.insertion_move_arm_4.add_precondition(Not(self.inserted_insert()))
        self.insertion_move_arm_4.add_precondition(
            self.current_arm_pose(self.over_fixture)
        )
        self.insertion_move_arm_4.add_precondition(self.holding(self.nothing))
        self.insertion_move_arm_4.add_precondition(
            self.current_item_size(self.insertion_move_arm_4.size)
        )
        self.insertion_move_arm_4.add_precondition(
            self.inserts_available(self.insertion_move_arm_4.size)
        )
        self.insertion_move_arm_4.add_precondition(Not(self.perceived_insert()))
        s1 = self.insertion_move_arm_4.add_subtask(self.move_arm, self.over_pallet)
        s2 = self.insertion_move_arm_4.add_subtask(self.perceive_insert)
        s3 = self.insertion_move_arm_4.add_subtask(
            self.pick_insert,
            self.insertion_move_arm_4.insert,
            self.insertion_move_arm_4.size,
        )
        s4 = self.insertion_move_arm_4.add_subtask(self.move_arm, self.arm_up)
        s5 = self.insertion_move_arm_4.add_subtask(self.move_arm, self.over_pallet)
        s6 = self.insertion_move_arm_4.add_subtask(self.move_arm, self.over_fixture)
        s7 = self.insertion_move_arm_4.add_subtask(
            self.insert,
            self.insertion_move_arm_4.case,
            self.insertion_move_arm_4.insert,
            self.insertion_move_arm_4.size,
        )
        s8 = self.insertion_move_arm_4.add_subtask(self.move_arm, self.arm_up)
        s9 = self.insertion_move_arm_4.add_subtask(self.move_arm, self.over_fixture)
        s10 = self.insertion_move_arm_4.add_subtask(self.inspect)
        s11 = self.insertion_move_arm_4.add_subtask(self.perceive_set)
        self.insertion_move_arm_4.set_ordered(
            s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11
        )

        # placed_case, arm moved up, move over pallet, pick insert, move to fixture,
        # insert, inspect, perceive
        self.insertion_move_arm_5 = Method(
            "insertion_move_arm_5", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_move_arm_5.set_task(
            self.t_insertion,
            self.insertion_move_arm_5.case,
            self.insertion_move_arm_5.insert,
        )
        self.insertion_move_arm_5.add_precondition(self.case_is_placed())
        self.insertion_move_arm_5.add_precondition(Not(self.inserted_insert()))
        self.insertion_move_arm_5.add_precondition(self.current_arm_pose(self.arm_up))
        self.insertion_move_arm_5.add_precondition(self.holding(self.nothing))
        self.insertion_move_arm_5.add_precondition(
            self.current_item_size(self.insertion_move_arm_5.size)
        )
        self.insertion_move_arm_5.add_precondition(
            self.inserts_available(self.insertion_move_arm_5.size)
        )
        self.insertion_move_arm_5.add_precondition(Not(self.perceived_insert()))
        s1 = self.insertion_move_arm_5.add_subtask(self.move_arm, self.over_fixture)
        s2 = self.insertion_move_arm_5.add_subtask(self.move_arm, self.over_pallet)
        s3 = self.insertion_move_arm_5.add_subtask(self.perceive_insert)
        s4 = self.insertion_move_arm_5.add_subtask(
            self.pick_insert,
            self.insertion_move_arm_5.insert,
            self.insertion_move_arm_5.size,
        )
        s5 = self.insertion_move_arm_5.add_subtask(self.move_arm, self.arm_up)
        s6 = self.insertion_move_arm_5.add_subtask(self.move_arm, self.over_pallet)
        s7 = self.insertion_move_arm_5.add_subtask(self.move_arm, self.over_fixture)
        s8 = self.insertion_move_arm_5.add_subtask(
            self.insert,
            self.insertion_move_arm_5.case,
            self.insertion_move_arm_5.insert,
            self.insertion_move_arm_5.size,
        )
        s9 = self.insertion_move_arm_5.add_subtask(self.move_arm, self.arm_up)
        s10 = self.insertion_move_arm_5.add_subtask(self.move_arm, self.over_fixture)
        s11 = self.insertion_move_arm_5.add_subtask(self.inspect)
        s12 = self.insertion_move_arm_5.add_subtask(self.perceive_set)
        self.insertion_move_arm_5.set_ordered(
            s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12
        )

        # placed case, move arm up
        self.insertion_move_arm_10 = Method(
            "insertion_move_arm_10", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_move_arm_10.set_task(
            self.t_insertion,
            self.insertion_move_arm_10.case,
            self.insertion_move_arm_10.insert,
        )
        self.insertion_move_arm_10.add_precondition(self.case_is_placed())
        self.insertion_move_arm_10.add_precondition(Not(self.inserted_insert()))
        self.insertion_move_arm_10.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.insertion_move_arm_10.add_precondition(self.holding(self.nothing))
        self.insertion_move_arm_10.add_precondition(
            self.current_item_size(self.insertion_move_arm_10.size)
        )
        self.insertion_move_arm_10.add_precondition(
            self.inserts_available(self.insertion_move_arm_10.size)
        )
        self.insertion_move_arm_10.add_precondition(Not(self.perceived_insert()))
        s1 = self.insertion_move_arm_10.add_subtask(self.move_arm, self.arm_up)
        s2 = self.insertion_move_arm_10.add_subtask(self.move_arm, self.over_fixture)
        s3 = self.insertion_move_arm_10.add_subtask(self.move_arm, self.over_pallet)
        s4 = self.insertion_move_arm_10.add_subtask(self.perceive_insert)
        s5 = self.insertion_move_arm_10.add_subtask(
            self.pick_insert,
            self.insertion_move_arm_10.insert,
            self.insertion_move_arm_10.size,
        )
        s6 = self.insertion_move_arm_10.add_subtask(self.move_arm, self.arm_up)
        s7 = self.insertion_move_arm_10.add_subtask(self.move_arm, self.over_pallet)
        s8 = self.insertion_move_arm_10.add_subtask(self.move_arm, self.over_fixture)
        s9 = self.insertion_move_arm_10.add_subtask(
            self.insert,
            self.insertion_move_arm_10.case,
            self.insertion_move_arm_10.insert,
            self.insertion_move_arm_10.size,
        )
        s10 = self.insertion_move_arm_10.add_subtask(self.move_arm, self.arm_up)
        s11 = self.insertion_move_arm_10.add_subtask(self.move_arm, self.over_fixture)
        s12 = self.insertion_move_arm_10.add_subtask(self.inspect)
        s13 = self.insertion_move_arm_10.add_subtask(self.perceive_set)
        self.insertion_move_arm_10.set_ordered(
            s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13
        )

        # case in hand, place it
        self.insertion_place_case = Method(
            "insertion_place_case", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_place_case.set_task(
            self.t_insertion,
            self.insertion_place_case.case,
            self.insertion_place_case.insert,
        )
        self.insertion_place_case.add_precondition(Not(self.case_is_placed()))
        self.insertion_place_case.add_precondition(Not(self.inserted_insert()))
        self.insertion_place_case.add_precondition(
            self.current_arm_pose(self.over_fixture)
        )
        self.insertion_place_case.add_precondition(
            self.holding(self.insertion_place_case.case)
        )
        self.insertion_place_case.add_precondition(
            self.current_item_size(self.insertion_place_case.size)
        )
        self.insertion_place_case.add_precondition(
            self.inserts_available(self.insertion_place_case.size)
        )
        self.insertion_place_case.add_precondition(Not(self.perceived_insert()))
        s1 = self.insertion_place_case.add_subtask(
            self.place_case,
            self.insertion_place_case.case,
            self.insertion_place_case.size,
        )
        s2 = self.insertion_place_case.add_subtask(self.move_arm, self.arm_up)
        s3 = self.insertion_place_case.add_subtask(self.move_arm, self.over_fixture)
        s4 = self.insertion_place_case.add_subtask(self.move_arm, self.over_pallet)
        s5 = self.insertion_place_case.add_subtask(self.perceive_insert)
        s6 = self.insertion_place_case.add_subtask(
            self.pick_insert,
            self.insertion_place_case.insert,
            self.insertion_place_case.size,
        )
        s7 = self.insertion_place_case.add_subtask(self.move_arm, self.arm_up)
        s8 = self.insertion_place_case.add_subtask(self.move_arm, self.over_pallet)
        s9 = self.insertion_place_case.add_subtask(self.move_arm, self.over_fixture)
        s10 = self.insertion_place_case.add_subtask(
            self.insert,
            self.insertion_place_case.case,
            self.insertion_place_case.insert,
            self.insertion_place_case.size,
        )
        s11 = self.insertion_place_case.add_subtask(self.move_arm, self.arm_up)
        s12 = self.insertion_place_case.add_subtask(self.move_arm, self.over_fixture)
        s13 = self.insertion_place_case.add_subtask(self.inspect)
        s14 = self.insertion_place_case.add_subtask(self.perceive_set)
        self.insertion_place_case.set_ordered(
            s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14
        )

        # case in hand, move over fixture, place case
        self.insertion_move_arm_6 = Method(
            "insertion_move_arm_6", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_move_arm_6.set_task(
            self.t_insertion,
            self.insertion_move_arm_6.case,
            self.insertion_move_arm_6.insert,
        )
        self.insertion_move_arm_6.add_precondition(Not(self.case_is_placed()))
        self.insertion_move_arm_6.add_precondition(Not(self.inserted_insert()))
        self.insertion_move_arm_6.add_precondition(
            self.current_arm_pose(self.over_conveyor)
        )
        self.insertion_move_arm_6.add_precondition(
            self.holding(self.insertion_move_arm_6.case)
        )
        self.insertion_move_arm_6.add_precondition(
            self.current_item_size(self.insertion_move_arm_6.size)
        )
        self.insertion_move_arm_6.add_precondition(
            self.inserts_available(self.insertion_move_arm_6.size)
        )
        self.insertion_move_arm_6.add_precondition(Not(self.perceived_insert()))
        s1 = self.insertion_move_arm_6.add_subtask(self.move_arm, self.over_fixture)
        s2 = self.insertion_move_arm_6.add_subtask(
            self.place_case,
            self.insertion_move_arm_6.case,
            self.insertion_move_arm_6.size,
        )
        s3 = self.insertion_move_arm_6.add_subtask(self.move_arm, self.arm_up)
        s4 = self.insertion_move_arm_6.add_subtask(self.move_arm, self.over_fixture)
        s5 = self.insertion_move_arm_6.add_subtask(self.move_arm, self.over_pallet)
        s6 = self.insertion_move_arm_6.add_subtask(self.perceive_insert)
        s7 = self.insertion_move_arm_6.add_subtask(
            self.pick_insert,
            self.insertion_move_arm_6.insert,
            self.insertion_move_arm_6.size,
        )
        s8 = self.insertion_move_arm_6.add_subtask(self.move_arm, self.arm_up)
        s9 = self.insertion_move_arm_6.add_subtask(self.move_arm, self.over_pallet)
        s10 = self.insertion_move_arm_6.add_subtask(self.move_arm, self.over_fixture)
        s11 = self.insertion_move_arm_6.add_subtask(
            self.insert,
            self.insertion_move_arm_6.case,
            self.insertion_move_arm_6.insert,
            self.insertion_move_arm_6.size,
        )
        s12 = self.insertion_move_arm_6.add_subtask(self.move_arm, self.arm_up)
        s13 = self.insertion_move_arm_6.add_subtask(self.move_arm, self.over_fixture)
        s14 = self.insertion_move_arm_6.add_subtask(self.inspect)
        s15 = self.insertion_move_arm_6.add_subtask(self.perceive_set)
        self.insertion_move_arm_6.set_ordered(
            s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15
        )

        # case in hand, move over conveyor, then over fixture, place case
        self.insertion_move_arm_7 = Method(
            "insertion_move_arm_7", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_move_arm_7.set_task(
            self.t_insertion,
            self.insertion_move_arm_7.case,
            self.insertion_move_arm_7.insert,
        )
        self.insertion_move_arm_7.add_precondition(Not(self.case_is_placed()))
        self.insertion_move_arm_7.add_precondition(Not(self.inserted_insert()))
        self.insertion_move_arm_7.add_precondition(self.current_arm_pose(self.arm_up))
        self.insertion_move_arm_7.add_precondition(
            self.holding(self.insertion_move_arm_7.case)
        )
        self.insertion_move_arm_7.add_precondition(
            self.current_item_size(self.insertion_move_arm_7.size)
        )
        self.insertion_move_arm_7.add_precondition(
            self.inserts_available(self.insertion_move_arm_7.size)
        )
        self.insertion_move_arm_7.add_precondition(Not(self.perceived_insert()))
        s1 = self.insertion_move_arm_7.add_subtask(self.move_arm, self.over_conveyor)
        s2 = self.insertion_move_arm_7.add_subtask(self.move_arm, self.over_fixture)
        s3 = self.insertion_move_arm_7.add_subtask(
            self.place_case,
            self.insertion_move_arm_7.case,
            self.insertion_move_arm_7.size,
        )
        s4 = self.insertion_move_arm_7.add_subtask(self.move_arm, self.arm_up)
        s5 = self.insertion_move_arm_7.add_subtask(self.move_arm, self.over_fixture)
        s6 = self.insertion_move_arm_7.add_subtask(self.move_arm, self.over_pallet)
        s7 = self.insertion_move_arm_7.add_subtask(self.perceive_insert)
        s8 = self.insertion_move_arm_7.add_subtask(
            self.pick_insert,
            self.insertion_move_arm_7.insert,
            self.insertion_move_arm_7.size,
        )
        s9 = self.insertion_move_arm_7.add_subtask(self.move_arm, self.arm_up)
        s10 = self.insertion_move_arm_7.add_subtask(self.move_arm, self.over_pallet)
        s11 = self.insertion_move_arm_7.add_subtask(self.move_arm, self.over_fixture)
        s12 = self.insertion_move_arm_7.add_subtask(
            self.insert,
            self.insertion_move_arm_7.case,
            self.insertion_move_arm_7.insert,
            self.insertion_move_arm_7.size,
        )
        s13 = self.insertion_move_arm_7.add_subtask(self.move_arm, self.arm_up)
        s14 = self.insertion_move_arm_7.add_subtask(self.move_arm, self.over_fixture)
        s15 = self.insertion_move_arm_7.add_subtask(self.inspect)
        s16 = self.insertion_move_arm_7.add_subtask(self.perceive_set)
        self.insertion_move_arm_7.set_ordered(
            s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16
        )

        # case location and type known, pick, insert, inspect, perceive
        self.insertion_move_arm_11 = Method(
            "insertion_move_arm_11", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_move_arm_11.set_task(
            self.t_insertion,
            self.insertion_move_arm_11.case,
            self.insertion_move_arm_11.insert,
        )

        self.insertion_move_arm_11.add_precondition(Not(self.case_is_placed()))
        self.insertion_move_arm_11.add_precondition(Not(self.inserted_insert()))
        self.insertion_move_arm_11.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.insertion_move_arm_11.add_precondition(
            self.holding(self.insertion_move_arm_11.case)
        )
        self.insertion_move_arm_11.add_precondition(
            self.current_item_size(self.insertion_move_arm_11.size)
        )
        self.insertion_move_arm_11.add_precondition(
            self.inserts_available(self.insertion_move_arm_11.size)
        )
        self.insertion_move_arm_11.add_precondition(Not(self.perceived_insert()))
        s1 = self.insertion_move_arm_11.add_subtask(self.move_arm, self.arm_up)
        s2 = self.insertion_move_arm_11.add_subtask(self.move_arm, self.over_conveyor)
        s3 = self.insertion_move_arm_11.add_subtask(self.move_arm, self.over_fixture)
        s4 = self.insertion_move_arm_11.add_subtask(
            self.place_case,
            self.insertion_move_arm_11.case,
            self.insertion_move_arm_11.size,
        )
        s5 = self.insertion_move_arm_11.add_subtask(self.move_arm, self.arm_up)
        s6 = self.insertion_move_arm_11.add_subtask(self.move_arm, self.over_fixture)
        s7 = self.insertion_move_arm_11.add_subtask(self.move_arm, self.over_pallet)
        s8 = self.insertion_move_arm_11.add_subtask(self.perceive_insert)
        s9 = self.insertion_move_arm_11.add_subtask(
            self.pick_insert,
            self.insertion_move_arm_11.insert,
            self.insertion_move_arm_11.size,
        )
        s10 = self.insertion_move_arm_11.add_subtask(self.move_arm, self.arm_up)
        s11 = self.insertion_move_arm_11.add_subtask(self.move_arm, self.over_pallet)
        s12 = self.insertion_move_arm_11.add_subtask(self.move_arm, self.over_fixture)
        s13 = self.insertion_move_arm_11.add_subtask(
            self.insert,
            self.insertion_move_arm_11.case,
            self.insertion_move_arm_11.insert,
            self.insertion_move_arm_11.size,
        )
        s14 = self.insertion_move_arm_11.add_subtask(self.move_arm, self.arm_up)
        s15 = self.insertion_move_arm_11.add_subtask(self.move_arm, self.over_fixture)
        s16 = self.insertion_move_arm_11.add_subtask(self.inspect)
        s17 = self.insertion_move_arm_11.add_subtask(self.perceive_set)
        self.insertion_move_arm_11.set_ordered(
            s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17
        )

        # case location and type known, pick, insert, inspect, perceive
        self.insertion_full = Method(
            "insertion_full", case=type_item, insert=type_item, size=type_size
        )
        self.insertion_full.set_task(
            self.t_insertion,
            self.insertion_full.case,
            self.insertion_full.insert,
        )
        self.insertion_full.add_precondition(Not(self.case_is_placed()))
        self.insertion_full.add_precondition(Not(self.inserted_insert()))
        self.insertion_full.add_precondition(self.current_arm_pose(self.over_conveyor))
        self.insertion_full.add_precondition(self.holding(self.nothing))
        self.insertion_full.add_precondition(
            self.current_item_size(self.insertion_full.size)
        )
        self.insertion_full.add_precondition(
            self.inserts_available(self.insertion_full.size)
        )
        self.insertion_full.add_precondition(Not(self.perceived_insert()))
        s1 = self.insertion_full.add_subtask(
            self.pick_case, self.insertion_full.case, self.insertion_full.size
        )
        s2 = self.insertion_full.add_subtask(self.move_arm, self.arm_up)
        s3 = self.insertion_full.add_subtask(self.move_arm, self.over_conveyor)
        s4 = self.insertion_full.add_subtask(self.move_arm, self.over_fixture)
        s5 = self.insertion_full.add_subtask(
            self.place_case, self.insertion_full.case, self.insertion_full.size
        )
        s6 = self.insertion_full.add_subtask(self.move_arm, self.arm_up)
        s7 = self.insertion_full.add_subtask(self.move_arm, self.over_fixture)
        s8 = self.insertion_full.add_subtask(self.move_arm, self.over_pallet)
        s9 = self.insertion_full.add_subtask(self.perceive_insert)
        s10 = self.insertion_full.add_subtask(
            self.pick_insert, self.insertion_full.insert, self.insertion_full.size
        )
        s11 = self.insertion_full.add_subtask(self.move_arm, self.arm_up)
        s12 = self.insertion_full.add_subtask(self.move_arm, self.over_pallet)
        s13 = self.insertion_full.add_subtask(self.move_arm, self.over_fixture)
        s14 = self.insertion_full.add_subtask(
            self.insert,
            self.insertion_full.case,
            self.insertion_full.insert,
            self.insertion_full.size,
        )
        s15 = self.insertion_full.add_subtask(self.move_arm, self.arm_up)
        s16 = self.insertion_full.add_subtask(self.move_arm, self.over_fixture)
        s17 = self.insertion_full.add_subtask(self.inspect)
        s18 = self.insertion_full.add_subtask(self.perceive_set)
        self.insertion_full.set_ordered(
            s1,
            s2,
            s3,
            s4,
            s5,
            s6,
            s7,
            s8,
            s9,
            s10,
            s11,
            s12,
            s13,
            s14,
            s15,
            s16,
            s17,
            s18,
        )

        # PLACE SET
        # set already placed, moved arm up, move over boxes
        self.place_set_move_arm_1 = Method(
            "place_set_move_arm_1", set=type_item, status=type_status
        )
        self.place_set_move_arm_1.set_task(
            self.t_place_set, self.place_set_move_arm_1.set
        )
        self.place_set_move_arm_1.add_precondition(self.set_status_known())
        self.place_set_move_arm_1.add_precondition(self.holding(self.nothing))
        self.place_set_move_arm_1.add_precondition(self.current_arm_pose(self.arm_up))
        self.place_set_move_arm_1.add_subtask(self.move_arm_end)

        # set already placed, move arm up and over boxes
        self.place_set_move_arm_4 = Method(
            "place_set_move_arm_4", set=type_item, status=type_status
        )
        self.place_set_move_arm_4.set_task(
            self.t_place_set, self.place_set_move_arm_4.set
        )
        self.place_set_move_arm_4.add_precondition(self.set_status_known())
        self.place_set_move_arm_4.add_precondition(self.holding(self.nothing))
        self.place_set_move_arm_4.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        s1 = self.place_set_move_arm_4.add_subtask(self.move_arm, self.arm_up)
        s2 = self.place_set_move_arm_4.add_subtask(self.move_arm_end)
        self.place_set_move_arm_4.set_ordered(s1, s2)

        # set in hand, place in box
        self.place_set_box = Method("place_set_box", set=type_item, status=type_status)
        self.place_set_box.set_task(self.t_place_set, self.place_set_box.set)
        self.place_set_box.add_precondition(self.set_status_known())
        self.place_set_box.add_precondition(self.holding(self.place_set_box.set))
        self.place_set_box.add_precondition(self.current_arm_pose(self.over_boxes))
        self.place_set_box.add_precondition(
            self.space_in_box(self.place_set_box.status)
        )
        s1 = self.place_set_box.add_subtask(
            self.place_set_in_box, self.place_set_box.set, self.place_set_box.status
        )
        s2 = self.place_set_box.add_subtask(self.move_arm, self.arm_up)
        s3 = self.place_set_box.add_subtask(self.move_arm_end)
        self.place_set_box.set_ordered(s1, s2, s3)

        # set in hand, move arm over boxes and place
        self.place_set_move_arm_2 = Method(
            "place_set_move_arm_2", set=type_item, status=type_status
        )
        self.place_set_move_arm_2.set_task(
            self.t_place_set, self.place_set_move_arm_2.set
        )
        self.place_set_move_arm_2.add_precondition(self.set_status_known())
        self.place_set_move_arm_2.add_precondition(
            self.space_in_box(self.place_set_move_arm_2.status)
        )
        self.place_set_move_arm_2.add_precondition(
            self.holding(self.place_set_move_arm_2.set)
        )
        self.place_set_move_arm_2.add_precondition(
            self.current_arm_pose(self.over_fixture)
        )
        s1 = self.place_set_move_arm_2.add_subtask(self.move_arm, self.over_boxes)
        s2 = self.place_set_move_arm_2.add_subtask(
            self.place_set_in_box,
            self.place_set_move_arm_2.set,
            self.place_set_move_arm_2.status,
        )
        s3 = self.place_set_move_arm_2.add_subtask(self.move_arm, self.arm_up)
        s4 = self.place_set_move_arm_2.add_subtask(self.move_arm_end)
        self.place_set_move_arm_2.set_ordered(s1, s2, s3, s4)

        # set in hand, move arm over fixture, then over boxes and place
        self.place_set_move_arm_3 = Method(
            "place_set_move_arm_3", set=type_item, status=type_status
        )
        self.place_set_move_arm_3.set_task(
            self.t_place_set, self.place_set_move_arm_3.set
        )
        self.place_set_move_arm_3.add_precondition(self.set_status_known())
        self.place_set_move_arm_3.add_precondition(
            self.holding(self.place_set_move_arm_3.set)
        )
        self.place_set_move_arm_3.add_precondition(self.current_arm_pose(self.arm_up))
        self.place_set_move_arm_3.add_precondition(
            self.space_in_box(self.place_set_move_arm_3.status)
        )
        s1 = self.place_set_move_arm_3.add_subtask(self.move_arm, self.over_fixture)
        s2 = self.place_set_move_arm_3.add_subtask(self.move_arm, self.over_boxes)
        s3 = self.place_set_move_arm_3.add_subtask(
            self.place_set_in_box,
            self.place_set_move_arm_3.set,
            self.place_set_move_arm_3.status,
        )
        s4 = self.place_set_move_arm_3.add_subtask(self.move_arm, self.arm_up)
        s5 = self.place_set_move_arm_3.add_subtask(self.move_arm_end)
        self.place_set_move_arm_3.set_ordered(s1, s2, s3, s4, s5)

        # set in hand, move arm up, then over fixture, then over boxes and place
        self.place_set_move_arm_5 = Method(
            "place_set_move_arm_5", set=type_item, status=type_status
        )
        self.place_set_move_arm_5.set_task(
            self.t_place_set, self.place_set_move_arm_5.set
        )
        self.place_set_move_arm_5.add_precondition(self.set_status_known())
        self.place_set_move_arm_5.add_precondition(
            self.holding(self.place_set_move_arm_3.set)
        )
        self.place_set_move_arm_5.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.place_set_move_arm_5.add_precondition(
            self.space_in_box(self.place_set_move_arm_5.status)
        )
        s1 = self.place_set_move_arm_5.add_subtask(self.move_arm, self.arm_up)
        s2 = self.place_set_move_arm_5.add_subtask(self.move_arm, self.over_fixture)
        s3 = self.place_set_move_arm_5.add_subtask(self.move_arm, self.over_boxes)
        s4 = self.place_set_move_arm_5.add_subtask(
            self.place_set_in_box,
            self.place_set_move_arm_5.set,
            self.place_set_move_arm_5.status,
        )
        s5 = self.place_set_move_arm_5.add_subtask(self.move_arm, self.arm_up)
        s6 = self.place_set_move_arm_5.add_subtask(self.move_arm_end)
        self.place_set_move_arm_5.set_ordered(s1, s2, s3, s4, s5, s6)

        # pick and place set
        self.place_set_full = Method(
            "place_set_full", set=type_item, status=type_status
        )
        self.place_set_full.set_task(self.t_place_set, self.place_set_full.set)
        self.place_set_full.add_precondition(self.set_status_known())
        self.place_set_full.add_precondition(self.holding(self.nothing))
        self.place_set_full.add_precondition(self.current_arm_pose(self.over_fixture))
        self.place_set_full.add_precondition(
            self.space_in_box(self.place_set_full.status)
        )
        self.place_set_full.add_precondition(self.perceived_set())
        s1 = self.place_set_full.add_subtask(self.pick_set, self.place_set_full.set)
        s2 = self.place_set_full.add_subtask(self.move_arm, self.arm_up)
        s3 = self.place_set_full.add_subtask(self.move_arm, self.over_fixture)
        s4 = self.place_set_full.add_subtask(self.move_arm, self.over_boxes)
        s5 = self.place_set_full.add_subtask(
            self.place_set_in_box, self.place_set_full.set, self.place_set_full.status
        )
        s6 = self.place_set_full.add_subtask(self.move_arm, self.arm_up)
        s7 = self.place_set_full.add_subtask(self.move_arm_end)
        self.place_set_full.set_ordered(s1, s2, s3, s4, s5, s6, s7)

        # box is full
        self.place_set_empty_box = Method(
            "place_set_empty_box", set=type_item, status=type_status
        )
        self.place_set_empty_box.set_task(
            self.t_place_set, self.place_set_empty_box.set
        )
        self.place_set_empty_box.add_precondition(self.set_status_known())
        self.place_set_empty_box.add_precondition(
            Not(self.space_in_box(self.place_set_empty_box.status))
        )
        self.place_set_empty_box.add_precondition(self.holding(self.nothing))
        self.place_set_empty_box.add_subtask(
            self.empty_box, self.place_set_empty_box.status
        )

        # ASSEMBLE SET
        # Get case, perceive to get size
        self.assemble_set_case = Method(
            "assemble_set_case", case=type_item, insert=type_item, set=type_item
        )
        self.assemble_set_case.set_task(
            self.t_assemble_set,
            self.assemble_set_case.case,
            self.assemble_set_case.insert,
            self.assemble_set_case.set,
        )
        self.assemble_set_case.add_precondition(Not(self.item_size_known()))
        self.assemble_set_case.add_subtask(self.t_get_case, self.assemble_set_case.case)

        # pick case, place case, pick insert, insert into case, inspect
        self.assemble_set_insert = Method(
            "assemble_set_insert",
            case=type_item,
            insert=type_item,
            set=type_item,
            size=type_size,
        )
        self.assemble_set_insert.set_task(
            self.t_assemble_set,
            self.assemble_set_insert.case,
            self.assemble_set_insert.insert,
            self.assemble_set_insert.set,
        )
        self.assemble_set_insert.add_precondition(self.item_size_known())
        self.assemble_set_insert.add_precondition(
            self.inserts_available(self.assemble_set_insert.size)
        )
        self.assemble_set_insert.add_precondition(Not(self.perceived_set()))
        self.assemble_set_insert.add_subtask(
            self.t_insertion,
            self.assemble_set_insert.case,
            self.assemble_set_insert.insert,
        )

        # place set in box
        self.assemble_set_place = Method(
            "assemble_set_place",
            case=type_item,
            insert=type_item,
            set=type_item,
        )
        self.assemble_set_place.set_task(
            self.t_assemble_set,
            self.assemble_set_place.case,
            self.assemble_set_place.insert,
            self.assemble_set_place.set,
        )
        self.assemble_set_place.add_precondition(self.item_size_known())
        self.assemble_set_place.add_precondition(self.set_status_known())
        self.assemble_set_place.add_subtask(
            self.t_place_set, self.assemble_set_place.set
        )

        # inserts empty, restock
        self.assemble_set_restock = Method(
            "assemble_set_restock",
            case=type_item,
            insert=type_item,
            set=type_item,
            size=type_size,
        )
        self.assemble_set_restock.set_task(
            self.t_assemble_set,
            self.assemble_set_restock.case,
            self.assemble_set_restock.insert,
            self.assemble_set_restock.set,
        )
        self.assemble_set_restock.add_precondition(
            self.current_item_size(self.assemble_set_restock.size)
        )
        self.assemble_set_restock.add_precondition(
            Not(self.inserts_available(self.assemble_set_restock.size))
        )
        self.assemble_set_restock.add_subtask(
            self.restock_inserts,
            self.assemble_set_restock.size,
        )

        self.methods = (
            self.get_case_noop,
            self.get_case_perceive,
            self.get_case_wait,
            self.get_case_full,
            self.insertion_noop,
            self.insertion_perceive_set,
            self.insertion_inspect,
            self.insertion_insert,
            self.insertion_pick_insert,
            self.insertion_place_case,
            self.insertion_perceive_insert,
            self.insertion_move_arm_1,
            self.insertion_move_arm_2,
            self.insertion_move_arm_3,
            self.insertion_move_arm_4,
            self.insertion_move_arm_5,
            self.insertion_move_arm_6,
            self.insertion_move_arm_7,
            self.insertion_move_arm_8,
            self.insertion_move_arm_9,
            self.insertion_move_arm_10,
            self.insertion_move_arm_11,
            self.insertion_full,
            self.place_set_move_arm_1,
            self.place_set_move_arm_2,
            self.place_set_move_arm_3,
            self.place_set_move_arm_4,
            self.place_set_move_arm_5,
            self.place_set_box,
            self.place_set_full,
            self.place_set_empty_box,
            self.assemble_set_case,
            self.assemble_set_insert,
            self.assemble_set_place,
            self.assemble_set_restock,
        )

    def _create_domain_actions(self, temporal: bool = False) -> None:
        actions = Actions(self._env)

        if temporal:
            # TODO TEMPORAL
            pass
        else:
            self.get_next_case, _ = self.create_action(
                "get_next_case",
                _callable=actions.get_next_case,
            )
            self.get_next_case.add_precondition(Not(self.item_in_fov()))
            self.get_next_case.add_effect(self.item_in_fov(), True)

            self.perceive_case, _ = self.create_action(
                "perceive_case",
                _callable=actions.perceive_case,
            )
            self.perceive_case.add_precondition(Not(self.item_size_known()))
            self.perceive_case.add_precondition(self.item_in_fov())
            self.perceive_case.add_effect(self.item_size_known(), True)

            self.perceive_insert, _ = self.create_action(
                "perceive_insert", _callable=actions.perceive_insert
            )
            self.perceive_insert.add_precondition(Not(self.perceived_insert()))
            self.perceive_insert.add_precondition(
                self.current_arm_pose(self.over_pallet)
            )
            self.perceive_insert.add_effect(self.perceived_insert(), True)

            self.perceive_set, _ = self.create_action(
                "perceive_set", _callable=actions.perceive_set
            )
            self.perceive_set.add_precondition(Not(self.perceived_set()))
            self.perceive_set.add_precondition(self.current_arm_pose(self.over_fixture))
            self.perceive_set.add_effect(self.perceived_set(), True)

            self.move_arm, [a] = self.create_action(
                "move_arm",
                arm_pose=ArmPose,
                _callable=actions.move_arm,
            )
            self.move_arm.add_effect(self.current_arm_pose(a), True)

            self.move_arm_end, _ = self.create_action(
                "move_arm_end", _callable=actions.move_arm_end
            )
            self.move_arm_end.add_effect(self.current_arm_pose(self.over_boxes), True)
            self.move_arm_end.add_effect(self.set_status_known(), False)

            self.pick_case, [c, s] = self.create_action(
                "pick_case", case=Item, size=Size, _callable=actions.pick_case
            )
            self.pick_case.add_precondition(self.holding(self.nothing))
            self.pick_case.add_precondition(self.item_in_fov())
            self.pick_case.add_precondition(self.item_size_known())
            self.pick_case.add_precondition(self.current_item_size(s))
            self.pick_case.add_precondition(self.current_arm_pose(self.over_conveyor))
            self.pick_case.add_effect(self.holding(c), True)
            self.pick_case.add_effect(self.holding(self.nothing), False)
            self.pick_case.add_effect(self.current_arm_pose(self.unknown_pose), True)
            self.pick_case.add_effect(self.item_in_fov(), False)
            self.pick_case.add_effect(self.current_arm_pose(self.over_conveyor), False)

            self.pick_insert, [i, s] = self.create_action(
                "pick_insert", insert=Item, size=Size, _callable=actions.pick_insert
            )
            self.pick_insert.add_precondition(self.holding(self.nothing))
            self.pick_insert.add_precondition(self.item_size_known())
            self.pick_insert.add_precondition(self.current_item_size(s))
            self.pick_insert.add_precondition(self.current_arm_pose(self.over_pallet))
            self.pick_insert.add_precondition(self.inserts_available(s))
            self.pick_insert.add_precondition(self.perceived_insert())
            self.pick_insert.add_effect(self.holding(i), True)
            self.pick_insert.add_effect(self.holding(self.nothing), False)
            self.pick_insert.add_effect(self.current_arm_pose(self.unknown_pose), True)
            self.pick_insert.add_effect(self.current_arm_pose(self.over_pallet), False)

            self.pick_set, [s] = self.create_action(
                "pick_set", set=Item, _callable=actions.pick_set
            )
            self.pick_set.add_precondition(self.holding(self.nothing))
            self.pick_set.add_precondition(self.set_status_known())
            self.pick_set.add_precondition(self.perceived_set())
            self.pick_set.add_precondition(self.current_arm_pose(self.over_fixture))
            self.pick_set.add_effect(self.holding(s), True)
            self.pick_set.add_effect(self.holding(self.nothing), False)
            self.pick_set.add_effect(self.current_arm_pose(self.unknown_pose), True)
            self.pick_set.add_effect(self.current_arm_pose(self.over_fixture), False)
            self.pick_set.add_effect(self.perceived_set(), False)

            self.place_case, [c, s] = self.create_action(
                "place_case", case=Item, size=Size, _callable=actions.place_case
            )
            self.place_case.add_precondition(self.item_size_known())
            self.place_case.add_precondition(self.current_item_size(s))
            self.place_case.add_precondition(self.holding(c))
            self.place_case.add_precondition(self.current_arm_pose(self.over_fixture))
            self.place_case.add_precondition(Not(self.case_is_placed()))
            self.place_case.add_effect(self.holding(self.nothing), True)
            self.place_case.add_effect(self.current_arm_pose(self.unknown_pose), True)
            self.place_case.add_effect(self.case_is_placed(), True)
            self.place_case.add_effect(self.current_arm_pose(self.over_fixture), False)
            self.place_case.add_effect(self.holding(c), False)

            self.place_set_in_box, [s, st] = self.create_action(
                "place_set_in_box",
                set=Item,
                status=Status,
                _callable=actions.place_set_in_box,
            )
            self.place_set_in_box.add_precondition(self.set_status_known())
            self.place_set_in_box.add_precondition(self.space_in_box(st))
            self.place_set_in_box.add_precondition(self.status_of_set(st))
            self.place_set_in_box.add_precondition(
                self.current_arm_pose(self.over_boxes)
            )
            self.place_set_in_box.add_precondition(self.holding(s))
            self.place_set_in_box.add_effect(self.holding(self.nothing), True)
            self.place_set_in_box.add_effect(
                self.current_arm_pose(self.unknown_pose), True
            )
            self.place_set_in_box.add_effect(
                self.current_arm_pose(self.over_boxes), False
            )
            self.place_set_in_box.add_effect(self.holding(s), False)

            self.empty_box, [st] = self.create_action(
                "empty_box",
                status=Status,
                _callable=actions.empty_box,
            )
            self.empty_box.add_precondition(Not(self.space_in_box(st)))
            self.empty_box.add_precondition(self.set_status_known())
            self.empty_box.add_effect(self.space_in_box(st), True)

            self.insert, [c, i, s] = self.create_action(
                "insert",
                case=Item,
                insert=Item,
                size=Size,
                _callable=actions.insert,
            )
            self.insert.add_precondition(self.holding(i))
            self.insert.add_precondition(self.item_size_known())
            self.insert.add_precondition(self.current_item_size(s))
            self.insert.add_precondition(self.current_arm_pose(self.over_fixture))
            self.insert.add_effect(self.holding(i), False)
            self.insert.add_effect(self.holding(self.nothing), True)
            self.insert.add_effect(self.current_arm_pose(self.unknown_pose), True)
            self.insert.add_effect(self.current_arm_pose(self.over_fixture), False)
            self.insert.add_effect(self.inserted_insert(), True)

            self.inspect, _ = self.create_action("inspect", _callable=actions.inspect)
            self.inspect.add_precondition(Not(self.set_status_known()))
            self.inspect.add_precondition(self.current_arm_pose(self.over_fixture))
            self.inspect.add_effect(self.set_status_known(), True)

            self.restock_inserts, [s] = self.create_action(
                "restock_inserts", size=Size, _callable=actions.restock_inserts
            )
            self.restock_inserts.add_precondition(self.item_size_known())
            self.restock_inserts.add_precondition(self.current_item_size(s))
            self.restock_inserts.add_precondition(Not(self.inserts_available(s)))
            self.restock_inserts.add_effect(self.inserts_available(s), True)

    def set_state_and_goal(self, problem, goal=None) -> None:
        success = True
        if goal is None:
            problem.task_network.add_subtask(
                self.t_assemble_set(self.case, self.insert_o, self.set)
            )
        else:
            logerr((f"Task ({goal}) is unknown! Please leave the goal empty."))
            success = False
        return success
