from rospy import logerr
from unified_planning.model.htn import Task, Method

from unified_planning.shortcuts import Not
from april_krem.domains.PEMU_components import (
    Item,
    Location,
    ArmPose,
    Size,
    Status,
    Actions,
    Environment,
)
from up_esb.bridge import Bridge


class PEMUDomain(Bridge):
    def __init__(self, krem_logging, temporal: bool = False) -> None:
        Bridge.__init__(self)

        self._env = Environment(krem_logging)

        # Create types for planning based on class types
        self.create_types([Item, Location, ArmPose, Size, Status])
        type_item = self.get_type(Item)
        type_size = self.get_type(Size)
        type_status = self.get_type(Status)

        # Create fluents for planning
        self.holding = self.create_fluent_from_function(self._env.holding)
        self.current_arm_pose = self.create_fluent_from_function(
            self._env.current_arm_pose
        )
        self.perceived_pillow = self.create_fluent_from_function(
            self._env.perceived_pillow
        )
        self.item_size_known = self.create_fluent_from_function(
            self._env.item_size_known
        )
        self.current_item_size = self.create_fluent_from_function(
            self._env.current_item_size
        )
        self.pillow_is_on = self.create_fluent_from_function(self._env.pillow_is_on)
        self.pillow_weight_known = self.create_fluent_from_function(
            self._env.pillow_weight_known
        )
        self.pillow_status_known = self.create_fluent_from_function(
            self._env.pillow_status_known
        )
        self.status_of_pillow = self.create_fluent_from_function(
            self._env.status_of_pillow
        )
        self.space_in_box = self.create_fluent_from_function(self._env.space_in_box)

        # Create objects for both planning and execution
        self.create_enum_objects(Item)
        self.nothing = self.objects[Item.nothing.name]
        self.pillow = self.objects[Item.pillow.name]

        self.create_enum_objects(Location)
        self.table = self.objects[Location.table.name]
        self.scale = self.objects[Location.scale.name]
        self.box = self.objects[Location.box.name]

        self.create_enum_objects(ArmPose)
        self.unknown_pose = self.objects[ArmPose.unknown.name]
        self.home = self.objects[ArmPose.home.name]
        self.arm_up = self.objects[ArmPose.arm_up.name]
        self.over_table = self.objects[ArmPose.over_table.name]
        self.over_scale = self.objects[ArmPose.over_scale.name]
        self.over_boxes = self.objects[ArmPose.over_boxes.name]

        self.create_enum_objects(Size)
        self.small = self.objects[Size.small.name]
        self.big = self.objects[Size.big.name]

        self.create_enum_objects(Status)
        self.ok = self.objects[Status.ok.name]
        self.nok = self.objects[Status.nok.name]

        # Create actions for planning
        self._create_domain_actions(temporal)

        # Tasks
        self.t_perceive_pillow = Task("t_perceive_pillow", pillow=type_item)
        self.t_inspect_pillow = Task("t_inspect_pillow", pillow=type_item)
        self.t_place_pillow_in_box = Task("t_place_pillow_in_box", pillow=type_item)
        self.t_rate_pillow = Task("t_rate_pillow", pillow=type_item)

        self.tasks = (
            self.t_perceive_pillow,
            self.t_inspect_pillow,
            self.t_place_pillow_in_box,
            self.t_rate_pillow,
        )

        # Methods
        # PERCEIVE PILLOW
        # arm over table, perceive pillow
        self.perceive_pillow_perceive = Method(
            "perceive_pillow_perceive", pillow=type_item
        )
        self.perceive_pillow_perceive.set_task(
            self.t_perceive_pillow, self.perceive_pillow_perceive.pillow
        )
        self.perceive_pillow_perceive.add_precondition(
            self.current_arm_pose(self.over_table)
        )
        self.perceive_pillow_perceive.add_precondition(self.holding(self.nothing))
        self.perceive_pillow_perceive.add_precondition(self.pillow_is_on(self.table))
        self.perceive_pillow_perceive.add_precondition(
            Not(self.perceived_pillow(self.table))
        )
        self.perceive_pillow_perceive.add_subtask(self.perceive_pillow, self.table)

        # move arm and perceive pillow
        self.perceive_pillow_move_arm = Method(
            "perceive_pillow_move_arm", pillow=type_item
        )
        self.perceive_pillow_move_arm.set_task(
            self.t_perceive_pillow, self.perceive_pillow_move_arm.pillow
        )
        self.perceive_pillow_move_arm.add_precondition(self.current_arm_pose(self.home))
        self.perceive_pillow_move_arm.add_precondition(self.holding(self.nothing))
        self.perceive_pillow_move_arm.add_precondition(self.pillow_is_on(self.table))
        self.perceive_pillow_move_arm.add_precondition(
            Not(self.perceived_pillow(self.table))
        )
        s1 = self.perceive_pillow_move_arm.add_subtask(self.move_arm, self.over_table)
        s2 = self.perceive_pillow_move_arm.add_subtask(self.perceive_pillow, self.table)
        self.perceive_pillow_move_arm.set_ordered(s1, s2)

        # get next pillow
        self.perceive_pillow_get = Method("perceive_pillow_get", pillow=type_item)
        self.perceive_pillow_get.set_task(
            self.t_perceive_pillow, self.perceive_pillow_get.pillow
        )
        self.perceive_pillow_get.add_precondition(self.current_arm_pose(self.home))
        self.perceive_pillow_get.add_precondition(self.holding(self.nothing))
        self.perceive_pillow_get.add_precondition(Not(self.pillow_is_on(self.table)))
        self.perceive_pillow_get.add_precondition(
            Not(self.perceived_pillow(self.table))
        )
        s1 = self.perceive_pillow_get.add_subtask(self.get_next_pillow)
        s2 = self.perceive_pillow_get.add_subtask(self.move_arm, self.over_table)
        s3 = self.perceive_pillow_get.add_subtask(self.perceive_pillow, self.table)
        self.perceive_pillow_get.set_ordered(s1, s2, s3)

        # move arm and perceive pillow
        self.perceive_pillow_full = Method("perceive_pillow_full", pillow=type_item)
        self.perceive_pillow_full.set_task(
            self.t_perceive_pillow, self.perceive_pillow_full.pillow
        )
        self.perceive_pillow_full.add_precondition(
            Not(self.current_arm_pose(self.home))
        )
        self.perceive_pillow_full.add_precondition(self.holding(self.nothing))
        self.perceive_pillow_full.add_precondition(Not(self.pillow_is_on(self.table)))
        self.perceive_pillow_full.add_precondition(
            Not(self.perceived_pillow(self.table))
        )
        s1 = self.perceive_pillow_full.add_subtask(self.move_arm, self.home)
        s2 = self.perceive_pillow_full.add_subtask(self.get_next_pillow)
        s3 = self.perceive_pillow_full.add_subtask(self.move_arm, self.over_table)
        s4 = self.perceive_pillow_full.add_subtask(self.perceive_pillow, self.table)
        self.perceive_pillow_full.set_ordered(s1, s2, s3, s4)

        # INSPECT PILLOW
        # weighted, inspect and perceive
        self.inspect_pillow_inspect = Method("inspect_pillow_inspect", pillow=type_item)
        self.inspect_pillow_inspect.set_task(
            self.t_inspect_pillow, self.inspect_pillow_inspect.pillow
        )
        self.inspect_pillow_inspect.add_precondition(Not(self.pillow_status_known()))
        self.inspect_pillow_inspect.add_precondition(self.pillow_weight_known())
        self.inspect_pillow_inspect.add_precondition(self.perceived_pillow(self.scale))
        self.inspect_pillow_inspect.add_precondition(self.pillow_is_on(self.scale))
        self.inspect_pillow_inspect.add_precondition(self.holding(self.nothing))
        self.inspect_pillow_inspect.add_precondition(
            self.current_arm_pose(self.over_scale)
        )
        self.inspect_pillow_inspect.add_subtask(self.inspect)

        # weighted and inspected, perceive
        self.inspect_pillow_perceive = Method(
            "inspect_pillow_perceive", pillow=type_item
        )
        self.inspect_pillow_perceive.set_task(
            self.t_inspect_pillow, self.inspect_pillow_perceive.pillow
        )
        self.inspect_pillow_perceive.add_precondition(Not(self.pillow_status_known()))
        self.inspect_pillow_perceive.add_precondition(self.pillow_weight_known())
        self.inspect_pillow_perceive.add_precondition(self.pillow_is_on(self.scale))
        self.inspect_pillow_perceive.add_precondition(
            Not(self.perceived_pillow(self.scale))
        )
        self.inspect_pillow_perceive.add_precondition(self.holding(self.nothing))
        self.inspect_pillow_perceive.add_precondition(
            self.current_arm_pose(self.over_scale)
        )
        s1 = self.inspect_pillow_perceive.add_subtask(self.perceive_pillow, self.scale)
        s2 = self.inspect_pillow_perceive.add_subtask(self.inspect)

        # pillow on scale, arm moved over scale, weight, perceive and inspect
        self.inspect_pillow_weight = Method("inspect_pillow_weight", pillow=type_item)
        self.inspect_pillow_weight.set_task(
            self.t_inspect_pillow, self.inspect_pillow_weight.pillow
        )
        self.inspect_pillow_weight.add_precondition(Not(self.pillow_status_known()))
        self.inspect_pillow_weight.add_precondition(Not(self.pillow_weight_known()))
        self.inspect_pillow_weight.add_precondition(self.holding(self.nothing))
        self.inspect_pillow_weight.add_precondition(self.pillow_is_on(self.scale))
        self.inspect_pillow_weight.add_precondition(
            self.current_arm_pose(self.over_scale)
        )
        s1 = self.inspect_pillow_weight.add_subtask(self.weigh_pillow)
        s2 = self.inspect_pillow_weight.add_subtask(self.perceive_pillow, self.scale)
        s3 = self.inspect_pillow_weight.add_subtask(self.inspect)
        self.inspect_pillow_weight.set_ordered(s1, s2, s3)

        # pillow on scale, move arm, weight, perceive and inspect
        self.inspect_pillow_move_arm_3 = Method(
            "inspect_pillow_move_arm_3", pillow=type_item
        )
        self.inspect_pillow_move_arm_3.set_task(
            self.t_inspect_pillow, self.inspect_pillow_move_arm_3.pillow
        )
        self.inspect_pillow_move_arm_3.add_precondition(Not(self.pillow_status_known()))
        self.inspect_pillow_move_arm_3.add_precondition(Not(self.pillow_weight_known()))
        self.inspect_pillow_move_arm_3.add_precondition(self.holding(self.nothing))
        self.inspect_pillow_move_arm_3.add_precondition(self.pillow_is_on(self.scale))
        self.inspect_pillow_move_arm_3.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        s1 = self.inspect_pillow_move_arm_3.add_subtask(self.move_arm, self.over_scale)
        s2 = self.inspect_pillow_move_arm_3.add_subtask(self.weigh_pillow)
        s3 = self.inspect_pillow_move_arm_3.add_subtask(
            self.perceive_pillow, self.scale
        )
        s4 = self.inspect_pillow_move_arm_3.add_subtask(self.inspect)
        self.inspect_pillow_move_arm_3.set_ordered(s1, s2, s3, s4)

        # arm is over scale, place on scale, weight, perceive and inspect
        self.inspect_pillow_place = Method(
            "inspect_pillow_place", pillow=type_item, size=type_size
        )
        self.inspect_pillow_place.set_task(
            self.t_inspect_pillow, self.inspect_pillow_place.pillow
        )
        self.inspect_pillow_place.add_precondition(Not(self.pillow_status_known()))
        self.inspect_pillow_place.add_precondition(Not(self.pillow_weight_known()))
        self.inspect_pillow_place.add_precondition(
            self.holding(self.inspect_pillow_place.pillow)
        )
        self.inspect_pillow_place.add_precondition(
            self.current_arm_pose(self.over_scale)
        )
        s1 = self.inspect_pillow_place.add_subtask(
            self.place_pillow_on_scale,
            self.inspect_pillow_place.pillow,
            self.inspect_pillow_place.size,
        )
        s2 = self.inspect_pillow_place.add_subtask(self.move_arm, self.over_scale)
        s3 = self.inspect_pillow_place.add_subtask(self.weigh_pillow)
        s4 = self.inspect_pillow_place.add_subtask(self.perceive_pillow, self.scale)
        s5 = self.inspect_pillow_place.add_subtask(self.inspect)
        self.inspect_pillow_place.set_ordered(s1, s2, s3, s4, s5)

        # arm is up, move over scale, place on scale, weight, perceive and inspect
        self.inspect_pillow_move_arm_2 = Method(
            "inspect_pillow_move_arm_2", pillow=type_item, size=type_size
        )
        self.inspect_pillow_move_arm_2.set_task(
            self.t_inspect_pillow, self.inspect_pillow_move_arm_2.pillow
        )
        self.inspect_pillow_move_arm_2.add_precondition(Not(self.pillow_status_known()))
        self.inspect_pillow_move_arm_2.add_precondition(Not(self.pillow_weight_known()))
        self.inspect_pillow_move_arm_2.add_precondition(
            self.holding(self.inspect_pillow_move_arm_2.pillow)
        )
        self.inspect_pillow_move_arm_2.add_precondition(
            self.current_arm_pose(self.arm_up)
        )
        s1 = self.inspect_pillow_move_arm_2.add_subtask(self.move_arm, self.over_scale)
        s2 = self.inspect_pillow_move_arm_2.add_subtask(
            self.place_pillow_on_scale,
            self.inspect_pillow_move_arm_2.pillow,
            self.inspect_pillow_move_arm_2.size,
        )
        s3 = self.inspect_pillow_move_arm_2.add_subtask(self.move_arm, self.over_scale)
        s4 = self.inspect_pillow_move_arm_2.add_subtask(self.weigh_pillow)
        s5 = self.inspect_pillow_move_arm_2.add_subtask(
            self.perceive_pillow, self.scale
        )
        s6 = self.inspect_pillow_move_arm_2.add_subtask(self.inspect)
        self.inspect_pillow_move_arm_2.set_ordered(s1, s2, s3, s4, s5, s6)

        # pillow in hand, move arm up, place on scale, weight, perceive and inspect
        self.inspect_pillow_move_arm_1 = Method(
            "inspect_pillow_move_arm_1", pillow=type_item, size=type_size
        )
        self.inspect_pillow_move_arm_1.set_task(
            self.t_inspect_pillow, self.inspect_pillow_move_arm_1.pillow
        )
        self.inspect_pillow_move_arm_1.add_precondition(Not(self.pillow_status_known()))
        self.inspect_pillow_move_arm_1.add_precondition(Not(self.pillow_weight_known()))
        self.inspect_pillow_move_arm_1.add_precondition(
            self.holding(self.inspect_pillow_move_arm_1.pillow)
        )
        self.inspect_pillow_move_arm_1.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        s1 = self.inspect_pillow_move_arm_1.add_subtask(self.move_arm, self.arm_up)
        s2 = self.inspect_pillow_move_arm_1.add_subtask(self.move_arm, self.over_scale)
        s3 = self.inspect_pillow_move_arm_1.add_subtask(
            self.place_pillow_on_scale,
            self.inspect_pillow_move_arm_1.pillow,
            self.inspect_pillow_move_arm_1.size,
        )
        s4 = self.inspect_pillow_move_arm_1.add_subtask(self.move_arm, self.over_scale)
        s5 = self.inspect_pillow_move_arm_1.add_subtask(self.weigh_pillow)
        s6 = self.inspect_pillow_move_arm_1.add_subtask(
            self.perceive_pillow, self.scale
        )
        s7 = self.inspect_pillow_move_arm_1.add_subtask(self.inspect)
        self.inspect_pillow_move_arm_1.set_ordered(s1, s2, s3, s4, s5, s6, s7)

        # pick from table, place on scale, weight, perceive and inspect
        self.inspect_pillow_full = Method(
            "inspect_pillow_full", pillow=type_item, size=type_size
        )
        self.inspect_pillow_full.set_task(
            self.t_inspect_pillow, self.inspect_pillow_full.pillow
        )
        self.inspect_pillow_full.add_precondition(Not(self.pillow_status_known()))
        self.inspect_pillow_full.add_precondition(Not(self.pillow_weight_known()))
        self.inspect_pillow_full.add_precondition(self.perceived_pillow(self.table))
        self.inspect_pillow_full.add_precondition(self.pillow_is_on(self.table))
        self.inspect_pillow_full.add_precondition(self.holding(self.nothing))
        s1 = self.inspect_pillow_full.add_subtask(
            self.pick_pillow,
            self.inspect_pillow_full.pillow,
            self.inspect_pillow_full.size,
            self.table,
        )
        s2 = self.inspect_pillow_full.add_subtask(self.move_arm, self.arm_up)
        s3 = self.inspect_pillow_full.add_subtask(self.move_arm, self.over_scale)
        s4 = self.inspect_pillow_full.add_subtask(
            self.place_pillow_on_scale,
            self.inspect_pillow_full.pillow,
            self.inspect_pillow_full.size,
        )
        s5 = self.inspect_pillow_full.add_subtask(self.move_arm, self.over_scale)
        s6 = self.inspect_pillow_full.add_subtask(self.weigh_pillow)
        s7 = self.inspect_pillow_full.add_subtask(self.perceive_pillow, self.scale)
        s8 = self.inspect_pillow_full.add_subtask(self.inspect)
        self.inspect_pillow_full.set_ordered(s1, s2, s3, s4, s5, s6, s7, s8)

        # PLACE PILLOW IN BOX
        # already done, update server
        self.place_pillow_in_box_update = Method(
            "place_pillow_in_box_update", pillow=type_item
        )
        self.place_pillow_in_box_update.set_task(
            self.t_place_pillow_in_box, self.place_pillow_in_box_update.pillow
        )
        self.place_pillow_in_box_update.add_precondition(self.pillow_is_on(self.box))
        self.place_pillow_in_box_update.add_precondition(
            self.current_arm_pose(self.over_boxes)
        )
        self.place_pillow_in_box_update.add_subtask(self.update_pemu_server)

        # placed in box, move over boxes, update server
        self.place_pillow_in_box_move_arm_3 = Method(
            "place_pillow_in_box_move_arm_3", pillow=type_item, status=type_status
        )
        self.place_pillow_in_box_move_arm_3.set_task(
            self.t_place_pillow_in_box, self.place_pillow_in_box_move_arm_3.pillow
        )
        self.place_pillow_in_box_move_arm_3.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.place_pillow_in_box_move_arm_3.add_precondition(self.holding(self.nothing))
        self.place_pillow_in_box_move_arm_3.add_precondition(
            self.pillow_is_on(self.box)
        )
        s1 = self.place_pillow_in_box_move_arm_3.add_subtask(
            self.move_arm, self.over_boxes
        )
        s2 = self.place_pillow_in_box_move_arm_3.add_subtask(self.update_pemu_server)
        self.place_pillow_in_box_move_arm_3.set_ordered(s1, s2)

        # pillow in hand and over boxes, place in box, update server
        self.place_pillow_in_box_place = Method(
            "place_pillow_in_box_place", pillow=type_item, status=type_status
        )
        self.place_pillow_in_box_place.set_task(
            self.t_place_pillow_in_box, self.place_pillow_in_box_place.pillow
        )
        self.place_pillow_in_box_place.add_precondition(
            self.space_in_box(self.place_pillow_in_box_place.status)
        )
        self.place_pillow_in_box_place.add_precondition(
            self.status_of_pillow(self.place_pillow_in_box_place.status)
        )
        self.place_pillow_in_box_place.add_precondition(
            self.current_arm_pose(self.over_boxes)
        )
        self.place_pillow_in_box_place.add_precondition(
            self.holding(self.place_pillow_in_box_place.pillow)
        )
        self.place_pillow_in_box_place.add_precondition(self.pillow_status_known())
        self.place_pillow_in_box_place.add_precondition(self.pillow_weight_known())
        self.place_pillow_in_box_place.add_precondition(
            Not(self.pillow_is_on(self.box))
        )
        s1 = self.place_pillow_in_box_place.add_subtask(
            self.place_pillow_in_box,
            self.place_pillow_in_box_place.pillow,
            self.place_pillow_in_box_place.status,
        )
        s2 = self.place_pillow_in_box_place.add_subtask(self.move_arm, self.over_boxes)
        s3 = self.place_pillow_in_box_place.add_subtask(self.update_pemu_server)
        self.place_pillow_in_box_place.set_ordered(s1, s2, s3)

        # pillow in hand and over boxes, place in box, update server
        self.place_pillow_in_box_move_arm_1 = Method(
            "place_pillow_in_box_move_arm_1", pillow=type_item, status=type_status
        )
        self.place_pillow_in_box_move_arm_1.set_task(
            self.t_place_pillow_in_box, self.place_pillow_in_box_move_arm_1.pillow
        )
        self.place_pillow_in_box_move_arm_1.add_precondition(
            self.space_in_box(self.place_pillow_in_box_move_arm_1.status)
        )
        self.place_pillow_in_box_move_arm_1.add_precondition(
            self.status_of_pillow(self.place_pillow_in_box_move_arm_1.status)
        )
        self.place_pillow_in_box_move_arm_1.add_precondition(
            self.current_arm_pose(self.arm_up)
        )
        self.place_pillow_in_box_move_arm_1.add_precondition(
            self.holding(self.place_pillow_in_box_move_arm_1.pillow)
        )
        self.place_pillow_in_box_move_arm_1.add_precondition(self.pillow_status_known())
        self.place_pillow_in_box_move_arm_1.add_precondition(self.pillow_weight_known())
        self.place_pillow_in_box_move_arm_1.add_precondition(
            Not(self.pillow_is_on(self.box))
        )
        s1 = self.place_pillow_in_box_move_arm_1.add_subtask(
            self.move_arm, self.over_boxes
        )
        s2 = self.place_pillow_in_box_move_arm_1.add_subtask(
            self.place_pillow_in_box,
            self.place_pillow_in_box_move_arm_1.pillow,
            self.place_pillow_in_box_move_arm_1.status,
        )
        s3 = self.place_pillow_in_box_move_arm_1.add_subtask(
            self.move_arm, self.over_boxes
        )
        s4 = self.place_pillow_in_box_move_arm_1.add_subtask(self.update_pemu_server)
        self.place_pillow_in_box_move_arm_1.set_ordered(s1, s2, s3, s4)

        # pillow in hand, move arm up, move over boxes, place in box, update server
        self.place_pillow_in_box_move_arm_2 = Method(
            "place_pillow_in_box_move_arm_2", pillow=type_item, status=type_status
        )
        self.place_pillow_in_box_move_arm_2.set_task(
            self.t_place_pillow_in_box, self.place_pillow_in_box_move_arm_2.pillow
        )
        self.place_pillow_in_box_move_arm_2.add_precondition(
            self.space_in_box(self.place_pillow_in_box_move_arm_2.status)
        )
        self.place_pillow_in_box_move_arm_2.add_precondition(
            self.status_of_pillow(self.place_pillow_in_box_move_arm_2.status)
        )
        self.place_pillow_in_box_move_arm_2.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.place_pillow_in_box_move_arm_2.add_precondition(
            self.holding(self.place_pillow_in_box_move_arm_2.pillow)
        )
        self.place_pillow_in_box_move_arm_2.add_precondition(self.pillow_status_known())
        self.place_pillow_in_box_move_arm_2.add_precondition(self.pillow_weight_known())
        self.place_pillow_in_box_move_arm_2.add_precondition(
            Not(self.pillow_is_on(self.box))
        )
        s1 = self.place_pillow_in_box_move_arm_2.add_subtask(self.move_arm, self.arm_up)
        s2 = self.place_pillow_in_box_move_arm_2.add_subtask(
            self.move_arm, self.over_boxes
        )
        s3 = self.place_pillow_in_box_move_arm_2.add_subtask(
            self.place_pillow_in_box,
            self.place_pillow_in_box_move_arm_2.pillow,
            self.place_pillow_in_box_move_arm_2.status,
        )
        s4 = self.place_pillow_in_box_move_arm_2.add_subtask(
            self.move_arm, self.over_boxes
        )
        s5 = self.place_pillow_in_box_move_arm_2.add_subtask(self.update_pemu_server)
        self.place_pillow_in_box_move_arm_2.set_ordered(s1, s2, s3, s4, s5)

        # inspected pillow, pick it up, move arm up then over boxes, place in box, update server
        self.place_pillow_in_box_full = Method(
            "place_pillow_in_box_full",
            pillow=type_item,
            status=type_status,
            size=type_size,
        )
        self.place_pillow_in_box_full.set_task(
            self.t_place_pillow_in_box, self.place_pillow_in_box_full.pillow
        )
        self.place_pillow_in_box_full.add_precondition(
            self.space_in_box(self.place_pillow_in_box_full.status)
        )
        self.place_pillow_in_box_full.add_precondition(
            self.status_of_pillow(self.place_pillow_in_box_full.status)
        )
        self.place_pillow_in_box_full.add_precondition(
            self.current_arm_pose(self.over_scale)
        )
        self.place_pillow_in_box_full.add_precondition(self.holding(self.nothing))
        self.place_pillow_in_box_full.add_precondition(self.pillow_status_known())
        self.place_pillow_in_box_full.add_precondition(self.pillow_weight_known())
        self.place_pillow_in_box_full.add_precondition(self.pillow_is_on(self.scale))
        self.place_pillow_in_box_full.add_precondition(Not(self.pillow_is_on(self.box)))
        s1 = self.place_pillow_in_box_full.add_subtask(
            self.pick_pillow,
            self.place_pillow_in_box_full.pillow,
            self.place_pillow_in_box_full.size,
            self.scale,
        )
        s2 = self.place_pillow_in_box_full.add_subtask(self.move_arm, self.arm_up)
        s3 = self.place_pillow_in_box_full.add_subtask(self.move_arm, self.over_boxes)
        s4 = self.place_pillow_in_box_full.add_subtask(
            self.place_pillow_in_box,
            self.place_pillow_in_box_full.pillow,
            self.place_pillow_in_box_full.status,
        )
        s5 = self.place_pillow_in_box_full.add_subtask(self.move_arm, self.over_boxes)
        s6 = self.place_pillow_in_box_full.add_subtask(self.update_pemu_server)
        self.place_pillow_in_box_full.set_ordered(s1, s2, s3, s4, s5, s6)

        # box is full
        self.place_pillow_in_box_empty_box = Method(
            "place_pillow_in_box_empty_box", pillow=type_item, status=type_status
        )
        self.place_pillow_in_box_empty_box.set_task(
            self.t_place_pillow_in_box, self.place_pillow_in_box_empty_box.pillow
        )
        self.place_pillow_in_box_empty_box.add_precondition(self.pillow_status_known())
        self.place_pillow_in_box_empty_box.add_precondition(
            Not(self.space_in_box(self.place_pillow_in_box_empty_box.status))
        )
        self.place_pillow_in_box_empty_box.add_precondition(
            self.status_of_pillow(self.place_pillow_in_box_empty_box.status)
        )
        self.place_pillow_in_box_empty_box.add_subtask(
            self.empty_box, self.place_pillow_in_box_empty_box.status
        )

        # RATE PILLOW
        # perceive the pillow to get the size
        self.rate_pillow_perceive = Method("rate_pillow_perceive", pillow=type_item)
        self.rate_pillow_perceive.set_task(
            self.t_rate_pillow, self.rate_pillow_perceive.pillow
        )
        self.rate_pillow_perceive.add_precondition(Not(self.item_size_known()))
        self.rate_pillow_perceive.add_subtask(
            self.t_perceive_pillow, self.rate_pillow_perceive.pillow
        )

        # pick the pillow from the table, place it on the scale and inspect it
        self.rate_pillow_inspect = Method("rate_pillow_inspect", pillow=type_item)
        self.rate_pillow_inspect.set_task(
            self.t_rate_pillow, self.rate_pillow_inspect.pillow
        )
        self.rate_pillow_inspect.add_precondition(self.item_size_known())
        self.rate_pillow_inspect.add_precondition(Not(self.pillow_status_known()))
        self.rate_pillow_inspect.add_subtask(
            self.t_inspect_pillow, self.rate_pillow_inspect.pillow
        )

        # place in box
        self.rate_pillow_place = Method("rate_pillow_place", pillow=type_item)
        self.rate_pillow_place.set_task(
            self.t_rate_pillow, self.rate_pillow_place.pillow
        )
        self.rate_pillow_place.add_precondition(self.pillow_weight_known())
        self.rate_pillow_place.add_precondition(self.pillow_status_known())
        self.rate_pillow_place.add_subtask(
            self.t_place_pillow_in_box, self.rate_pillow_place.pillow
        )

        self.methods = (
            self.perceive_pillow_perceive,
            self.perceive_pillow_move_arm,
            self.perceive_pillow_get,
            self.perceive_pillow_full,
            self.inspect_pillow_move_arm_1,
            self.inspect_pillow_move_arm_2,
            self.inspect_pillow_move_arm_3,
            self.inspect_pillow_weight,
            self.inspect_pillow_place,
            self.inspect_pillow_perceive,
            self.inspect_pillow_inspect,
            self.inspect_pillow_full,
            self.place_pillow_in_box_update,
            self.place_pillow_in_box_move_arm_3,
            self.place_pillow_in_box_move_arm_1,
            self.place_pillow_in_box_move_arm_2,
            self.place_pillow_in_box_place,
            self.place_pillow_in_box_full,
            self.place_pillow_in_box_empty_box,
            self.rate_pillow_perceive,
            self.rate_pillow_inspect,
            self.rate_pillow_place,
        )

    def _create_domain_actions(self, temporal: bool = False) -> None:
        actions = Actions(self._env)

        if temporal:
            # TODO TEMPORAL
            pass
        else:
            self.perceive_pillow, [l] = self.create_action(
                "perceive_pillow",
                location=Location,
                _callable=actions.perceive_pillow,
            )
            self.perceive_pillow.add_precondition(Not(self.perceived_pillow(l)))
            self.perceive_pillow.add_effect(self.item_size_known(), True)
            self.perceive_pillow.add_effect(self.perceived_pillow(l), True)

            self.get_next_pillow, _ = self.create_action(
                "get_next_pillow",
                _callable=actions.get_next_pillow,
            )
            self.get_next_pillow.add_precondition(Not(self.pillow_is_on(self.table)))
            self.get_next_pillow.add_effect(self.pillow_is_on(self.table), True)

            self.move_arm, [a] = self.create_action(
                "move_arm",
                arm_pose=ArmPose,
                _callable=actions.move_arm,
            )
            self.move_arm.add_effect(self.current_arm_pose(a), True)

            self.pick_pillow, [p, s, l] = self.create_action(
                "pick_pillow",
                pillow=Item,
                size=Size,
                location=Location,
                _callable=actions.pick_pillow,
            )
            self.pick_pillow.add_precondition(self.holding(self.nothing))
            self.pick_pillow.add_precondition(self.perceived_pillow(l))
            self.pick_pillow.add_precondition(self.item_size_known())
            self.pick_pillow.add_precondition(self.current_item_size(s))
            self.pick_pillow.add_precondition(self.pillow_is_on(l))
            self.pick_pillow.add_effect(self.holding(p), True)
            self.pick_pillow.add_effect(self.holding(self.nothing), False)
            self.pick_pillow.add_effect(self.current_arm_pose(self.unknown_pose), True)
            self.pick_pillow.add_effect(self.perceived_pillow(l), False)
            self.pick_pillow.add_effect(self.pillow_is_on(l), False)

            self.place_pillow_on_scale, [p, s] = self.create_action(
                "place_pillow_on_scale",
                pillow=Item,
                size=Size,
                _callable=actions.place_pillow_on_scale,
            )
            self.place_pillow_on_scale.add_precondition(self.item_size_known())
            self.place_pillow_on_scale.add_precondition(self.current_item_size(s))
            self.place_pillow_on_scale.add_precondition(self.holding(p))
            self.place_pillow_on_scale.add_precondition(
                self.current_arm_pose(self.over_scale)
            )
            self.place_pillow_on_scale.add_precondition(Not(self.pillow_weight_known()))
            self.place_pillow_on_scale.add_effect(self.holding(self.nothing), True)
            self.place_pillow_on_scale.add_effect(
                self.current_arm_pose(self.unknown_pose), True
            )
            self.place_pillow_on_scale.add_effect(
                self.current_arm_pose(self.over_scale), False
            )
            self.place_pillow_on_scale.add_effect(self.holding(p), False)
            self.place_pillow_on_scale.add_effect(self.pillow_is_on(self.scale), True)

            self.weigh_pillow, _ = self.create_action(
                "weigh_pillow", _callable=actions.weigh_pillow
            )
            self.weigh_pillow.add_precondition(Not(self.pillow_weight_known()))
            self.weigh_pillow.add_precondition(self.holding(self.nothing))
            self.weigh_pillow.add_precondition(self.pillow_is_on(self.scale))
            self.weigh_pillow.add_effect(self.pillow_weight_known(), True)

            self.place_pillow_in_box, [p, st] = self.create_action(
                "place_pillow_in_box",
                pillow=Item,
                status=Status,
                _callable=actions.place_pillow_in_box,
            )
            self.place_pillow_in_box.add_precondition(self.pillow_status_known())
            self.place_pillow_in_box.add_precondition(self.space_in_box(st))
            self.place_pillow_in_box.add_precondition(self.status_of_pillow(st))
            self.place_pillow_in_box.add_precondition(
                self.current_arm_pose(self.over_boxes)
            )
            self.place_pillow_in_box.add_precondition(self.holding(p))
            self.place_pillow_in_box.add_effect(self.holding(self.nothing), True)
            self.place_pillow_in_box.add_effect(
                self.current_arm_pose(self.unknown_pose), True
            )
            self.place_pillow_in_box.add_effect(
                self.current_arm_pose(self.over_boxes), False
            )
            self.place_pillow_in_box.add_effect(self.holding(p), False)
            self.place_pillow_in_box.add_effect(self.pillow_is_on(self.box), True)

            self.empty_box, [st] = self.create_action(
                "empty_box",
                status=Status,
                _callable=actions.empty_box,
            )
            self.empty_box.add_precondition(Not(self.space_in_box(st)))
            self.empty_box.add_precondition(self.pillow_status_known())
            self.empty_box.add_effect(self.space_in_box(st), True)

            self.inspect, _ = self.create_action("inspect", _callable=actions.inspect)
            self.inspect.add_precondition(Not(self.pillow_status_known()))
            self.inspect.add_precondition(self.current_arm_pose(self.over_scale))
            self.inspect.add_effect(self.pillow_status_known(), True)

            self.update_pemu_server, _ = self.create_action(
                "update_pemu_server", _callable=actions.update_pemu_server
            )
            self.update_pemu_server.add_precondition(
                self.current_arm_pose(self.over_boxes)
            )
            self.update_pemu_server.add_precondition(self.pillow_is_on(self.box))
            self.update_pemu_server.add_effect(self.pillow_status_known(), False)
            self.update_pemu_server.add_effect(self.pillow_weight_known(), False)
            self.update_pemu_server.add_effect(self.pillow_is_on(self.box), False)

    def set_state_and_goal(self, problem, goal=None) -> None:
        success = True
        if goal is None:
            problem.task_network.add_subtask(self.t_rate_pillow(self.pillow))
        else:
            logerr((f"Task ({goal}) is unknown! Please leave the goal empty."))
            success = False
        return success
