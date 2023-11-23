from rospy import logerr
from unified_planning.model.htn import Task, Method

from unified_planning.shortcuts import Not, Or, And
from april_krem.domains.PEMU_components import (
    Item,
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
        self.create_types([Item, ArmPose, Size, Status])
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
        self.pillow_on_scale = self.create_fluent_from_function(
            self._env.pillow_on_scale
        )
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
        self.pillow_in_box = self.create_fluent_from_function(self._env.pillow_in_box)

        # Create objects for both planning and execution
        self.create_enum_objects(Item)
        self.nothing = self.objects[Item.nothing.name]
        self.pillow = self.objects[Item.pillow.name]

        self.create_enum_objects(ArmPose)
        self.unknown_pose = self.objects[ArmPose.unknown.name]
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
        self.t_pick_pillow = Task("t_pick_pillow", pillow=type_item)
        self.t_place_pillow_on_scale = Task("t_place_pillow_on_scale", pillow=type_item)
        self.t_inspect_pillow = Task("t_inspect_pillow")
        self.t_place_pillow_in_box = Task("t_place_pillow_in_box", pillow=type_item)
        self.t_rate_pillow = Task("t_rate_pillow", pillow=type_item)

        self.tasks = (
            self.t_perceive_pillow,
            self.t_pick_pillow,
            self.t_place_pillow_on_scale,
            self.t_inspect_pillow,
            self.t_place_pillow_in_box,
            self.t_rate_pillow,
        )

        # Methods

        # PERCEIVE PILLOW
        # pillow perceived
        self.perceive_pillow_noop = Method("perceive_pillow_noop", pillow=type_item)
        self.perceive_pillow_noop.set_task(
            self.t_perceive_pillow, self.perceive_pillow_noop.pillow
        )
        self.perceive_pillow_noop.add_precondition(self.item_size_known())

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
        self.perceive_pillow_perceive.add_subtask(self.perceive_pillow)

        # move arm and perceive pillow
        self.perceive_pillow_full = Method("perceive_pillow_full", pillow=type_item)
        self.perceive_pillow_full.set_task(
            self.t_perceive_pillow, self.perceive_pillow_full.pillow
        )
        self.perceive_pillow_full.add_precondition(
            Not(self.current_arm_pose(self.over_table))
        )
        self.perceive_pillow_full.add_precondition(self.holding(self.nothing))
        s1 = self.perceive_pillow_full.add_subtask(self.move_arm, self.over_table)
        s2 = self.perceive_pillow_full.add_subtask(self.perceive_pillow)
        self.perceive_pillow_full.set_ordered(s1, s2)

        # PICK PILLOW
        # pillow in hand, arm over table
        self.pick_pillow_noop = Method("pick_pillow_noop", pillow=type_item)
        self.pick_pillow_noop.set_task(self.t_pick_pillow, self.pick_pillow_noop.pillow)
        self.pick_pillow_noop.add_precondition(self.item_size_known())
        self.pick_pillow_noop.add_precondition(Not(self.perceived_pillow()))

        # picked pillow from table, move arm over table
        self.pick_pillow_from_table_move_arm = Method(
            "pick_pillow_from_table_move_arm", pillow=type_item
        )
        self.pick_pillow_from_table_move_arm.set_task(
            self.t_pick_pillow, self.pick_pillow_from_table_move_arm.pillow
        )
        self.pick_pillow_from_table_move_arm.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.pick_pillow_from_table_move_arm.add_precondition(
            self.holding(self.pick_pillow_from_table_move_arm.pillow)
        )
        self.pick_pillow_from_table_move_arm.add_precondition(self.item_size_known())
        self.pick_pillow_from_table_move_arm.add_subtask(self.move_arm, self.over_table)

        # pick pillow from table and move arm
        self.pick_pillow_from_table_full = Method(
            "pick_pillow_from_table_full", pillow=type_item, size=type_size
        )
        self.pick_pillow_from_table_full.set_task(
            self.t_pick_pillow, self.pick_pillow_from_table_full.pillow
        )
        self.pick_pillow_from_table_full.add_precondition(
            self.current_arm_pose(self.over_table)
        )
        self.pick_pillow_from_table_full.add_precondition(self.holding(self.nothing))
        self.pick_pillow_from_table_full.add_precondition(self.perceived_pillow())
        self.pick_pillow_from_table_full.add_precondition(self.item_size_known())
        s1 = self.pick_pillow_from_table_full.add_subtask(
            self.pick_pillow,
            self.pick_pillow_from_table_full.pillow,
            self.pick_pillow_from_table_full.size,
        )
        s2 = self.pick_pillow_from_table_full.add_subtask(
            self.move_arm, self.over_table
        )
        self.pick_pillow_from_table_full.set_ordered(s1, s2)

        # picked pillow from scale, move arm over scale
        self.pick_pillow_from_scale_move_arm = Method(
            "pick_pillow_from_scale_move_arm", pillow=type_item
        )
        self.pick_pillow_from_scale_move_arm.set_task(
            self.t_pick_pillow, self.pick_pillow_from_scale_move_arm.pillow
        )
        self.pick_pillow_from_scale_move_arm.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.pick_pillow_from_scale_move_arm.add_precondition(
            self.holding(self.pick_pillow_from_scale_move_arm.pillow)
        )
        self.pick_pillow_from_scale_move_arm.add_precondition(self.item_size_known())
        self.pick_pillow_from_scale_move_arm.add_precondition(
            self.pillow_status_known()
        )
        self.pick_pillow_from_scale_move_arm.add_subtask(self.move_arm, self.over_scale)

        # pillow on scale, pick it and move arm
        self.pick_pillow_from_scale_full = Method(
            "pick_pillow_from_scale_full", pillow=type_item, size=type_size
        )
        self.pick_pillow_from_scale_full.set_task(
            self.t_pick_pillow, self.pick_pillow_from_scale_full.pillow
        )
        self.pick_pillow_from_scale_full.add_precondition(
            self.current_arm_pose(self.over_scale)
        )
        self.pick_pillow_from_scale_full.add_precondition(self.holding(self.nothing))
        self.pick_pillow_from_scale_full.add_precondition(self.perceived_pillow())
        self.pick_pillow_from_scale_full.add_precondition(self.item_size_known())
        self.pick_pillow_from_scale_full.add_precondition(self.pillow_status_known())
        s1 = self.pick_pillow_from_scale_full.add_subtask(
            self.pick_pillow,
            self.pick_pillow_from_scale_full.pillow,
            self.pick_pillow_from_scale_full.size,
        )
        s2 = self.pick_pillow_from_scale_full.add_subtask(
            self.move_arm, self.over_scale
        )
        self.pick_pillow_from_scale_full.set_ordered(s1, s2)

        # box is full
        self.pick_pillow_empty_box = Method(
            "pick_pillow_empty_box", pillow=type_item, status=type_status
        )
        self.pick_pillow_empty_box.set_task(
            self.t_pick_pillow, self.pick_pillow_empty_box.pillow
        )
        self.pick_pillow_empty_box.add_precondition(self.pillow_status_known())
        self.pick_pillow_empty_box.add_precondition(
            Not(self.space_in_box(self.pick_pillow_empty_box.status))
        )
        self.pick_pillow_empty_box.add_subtask(
            self.empty_box, self.pick_pillow_empty_box.status
        )

        # PLACE PILLOW ON SCALE
        # pillow already weighted or already on scale
        self.place_pillow_on_scale_noop = Method(
            "place_pillow_on_scale_noop", pillow=type_item, size=type_size
        )
        self.place_pillow_on_scale_noop.set_task(
            self.t_place_pillow_on_scale, self.place_pillow_on_scale_noop.pillow
        )
        self.place_pillow_on_scale_noop.add_precondition(
            Or(self.pillow_on_scale(), self.pillow_weight_known())
        )

        # pillow in hand, already placed pillow, move to over scale
        self.place_pillow_on_scale_move_arm = Method(
            "place_pillow_on_scale_move_arm", pillow=type_item, size=type_size
        )
        self.place_pillow_on_scale_move_arm.set_task(
            self.t_place_pillow_on_scale, self.place_pillow_on_scale_move_arm.pillow
        )
        self.place_pillow_on_scale_move_arm.add_precondition(self.holding(self.nothing))
        self.place_pillow_on_scale_move_arm.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.place_pillow_on_scale_move_arm.add_precondition(
            Not(self.pillow_weight_known())
        )
        self.place_pillow_on_scale_move_arm.add_subtask(self.move_arm, self.over_scale)

        # pillow in hand, already over scale, place pillow, move back to over scale
        self.place_pillow_on_scale_place = Method(
            "place_pillow_on_scale_place", pillow=type_item, size=type_size
        )
        self.place_pillow_on_scale_place.set_task(
            self.t_place_pillow_on_scale, self.place_pillow_on_scale_place.pillow
        )
        self.place_pillow_on_scale_place.add_precondition(
            self.holding(self.place_pillow_on_scale_place.pillow)
        )
        self.place_pillow_on_scale_place.add_precondition(
            self.current_arm_pose(self.over_scale)
        )
        self.place_pillow_on_scale_place.add_precondition(
            Not(self.pillow_weight_known())
        )
        s1 = self.place_pillow_on_scale_place.add_subtask(
            self.place_pillow_on_scale,
            self.place_pillow_on_scale_place.pillow,
            self.place_pillow_on_scale_place.size,
        )
        s2 = self.place_pillow_on_scale_place.add_subtask(
            self.move_arm, self.over_scale
        )
        self.place_pillow_on_scale_place.set_ordered(s1, s2)

        # pillow in hand, move over scale, place pillow, move back to over scale
        self.place_pillow_on_scale_full = Method(
            "place_pillow_on_scale_full", pillow=type_item, size=type_size
        )
        self.place_pillow_on_scale_full.set_task(
            self.t_place_pillow_on_scale, self.place_pillow_on_scale_full.pillow
        )
        self.place_pillow_on_scale_full.add_precondition(
            self.holding(self.place_pillow_on_scale_full.pillow)
        )
        self.place_pillow_on_scale_full.add_precondition(
            self.current_arm_pose(self.over_table)
        )
        self.place_pillow_on_scale_full.add_precondition(
            Not(self.pillow_weight_known())
        )
        s1 = self.place_pillow_on_scale_full.add_subtask(self.move_arm, self.over_scale)
        s2 = self.place_pillow_on_scale_full.add_subtask(
            self.place_pillow_on_scale,
            self.place_pillow_on_scale_full.pillow,
            self.place_pillow_on_scale_full.size,
        )
        s3 = self.place_pillow_on_scale_full.add_subtask(self.move_arm, self.over_scale)
        self.place_pillow_on_scale_full.set_ordered(s1, s2, s3)

        # INSPECT PILLOW
        # weighted and inspected, perceive
        self.inspect_pillow_perceive = Method(
            "inspect_pillow_perceive", pillow=type_item
        )
        self.inspect_pillow_perceive.set_task(self.t_inspect_pillow)
        self.inspect_pillow_perceive.add_precondition(self.pillow_status_known())
        self.inspect_pillow_perceive.add_precondition(self.pillow_weight_known())
        self.inspect_pillow_perceive.add_precondition(Not(self.perceived_pillow()))
        self.inspect_pillow_perceive.add_precondition(self.holding(self.nothing))
        self.inspect_pillow_perceive.add_subtask(self.perceive_pillow)

        # weighted, inspect and perceive
        self.inspect_pillow_inspect = Method("inspect_pillow_inspect", pillow=type_item)
        self.inspect_pillow_inspect.set_task(self.t_inspect_pillow)
        self.inspect_pillow_inspect.add_precondition(Not(self.pillow_status_known()))
        self.inspect_pillow_inspect.add_precondition(self.pillow_weight_known())
        self.inspect_pillow_inspect.add_precondition(Not(self.perceived_pillow()))
        self.inspect_pillow_inspect.add_precondition(self.holding(self.nothing))
        s1 = self.inspect_pillow_inspect.add_subtask(self.inspect)
        s2 = self.inspect_pillow_inspect.add_subtask(self.perceive_pillow)
        self.inspect_pillow_inspect.set_ordered(s1, s2)

        # weight, inspect and perceive
        self.inspect_pillow_full = Method("inspect_pillow_full", pillow=type_item)
        self.inspect_pillow_full.set_task(self.t_inspect_pillow)
        self.inspect_pillow_full.add_precondition(Not(self.pillow_status_known()))
        self.inspect_pillow_full.add_precondition(Not(self.pillow_weight_known()))
        self.inspect_pillow_full.add_precondition(self.holding(self.nothing))
        s1 = self.inspect_pillow_full.add_subtask(self.weigh_pillow)
        s2 = self.inspect_pillow_full.add_subtask(self.inspect)
        s3 = self.inspect_pillow_full.add_subtask(self.perceive_pillow)
        self.inspect_pillow_full.set_ordered(s1, s2, s3)

        # PLACE PILLOW IN BOX
        # already done, update server
        self.place_pillow_in_box_update = Method(
            "place_pillow_in_box_update", pillow=type_item
        )
        self.place_pillow_in_box_update.set_task(
            self.t_place_pillow_in_box, self.place_pillow_in_box_update.pillow
        )
        self.place_pillow_in_box_update.add_precondition(self.pillow_in_box())
        self.place_pillow_in_box_update.add_precondition(
            self.current_arm_pose(self.over_boxes)
        )
        self.place_pillow_in_box_update.add_subtask(self.update_pemu_server)

        # pillow placed, move back to boxes, update server
        self.place_pillow_in_box_move_arm = Method(
            "place_pillow_in_box_move_arm", pillow=type_item
        )
        self.place_pillow_in_box_move_arm.set_task(
            self.t_place_pillow_in_box, self.place_pillow_in_box_move_arm.pillow
        )
        self.place_pillow_in_box_move_arm.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.place_pillow_in_box_move_arm.add_precondition(self.holding(self.nothing))
        self.place_pillow_in_box_move_arm.add_precondition(self.pillow_in_box())
        s1 = self.place_pillow_in_box_move_arm.add_subtask(
            self.move_arm, self.over_boxes
        )
        s2 = self.place_pillow_in_box_move_arm.add_subtask(self.update_pemu_server)
        self.place_pillow_in_box_move_arm.set_ordered(s1, s2)

        # pillow in hand and over boxes, place in box, move back to boxes, update server
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
            self.current_arm_pose(self.over_boxes)
        )
        self.place_pillow_in_box_place.add_precondition(
            self.holding(self.place_pillow_in_box_place.pillow)
        )
        self.place_pillow_in_box_place.add_precondition(self.pillow_status_known())
        self.place_pillow_in_box_place.add_precondition(self.pillow_weight_known())
        self.place_pillow_in_box_place.add_precondition(Not(self.pillow_in_box()))
        s1 = self.place_pillow_in_box_place.add_subtask(
            self.place_pillow_in_box,
            self.place_pillow_in_box_place.pillow,
            self.place_pillow_in_box_place.status,
        )
        s2 = self.place_pillow_in_box_place.add_subtask(self.move_arm, self.over_boxes)
        s3 = self.place_pillow_in_box_place.add_subtask(self.update_pemu_server)
        self.place_pillow_in_box_place.set_ordered(s1, s2, s3)

        # pillow in hand, move over boxes, place in box, move back to boxes, update server
        self.place_pillow_in_box_full = Method(
            "place_pillow_in_box_full", pillow=type_item, status=type_status
        )
        self.place_pillow_in_box_full.set_task(
            self.t_place_pillow_in_box, self.place_pillow_in_box_full.pillow
        )
        self.place_pillow_in_box_full.add_precondition(
            self.space_in_box(self.place_pillow_in_box_full.status)
        )
        self.place_pillow_in_box_full.add_precondition(
            self.current_arm_pose(self.over_scale)
        )
        self.place_pillow_in_box_full.add_precondition(
            self.holding(self.place_pillow_in_box_full.pillow)
        )
        self.place_pillow_in_box_full.add_precondition(self.pillow_status_known())
        self.place_pillow_in_box_full.add_precondition(self.pillow_weight_known())
        self.place_pillow_in_box_full.add_precondition(Not(self.pillow_in_box()))
        s1 = self.place_pillow_in_box_full.add_subtask(self.move_arm, self.over_boxes)
        s2 = self.place_pillow_in_box_full.add_subtask(
            self.place_pillow_in_box,
            self.place_pillow_in_box_full.pillow,
            self.place_pillow_in_box_full.status,
        )
        s3 = self.place_pillow_in_box_full.add_subtask(self.move_arm, self.over_boxes)
        s4 = self.place_pillow_in_box_full.add_subtask(self.update_pemu_server)
        self.place_pillow_in_box_full.set_ordered(s1, s2, s3, s4)

        # RATE PILLOW
        # perceive the pillow to get the size
        self.rate_pillow_perceive = Method("rate_pillow_perceive", pillow=type_item)
        self.rate_pillow_perceive.set_task(
            self.t_rate_pillow, self.rate_pillow_perceive.pillow
        )
        self.rate_pillow_perceive.add_precondition(Not(self.item_size_known()))
        self.rate_pillow_perceive.add_precondition(Not(self.pillow_in_box()))
        self.rate_pillow_perceive.add_subtask(
            self.t_perceive_pillow, self.rate_pillow_perceive.pillow
        )

        # pick the pillow from the table, place it on the scale and inspect it
        self.rate_pillow_inspect = Method("rate_pillow_inspect", pillow=type_item)
        self.rate_pillow_inspect.set_task(
            self.t_rate_pillow, self.rate_pillow_inspect.pillow
        )
        self.rate_pillow_inspect.add_precondition(self.item_size_known())
        self.rate_pillow_inspect.add_precondition(Not(self.pillow_in_box()))
        s1 = self.rate_pillow_inspect.add_subtask(
            self.t_pick_pillow, self.rate_pillow_inspect.pillow
        )
        s2 = self.rate_pillow_inspect.add_subtask(
            self.t_place_pillow_on_scale, self.rate_pillow_inspect.pillow
        )
        s3 = self.rate_pillow_inspect.add_subtask(self.t_inspect_pillow)
        self.rate_pillow_inspect.set_ordered(s1, s2, s3)

        # pick the pillow from the scale
        self.rate_pillow_pick_scale = Method("rate_pillow_pick_scale", pillow=type_item)
        self.rate_pillow_pick_scale.set_task(
            self.t_rate_pillow, self.rate_pillow_pick_scale.pillow
        )
        self.rate_pillow_pick_scale.add_precondition(self.item_size_known())
        self.rate_pillow_pick_scale.add_precondition(
            Or(
                Not(self.holding(self.rate_pillow_pick_scale.pillow)),
                self.pillow_on_scale(),
            )
        )
        self.rate_pillow_pick_scale.add_precondition(self.pillow_weight_known())
        self.rate_pillow_pick_scale.add_precondition(self.pillow_status_known())
        self.rate_pillow_pick_scale.add_subtask(
            self.t_pick_pillow, self.rate_pillow_pick_scale.pillow
        )

        # place in box
        self.rate_pillow_place = Method("rate_pillow_place", pillow=type_item)
        self.rate_pillow_place.set_task(
            self.t_rate_pillow, self.rate_pillow_place.pillow
        )
        self.rate_pillow_place.add_precondition(
            Or(
                self.holding(self.rate_pillow_place.pillow),
                self.pillow_in_box(),
            )
        )
        self.rate_pillow_place.add_precondition(
            Or(
                self.pillow_in_box(),
                And(self.pillow_weight_known(), self.pillow_status_known()),
            )
        )
        self.rate_pillow_place.add_subtask(
            self.t_place_pillow_in_box, self.rate_pillow_place.pillow
        )

        self.methods = (
            self.perceive_pillow_noop,
            self.perceive_pillow_perceive,
            self.perceive_pillow_full,
            self.pick_pillow_noop,
            self.pick_pillow_from_table_move_arm,
            self.pick_pillow_from_table_full,
            self.pick_pillow_from_scale_move_arm,
            self.pick_pillow_from_scale_full,
            self.pick_pillow_empty_box,
            self.place_pillow_on_scale_noop,
            self.place_pillow_on_scale_move_arm,
            self.place_pillow_on_scale_place,
            self.place_pillow_on_scale_full,
            self.inspect_pillow_perceive,
            self.inspect_pillow_inspect,
            self.inspect_pillow_full,
            self.place_pillow_in_box_update,
            self.place_pillow_in_box_move_arm,
            self.place_pillow_in_box_place,
            self.place_pillow_in_box_full,
            self.rate_pillow_perceive,
            self.rate_pillow_pick_scale,
            self.rate_pillow_inspect,
            self.rate_pillow_place,
        )

    def _create_domain_actions(self, temporal: bool = False) -> None:
        actions = Actions(self._env)

        if temporal:
            # TODO TEMPORAL
            pass
        else:
            self.perceive_pillow, _ = self.create_action(
                "perceive_pillow",
                _callable=actions.perceive_pillow,
            )
            self.perceive_pillow.add_precondition(Not(self.perceived_pillow()))
            self.perceive_pillow.add_effect(self.item_size_known(), True)
            self.perceive_pillow.add_effect(self.perceived_pillow(), True)

            self.move_arm, [a] = self.create_action(
                "move_arm",
                arm_pose=ArmPose,
                _callable=actions.move_arm,
            )
            self.move_arm.add_effect(self.current_arm_pose(a), True)

            self.pick_pillow, [p, s] = self.create_action(
                "pick_pillow",
                pillow=Item,
                size=Size,
                _callable=actions.pick_pillow,
            )
            self.pick_pillow.add_precondition(self.holding(self.nothing))
            self.pick_pillow.add_precondition(self.perceived_pillow())
            self.pick_pillow.add_precondition(self.item_size_known())
            self.pick_pillow.add_precondition(self.current_item_size(s))
            self.pick_pillow.add_effect(self.holding(p), True)
            self.pick_pillow.add_effect(self.holding(self.nothing), False)
            self.pick_pillow.add_effect(self.current_arm_pose(self.unknown_pose), True)
            self.pick_pillow.add_effect(self.perceived_pillow(), False)

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
            self.place_pillow_on_scale.add_effect(self.pillow_on_scale(), True)

            self.weigh_pillow, _ = self.create_action(
                "weigh_pillow", _callable=actions.weigh_pillow
            )
            self.weigh_pillow.add_precondition(Not(self.pillow_weight_known()))
            self.weigh_pillow.add_precondition(self.holding(self.nothing))
            self.weigh_pillow.add_precondition(self.pillow_on_scale())
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
            self.place_pillow_in_box.add_effect(self.pillow_status_known(), False)
            self.place_pillow_in_box.add_effect(self.pillow_weight_known(), False)
            self.place_pillow_in_box.add_effect(self.pillow_in_box(), True)

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

    def set_state_and_goal(self, problem, goal=None) -> None:
        success = True
        if goal is None:
            problem.task_network.add_subtask(self.t_rate_pillow(self.pillow))
        else:
            logerr((f"Task ({goal}) is unknown! Please leave the goal empty."))
            success = False
        return success
