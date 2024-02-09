from rospy import logerr
from unified_planning.model.htn import Task, Method
from unified_planning.shortcuts import Not, Equals
from april_krem.domains.SLVR_components import (
    Item,
    ArmPose,
    Status,
    Color,
    Epic,
    Actions,
    Environment,
)
from up_esb.bridge import Bridge


class SLVRDomain(Bridge):
    def __init__(
        self, krem_logging, temporal: bool = False, use_case: str = "uc5"
    ) -> None:
        Bridge.__init__(self)

        self._env = Environment(krem_logging, use_case)

        # Create types for planning based on class types
        self.create_types([Item, ArmPose, Status, Color, Epic])
        type_item = self.get_type(Item)
        type_color = self.get_type(Color)

        # Create fluents for planning
        self.epic_complete = self.create_fluent_from_function(self._env.epic_complete)
        self.epic_active = self.create_fluent_from_function(self._env.epic_active)

        self.holding = self.create_fluent_from_function(self._env.holding)

        self.current_arm_pose = self.create_fluent_from_function(
            self._env.current_arm_pose
        )
        self.current_cable_color = self.create_fluent_from_function(
            self._env.current_cable_color
        )
        self.perceived_item = self.create_fluent_from_function(self._env.perceived_item)
        self.item_status_known = self.create_fluent_from_function(
            self._env.item_status_known
        )
        self.cable_color_available = self.create_fluent_from_function(
            self._env.cable_color_available
        )
        self.cable_soldered = self.create_fluent_from_function(self._env.cable_soldered)
        self.status_of_item = self.create_fluent_from_function(self._env.status_of_item)
        self.pallet_is_available = self.create_fluent_from_function(
            self._env.pallet_is_available
        )
        self.cover_is_available = self.create_fluent_from_function(
            self._env.cover_is_available
        )
        self.propeller_is_available = self.create_fluent_from_function(
            self._env.propeller_is_available
        )
        self.cable_pose_known = self.create_fluent_from_function(
            self._env.cable_pose_known
        )
        self.cover_pose_known = self.create_fluent_from_function(
            self._env.cover_pose_known
        )
        self.cover_is_leveled = self.create_fluent_from_function(
            self._env.cover_is_leveled
        )
        self.cover_is_assembled = self.create_fluent_from_function(
            self._env.cover_is_assembled
        )
        self.propeller_is_assembled = self.create_fluent_from_function(
            self._env.propeller_is_assembled
        )
        self.space_in_reject_box = self.create_fluent_from_function(
            self._env.space_in_reject_box
        )

        # Create objects for both planning and execution
        self.create_enum_objects(Item)
        self.nothing = self.objects[Item.nothing.name]
        self.cable = self.objects[Item.cable.name]
        self.cover = self.objects[Item.cover.name]
        self.propeller = self.objects[Item.propeller.name]

        self.create_enum_objects(ArmPose)
        self.unknown_pose = self.objects[ArmPose.unknown.name]
        self.home = self.objects[ArmPose.home.name]
        self.arm_up = self.objects[ArmPose.arm_up.name]
        self.over_reject_box = self.objects[ArmPose.over_reject_box.name]
        self.over_cable_dispenser = self.objects[ArmPose.over_cable_dispenser.name]
        self.over_cable_station = self.objects[ArmPose.over_cable_station.name]
        self.soldering_pose = self.objects[ArmPose.soldering_pose.name]
        self.over_feeding_conveyor = self.objects[ArmPose.over_feeding_conveyor.name]
        self.cover_transition_pose = self.objects[ArmPose.cover_transition_pose.name]
        self.over_cover_station = self.objects[ArmPose.over_cover_station.name]
        self.propeller_transition_pose = self.objects[
            ArmPose.propeller_transition_pose.name
        ]
        self.over_propeller_station = self.objects[ArmPose.over_propeller_station.name]

        self.create_enum_objects(Status)
        self.ok = self.objects[Status.ok.name]
        self.nok = self.objects[Status.nok.name]

        self.create_enum_objects(Color)
        self.red = self.objects[Color.red.name]
        self.blue = self.objects[Color.blue.name]
        self.brown = self.objects[Color.brown.name]
        self.white = self.objects[Color.white.name]

        self.create_enum_objects(Epic)
        self.epic2 = self.objects[Epic.epic2.name]
        self.epic3 = self.objects[Epic.epic3.name]
        self.epic4 = self.objects[Epic.epic4.name]

        # Create actions for planning
        self._create_domain_actions(temporal)

        # Tasks
        # Epic 2
        self.t_get_cable = Task("t_get_cable", cable=type_item)
        self.t_pick_cable = Task("t_pick_cable", cable=type_item)
        self.t_solder_cable = Task("t_solder_cable", cable=type_item)
        self.t_solder_all_cables = Task("t_solder_all_cables", cable=type_item)

        # Epic 3
        self.t_get_cover = Task("t_get_cover", cover=type_item)
        self.t_pick_cover = Task("t_pick_cover", cover=type_item)
        self.t_put_cover = Task("t_put_cover", cover=type_item)
        self.t_assemble_cover = Task("t_assemble_cover", cover=type_item)

        # Epic 4
        self.t_get_propeller = Task("t_get_propeller", propeller=type_item)
        self.t_pick_propeller = Task("t_pick_propeller", propeller=type_item)
        self.t_put_propeller = Task("t_put_propeller", propeller=type_item)
        self.t_assemble_propeller = Task("t_assemble_propeller", propeller=type_item)

        # Combined
        self.t_assemble_blower = Task("t_assemble_blower")

        self.tasks = (
            self.t_get_cable,
            self.t_pick_cable,
            self.t_solder_cable,
            self.t_solder_all_cables,
            self.t_get_cover,
            self.t_pick_cover,
            self.t_put_cover,
            self.t_assemble_cover,
            self.t_get_propeller,
            self.t_pick_propeller,
            self.t_put_propeller,
            self.t_assemble_propeller,
            self.t_assemble_blower,
        )

        # Methods

        #
        # EPIC 2
        #

        # GET CABLE
        # already at home pose, pallet available, cables available, perceive cables
        self.get_cable_perceive = Method(
            "get_cable_perceive", cable=type_item, color=type_color
        )
        self.get_cable_perceive.set_task(
            self.t_get_cable, self.get_cable_perceive.cable
        )
        self.get_cable_perceive.add_precondition(self.current_arm_pose(self.home))
        self.get_cable_perceive.add_precondition(self.pallet_is_available())
        self.get_cable_perceive.add_precondition(
            Not(self.perceived_item(self.get_cable_perceive.cable))
        )
        self.get_cable_perceive.add_precondition(
            self.current_cable_color(self.get_cable_perceive.color)
        )
        self.get_cable_perceive.add_precondition(
            self.cable_color_available(self.get_cable_perceive.color)
        )
        self.get_cable_perceive.add_precondition(self.holding(self.nothing))
        self.get_cable_perceive.add_precondition(
            Not(self.cable_soldered(self.get_cable_perceive.color))
        )
        self.get_cable_perceive.add_subtask(
            self.perceive_item, self.get_cable_perceive.cable
        )

        # move to home pose, pallet available, cables available, perceive them
        self.get_cable_partial = Method(
            "get_cable_partial", cable=type_item, color=type_color
        )
        self.get_cable_partial.set_task(self.t_get_cable, self.get_cable_partial.cable)
        self.get_cable_partial.add_precondition(Not(self.current_arm_pose(self.home)))
        self.get_cable_partial.add_precondition(self.pallet_is_available())
        self.get_cable_partial.add_precondition(
            Not(self.perceived_item(self.get_cable_partial.cable))
        )
        self.get_cable_partial.add_precondition(
            self.current_cable_color(self.get_cable_partial.color)
        )
        self.get_cable_partial.add_precondition(
            self.cable_color_available(self.get_cable_partial.color)
        )
        self.get_cable_partial.add_precondition(self.holding(self.nothing))
        self.get_cable_partial.add_precondition(
            Not(self.cable_soldered(self.get_cable_partial.color))
        )
        s1 = self.get_cable_partial.add_subtask(self.move_arm, self.home)
        s2 = self.get_cable_partial.add_subtask(
            self.perceive_item, self.get_cable_partial.cable
        )
        self.get_cable_partial.set_ordered(s1, s2)

        # move to home pose, pallet available, cable missing, perceive them
        self.get_cable_redo = Method(
            "get_cable_redo", cable=type_item, color=type_color
        )
        self.get_cable_redo.set_task(self.t_get_cable, self.get_cable_redo.cable)
        self.get_cable_redo.add_precondition(Not(self.current_arm_pose(self.home)))
        self.get_cable_redo.add_precondition(self.pallet_is_available())
        self.get_cable_redo.add_precondition(
            Not(self.perceived_item(self.get_cable_redo.cable))
        )
        self.get_cable_redo.add_precondition(
            self.current_cable_color(self.get_cable_redo.color)
        )
        self.get_cable_redo.add_precondition(
            Not(self.cable_color_available(self.get_cable_redo.color))
        )
        self.get_cable_redo.add_precondition(self.holding(self.nothing))
        self.get_cable_redo.add_precondition(
            Not(self.cable_soldered(self.get_cable_redo.color))
        )
        s1 = self.get_cable_redo.add_subtask(self.move_arm, self.home)
        s2 = self.get_cable_redo.add_subtask(
            self.get_next_cables, self.get_cable_redo.color
        )
        s3 = self.get_cable_redo.add_subtask(
            self.perceive_item, self.get_cable_redo.cable
        )
        self.get_cable_redo.set_ordered(s1, s2, s3)

        # already at home pose, pallet available, cable missing, perceive them
        self.get_cable_cables = Method(
            "get_cable_cables", cable=type_item, color=type_color
        )
        self.get_cable_cables.set_task(self.t_get_cable, self.get_cable_cables.cable)
        self.get_cable_cables.add_precondition(self.current_arm_pose(self.home))
        self.get_cable_cables.add_precondition(self.pallet_is_available())
        self.get_cable_cables.add_precondition(
            Not(self.perceived_item(self.get_cable_cables.cable))
        )
        self.get_cable_cables.add_precondition(
            self.current_cable_color(self.get_cable_cables.color)
        )
        self.get_cable_cables.add_precondition(
            Not(self.cable_color_available(self.get_cable_cables.color))
        )
        self.get_cable_cables.add_precondition(self.holding(self.nothing))
        self.get_cable_cables.add_precondition(
            Not(self.cable_soldered(self.get_cable_cables.color))
        )
        s1 = self.get_cable_cables.add_subtask(
            self.get_next_cables, self.get_cable_cables.color
        )
        s2 = self.get_cable_cables.add_subtask(
            self.perceive_item, self.get_cable_cables.cable
        )
        self.get_cable_cables.set_ordered(s1, s2)

        # already at home pose, pallet unavailable, cable missing, perceive them
        self.get_cable_pallet = Method(
            "get_cable_pallet", cable=type_item, color=type_color
        )
        self.get_cable_pallet.set_task(self.t_get_cable, self.get_cable_pallet.cable)
        self.get_cable_pallet.add_precondition(self.current_arm_pose(self.home))
        self.get_cable_pallet.add_precondition(Not(self.pallet_is_available()))
        self.get_cable_pallet.add_precondition(
            Not(self.perceived_item(self.get_cable_pallet.cable))
        )
        self.get_cable_pallet.add_precondition(
            self.current_cable_color(self.get_cable_pallet.color)
        )
        self.get_cable_pallet.add_precondition(
            Not(self.cable_color_available(self.get_cable_pallet.color))
        )
        self.get_cable_pallet.add_precondition(self.holding(self.nothing))
        self.get_cable_pallet.add_precondition(
            Not(self.cable_soldered(self.get_cable_pallet.color))
        )
        s1 = self.get_cable_pallet.add_subtask(self.get_next_pallet)
        s2 = self.get_cable_pallet.add_subtask(
            self.get_next_cables, self.get_cable_pallet.color
        )
        s3 = self.get_cable_pallet.add_subtask(
            self.perceive_item, self.get_cable_pallet.cable
        )
        self.get_cable_pallet.set_ordered(s1, s2, s3)

        # move to home pose, get new pallet, new cables and perceive them
        self.get_cable_full = Method(
            "get_cables_full", cable=type_item, color=type_color
        )
        self.get_cable_full.set_task(self.t_get_cable, self.get_cable_full.cable)
        self.get_cable_full.add_precondition(Not(self.current_arm_pose(self.home)))
        self.get_cable_full.add_precondition(Not(self.pallet_is_available()))
        self.get_cable_full.add_precondition(
            Not(self.perceived_item(self.get_cable_full.cable))
        )
        self.get_cable_full.add_precondition(
            self.current_cable_color(self.get_cable_full.color)
        )
        self.get_cable_full.add_precondition(
            Not(self.cable_color_available(self.get_cable_full.color))
        )
        self.get_cable_full.add_precondition(self.holding(self.nothing))
        self.get_cable_full.add_precondition(
            Not(self.cable_soldered(self.get_cable_full.color))
        )
        s1 = self.get_cable_full.add_subtask(self.move_arm, self.home)
        s2 = self.get_cable_full.add_subtask(self.get_next_pallet)
        s3 = self.get_cable_full.add_subtask(
            self.get_next_cables, self.get_cable_full.color
        )
        s4 = self.get_cable_full.add_subtask(
            self.perceive_item, self.get_cable_full.cable
        )
        self.get_cable_full.set_ordered(s1, s2, s3, s4)

        # PICK CABLE
        self.pick_cable_inspect = Method("pick_cable_inspect", cable=type_item)
        self.pick_cable_inspect.set_task(
            self.t_pick_cable, self.pick_cable_inspect.cable
        )
        self.pick_cable_inspect.add_precondition(
            self.holding(self.pick_cable_inspect.cable)
        )
        self.pick_cable_inspect.add_precondition(self.pallet_is_available())
        self.pick_cable_inspect.add_precondition(
            self.current_arm_pose(self.over_cable_station)
        )
        self.pick_cable_inspect.add_precondition(self.cable_pose_known())
        self.pick_cable_inspect.add_subtask(self.inspect, self.pick_cable_inspect.cable)

        # over station
        self.pick_cable_over_station = Method(
            "pick_cable_over_station", cable=type_item
        )
        self.pick_cable_over_station.set_task(
            self.t_pick_cable, self.pick_cable_over_station.cable
        )
        self.pick_cable_over_station.add_precondition(
            self.holding(self.pick_cable_over_station.cable)
        )
        self.pick_cable_over_station.add_precondition(self.pallet_is_available())
        self.pick_cable_over_station.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.pick_cable_over_station.add_precondition(self.cable_pose_known())
        s1 = self.pick_cable_over_station.add_subtask(
            self.move_arm, self.over_cable_station
        )
        s2 = self.pick_cable_over_station.add_subtask(
            self.inspect, self.pick_cable_over_station.cable
        )
        self.pick_cable_over_station.set_ordered(s1, s2)

        # get cable pose
        self.pick_cable_pose = Method("pick_cable_pose", cable=type_item)
        self.pick_cable_pose.set_task(self.t_pick_cable, self.pick_cable_pose.cable)
        self.pick_cable_pose.add_precondition(self.holding(self.pick_cable_pose.cable))
        self.pick_cable_pose.add_precondition(self.pallet_is_available())
        self.pick_cable_pose.add_precondition(self.current_arm_pose(self.arm_up))
        self.pick_cable_pose.add_precondition(Not(self.cable_pose_known()))
        s1 = self.pick_cable_pose.add_subtask(self.get_cable_pose)
        s2 = self.pick_cable_pose.add_subtask(self.move_arm, self.over_cable_station)
        s3 = self.pick_cable_pose.add_subtask(self.inspect, self.pick_cable_pose.cable)
        self.pick_cable_pose.set_ordered(s1, s2, s3)

        # arm up
        self.pick_cable_arm_up = Method("pick_cable_arm_up", cable=type_item)
        self.pick_cable_arm_up.set_task(self.t_pick_cable, self.pick_cable_arm_up.cable)
        self.pick_cable_arm_up.add_precondition(
            self.holding(self.pick_cable_arm_up.cable)
        )
        self.pick_cable_arm_up.add_precondition(self.pallet_is_available())
        self.pick_cable_arm_up.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.pick_cable_arm_up.add_precondition(Not(self.cable_pose_known()))
        s1 = self.pick_cable_arm_up.add_subtask(self.move_arm, self.arm_up)
        s2 = self.pick_cable_arm_up.add_subtask(self.get_cable_pose)
        s3 = self.pick_cable_arm_up.add_subtask(self.move_arm, self.over_cable_station)
        s4 = self.pick_cable_arm_up.add_subtask(
            self.inspect, self.pick_cable_arm_up.cable
        )
        self.pick_cable_arm_up.set_ordered(s1, s2, s3, s4)

        # pick
        self.pick_cable_pick = Method(
            "pick_cable_pick", cable=type_item, color=type_color
        )
        self.pick_cable_pick.set_task(self.t_pick_cable, self.pick_cable_pick.cable)
        self.pick_cable_pick.add_precondition(
            self.current_cable_color(self.pick_cable_pick.color)
        )
        self.pick_cable_pick.add_precondition(
            self.cable_color_available(self.pick_cable_pick.color)
        )
        self.pick_cable_pick.add_precondition(
            self.perceived_item(self.pick_cable_pick.cable)
        )
        self.pick_cable_pick.add_precondition(self.holding(self.nothing))
        self.pick_cable_pick.add_precondition(self.pallet_is_available())
        self.pick_cable_pick.add_precondition(
            self.current_arm_pose(self.over_cable_dispenser)
        )
        self.pick_cable_pick.add_precondition(Not(self.cable_pose_known()))
        s1 = self.pick_cable_pick.add_subtask(
            self.pick_cable, self.pick_cable_pick.cable, self.pick_cable_pick.color
        )
        s2 = self.pick_cable_pick.add_subtask(self.move_arm, self.arm_up)
        s3 = self.pick_cable_pick.add_subtask(self.get_cable_pose)
        s4 = self.pick_cable_pick.add_subtask(self.move_arm, self.over_cable_station)
        s5 = self.pick_cable_pick.add_subtask(self.inspect, self.pick_cable_pick.cable)
        self.pick_cable_pick.set_ordered(s1, s2, s3, s4, s5)

        # move arm over cable dispenser, pick cable, move arm up, move arm over soldering station, inspect cable
        self.pick_cable_full = Method(
            "pick_cable_full", cable=type_item, color=type_color
        )
        self.pick_cable_full.set_task(self.t_pick_cable, self.pick_cable_full.cable)
        self.pick_cable_full.add_precondition(
            self.current_cable_color(self.pick_cable_full.color)
        )
        self.pick_cable_full.add_precondition(
            self.cable_color_available(self.pick_cable_full.color)
        )
        self.pick_cable_full.add_precondition(
            self.perceived_item(self.pick_cable_full.cable)
        )
        self.pick_cable_full.add_precondition(self.holding(self.nothing))
        self.pick_cable_full.add_precondition(self.pallet_is_available())
        self.pick_cable_full.add_precondition(self.current_arm_pose(self.home))
        self.pick_cable_full.add_precondition(Not(self.cable_pose_known()))
        s1 = self.pick_cable_full.add_subtask(self.move_arm, self.over_cable_dispenser)
        s2 = self.pick_cable_full.add_subtask(
            self.pick_cable, self.pick_cable_full.cable, self.pick_cable_full.color
        )
        s3 = self.pick_cable_full.add_subtask(self.move_arm, self.arm_up)
        s4 = self.pick_cable_full.add_subtask(self.get_cable_pose)
        s5 = self.pick_cable_full.add_subtask(self.move_arm, self.over_cable_station)
        s6 = self.pick_cable_full.add_subtask(self.inspect, self.pick_cable_full.cable)
        self.pick_cable_full.set_ordered(s1, s2, s3, s4, s5, s6)

        # SOLDER CABLE
        self.solder_cable_move_arm = Method(
            "solder_cable_move_arm", cable=type_item, color=type_color
        )
        self.solder_cable_move_arm.set_task(
            self.t_solder_cable, self.solder_cable_move_arm.cable
        )
        self.solder_cable_move_arm.add_precondition(
            self.current_arm_pose(self.soldering_pose)
        )
        self.solder_cable_move_arm.add_precondition(self.pallet_is_available())
        self.solder_cable_move_arm.add_precondition(self.holding(self.nothing))
        self.solder_cable_move_arm.add_precondition(
            self.cable_soldered(self.solder_cable_move_arm.color)
        )
        self.solder_cable_move_arm.add_subtask(self.move_arm_cable_end)

        # release
        self.solder_cable_release = Method(
            "solder_cable_release", cable=type_item, color=type_color
        )
        self.solder_cable_release.set_task(
            self.t_solder_cable, self.solder_cable_release.cable
        )
        self.solder_cable_release.add_precondition(
            self.current_cable_color(self.solder_cable_release.color)
        )
        self.solder_cable_release.add_precondition(
            self.item_status_known(self.solder_cable_release.cable)
        )
        self.solder_cable_release.add_precondition(
            self.status_of_item(self.solder_cable_release.cable, self.ok)
        )
        self.solder_cable_release.add_precondition(
            self.current_arm_pose(self.soldering_pose)
        )
        self.solder_cable_release.add_precondition(self.pallet_is_available())
        self.solder_cable_release.add_precondition(
            self.holding(self.solder_cable_release.cable)
        )
        self.solder_cable_release.add_precondition(
            self.cable_soldered(self.solder_cable_release.color)
        )
        s1 = self.solder_cable_release.add_subtask(
            self.release_cable, self.solder_cable_release.color
        )
        s2 = self.solder_cable_release.add_subtask(self.move_arm_cable_end)
        self.solder_cable_release.set_ordered(s1, s2)

        # wait
        self.solder_cable_wait = Method(
            "solder_cable_wait", cable=type_item, color=type_color
        )
        self.solder_cable_wait.set_task(
            self.t_solder_cable, self.solder_cable_wait.cable
        )
        self.solder_cable_wait.add_precondition(
            self.current_cable_color(self.solder_cable_wait.color)
        )
        self.solder_cable_wait.add_precondition(
            self.item_status_known(self.solder_cable_wait.cable)
        )
        self.solder_cable_wait.add_precondition(
            self.status_of_item(self.solder_cable_wait.cable, self.ok)
        )
        self.solder_cable_wait.add_precondition(
            self.current_arm_pose(self.soldering_pose)
        )
        self.solder_cable_wait.add_precondition(self.pallet_is_available())
        self.solder_cable_wait.add_precondition(
            self.holding(self.solder_cable_wait.cable)
        )
        self.solder_cable_wait.add_precondition(
            Not(self.cable_soldered(self.solder_cable_wait.color))
        )
        s1 = self.solder_cable_wait.add_subtask(
            self.wait_for_soldering, self.solder_cable_wait.color
        )
        s2 = self.solder_cable_wait.add_subtask(
            self.release_cable, self.solder_cable_wait.color
        )
        s3 = self.solder_cable_wait.add_subtask(self.move_arm_cable_end)
        self.solder_cable_wait.set_ordered(s1, s2, s3)

        # full
        self.solder_cable_full = Method(
            "solder_cable_full", cable=type_item, color=type_color
        )
        self.solder_cable_full.set_task(
            self.t_solder_cable, self.solder_cable_full.cable
        )
        self.solder_cable_full.add_precondition(
            self.current_cable_color(self.solder_cable_full.color)
        )
        self.solder_cable_full.add_precondition(
            self.item_status_known(self.solder_cable_full.cable)
        )
        self.solder_cable_full.add_precondition(
            self.status_of_item(self.solder_cable_full.cable, self.ok)
        )
        self.solder_cable_full.add_precondition(
            self.current_arm_pose(self.over_cable_station)
        )
        self.solder_cable_full.add_precondition(self.pallet_is_available())
        self.solder_cable_full.add_precondition(
            self.holding(self.solder_cable_full.cable)
        )
        self.solder_cable_full.add_precondition(
            Not(self.cable_soldered(self.solder_cable_full.color))
        )
        s1 = self.solder_cable_full.add_subtask(self.move_arm, self.soldering_pose)
        s2 = self.solder_cable_full.add_subtask(
            self.wait_for_soldering, self.solder_cable_full.color
        )
        s3 = self.solder_cable_full.add_subtask(
            self.release_cable, self.solder_cable_full.color
        )
        s4 = self.solder_cable_full.add_subtask(self.move_arm_cable_end)
        self.solder_cable_full.set_ordered(s1, s2, s3, s4)

        # NOK Reject
        self.solder_cable_reject = Method(
            "solder_cable_reject", cable=type_item, color=type_color
        )
        self.solder_cable_reject.set_task(
            self.t_solder_cable, self.solder_cable_reject.cable
        )
        self.solder_cable_reject.add_precondition(
            self.current_cable_color(self.solder_cable_reject.color)
        )
        self.solder_cable_reject.add_precondition(
            self.item_status_known(self.solder_cable_reject.cable)
        )
        self.solder_cable_reject.add_precondition(
            self.status_of_item(self.solder_cable_reject.cable, self.nok)
        )
        self.solder_cable_reject.add_precondition(
            self.current_arm_pose(self.over_reject_box)
        )
        self.solder_cable_reject.add_precondition(
            self.holding(self.solder_cable_reject.cable)
        )
        self.solder_cable_reject.add_precondition(
            Not(self.cable_soldered(self.solder_cable_reject.color))
        )
        self.solder_cable_reject.add_precondition(self.space_in_reject_box())
        self.solder_cable_reject.add_subtask(
            self.reject_item, self.solder_cable_reject.cable
        )

        # reject full
        self.solder_cable_reject_full = Method(
            "solder_cable_reject_full", cable=type_item, color=type_color
        )
        self.solder_cable_reject_full.set_task(
            self.t_solder_cable, self.solder_cable_reject_full.cable
        )
        self.solder_cable_reject_full.add_precondition(
            self.current_cable_color(self.solder_cable_reject_full.color)
        )
        self.solder_cable_reject_full.add_precondition(
            self.item_status_known(self.solder_cable_reject_full.cable)
        )
        self.solder_cable_reject_full.add_precondition(
            self.status_of_item(self.solder_cable_reject_full.cable, self.nok)
        )
        self.solder_cable_reject_full.add_precondition(
            self.current_arm_pose(self.over_cable_station)
        )
        self.solder_cable_reject_full.add_precondition(
            self.holding(self.solder_cable_reject_full.cable)
        )
        self.solder_cable_reject_full.add_precondition(
            Not(self.cable_soldered(self.solder_cable_reject_full.color))
        )
        self.solder_cable_reject_full.add_precondition(self.space_in_reject_box())
        s1 = self.solder_cable_reject_full.add_subtask(
            self.move_arm, self.over_reject_box
        )
        s2 = self.solder_cable_reject_full.add_subtask(
            self.reject_item, self.solder_cable_reject_full.cable
        )
        self.solder_cable_reject_full.set_ordered(s1, s2)

        # SOLDER ALL CABLES
        # get cables
        self.solder_all_cables_get = Method("solder_all_cables_get", cable=type_item)
        self.solder_all_cables_get.set_task(
            self.t_solder_all_cables, self.solder_all_cables_get.cable
        )
        self.solder_all_cables_get.add_precondition(self.epic_active(self.epic2))
        self.solder_all_cables_get.add_precondition(Not(self.epic_complete(self.epic2)))
        self.solder_all_cables_get.add_precondition(
            Not(self.perceived_item(self.solder_all_cables_get.cable))
        )
        self.solder_all_cables_get.add_subtask(
            self.t_get_cable, self.solder_all_cables_get.cable
        )

        # pick cable
        self.solder_all_cables_pick = Method("solder_all_cables_pick", cable=type_item)
        self.solder_all_cables_pick.set_task(
            self.t_solder_all_cables, self.solder_all_cables_pick.cable
        )
        self.solder_all_cables_pick.add_precondition(self.epic_active(self.epic2))
        self.solder_all_cables_pick.add_precondition(
            Not(self.epic_complete(self.epic2))
        )
        self.solder_all_cables_pick.add_precondition(
            Not(self.item_status_known(self.solder_all_cables_pick.cable))
        )
        self.solder_all_cables_pick.add_subtask(
            self.t_pick_cable, self.solder_all_cables_pick.cable
        )

        # solder cable
        self.solder_all_cables_solder = Method(
            "solder_all_cables_solder", cable=type_item
        )
        self.solder_all_cables_solder.set_task(
            self.t_solder_all_cables, self.solder_all_cables_solder.cable
        )
        self.solder_all_cables_solder.add_precondition(self.epic_active(self.epic2))
        self.solder_all_cables_solder.add_precondition(
            Not(self.epic_complete(self.epic2))
        )
        self.solder_all_cables_solder.add_precondition(
            self.item_status_known(self.solder_all_cables_solder.cable)
        )
        self.solder_all_cables_solder.add_subtask(
            self.t_solder_cable, self.solder_all_cables_solder.cable
        )

        self.solder_all_cables_switch = Method(
            "solder_all_cables_switch", cable=type_item
        )
        self.solder_all_cables_switch.set_task(
            self.t_solder_all_cables, self.solder_all_cables_switch.cable
        )
        self.solder_all_cables_switch.add_precondition(
            Not(self.epic_active(self.epic2))
        )
        self.solder_all_cables_switch.add_precondition(
            Not(self.epic_complete(self.epic2))
        )
        self.solder_all_cables_switch.add_subtask(self.switch_to_epic, self.epic2)

        #
        # EPIC 3
        #

        # GET COVER
        # perceive
        self.get_cover_perceive = Method("get_cover_perceive", cover=type_item)
        self.get_cover_perceive.set_task(
            self.t_get_cover, self.get_cover_perceive.cover
        )
        self.get_cover_perceive.add_precondition(self.current_arm_pose(self.home))
        self.get_cover_perceive.add_precondition(self.cover_is_available())
        self.get_cover_perceive.add_precondition(self.pallet_is_available())
        self.get_cover_perceive.add_precondition(
            Not(self.perceived_item(self.get_cover_perceive.cover))
        )
        self.get_cover_perceive.add_precondition(self.holding(self.nothing))
        self.get_cover_perceive.add_subtask(
            self.perceive_item, self.get_cover_perceive.cover
        )

        # get cover
        self.get_cover_new = Method("get_cover_new", cover=type_item)
        self.get_cover_new.set_task(self.t_get_cover, self.get_cover_new.cover)
        self.get_cover_new.add_precondition(self.current_arm_pose(self.home))
        self.get_cover_new.add_precondition(Not(self.cover_is_available()))
        self.get_cover_new.add_precondition(self.pallet_is_available())
        self.get_cover_new.add_precondition(
            Not(self.perceived_item(self.get_cover_new.cover))
        )
        self.get_cover_new.add_precondition(self.holding(self.nothing))
        s1 = self.get_cover_new.add_subtask(self.get_next_cover)
        s2 = self.get_cover_new.add_subtask(
            self.perceive_item, self.get_cover_new.cover
        )
        self.get_cover_new.set_ordered(s1, s2)

        # redo
        self.get_cover_redo = Method("get_cover_redo", cover=type_item)
        self.get_cover_redo.set_task(self.t_get_cover, self.get_cover_redo.cover)
        self.get_cover_redo.add_precondition(Not(self.current_arm_pose(self.home)))
        self.get_cover_redo.add_precondition(Not(self.cover_is_available()))
        self.get_cover_redo.add_precondition(self.pallet_is_available())
        self.get_cover_redo.add_precondition(
            Not(self.perceived_item(self.get_cover_redo.cover))
        )
        self.get_cover_redo.add_precondition(self.holding(self.nothing))
        s1 = self.get_cover_redo.add_subtask(self.move_arm, self.home)
        s2 = self.get_cover_redo.add_subtask(self.get_next_cover)
        s3 = self.get_cover_redo.add_subtask(
            self.perceive_item, self.get_cover_redo.cover
        )
        self.get_cover_redo.set_ordered(s1, s2, s3)

        # get pallet
        self.get_cover_pallet = Method("get_cover_pallet", cover=type_item)
        self.get_cover_pallet.set_task(self.t_get_cover, self.get_cover_pallet.cover)
        self.get_cover_pallet.add_precondition(self.current_arm_pose(self.home))
        self.get_cover_pallet.add_precondition(Not(self.cover_is_available()))
        self.get_cover_pallet.add_precondition(Not(self.pallet_is_available()))
        self.get_cover_pallet.add_precondition(
            Not(self.perceived_item(self.get_cover_pallet.cover))
        )
        self.get_cover_pallet.add_precondition(self.holding(self.nothing))
        s1 = self.get_cover_pallet.add_subtask(self.get_next_pallet)
        s2 = self.get_cover_pallet.add_subtask(self.get_next_cover)
        s3 = self.get_cover_pallet.add_subtask(
            self.perceive_item, self.get_cover_pallet.cover
        )
        self.get_cover_pallet.set_ordered(s1, s2, s3)

        # full
        self.get_cover_full = Method("get_cover_full", cover=type_item)
        self.get_cover_full.set_task(self.t_get_cover, self.get_cover_full.cover)
        self.get_cover_full.add_precondition(Not(self.current_arm_pose(self.home)))
        self.get_cover_full.add_precondition(Not(self.cover_is_available()))
        self.get_cover_full.add_precondition(Not(self.pallet_is_available()))
        self.get_cover_full.add_precondition(
            Not(self.perceived_item(self.get_cover_full.cover))
        )
        self.get_cover_full.add_precondition(self.holding(self.nothing))
        s1 = self.get_cover_full.add_subtask(self.move_arm, self.home)
        s2 = self.get_cover_full.add_subtask(self.get_next_pallet)
        s3 = self.get_cover_full.add_subtask(self.get_next_cover)
        s4 = self.get_cover_full.add_subtask(
            self.perceive_item, self.get_cover_full.cover
        )
        self.get_cover_full.set_ordered(s1, s2, s3, s4)

        # PICK COVER
        # inspect
        self.pick_cover_inspect = Method("pick_cover_inspect", cover=type_item)
        self.pick_cover_inspect.set_task(
            self.t_pick_cover, self.pick_cover_inspect.cover
        )
        self.pick_cover_inspect.add_precondition(
            self.current_arm_pose(self.over_cover_station)
        )
        self.pick_cover_inspect.add_precondition(self.pallet_is_available())
        self.pick_cover_inspect.add_precondition(
            self.holding(self.pick_cover_inspect.cover)
        )
        self.pick_cover_inspect.add_precondition(self.cover_pose_known())
        self.pick_cover_inspect.add_precondition(self.cover_is_leveled())
        self.pick_cover_inspect.add_precondition(
            Not(self.item_status_known(self.pick_cover_inspect.cover))
        )
        self.pick_cover_inspect.add_subtask(self.inspect, self.pick_cover_inspect.cover)

        # cover station
        self.pick_cover_station = Method("pick_cover_station", cover=type_item)
        self.pick_cover_station.set_task(
            self.t_pick_cover, self.pick_cover_station.cover
        )
        self.pick_cover_station.add_precondition(
            self.current_arm_pose(self.cover_transition_pose)
        )
        self.pick_cover_station.add_precondition(self.pallet_is_available())
        self.pick_cover_station.add_precondition(
            self.holding(self.pick_cover_station.cover)
        )
        self.pick_cover_station.add_precondition(self.cover_pose_known())
        self.pick_cover_station.add_precondition(self.cover_is_leveled())
        self.pick_cover_station.add_precondition(
            Not(self.item_status_known(self.pick_cover_station.cover))
        )
        s1 = self.pick_cover_station.add_subtask(self.move_arm, self.over_cover_station)
        s2 = self.pick_cover_station.add_subtask(
            self.inspect, self.pick_cover_station.cover
        )
        self.pick_cover_station.set_ordered(s1, s2)

        # cover transition pose
        self.pick_cover_transition = Method("pick_cover_transition", cover=type_item)
        self.pick_cover_transition.set_task(
            self.t_pick_cover, self.pick_cover_transition.cover
        )
        self.pick_cover_transition.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.pick_cover_transition.add_precondition(self.pallet_is_available())
        self.pick_cover_transition.add_precondition(
            self.holding(self.pick_cover_transition.cover)
        )
        self.pick_cover_transition.add_precondition(self.cover_pose_known())
        self.pick_cover_transition.add_precondition(self.cover_is_leveled())
        self.pick_cover_transition.add_precondition(
            Not(self.item_status_known(self.pick_cover_transition.cover))
        )
        s1 = self.pick_cover_transition.add_subtask(
            self.move_arm, self.cover_transition_pose
        )
        s2 = self.pick_cover_transition.add_subtask(
            self.move_arm, self.over_cover_station
        )
        s3 = self.pick_cover_transition.add_subtask(
            self.inspect, self.pick_cover_transition.cover
        )
        self.pick_cover_transition.set_ordered(s1, s2, s3)

        # level cover
        self.pick_cover_level = Method("pick_cover_level", cover=type_item)
        self.pick_cover_level.set_task(self.t_pick_cover, self.pick_cover_level.cover)
        self.pick_cover_level.add_precondition(self.current_arm_pose(self.unknown_pose))
        self.pick_cover_level.add_precondition(self.pallet_is_available())
        self.pick_cover_level.add_precondition(
            self.holding(self.pick_cover_level.cover)
        )
        self.pick_cover_level.add_precondition(self.cover_pose_known())
        self.pick_cover_level.add_precondition(Not(self.cover_is_leveled()))
        self.pick_cover_level.add_precondition(
            Not(self.item_status_known(self.pick_cover_level.cover))
        )
        s1 = self.pick_cover_level.add_subtask(self.level_cover)
        s2 = self.pick_cover_level.add_subtask(
            self.move_arm, self.cover_transition_pose
        )
        s3 = self.pick_cover_level.add_subtask(self.move_arm, self.over_cover_station)
        s4 = self.pick_cover_level.add_subtask(
            self.inspect, self.pick_cover_level.cover
        )
        self.pick_cover_level.set_ordered(s1, s2, s3, s4)

        # get pose
        self.pick_cover_pose = Method("pick_cover_pose", cover=type_item)
        self.pick_cover_pose.set_task(self.t_pick_cover, self.pick_cover_pose.cover)
        self.pick_cover_pose.add_precondition(self.current_arm_pose(self.arm_up))
        self.pick_cover_pose.add_precondition(self.pallet_is_available())
        self.pick_cover_pose.add_precondition(self.holding(self.pick_cover_pose.cover))
        self.pick_cover_pose.add_precondition(Not(self.cover_pose_known()))
        self.pick_cover_pose.add_precondition(Not(self.cover_is_leveled()))
        self.pick_cover_pose.add_precondition(
            Not(self.item_status_known(self.pick_cover_pose.cover))
        )
        s1 = self.pick_cover_pose.add_subtask(self.get_cover_pose)
        s2 = self.pick_cover_pose.add_subtask(self.level_cover)
        s3 = self.pick_cover_pose.add_subtask(self.move_arm, self.cover_transition_pose)
        s4 = self.pick_cover_pose.add_subtask(self.move_arm, self.over_cover_station)
        s5 = self.pick_cover_pose.add_subtask(self.inspect, self.pick_cover_pose.cover)
        self.pick_cover_pose.set_ordered(s1, s2, s3, s4, s5)

        # arm up
        self.pick_cover_move_arm_up = Method("pick_cover_move_arm_up", cover=type_item)
        self.pick_cover_move_arm_up.set_task(self.t_pick_cover, self.pick_cover_move_arm_up.cover)
        self.pick_cover_move_arm_up.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.pick_cover_move_arm_up.add_precondition(self.pallet_is_available())
        self.pick_cover_move_arm_up.add_precondition(self.holding(self.pick_cover_move_arm_up.cover))
        self.pick_cover_move_arm_up.add_precondition(Not(self.cover_pose_known()))
        self.pick_cover_move_arm_up.add_precondition(Not(self.cover_is_leveled()))
        self.pick_cover_move_arm_up.add_precondition(
            Not(self.item_status_known(self.pick_cover_move_arm_up.cover))
        )
        s1 = self.pick_cover_move_arm_up.add_subtask(self.move_arm, self.arm_up)
        s2 = self.pick_cover_move_arm_up.add_subtask(self.get_cover_pose)
        s3 = self.pick_cover_move_arm_up.add_subtask(self.level_cover)
        s4 = self.pick_cover_move_arm_up.add_subtask(self.move_arm, self.cover_transition_pose)
        s5 = self.pick_cover_move_arm_up.add_subtask(self.move_arm, self.over_cover_station)
        s6 = self.pick_cover_move_arm_up.add_subtask(self.inspect, self.pick_cover_move_arm_up.cover)
        self.pick_cover_move_arm_up.set_ordered(s1, s2, s3, s4, s5, s6)

        # pick
        self.pick_cover_pick = Method("pick_cover_pick", cover=type_item)
        self.pick_cover_pick.set_task(self.t_pick_cover, self.pick_cover_pick.cover)
        self.pick_cover_pick.add_precondition(
            self.current_arm_pose(self.over_feeding_conveyor)
        )
        self.pick_cover_pick.add_precondition(self.cover_is_available())
        self.pick_cover_pick.add_precondition(self.pallet_is_available())
        self.pick_cover_pick.add_precondition(
            self.perceived_item(self.pick_cover_pick.cover)
        )
        self.pick_cover_pick.add_precondition(self.holding(self.nothing))
        self.pick_cover_pick.add_precondition(Not(self.cover_pose_known()))
        self.pick_cover_pick.add_precondition(Not(self.cover_is_leveled()))
        self.pick_cover_pick.add_precondition(
            Not(self.item_status_known(self.pick_cover_pick.cover))
        )
        s1 = self.pick_cover_pick.add_subtask(
            self.pick_cover, self.pick_cover_pick.cover
        )
        s2 = self.pick_cover_pick.add_subtask(self.move_arm, self.arm_up)
        s3 = self.pick_cover_pick.add_subtask(self.get_cover_pose)
        s4 = self.pick_cover_pick.add_subtask(self.level_cover)
        s5 = self.pick_cover_pick.add_subtask(self.move_arm, self.cover_transition_pose)
        s6 = self.pick_cover_pick.add_subtask(self.move_arm, self.over_cover_station)
        s7 = self.pick_cover_pick.add_subtask(self.inspect, self.pick_cover_pick.cover)
        self.pick_cover_pick.set_ordered(s1, s2, s3, s4, s5, s6, s7)

        # full
        self.pick_cover_full = Method("pick_cover_full", cover=type_item)
        self.pick_cover_full.set_task(self.t_pick_cover, self.pick_cover_full.cover)
        self.pick_cover_full.add_precondition(self.current_arm_pose(self.home))
        self.pick_cover_full.add_precondition(self.cover_is_available())
        self.pick_cover_full.add_precondition(self.pallet_is_available())
        self.pick_cover_full.add_precondition(
            self.perceived_item(self.pick_cover_full.cover)
        )
        self.pick_cover_full.add_precondition(self.holding(self.nothing))
        self.pick_cover_full.add_precondition(Not(self.cover_pose_known()))
        self.pick_cover_full.add_precondition(Not(self.cover_is_leveled()))
        self.pick_cover_full.add_precondition(
            Not(self.item_status_known(self.pick_cover_full.cover))
        )
        s1 = self.pick_cover_full.add_subtask(self.move_arm, self.over_feeding_conveyor)
        s2 = self.pick_cover_full.add_subtask(
            self.pick_cover, self.pick_cover_full.cover
        )
        s3 = self.pick_cover_full.add_subtask(self.move_arm, self.arm_up)
        s4 = self.pick_cover_full.add_subtask(self.get_cover_pose)
        s5 = self.pick_cover_full.add_subtask(self.level_cover)
        s6 = self.pick_cover_full.add_subtask(self.move_arm, self.cover_transition_pose)
        s7 = self.pick_cover_full.add_subtask(self.move_arm, self.over_cover_station)
        s8 = self.pick_cover_full.add_subtask(self.inspect, self.pick_cover_full.cover)
        self.pick_cover_full.set_ordered(s1, s2, s3, s4, s5, s6, s7, s8)

        # PUT COVER
        # wait
        self.put_cover_wait = Method("put_cover_wait", cover=type_item)
        self.put_cover_wait.set_task(self.t_put_cover, self.put_cover_wait.cover)
        self.put_cover_wait.add_precondition(self.current_arm_pose(self.unknown_pose))
        self.put_cover_wait.add_precondition(self.pallet_is_available())
        self.put_cover_wait.add_precondition(self.holding(self.nothing))
        self.put_cover_wait.add_precondition(
            self.item_status_known(self.put_cover_wait.cover)
        )
        self.put_cover_wait.add_precondition(
            self.status_of_item(self.put_cover_wait.cover, self.ok)
        )
        self.put_cover_wait.add_precondition(self.cover_is_assembled())
        self.put_cover_wait.add_subtask(self.wait_for_fixed_cover)

        # full
        self.put_cover_full = Method("put_cover_full", cover=type_item)
        self.put_cover_full.set_task(self.t_put_cover, self.put_cover_full.cover)
        self.put_cover_full.add_precondition(
            self.current_arm_pose(self.over_cover_station)
        )
        self.put_cover_full.add_precondition(self.pallet_is_available())
        self.put_cover_full.add_precondition(self.holding(self.put_cover_full.cover))
        self.put_cover_full.add_precondition(
            self.item_status_known(self.put_cover_full.cover)
        )
        self.put_cover_full.add_precondition(
            self.status_of_item(self.put_cover_full.cover, self.ok)
        )
        self.put_cover_full.add_precondition(Not(self.cover_is_assembled()))
        s1 = self.put_cover_full.add_subtask(
            self.assemble_cover, self.put_cover_full.cover
        )
        s2 = self.put_cover_full.add_subtask(self.wait_for_fixed_cover)
        self.put_cover_full.set_ordered(s1, s2)

        # reject
        self.put_cover_reject = Method("put_cover_reject", cover=type_item)
        self.put_cover_reject.set_task(self.t_put_cover, self.put_cover_reject.cover)
        self.put_cover_reject.add_precondition(
            self.current_arm_pose(self.over_reject_box)
        )
        self.put_cover_reject.add_precondition(self.space_in_reject_box())
        self.put_cover_reject.add_precondition(
            self.holding(self.put_cover_reject.cover)
        )
        self.put_cover_reject.add_precondition(
            self.item_status_known(self.put_cover_reject.cover)
        )
        self.put_cover_reject.add_precondition(
            self.status_of_item(self.put_cover_reject.cover, self.nok)
        )
        self.put_cover_reject.add_subtask(self.reject_item, self.put_cover_reject.cover)

        # reject full
        self.put_cover_reject_full = Method("put_cover_reject_full", cover=type_item)
        self.put_cover_reject_full.set_task(
            self.t_put_cover, self.put_cover_reject_full.cover
        )
        self.put_cover_reject_full.add_precondition(
            self.current_arm_pose(self.over_cover_station)
        )
        self.put_cover_reject_full.add_precondition(self.space_in_reject_box())
        self.put_cover_reject_full.add_precondition(
            self.holding(self.put_cover_reject_full.cover)
        )
        self.put_cover_reject_full.add_precondition(
            self.item_status_known(self.put_cover_reject_full.cover)
        )
        self.put_cover_reject_full.add_precondition(
            self.status_of_item(self.put_cover_reject_full.cover, self.nok)
        )
        s1 = self.put_cover_reject_full.add_subtask(self.move_arm, self.over_reject_box)
        s2 = self.put_cover_reject_full.add_subtask(
            self.reject_item, self.put_cover_reject_full.cover
        )
        self.put_cover_reject_full.set_ordered(s1, s2)

        # ASSEMBLE COVER
        # get cover
        self.assemble_cover_get = Method("assemble_cover_get", cover=type_item)
        self.assemble_cover_get.set_task(
            self.t_assemble_cover, self.assemble_cover_get.cover
        )
        self.assemble_cover_get.add_precondition(self.epic_active(self.epic3))
        self.assemble_cover_get.add_precondition(Not(self.epic_complete(self.epic3)))
        self.assemble_cover_get.add_precondition(
            Not(self.perceived_item(self.assemble_cover_get.cover))
        )
        self.assemble_cover_get.add_subtask(
            self.t_get_cover, self.assemble_cover_get.cover
        )

        # pick cover
        self.assemble_cover_pick = Method("assemble_cover_pick", cover=type_item)
        self.assemble_cover_pick.set_task(
            self.t_assemble_cover, self.assemble_cover_pick.cover
        )
        self.assemble_cover_pick.add_precondition(self.epic_active(self.epic3))
        self.assemble_cover_pick.add_precondition(Not(self.epic_complete(self.epic3)))
        self.assemble_cover_pick.add_precondition(
            Not(self.item_status_known(self.assemble_cover_pick.cover))
        )
        self.assemble_cover_pick.add_subtask(
            self.t_pick_cover, self.assemble_cover_pick.cover
        )

        # put cover
        self.assemble_cover_put = Method("assemble_cover_put", cover=type_item)
        self.assemble_cover_put.set_task(
            self.t_assemble_cover, self.assemble_cover_put.cover
        )
        self.assemble_cover_put.add_precondition(self.epic_active(self.epic3))
        self.assemble_cover_put.add_precondition(Not(self.epic_complete(self.epic3)))
        self.assemble_cover_put.add_precondition(
            self.item_status_known(self.assemble_cover_put.cover)
        )
        self.assemble_cover_put.add_subtask(
            self.t_put_cover, self.assemble_cover_put.cover
        )

        # switch
        self.assemble_cover_switch = Method("assemble_cover_switch", cover=type_item)
        self.assemble_cover_switch.set_task(
            self.t_assemble_cover, self.assemble_cover_switch.cover
        )
        self.assemble_cover_switch.add_precondition(Not(self.epic_active(self.epic3)))
        self.assemble_cover_switch.add_precondition(Not(self.epic_complete(self.epic3)))
        self.assemble_cover_switch.add_precondition(self.epic_complete(self.epic2))
        self.assemble_cover_switch.add_subtask(self.switch_to_epic, self.epic3)

        #
        # EPIC 4
        #

        # GET PROPELLER
        # perceive
        self.get_propeller_perceive = Method(
            "get_propeller_perceive", propeller=type_item
        )
        self.get_propeller_perceive.set_task(
            self.t_get_propeller, self.get_propeller_perceive.propeller
        )
        self.get_propeller_perceive.add_precondition(self.current_arm_pose(self.home))
        self.get_propeller_perceive.add_precondition(self.propeller_is_available())
        self.get_propeller_perceive.add_precondition(self.pallet_is_available())
        self.get_propeller_perceive.add_precondition(
            Not(self.perceived_item(self.get_propeller_perceive.propeller))
        )
        self.get_propeller_perceive.add_precondition(self.holding(self.nothing))
        self.get_propeller_perceive.add_subtask(
            self.perceive_item, self.get_propeller_perceive.propeller
        )

        # get propeller
        self.get_propeller_new = Method("get_propeller_new", propeller=type_item)
        self.get_propeller_new.set_task(
            self.t_get_propeller, self.get_propeller_new.propeller
        )
        self.get_propeller_new.add_precondition(self.current_arm_pose(self.home))
        self.get_propeller_new.add_precondition(Not(self.propeller_is_available()))
        self.get_propeller_new.add_precondition(self.pallet_is_available())
        self.get_propeller_new.add_precondition(
            Not(self.perceived_item(self.get_propeller_new.propeller))
        )
        self.get_propeller_new.add_precondition(self.holding(self.nothing))
        s1 = self.get_propeller_new.add_subtask(self.get_next_propeller)
        s2 = self.get_propeller_new.add_subtask(
            self.perceive_item, self.get_propeller_new.propeller
        )
        self.get_propeller_new.set_ordered(s1, s2)

        # redo
        self.get_propeller_redo = Method("get_propeller_redo", propeller=type_item)
        self.get_propeller_redo.set_task(
            self.t_get_propeller, self.get_propeller_redo.propeller
        )
        self.get_propeller_redo.add_precondition(Not(self.current_arm_pose(self.home)))
        self.get_propeller_redo.add_precondition(Not(self.propeller_is_available()))
        self.get_propeller_redo.add_precondition(self.pallet_is_available())
        self.get_propeller_redo.add_precondition(
            Not(self.perceived_item(self.get_propeller_redo.propeller))
        )
        self.get_propeller_redo.add_precondition(self.holding(self.nothing))
        s1 = self.get_propeller_redo.add_subtask(self.move_arm, self.home)
        s2 = self.get_propeller_redo.add_subtask(self.get_next_propeller)
        s3 = self.get_propeller_redo.add_subtask(
            self.perceive_item, self.get_propeller_redo.propeller
        )
        self.get_propeller_redo.set_ordered(s1, s2, s3)

        # get pallet
        self.get_propeller_pallet = Method("get_propeller_pallet", propeller=type_item)
        self.get_propeller_pallet.set_task(
            self.t_get_propeller, self.get_propeller_pallet.propeller
        )
        self.get_propeller_pallet.add_precondition(self.current_arm_pose(self.home))
        self.get_propeller_pallet.add_precondition(Not(self.propeller_is_available()))
        self.get_propeller_pallet.add_precondition(Not(self.pallet_is_available()))
        self.get_propeller_pallet.add_precondition(
            Not(self.perceived_item(self.get_propeller_pallet.propeller))
        )
        self.get_propeller_pallet.add_precondition(self.holding(self.nothing))
        s1 = self.get_propeller_pallet.add_subtask(self.get_next_pallet)
        s2 = self.get_propeller_pallet.add_subtask(self.get_next_propeller)
        s3 = self.get_propeller_pallet.add_subtask(
            self.perceive_item, self.get_propeller_pallet.propeller
        )
        self.get_propeller_pallet.set_ordered(s1, s2, s3)

        # full
        self.get_propeller_full = Method("get_propeller_full", propeller=type_item)
        self.get_propeller_full.set_task(
            self.t_get_propeller, self.get_propeller_full.propeller
        )
        self.get_propeller_full.add_precondition(Not(self.current_arm_pose(self.home)))
        self.get_propeller_full.add_precondition(Not(self.propeller_is_available()))
        self.get_propeller_full.add_precondition(Not(self.pallet_is_available()))
        self.get_propeller_full.add_precondition(
            Not(self.perceived_item(self.get_propeller_full.propeller))
        )
        self.get_propeller_full.add_precondition(self.holding(self.nothing))
        s1 = self.get_propeller_full.add_subtask(self.move_arm, self.home)
        s2 = self.get_propeller_full.add_subtask(self.get_next_pallet)
        s3 = self.get_propeller_full.add_subtask(self.get_next_propeller)
        s4 = self.get_propeller_full.add_subtask(
            self.perceive_item, self.get_propeller_full.propeller
        )
        self.get_propeller_full.set_ordered(s1, s2, s3, s4)

        # PICK PROPELLER
        # inspect
        self.pick_propeller_inspect = Method(
            "pick_propeller_inspect", propeller=type_item
        )
        self.pick_propeller_inspect.set_task(
            self.t_pick_propeller, self.pick_propeller_inspect.propeller
        )
        self.pick_propeller_inspect.add_precondition(
            self.current_arm_pose(self.over_propeller_station)
        )
        self.pick_propeller_inspect.add_precondition(self.pallet_is_available())
        self.pick_propeller_inspect.add_precondition(
            self.holding(self.pick_propeller_inspect.propeller)
        )
        self.pick_propeller_inspect.add_precondition(
            Not(self.item_status_known(self.pick_propeller_inspect.propeller))
        )
        self.pick_propeller_inspect.add_subtask(
            self.inspect, self.pick_propeller_inspect.propeller
        )

        # propeller station
        self.pick_propeller_station = Method(
            "pick_propeller_station", propeller=type_item
        )
        self.pick_propeller_station.set_task(
            self.t_pick_propeller, self.pick_propeller_station.propeller
        )
        self.pick_propeller_station.add_precondition(
            self.current_arm_pose(self.propeller_transition_pose)
        )
        self.pick_propeller_station.add_precondition(self.pallet_is_available())
        self.pick_propeller_station.add_precondition(
            self.holding(self.pick_propeller_station.propeller)
        )
        self.pick_propeller_station.add_precondition(
            Not(self.item_status_known(self.pick_propeller_station.propeller))
        )
        s1 = self.pick_propeller_station.add_subtask(
            self.move_arm, self.over_propeller_station
        )
        s2 = self.pick_propeller_station.add_subtask(
            self.inspect, self.pick_propeller_station.propeller
        )
        self.pick_propeller_station.set_ordered(s1, s2)

        # propeller transition pose
        self.pick_propeller_transition = Method(
            "pick_propeller_transition", propeller=type_item
        )
        self.pick_propeller_transition.set_task(
            self.t_pick_propeller, self.pick_propeller_transition.propeller
        )
        self.pick_propeller_transition.add_precondition(
            self.current_arm_pose(self.arm_up)
        )
        self.pick_propeller_transition.add_precondition(self.pallet_is_available())
        self.pick_propeller_transition.add_precondition(
            self.holding(self.pick_propeller_transition.propeller)
        )
        self.pick_propeller_transition.add_precondition(
            Not(self.item_status_known(self.pick_propeller_transition.propeller))
        )
        s1 = self.pick_propeller_transition.add_subtask(
            self.move_arm, self.propeller_transition_pose
        )
        s2 = self.pick_propeller_transition.add_subtask(
            self.move_arm, self.over_propeller_station
        )
        s3 = self.pick_propeller_transition.add_subtask(
            self.inspect, self.pick_propeller_transition.propeller
        )
        self.pick_propeller_transition.set_ordered(s1, s2, s3)

        # arm up
        self.pick_propeller_pose = Method("pick_propeller_pose", propeller=type_item)
        self.pick_propeller_pose.set_task(
            self.t_pick_propeller, self.pick_propeller_pose.propeller
        )
        self.pick_propeller_pose.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.pick_propeller_pose.add_precondition(self.pallet_is_available())
        self.pick_propeller_pose.add_precondition(
            self.holding(self.pick_propeller_pose.propeller)
        )
        self.pick_propeller_pose.add_precondition(
            Not(self.item_status_known(self.pick_propeller_pose.propeller))
        )
        s1 = self.pick_propeller_pose.add_subtask(self.move_arm, self.arm_up)
        s2 = self.pick_propeller_pose.add_subtask(
            self.move_arm, self.propeller_transition_pose
        )
        s3 = self.pick_propeller_pose.add_subtask(
            self.move_arm, self.over_propeller_station
        )
        s4 = self.pick_propeller_pose.add_subtask(
            self.inspect, self.pick_propeller_pose.propeller
        )
        self.pick_propeller_pose.set_ordered(s1, s2, s3, s4)

        # pick
        self.pick_propeller_pick = Method("pick_propeller_pick", propeller=type_item)
        self.pick_propeller_pick.set_task(
            self.t_pick_propeller, self.pick_propeller_pick.propeller
        )
        self.pick_propeller_pick.add_precondition(
            self.current_arm_pose(self.over_feeding_conveyor)
        )
        self.pick_propeller_pick.add_precondition(self.propeller_is_available())
        self.pick_propeller_pick.add_precondition(self.pallet_is_available())
        self.pick_propeller_pick.add_precondition(
            self.perceived_item(self.pick_propeller_pick.propeller)
        )
        self.pick_propeller_pick.add_precondition(self.holding(self.nothing))
        self.pick_propeller_pick.add_precondition(
            Not(self.item_status_known(self.pick_propeller_pick.propeller))
        )
        s1 = self.pick_propeller_pick.add_subtask(
            self.pick_propeller, self.pick_propeller_pick.propeller
        )
        s2 = self.pick_propeller_pick.add_subtask(self.move_arm, self.arm_up)
        s3 = self.pick_propeller_pick.add_subtask(
            self.move_arm, self.propeller_transition_pose
        )
        s4 = self.pick_propeller_pick.add_subtask(
            self.move_arm, self.over_propeller_station
        )
        s5 = self.pick_propeller_pick.add_subtask(
            self.inspect, self.pick_propeller_pick.propeller
        )
        self.pick_propeller_pick.set_ordered(s1, s2, s3, s4, s5)

        # full
        self.pick_propeller_full = Method("pick_propeller_full", propeller=type_item)
        self.pick_propeller_full.set_task(
            self.t_pick_propeller, self.pick_propeller_full.propeller
        )
        self.pick_propeller_full.add_precondition(self.current_arm_pose(self.home))
        self.pick_propeller_full.add_precondition(self.propeller_is_available())
        self.pick_propeller_full.add_precondition(self.pallet_is_available())
        self.pick_propeller_full.add_precondition(
            self.perceived_item(self.pick_propeller_full.propeller)
        )
        self.pick_propeller_full.add_precondition(self.holding(self.nothing))
        self.pick_propeller_full.add_precondition(
            Not(self.item_status_known(self.pick_propeller_full.propeller))
        )
        s1 = self.pick_propeller_full.add_subtask(
            self.move_arm, self.over_feeding_conveyor
        )
        s2 = self.pick_propeller_full.add_subtask(
            self.pick_propeller, self.pick_propeller_full.propeller
        )
        s3 = self.pick_propeller_full.add_subtask(self.move_arm, self.arm_up)
        s4 = self.pick_propeller_full.add_subtask(
            self.move_arm, self.propeller_transition_pose
        )
        s5 = self.pick_propeller_full.add_subtask(
            self.move_arm, self.over_propeller_station
        )
        s6 = self.pick_propeller_full.add_subtask(
            self.inspect, self.pick_propeller_full.propeller
        )
        self.pick_propeller_full.set_ordered(s1, s2, s3, s4, s5, s6)

        # PUT COVER
        # wait
        self.put_propeller_wait = Method("put_propeller_wait", propeller=type_item)
        self.put_propeller_wait.set_task(
            self.t_put_propeller, self.put_propeller_wait.propeller
        )
        self.put_propeller_wait.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.put_propeller_wait.add_precondition(self.pallet_is_available())
        self.put_propeller_wait.add_precondition(self.holding(self.nothing))
        self.put_propeller_wait.add_precondition(
            self.item_status_known(self.put_propeller_wait.propeller)
        )
        self.put_propeller_wait.add_precondition(
            self.status_of_item(self.put_propeller_wait.propeller, self.ok)
        )
        self.put_propeller_wait.add_precondition(self.propeller_is_assembled())
        self.put_propeller_wait.add_subtask(self.wait_for_fixed_propeller)

        # full
        self.put_propeller_full = Method("put_propeller_full", propeller=type_item)
        self.put_propeller_full.set_task(
            self.t_put_propeller, self.put_propeller_full.propeller
        )
        self.put_propeller_full.add_precondition(
            self.current_arm_pose(self.over_propeller_station)
        )
        self.put_propeller_full.add_precondition(self.pallet_is_available())
        self.put_propeller_full.add_precondition(
            self.holding(self.put_propeller_full.propeller)
        )
        self.put_propeller_full.add_precondition(
            self.item_status_known(self.put_propeller_full.propeller)
        )
        self.put_propeller_full.add_precondition(
            self.status_of_item(self.put_propeller_full.propeller, self.ok)
        )
        self.put_propeller_full.add_precondition(Not(self.propeller_is_assembled()))
        s1 = self.put_propeller_full.add_subtask(
            self.assemble_propeller, self.put_propeller_full.propeller
        )
        s2 = self.put_propeller_full.add_subtask(self.wait_for_fixed_propeller)
        self.put_propeller_full.set_ordered(s1, s2)

        # reject
        self.put_propeller_reject = Method("put_propeller_reject", propeller=type_item)
        self.put_propeller_reject.set_task(
            self.t_put_propeller, self.put_propeller_reject.propeller
        )
        self.put_propeller_reject.add_precondition(
            self.current_arm_pose(self.over_reject_box)
        )
        self.put_propeller_reject.add_precondition(self.space_in_reject_box())
        self.put_propeller_reject.add_precondition(
            self.holding(self.put_propeller_reject.propeller)
        )
        self.put_propeller_reject.add_precondition(
            self.item_status_known(self.put_propeller_reject.propeller)
        )
        self.put_propeller_reject.add_precondition(
            self.status_of_item(self.put_propeller_reject.propeller, self.nok)
        )
        self.put_propeller_reject.add_subtask(
            self.reject_item, self.put_propeller_reject.propeller
        )

        # reject full
        self.put_propeller_reject_full = Method(
            "put_propeller_reject_full", propeller=type_item
        )
        self.put_propeller_reject_full.set_task(
            self.t_put_propeller, self.put_propeller_reject_full.propeller
        )
        self.put_propeller_reject_full.add_precondition(
            self.current_arm_pose(self.over_propeller_station)
        )
        self.put_propeller_reject_full.add_precondition(self.space_in_reject_box())
        self.put_propeller_reject_full.add_precondition(
            self.holding(self.put_propeller_reject_full.propeller)
        )
        self.put_propeller_reject_full.add_precondition(
            self.item_status_known(self.put_propeller_reject_full.propeller)
        )
        self.put_propeller_reject_full.add_precondition(
            self.status_of_item(self.put_propeller_reject_full.propeller, self.nok)
        )
        s1 = self.put_propeller_reject_full.add_subtask(
            self.move_arm, self.over_reject_box
        )
        s2 = self.put_propeller_reject_full.add_subtask(
            self.reject_item, self.put_propeller_reject_full.propeller
        )
        self.put_propeller_reject_full.set_ordered(s1, s2)

        # ASSEMBLE PROPELLER
        # get propeller
        self.assemble_propeller_get = Method(
            "assemble_propeller_get", propeller=type_item
        )
        self.assemble_propeller_get.set_task(
            self.t_assemble_propeller, self.assemble_propeller_get.propeller
        )
        self.assemble_propeller_get.add_precondition(self.epic_active(self.epic4))
        self.assemble_propeller_get.add_precondition(
            Not(self.epic_complete(self.epic4))
        )
        self.assemble_propeller_get.add_precondition(
            Not(self.perceived_item(self.assemble_propeller_get.propeller))
        )
        self.assemble_propeller_get.add_subtask(
            self.t_get_propeller, self.assemble_propeller_get.propeller
        )

        # pick propeller
        self.assemble_propeller_pick = Method(
            "assemble_propeller_pick", propeller=type_item
        )
        self.assemble_propeller_pick.set_task(
            self.t_assemble_propeller, self.assemble_propeller_pick.propeller
        )
        self.assemble_propeller_pick.add_precondition(self.epic_active(self.epic4))
        self.assemble_propeller_pick.add_precondition(
            Not(self.epic_complete(self.epic4))
        )
        self.assemble_propeller_pick.add_precondition(
            Not(self.item_status_known(self.assemble_propeller_pick.propeller))
        )
        self.assemble_propeller_pick.add_subtask(
            self.t_pick_propeller, self.assemble_propeller_pick.propeller
        )

        # put propeller
        self.assemble_propeller_put = Method(
            "assemble_propeller_put", propeller=type_item
        )
        self.assemble_propeller_put.set_task(
            self.t_assemble_propeller, self.assemble_propeller_put.propeller
        )
        self.assemble_propeller_put.add_precondition(self.epic_active(self.epic4))
        self.assemble_propeller_put.add_precondition(
            Not(self.epic_complete(self.epic4))
        )
        self.assemble_propeller_put.add_precondition(
            self.item_status_known(self.assemble_propeller_put.propeller)
        )
        self.assemble_propeller_put.add_subtask(
            self.t_put_propeller, self.assemble_propeller_put.propeller
        )

        # switch
        self.assemble_propeller_switch = Method(
            "assemble_propeller_switch", propeller=type_item
        )
        self.assemble_propeller_switch.set_task(
            self.t_assemble_propeller, self.assemble_propeller_switch.propeller
        )
        self.assemble_propeller_switch.add_precondition(
            Not(self.epic_active(self.epic4))
        )
        self.assemble_propeller_switch.add_precondition(
            Not(self.epic_complete(self.epic4))
        )
        self.assemble_propeller_switch.add_precondition(self.epic_complete(self.epic2))
        self.assemble_propeller_switch.add_precondition(self.epic_complete(self.epic3))
        self.assemble_propeller_switch.add_subtask(self.switch_to_epic, self.epic4)

        #
        # ASSEMBLE BLOWER
        #

        self.assemble_blower_epic2 = Method("assemble_blower_epic2")
        self.assemble_blower_epic2.set_task(self.t_assemble_blower)
        self.assemble_blower_epic2.add_precondition(Not(self.epic_complete(self.epic2)))
        self.assemble_blower_epic2.add_precondition(self.space_in_reject_box())
        self.assemble_blower_epic2.add_subtask(self.t_solder_all_cables, self.cable)

        self.assemble_blower_epic3 = Method("assemble_blower_epic3")
        self.assemble_blower_epic3.set_task(self.t_assemble_blower)
        self.assemble_blower_epic3.add_precondition(self.epic_complete(self.epic2))
        self.assemble_blower_epic3.add_precondition(Not(self.epic_complete(self.epic3)))
        self.assemble_blower_epic3.add_precondition(self.space_in_reject_box())
        self.assemble_blower_epic3.add_subtask(self.t_assemble_cover, self.cover)

        self.assemble_blower_epic4 = Method("assemble_blower_epic4")
        self.assemble_blower_epic4.set_task(self.t_assemble_blower)
        self.assemble_blower_epic4.add_precondition(self.epic_complete(self.epic2))
        self.assemble_blower_epic4.add_precondition(self.epic_complete(self.epic3))
        self.assemble_blower_epic4.add_precondition(Not(self.epic_complete(self.epic4)))
        self.assemble_blower_epic4.add_precondition(self.space_in_reject_box())
        self.assemble_blower_epic4.add_subtask(
            self.t_assemble_propeller, self.propeller
        )

        self.assemble_blower_empty = Method("assemble_blower_empty")
        self.assemble_blower_empty.set_task(self.t_assemble_blower)
        self.assemble_blower_empty.add_precondition(Not(self.space_in_reject_box()))
        self.assemble_blower_empty.add_subtask(self.empty_reject_box)

        self.methods = (
            self.get_cable_perceive,
            self.get_cable_partial,
            self.get_cable_redo,
            self.get_cable_cables,
            self.get_cable_pallet,
            self.get_cable_full,
            self.pick_cable_inspect,
            self.pick_cable_over_station,
            self.pick_cable_pose,
            self.pick_cable_arm_up,
            self.pick_cable_pick,
            self.pick_cable_full,
            self.solder_cable_move_arm,
            self.solder_cable_release,
            self.solder_cable_wait,
            self.solder_cable_full,
            self.solder_cable_reject,
            self.solder_cable_reject_full,
            self.solder_all_cables_get,
            self.solder_all_cables_pick,
            self.solder_all_cables_solder,
            self.solder_all_cables_switch,
            self.get_cover_perceive,
            self.get_cover_new,
            self.get_cover_redo,
            self.get_cover_pallet,
            self.get_cover_full,
            self.pick_cover_inspect,
            self.pick_cover_station,
            self.pick_cover_transition,
            self.pick_cover_level,
            self.pick_cover_pose,
            self.pick_cover_move_arm_up,
            self.pick_cover_pick,
            self.pick_cover_full,
            self.put_cover_wait,
            self.put_cover_full,
            self.put_cover_reject,
            self.put_cover_reject_full,
            self.assemble_cover_get,
            self.assemble_cover_pick,
            self.assemble_cover_put,
            self.assemble_cover_switch,
            self.get_propeller_perceive,
            self.get_propeller_new,
            self.get_propeller_redo,
            self.get_propeller_pallet,
            self.get_propeller_full,
            self.pick_propeller_inspect,
            self.pick_propeller_station,
            self.pick_propeller_transition,
            self.pick_propeller_pose,
            self.pick_propeller_pick,
            self.pick_propeller_full,
            self.put_propeller_wait,
            self.put_propeller_full,
            self.put_propeller_reject,
            self.put_propeller_reject_full,
            self.assemble_propeller_get,
            self.assemble_propeller_pick,
            self.assemble_propeller_put,
            self.assemble_propeller_switch,
            self.assemble_blower_epic2,
            self.assemble_blower_epic3,
            self.assemble_blower_epic4,
            self.assemble_blower_empty,
        )

    def _create_domain_actions(self, temporal: bool = False) -> None:
        actions = Actions(self._env)

        if temporal:
            # TODO TEMPORAL
            pass
        else:
            self.switch_to_epic, [e] = self.create_action(
                "switch_to_epic", epic=Epic, _callable=actions.switch_to_epic
            )
            self.switch_to_epic.add_precondition(Not(self.epic_complete(e)))
            self.switch_to_epic.add_precondition(Not(self.epic_active(e)))
            self.switch_to_epic.add_effect(self.epic_active(e), True)

            self.move_arm, [a] = self.create_action(
                "move_arm",
                arm_pose=ArmPose,
                _callable=actions.move_arm,
            )
            self.move_arm.add_effect(self.current_arm_pose(a), True)

            self.get_next_pallet, _ = self.create_action(
                "get_next_pallet", _callable=actions.get_next_pallet
            )
            self.get_next_pallet.add_precondition(Not(self.pallet_is_available()))
            self.get_next_pallet.add_effect(self.pallet_is_available(), True)

            self.get_next_cables, [co] = self.create_action(
                "get_next_cables", color=Color, _callable=actions.get_next_cables
            )
            self.get_next_cables.add_precondition(Not(self.holding(self.cable)))
            self.get_next_cables.add_precondition(Not(self.cable_soldered(co)))
            self.get_next_cables.add_precondition(Not(self.cable_color_available(co)))
            self.get_next_cables.add_effect(self.cable_color_available(self.red), True)
            self.get_next_cables.add_effect(self.cable_color_available(self.blue), True)
            self.get_next_cables.add_effect(
                self.cable_color_available(self.brown), True
            )
            self.get_next_cables.add_effect(
                self.cable_color_available(self.white), True
            )
            self.get_next_cables.add_effect(self.perceived_item(self.cable), False)

            self.get_next_cover, _ = self.create_action(
                "get_next_cover", _callable=actions.get_next_cover
            )
            self.get_next_cover.add_precondition(Not(self.holding(self.cover)))
            self.get_next_cover.add_precondition(Not(self.cover_is_available()))
            self.get_next_cover.add_effect(self.cover_is_available(), True)
            self.get_next_cover.add_effect(self.perceived_item(self.cover), False)

            self.get_next_propeller, _ = self.create_action(
                "get_next_propeller", _callable=actions.get_next_propeller
            )
            self.get_next_propeller.add_precondition(Not(self.holding(self.propeller)))
            self.get_next_propeller.add_precondition(Not(self.propeller_is_available()))
            self.get_next_propeller.add_effect(self.propeller_is_available(), True)
            self.get_next_propeller.add_effect(
                self.perceived_item(self.propeller), False
            )

            self.perceive_item, [i] = self.create_action(
                "perceive_item",
                item=Item,
                _callable=actions.perceive_item,
            )
            self.perceive_item.add_precondition(Not(Equals(i, self.nothing)))
            self.perceive_item.add_precondition(Not(self.perceived_item(i)))
            self.perceive_item.add_effect(self.perceived_item(i), True)

            self.pick_cable, [ca, co] = self.create_action(
                "pick_cable",
                cable=Item,
                color=Color,
                _callable=actions.pick_cable,
            )
            self.pick_cable.add_precondition(self.holding(self.nothing))
            self.pick_cable.add_precondition(self.perceived_item(ca))
            self.pick_cable.add_precondition(
                self.current_arm_pose(self.over_cable_dispenser)
            )
            self.pick_cable.add_precondition(self.cable_color_available(co))
            self.pick_cable.add_effect(self.holding(ca), True)
            self.pick_cable.add_effect(self.holding(self.nothing), False)
            self.pick_cable.add_effect(self.current_arm_pose(self.unknown_pose), True)
            self.pick_cable.add_effect(
                self.current_arm_pose(self.over_cable_dispenser), False
            )
            self.pick_cable.add_effect(self.perceived_item(ca), False)
            self.pick_cable.add_effect(self.cable_color_available(co), False)

            self.pick_cover, [co] = self.create_action(
                "pick_cover",
                cover=Item,
                _callable=actions.pick_cover,
            )
            self.pick_cover.add_precondition(self.holding(self.nothing))
            self.pick_cover.add_precondition(self.perceived_item(co))
            self.pick_cover.add_precondition(
                self.current_arm_pose(self.over_feeding_conveyor)
            )
            self.pick_cover.add_precondition(self.cover_is_available())
            self.pick_cover.add_effect(self.holding(co), True)
            self.pick_cover.add_effect(self.holding(self.nothing), False)
            self.pick_cover.add_effect(self.current_arm_pose(self.unknown_pose), True)
            self.pick_cover.add_effect(
                self.current_arm_pose(self.over_feeding_conveyor), False
            )
            self.pick_cover.add_effect(self.perceived_item(co), False)
            self.pick_cover.add_effect(self.cover_is_available(), False)

            self.pick_propeller, [p] = self.create_action(
                "pick_propeller",
                propeller=Item,
                _callable=actions.pick_propeller,
            )
            self.pick_propeller.add_precondition(self.holding(self.nothing))
            self.pick_propeller.add_precondition(self.perceived_item(p))
            self.pick_propeller.add_precondition(
                self.current_arm_pose(self.over_feeding_conveyor)
            )
            self.pick_propeller.add_precondition(self.propeller_is_available())
            self.pick_propeller.add_effect(self.holding(p), True)
            self.pick_propeller.add_effect(self.holding(self.nothing), False)
            self.pick_propeller.add_effect(
                self.current_arm_pose(self.unknown_pose), True
            )
            self.pick_propeller.add_effect(
                self.current_arm_pose(self.over_feeding_conveyor), False
            )
            self.pick_propeller.add_effect(self.perceived_item(p), False)

            self.get_cable_pose, _ = self.create_action(
                "get_cable_pose",
                _callable=actions.get_cable_pose,
            )
            self.get_cable_pose.add_precondition(self.holding(self.cable))
            self.get_cable_pose.add_precondition(self.current_arm_pose(self.arm_up))
            self.get_cable_pose.add_precondition(Not(self.cable_pose_known()))
            self.get_cable_pose.add_effect(
                self.current_arm_pose(self.unknown_pose), True
            )
            self.get_cable_pose.add_effect(self.current_arm_pose(self.arm_up), False)
            self.get_cable_pose.add_effect(self.cable_pose_known(), True)

            self.get_cover_pose, _ = self.create_action(
                "get_cover_pose",
                _callable=actions.get_cover_pose,
            )
            self.get_cover_pose.add_precondition(self.holding(self.cover))
            self.get_cover_pose.add_precondition(
                self.current_arm_pose(self.unknown_pose)
            )
            self.get_cover_pose.add_precondition(Not(self.cover_pose_known()))
            self.get_cover_pose.add_effect(self.cover_pose_known(), True)

            self.level_cover, _ = self.create_action(
                "level_cover",
                _callable=actions.level_cover,
            )
            self.level_cover.add_precondition(self.holding(self.cover))
            self.level_cover.add_precondition(self.current_arm_pose(self.unknown_pose))
            self.level_cover.add_precondition(self.cover_pose_known())
            self.level_cover.add_precondition(Not(self.cover_is_leveled()))
            self.level_cover.add_effect(self.cover_is_leveled(), True)

            self.inspect, [i] = self.create_action(
                "inspect",
                item=Item,
                _callable=actions.inspect,
            )
            self.inspect.add_precondition(Not(self.item_status_known(i)))
            self.inspect.add_precondition(self.holding(i))
            self.inspect.add_effect(self.item_status_known(i), True)

            self.assemble_cover, [co] = self.create_action(
                "assemble_cover",
                cover=Item,
                _callable=actions.assemble_cover,
            )
            self.assemble_cover.add_precondition(self.holding(co))
            self.assemble_cover.add_precondition(Not(self.cover_is_assembled()))
            self.assemble_cover.add_precondition(
                self.current_arm_pose(self.over_cover_station)
            )
            self.assemble_cover.add_effect(self.holding(self.nothing), True)
            self.assemble_cover.add_effect(self.holding(co), False)
            self.assemble_cover.add_effect(self.cover_is_assembled(), True)
            self.assemble_cover.add_effect(
                self.current_arm_pose(self.unknown_pose), True
            )
            self.assemble_cover.add_effect(
                self.current_arm_pose(self.over_cover_station), False
            )

            self.assemble_propeller, [p] = self.create_action(
                "assemble_propeller",
                propeller=Item,
                _callable=actions.assemble_propeller,
            )
            self.assemble_propeller.add_precondition(self.holding(p))
            self.assemble_propeller.add_precondition(Not(self.propeller_is_assembled()))
            self.assemble_propeller.add_precondition(
                self.current_arm_pose(self.over_propeller_station)
            )
            self.assemble_propeller.add_effect(self.holding(self.nothing), True)
            self.assemble_propeller.add_effect(self.holding(p), False)
            self.assemble_propeller.add_effect(self.propeller_is_assembled(), True)
            self.assemble_propeller.add_effect(
                self.current_arm_pose(self.unknown_pose), True
            )
            self.assemble_propeller.add_effect(
                self.current_arm_pose(self.over_propeller_station), False
            )

            self.wait_for_soldering, [co] = self.create_action(
                "wait_for_soldering", color=Color, _callable=actions.wait_for_soldering
            )
            self.wait_for_soldering.add_precondition(
                self.current_arm_pose(self.soldering_pose)
            )
            self.wait_for_soldering.add_precondition(Not(self.cable_soldered(co)))
            self.wait_for_soldering.add_precondition(self.current_cable_color(co))
            self.wait_for_soldering.add_effect(self.cable_soldered(co), True)

            self.wait_for_fixed_cover, _ = self.create_action(
                "wait_for_fixed_cover",
                _callable=actions.wait_for_fixed_cover,
            )
            self.wait_for_fixed_cover.add_precondition(self.cover_is_assembled())
            self.wait_for_fixed_cover.add_effect(self.cover_is_assembled(), False)
            self.wait_for_fixed_cover.add_effect(
                self.item_status_known(self.cover), False
            )

            self.wait_for_fixed_propeller, _ = self.create_action(
                "wait_for_fixed_propeller",
                _callable=actions.wait_for_fixed_propeller,
            )
            self.wait_for_fixed_propeller.add_precondition(
                self.propeller_is_assembled()
            )
            self.wait_for_fixed_propeller.add_effect(
                self.propeller_is_assembled(), False
            )
            self.wait_for_fixed_propeller.add_effect(
                self.item_status_known(self.propeller), False
            )

            self.release_cable, [co] = self.create_action(
                "release_cable", color=Color, _callable=actions.release_cable
            )
            self.release_cable.add_precondition(self.holding(self.cable))
            self.release_cable.add_precondition(self.cable_soldered(co))
            self.release_cable.add_precondition(self.current_cable_color(co))
            self.release_cable.add_effect(self.holding(self.nothing), True)
            self.release_cable.add_effect(self.holding(self.cable), False)
            self.release_cable.add_effect(self.current_cable_color(co), False)

            self.move_arm_cable_end, _ = self.create_action(
                "move_arm_cable_end",
                _callable=actions.move_arm_cable_end,
            )
            self.move_arm_cable_end.add_precondition(self.holding(self.nothing))
            self.move_arm_cable_end.add_effect(
                self.current_arm_pose(self.arm_up), True
            )
            self.move_arm_cable_end.add_effect(
                self.item_status_known(self.cable), False
            )

            self.reject_item, [i] = self.create_action(
                "reject_item",
                item=Item,
                _callable=actions.reject_item,
            )
            self.reject_item.add_precondition(self.space_in_reject_box())
            self.reject_item.add_precondition(self.item_status_known(i))
            self.reject_item.add_precondition(
                self.current_arm_pose(self.over_reject_box)
            )
            self.reject_item.add_precondition(self.holding(i))
            self.reject_item.add_effect(self.holding(i), False)
            self.reject_item.add_effect(self.holding(self.nothing), True)
            self.reject_item.add_effect(
                self.current_arm_pose(self.over_reject_box), False
            )
            self.reject_item.add_effect(self.current_arm_pose(self.unknown_pose), True)
            self.reject_item.add_effect(self.item_status_known(i), False)

            self.empty_reject_box, _ = self.create_action(
                "empty_reject_box",
                _callable=actions.empty_reject_box,
            )
            self.empty_reject_box.add_precondition(Not(self.space_in_reject_box()))
            self.empty_reject_box.add_effect(self.space_in_reject_box(), True)

    def set_state_and_goal(self, problem, goal=None) -> None:
        success = True
        if goal is None:
            problem.task_network.add_subtask(self.t_assemble_blower())
        else:
            logerr((f"Task ({goal}) is unknown! Please leave the goal empty."))
            success = False
        return success
