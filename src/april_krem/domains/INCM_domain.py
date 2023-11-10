from rospy import logerr
from unified_planning.model.htn import Task, Method
from unified_planning.shortcuts import Not
from april_krem.domains.INCM_components import (
    Item,
    ArmPose,
    Status,
    Actions,
    Environment,
)
from up_esb.bridge import Bridge


class INCMDomain(Bridge):
    def __init__(self, krem_logging, temporal: bool = False) -> None:
        Bridge.__init__(self)

        self._env = Environment(krem_logging)

        # Create types for planning based on class types
        self.create_types([Item, ArmPose, Status])
        type_item = self.get_type(Item)
        type_status = self.get_type(Status)

        # Create fluents for planning
        self.holding = self.create_fluent_from_function(self._env.holding)

        self.current_arm_pose = self.create_fluent_from_function(
            self._env.current_arm_pose
        )
        self.perceived_passport = self.create_fluent_from_function(
            self._env.perceived_passport
        )
        self.passport_status_known = self.create_fluent_from_function(
            self._env.passport_status_known
        )
        self.status_of_passport = self.create_fluent_from_function(
            self._env.status_of_passport
        )
        self.mrz_reader_used = self.create_fluent_from_function(
            self._env.mrz_reader_used
        )
        self.chip_reader_used = self.create_fluent_from_function(
            self._env.chip_reader_used
        )
        self.space_in_box = self.create_fluent_from_function(self._env.space_in_box)

        # Create objects for both planning and execution
        self.items = self.create_enum_objects(Item)
        self.nothing = self.objects[Item.nothing.name]
        self.passport = self.objects[Item.passport.name]

        self.arm_poses = self.create_enum_objects(ArmPose)
        self.unknown = self.objects[ArmPose.unknown.name]
        self.over_passport = self.objects[ArmPose.over_passport.name]
        self.over_mrz = self.objects[ArmPose.over_mrz.name]
        self.over_chip = self.objects[ArmPose.over_chip.name]
        self.over_boxes = self.objects[ArmPose.over_boxes.name]

        self.status = self.create_enum_objects(Status)
        self.ok = self.objects[Status.ok.name]
        self.nok = self.objects[Status.nok.name]

        # Create actions for planning
        self._create_domain_actions(temporal)

        # Tasks
        self.t_get_passport = Task("t_get_passport", passport=type_item)
        self.t_read_mrz = Task("t_read_mrz", passport=type_item)
        self.t_read_chip = Task("t_read_chip", passport=type_item)
        self.t_scan_passport = Task("t_scan_passport", passport=type_item)
        self.t_place_passport = Task("t_place_passport", passport=type_item)

        self.tasks = (
            self.t_get_passport,
            self.t_read_mrz,
            self.t_read_chip,
            self.t_scan_passport,
            self.t_place_passport,
        )

        # Methods

        # GET PASSPORT
        # passport inspected, arm over boxes, nothing to do
        self.get_passport_noop = Method("get_passport_noop", passport=type_item)
        self.get_passport_noop.set_task(
            self.t_get_passport, self.get_passport_noop.passport
        )
        self.get_passport_noop.add_precondition(
            Not(self.current_arm_pose(self.unknown))
        )
        self.get_passport_noop.add_precondition(
            self.holding(self.get_passport_noop.passport)
        )

        # passport in hand, move arm
        self.get_passport_move_arm_1 = Method(
            "get_passport_move_arm_1", passport=type_item
        )
        self.get_passport_move_arm_1.set_task(
            self.t_get_passport, self.get_passport_move_arm_1.passport
        )
        self.get_passport_move_arm_1.add_precondition(
            self.current_arm_pose(self.unknown)
        )
        self.get_passport_move_arm_1.add_precondition(
            self.holding(self.get_passport_move_arm_1.passport)
        )
        self.get_passport_move_arm_1.add_subtask(self.move_arm, self.over_passport)

        # perceived passport, pick it up
        self.get_passport_pick = Method("get_passport_pick", passport=type_item)
        self.get_passport_pick.set_task(
            self.t_get_passport, self.get_passport_pick.passport
        )
        self.get_passport_pick.add_precondition(
            self.current_arm_pose(self.over_passport)
        )
        self.get_passport_pick.add_precondition(self.holding(self.nothing))
        self.get_passport_pick.add_precondition(self.perceived_passport())
        s1 = self.get_passport_pick.add_subtask(
            self.pick_passport, self.get_passport_pick.passport
        )
        s2 = self.get_passport_pick.add_subtask(self.move_arm, self.over_passport)
        self.get_passport_pick.set_ordered(s1, s2)

        # perceive passport, pick it up
        self.get_passport_perceive = Method("get_passport_perceive", passport=type_item)
        self.get_passport_perceive.set_task(
            self.t_get_passport, self.get_passport_perceive.passport
        )
        self.get_passport_perceive.add_precondition(
            self.current_arm_pose(self.over_passport)
        )
        self.get_passport_perceive.add_precondition(self.holding(self.nothing))
        s1 = self.get_passport_perceive.add_subtask(self.perceive_passport)
        s2 = self.get_passport_perceive.add_subtask(
            self.pick_passport, self.get_passport_perceive.passport
        )
        s3 = self.get_passport_perceive.add_subtask(self.move_arm, self.over_passport)
        self.get_passport_perceive.set_ordered(s1, s2, s3)

        # move arm over passport, perceive passport, pick it up, move arm over passport
        self.get_passport_full = Method("get_passport_full", passport=type_item)
        self.get_passport_full.set_task(
            self.t_get_passport, self.get_passport_full.passport
        )
        self.get_passport_full.add_precondition(
            Not(self.current_arm_pose(self.over_passport))
        )
        self.get_passport_full.add_precondition(self.holding(self.nothing))
        s1 = self.get_passport_full.add_subtask(self.move_arm, self.over_passport)
        s2 = self.get_passport_full.add_subtask(self.perceive_passport)
        s3 = self.get_passport_full.add_subtask(
            self.pick_passport, self.get_passport_full.passport
        )
        s4 = self.get_passport_full.add_subtask(self.move_arm, self.over_passport)
        self.get_passport_full.set_ordered(s1, s2, s3, s4)

        # READ MRZ
        # already read mrz, arm in position
        self.read_mrz_noop = Method("read_mrz_noop", passport=type_item)
        self.read_mrz_noop.set_task(self.t_read_mrz, self.read_mrz_noop.passport)
        self.read_mrz_noop.add_precondition(self.mrz_reader_used())
        self.read_mrz_noop.add_precondition(Not(self.current_arm_pose(self.unknown)))
        self.read_mrz_noop.add_precondition(self.holding(self.read_mrz_noop.passport))

        # already read mrz, move arm
        self.read_mrz_move_arm = Method("read_mrz_move_arm", passport=type_item)
        self.read_mrz_move_arm.set_task(
            self.t_read_mrz, self.read_mrz_move_arm.passport
        )
        self.read_mrz_move_arm.add_precondition(self.mrz_reader_used())
        self.read_mrz_move_arm.add_precondition(self.current_arm_pose(self.unknown))
        self.read_mrz_move_arm.add_precondition(
            self.holding(self.read_mrz_move_arm.passport)
        )
        self.read_mrz_move_arm.add_subtask(self.move_arm, self.over_mrz)

        # already read mrz, arm in position
        self.read_mrz_read = Method("read_mrz_read", passport=type_item)
        self.read_mrz_read.set_task(self.t_read_mrz, self.read_mrz_read.passport)
        self.read_mrz_read.add_precondition(Not(self.mrz_reader_used()))
        self.read_mrz_read.add_precondition(self.current_arm_pose(self.over_mrz))
        self.read_mrz_read.add_precondition(self.holding(self.read_mrz_read.passport))
        s1 = self.read_mrz_read.add_subtask(self.read_mrz, self.read_mrz_read.passport)
        s2 = self.read_mrz_read.add_subtask(self.move_arm, self.over_mrz)
        self.read_mrz_read.set_ordered(s1, s2)

        # already read mrz, arm in position
        self.read_mrz_full = Method("read_mrz_full", passport=type_item)
        self.read_mrz_full.set_task(self.t_read_mrz, self.read_mrz_full.passport)
        self.read_mrz_full.add_precondition(Not(self.mrz_reader_used()))
        self.read_mrz_full.add_precondition(self.current_arm_pose(self.over_passport))
        self.read_mrz_full.add_precondition(self.holding(self.read_mrz_full.passport))
        s1 = self.read_mrz_full.add_subtask(self.move_arm, self.over_mrz)
        s2 = self.read_mrz_full.add_subtask(self.read_mrz, self.read_mrz_full.passport)
        s3 = self.read_mrz_full.add_subtask(self.move_arm, self.over_mrz)
        self.read_mrz_full.set_ordered(s1, s2, s3)

        # READ CHIP
        # already read chip, arm in position
        self.read_chip_noop = Method("read_chip_noop", passport=type_item)
        self.read_chip_noop.set_task(self.t_read_chip, self.read_chip_noop.passport)
        self.read_chip_noop.add_precondition(self.chip_reader_used())
        self.read_chip_noop.add_precondition(self.mrz_reader_used())
        self.read_chip_noop.add_precondition(Not(self.current_arm_pose(self.unknown)))
        self.read_chip_noop.add_precondition(self.holding(self.read_chip_noop.passport))

        # already read chip, move arm
        self.read_chip_move_arm = Method("read_chip_move_arm", passport=type_item)
        self.read_chip_move_arm.set_task(
            self.t_read_chip, self.read_chip_move_arm.passport
        )
        self.read_chip_move_arm.add_precondition(self.chip_reader_used())
        self.read_chip_move_arm.add_precondition(self.mrz_reader_used())
        self.read_chip_move_arm.add_precondition(self.current_arm_pose(self.unknown))
        self.read_chip_move_arm.add_precondition(
            self.holding(self.read_chip_move_arm.passport)
        )
        self.read_chip_move_arm.add_subtask(self.move_arm, self.over_chip)

        # already read chip, arm in position
        self.read_chip_read = Method("read_chip_read", passport=type_item)
        self.read_chip_read.set_task(self.t_read_chip, self.read_chip_read.passport)
        self.read_chip_read.add_precondition(Not(self.chip_reader_used()))
        self.read_chip_read.add_precondition(self.mrz_reader_used())
        self.read_chip_read.add_precondition(self.current_arm_pose(self.over_chip))
        self.read_chip_read.add_precondition(self.holding(self.read_chip_read.passport))
        s1 = self.read_chip_read.add_subtask(
            self.read_chip, self.read_chip_read.passport
        )
        s2 = self.read_chip_read.add_subtask(self.move_arm, self.over_chip)
        self.read_chip_read.set_ordered(s1, s2)

        # already read chip, arm in position
        self.read_chip_full = Method("read_chip_full", passport=type_item)
        self.read_chip_full.set_task(self.t_read_chip, self.read_chip_full.passport)
        self.read_chip_full.add_precondition(Not(self.chip_reader_used()))
        self.read_chip_full.add_precondition(self.mrz_reader_used())
        self.read_chip_full.add_precondition(self.current_arm_pose(self.over_mrz))
        self.read_chip_full.add_precondition(self.holding(self.read_chip_full.passport))
        s1 = self.read_chip_full.add_subtask(self.move_arm, self.over_chip)
        s2 = self.read_chip_full.add_subtask(
            self.read_chip, self.read_chip_full.passport
        )
        s3 = self.read_chip_full.add_subtask(self.move_arm, self.over_chip)
        self.read_chip_full.set_ordered(s1, s2, s3)

        # PLACE PASSPORT
        # already placed passport, move arm
        self.place_passport_move_arm = Method(
            "place_passport_move_arm", passport=type_item, status=type_status
        )
        self.place_passport_move_arm.set_task(
            self.t_place_passport, self.place_passport_move_arm.passport
        )
        self.place_passport_move_arm.add_precondition(Not(self.passport_status_known()))
        self.place_passport_move_arm.add_precondition(
            self.current_arm_pose(self.unknown)
        )
        self.place_passport_move_arm.add_precondition(self.holding(self.nothing))
        self.place_passport_move_arm.add_subtask(self.move_arm, self.over_boxes)

        # arm in position, place passport
        self.place_passport_place = Method(
            "place_passport_place", passport=type_item, status=type_status
        )
        self.place_passport_place.set_task(
            self.t_place_passport, self.place_passport_place.passport
        )
        self.place_passport_place.add_precondition(self.passport_status_known())
        self.place_passport_place.add_precondition(
            self.space_in_box(self.place_passport_place.status)
        )
        self.place_passport_place.add_precondition(
            self.current_arm_pose(self.over_boxes)
        )
        self.place_passport_place.add_precondition(
            self.holding(self.place_passport_place.passport)
        )
        s1 = self.place_passport_place.add_subtask(
            self.place_passport_in_box,
            self.place_passport_place.passport,
            self.place_passport_place.status,
        )
        s2 = self.place_passport_place.add_subtask(self.move_arm, self.over_boxes)
        self.place_passport_place.set_ordered(s1, s2)

        # already read chip, arm in position
        self.place_passport_full = Method(
            "place_passport_full", passport=type_item, status=type_status
        )
        self.place_passport_full.set_task(
            self.t_place_passport, self.place_passport_full.passport
        )
        self.place_passport_full.add_precondition(self.passport_status_known())
        self.place_passport_full.add_precondition(
            self.space_in_box(self.place_passport_full.status)
        )
        self.place_passport_full.add_precondition(self.current_arm_pose(self.over_chip))
        self.place_passport_full.add_precondition(
            self.holding(self.place_passport_full.passport)
        )
        s1 = self.place_passport_full.add_subtask(self.move_arm, self.over_boxes)
        s2 = self.place_passport_full.add_subtask(
            self.place_passport_in_box,
            self.place_passport_full.passport,
            self.place_passport_full.status,
        )
        s3 = self.place_passport_full.add_subtask(self.move_arm, self.over_boxes)
        self.place_passport_full.set_ordered(s1, s2, s3)

        # box is full
        self.place_passport_empty_box = Method(
            "place_passport_empty_box", passport=type_item, status=type_status
        )
        self.place_passport_empty_box.set_task(
            self.t_place_passport, self.place_passport_empty_box.passport
        )
        self.place_passport_empty_box.add_precondition(self.passport_status_known())
        self.place_passport_empty_box.add_precondition(
            Not(self.space_in_box(self.place_passport_empty_box.status))
        )
        self.place_passport_empty_box.add_precondition(
            self.holding(self.place_passport_empty_box.passport)
        )
        self.place_passport_empty_box.add_subtask(
            self.empty_box, self.place_passport_empty_box.status
        )

        # SCAN PASSPORT
        # get and scan
        self.scan_passport_get = Method("scan_passport_get", passport=type_item)
        self.scan_passport_get.set_task(
            self.t_scan_passport, self.scan_passport_get.passport
        )
        s1 = self.scan_passport_get.add_subtask(
            self.t_get_passport, self.scan_passport_get.passport
        )
        s2 = self.scan_passport_get.add_subtask(
            self.t_read_mrz, self.scan_passport_get.passport
        )
        s3 = self.scan_passport_get.add_subtask(
            self.t_read_chip, self.scan_passport_get.passport
        )
        s4 = self.scan_passport_get.add_subtask(
            self.inspect, self.scan_passport_get.passport
        )
        self.scan_passport_get.set_ordered(s1, s2, s3, s4)

        # scan finished, place in box
        self.scan_passport_place = Method("scan_passport_place", passport=type_item)
        self.scan_passport_place.set_task(
            self.t_scan_passport, self.scan_passport_place.passport
        )
        self.scan_passport_place.add_precondition(self.passport_status_known())
        self.scan_passport_place.add_subtask(
            self.t_place_passport,
            self.scan_passport_place.passport,
        )

        self.methods = (
            self.get_passport_noop,
            self.get_passport_move_arm_1,
            self.get_passport_pick,
            self.get_passport_perceive,
            self.get_passport_full,
            self.read_mrz_noop,
            self.read_mrz_move_arm,
            self.read_mrz_read,
            self.read_mrz_full,
            self.read_chip_noop,
            self.read_chip_move_arm,
            self.read_chip_read,
            self.read_chip_full,
            self.place_passport_move_arm,
            self.place_passport_place,
            self.place_passport_full,
            self.place_passport_empty_box,
            self.scan_passport_get,
            self.scan_passport_place,
        )

    def _create_domain_actions(self, temporal: bool = False) -> None:
        actions = Actions(self._env)

        if temporal:
            # TODO TEMPORAL
            pass
        else:
            self.perceive_passport, _ = self.create_action(
                "perceive_passport",
                _callable=actions.perceive_passport,
            )
            self.perceive_passport.add_precondition(Not(self.perceived_passport()))
            self.perceive_passport.add_effect(self.perceived_passport(), True)

            self.move_arm, [a] = self.create_action(
                "move_arm",
                arm_pose=ArmPose,
                _callable=actions.move_arm,
            )
            self.move_arm.add_effect(self.current_arm_pose(a), True)

            self.pick_passport, [p] = self.create_action(
                "pick_passport",
                passport=Item,
                _callable=actions.pick_passport,
            )
            self.pick_passport.add_precondition(self.holding(self.nothing))
            self.pick_passport.add_precondition(self.perceived_passport())
            self.pick_passport.add_precondition(
                self.current_arm_pose(self.over_passport)
            )
            self.pick_passport.add_effect(self.holding(p), True)
            self.pick_passport.add_effect(self.holding(self.nothing), False)
            self.pick_passport.add_effect(self.current_arm_pose(self.unknown), True)
            self.pick_passport.add_effect(
                self.current_arm_pose(self.over_passport), False
            )
            self.pick_passport.add_effect(self.perceived_passport(), False)

            self.read_mrz, [p] = self.create_action(
                "read_mrz",
                passport=Item,
                _callable=actions.read_mrz,
            )
            self.read_mrz.add_precondition(Not(self.mrz_reader_used()))
            self.read_mrz.add_precondition(self.current_arm_pose(self.over_mrz))
            self.read_mrz.add_precondition(self.holding(p))
            self.read_mrz.add_effect(self.mrz_reader_used(), True)

            self.read_chip, [p] = self.create_action(
                "read_chip",
                passport=Item,
                _callable=actions.read_chip,
            )
            self.read_chip.add_precondition(Not(self.chip_reader_used()))
            self.read_chip.add_precondition(self.mrz_reader_used())
            self.read_chip.add_precondition(self.current_arm_pose(self.over_chip))
            self.read_chip.add_precondition(self.holding(p))
            self.read_chip.add_effect(self.chip_reader_used(), True)

            self.inspect, _ = self.create_action(
                "inspect",
                passport=Item,
                _callable=actions.inspect,
            )
            self.inspect.add_precondition(Not(self.passport_status_known()))
            self.inspect.add_precondition(self.mrz_reader_used())
            self.inspect.add_precondition(self.chip_reader_used())
            self.inspect.add_effect(self.passport_status_known(), True)

            self.place_passport_in_box, [p, s] = self.create_action(
                "place_passport_in_box",
                passport=Item,
                status=Status,
                _callable=actions.place_passport_in_box,
            )
            self.place_passport_in_box.add_precondition(self.space_in_box(s))
            self.place_passport_in_box.add_precondition(self.status_of_passport(s))
            self.place_passport_in_box.add_precondition(self.passport_status_known())
            self.place_passport_in_box.add_precondition(
                self.current_arm_pose(self.over_boxes)
            )
            self.place_passport_in_box.add_precondition(self.holding(p))
            self.place_passport_in_box.add_effect(self.holding(p), False)
            self.place_passport_in_box.add_effect(self.holding(self.nothing), True)
            self.place_passport_in_box.add_effect(
                self.current_arm_pose(self.over_boxes), False
            )
            self.place_passport_in_box.add_effect(
                self.current_arm_pose(self.unknown), True
            )
            self.place_passport_in_box.add_effect(self.passport_status_known(), False)

            self.empty_box, [s] = self.create_action(
                "empty_box",
                status=Status,
                _callable=actions.empty_box,
            )
            self.empty_box.add_precondition(Not(self.space_in_box(s)))
            self.empty_box.add_precondition(self.passport_status_known())
            self.empty_box.add_effect(self.space_in_box(s), True)

    def set_state_and_goal(self, problem, goal=None) -> None:
        success = True
        if goal is None:
            problem.task_network.add_subtask(self.t_scan_passport(self.passport))
        else:
            logerr((f"Task ({goal}) is unknown! Please leave the goal empty."))
            success = False
        return success
