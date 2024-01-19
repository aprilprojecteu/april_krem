from rospy import logerr
from unified_planning.model.htn import Task, Method
from unified_planning.shortcuts import Not, Or
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
        self.passport_corner_detected = self.create_fluent_from_function(
            self._env.passport_corner_detected
        )
        self.passports_available = self.create_fluent_from_function(self._env.passports_available)

        # Create objects for both planning and execution
        self.create_enum_objects(Item)
        self.nothing = self.objects[Item.nothing.name]
        self.passport = self.objects[Item.passport.name]

        self.create_enum_objects(ArmPose)
        self.unknown_pose = self.objects[ArmPose.unknown.name]
        self.home = self.objects[ArmPose.home.name]
        self.over_passport = self.objects[ArmPose.over_passport.name]
        self.over_mrz = self.objects[ArmPose.over_mrz.name]
        self.over_chip = self.objects[ArmPose.over_chip.name]
        self.over_boxes = self.objects[ArmPose.over_boxes.name]
        self.arm_up = self.objects[ArmPose.arm_up.name]

        self.create_enum_objects(Status)
        self.ok = self.objects[Status.ok.name]
        self.nok = self.objects[Status.nok.name]

        # Create actions for planning
        self._create_domain_actions(temporal)

        # Tasks
        self.t_get_passport = Task("t_get_passport", passport=type_item)
        self.t_read_mrz_chip = Task("t_read_mrz_chip", passport=type_item)
        self.t_scan_passport = Task("t_scan_passport", passport=type_item)
        self.t_place_passport = Task("t_place_passport", passport=type_item)

        self.tasks = (
            self.t_get_passport,
            self.t_read_mrz_chip,
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
            self.holding(self.get_passport_noop.passport)
        )
        self.get_passport_noop.add_precondition(self.passport_corner_detected())

        # passport in hand, moved arm up, detect corner
        self.get_passport_detect_corner = Method(
            "get_passport_detect_corner", passport=type_item
        )
        self.get_passport_detect_corner.set_task(
            self.t_get_passport, self.get_passport_detect_corner.passport
        )
        self.get_passport_detect_corner.add_precondition(
            Or(self.current_arm_pose(self.arm_up), self.current_arm_pose(self.over_mrz))
        )
        self.get_passport_detect_corner.add_precondition(
            self.holding(self.get_passport_detect_corner.passport)
        )
        self.get_passport_detect_corner.add_precondition(
            Not(self.passport_corner_detected())
        )
        self.get_passport_detect_corner.add_subtask(self.detect_passport_corner)

        # passport in hand, move arm
        self.get_passport_move_arm_1 = Method(
            "get_passport_move_arm_1", passport=type_item
        )
        self.get_passport_move_arm_1.set_task(
            self.t_get_passport, self.get_passport_move_arm_1.passport
        )
        self.get_passport_move_arm_1.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.get_passport_move_arm_1.add_precondition(
            self.holding(self.get_passport_move_arm_1.passport)
        )
        self.get_passport_move_arm_1.add_precondition(
            Not(self.passport_corner_detected())
        )
        s1 = self.get_passport_move_arm_1.add_subtask(self.move_arm, self.arm_up)
        s2 = self.get_passport_move_arm_1.add_subtask(self.detect_passport_corner)
        self.get_passport_move_arm_1.set_ordered(s1, s2)

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
        self.get_passport_pick.add_precondition(Not(self.passport_corner_detected()))
        s1 = self.get_passport_pick.add_subtask(
            self.pick_passport, self.get_passport_pick.passport
        )
        s2 = self.get_passport_pick.add_subtask(self.move_arm, self.arm_up)
        s3 = self.get_passport_pick.add_subtask(self.detect_passport_corner)
        self.get_passport_pick.set_ordered(s1, s2, s3)

        # perceive passport, pick it up
        self.get_passport_perceive = Method("get_passport_perceive", passport=type_item)
        self.get_passport_perceive.set_task(
            self.t_get_passport, self.get_passport_perceive.passport
        )
        self.get_passport_perceive.add_precondition(
            self.current_arm_pose(self.over_passport)
        )
        self.get_passport_perceive.add_precondition(self.holding(self.nothing))
        self.get_passport_perceive.add_precondition(
            Not(self.passport_corner_detected())
        )
        s1 = self.get_passport_perceive.add_subtask(self.perceive_passport)
        s2 = self.get_passport_perceive.add_subtask(
            self.pick_passport, self.get_passport_perceive.passport
        )
        s3 = self.get_passport_perceive.add_subtask(self.move_arm, self.arm_up)
        s4 = self.get_passport_perceive.add_subtask(self.detect_passport_corner)
        self.get_passport_perceive.set_ordered(s1, s2, s3, s4)

        # move arm over passport, perceive passport, pick it up, move arm over passport
        self.get_passport_move_arm_2 = Method(
            "get_passport_move_arm_2", passport=type_item
        )
        self.get_passport_move_arm_2.set_task(
            self.t_get_passport, self.get_passport_move_arm_2.passport
        )
        self.get_passport_move_arm_2.add_precondition(self.current_arm_pose(self.home))
        self.get_passport_move_arm_2.add_precondition(self.holding(self.nothing))
        self.get_passport_move_arm_2.add_precondition(
            Not(self.passport_corner_detected())
        )
        s1 = self.get_passport_move_arm_2.add_subtask(self.move_arm, self.over_passport)
        s2 = self.get_passport_move_arm_2.add_subtask(self.perceive_passport)
        s3 = self.get_passport_move_arm_2.add_subtask(
            self.pick_passport, self.get_passport_move_arm_2.passport
        )
        s4 = self.get_passport_move_arm_2.add_subtask(self.move_arm, self.arm_up)
        s5 = self.get_passport_move_arm_2.add_subtask(self.detect_passport_corner)
        self.get_passport_move_arm_2.set_ordered(s1, s2, s3, s4, s5)

        # move arm over passport, perceive passport, pick it up, move arm over passport
        self.get_passport_full = Method("get_passport_full", passport=type_item)
        self.get_passport_full.set_task(
            self.t_get_passport, self.get_passport_full.passport
        )
        self.get_passport_full.add_precondition(Not(self.current_arm_pose(self.home)))
        self.get_passport_full.add_precondition(self.holding(self.nothing))
        self.get_passport_full.add_precondition(Not(self.passport_corner_detected()))
        s1 = self.get_passport_full.add_subtask(self.move_arm, self.home)
        s2 = self.get_passport_full.add_subtask(self.move_arm, self.over_passport)
        s3 = self.get_passport_full.add_subtask(self.perceive_passport)
        s4 = self.get_passport_full.add_subtask(
            self.pick_passport, self.get_passport_full.passport
        )
        s5 = self.get_passport_full.add_subtask(self.move_arm, self.arm_up)
        s6 = self.get_passport_full.add_subtask(self.detect_passport_corner)
        self.get_passport_full.set_ordered(s1, s2, s3, s4, s5, s6)

        # READ MRZ AND CHIP
        # already read chip, arm in position
        self.read_mrz_chip_noop = Method("read_mrz_chip_noop", passport=type_item)
        self.read_mrz_chip_noop.set_task(
            self.t_read_mrz_chip, self.read_mrz_chip_noop.passport
        )
        self.read_mrz_chip_noop.add_precondition(self.chip_reader_used())
        self.read_mrz_chip_noop.add_precondition(self.mrz_reader_used())
        self.read_mrz_chip_noop.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.read_mrz_chip_noop.add_precondition(
            self.holding(self.read_mrz_chip_noop.passport)
        )

        # arm at chip reader, read chip
        self.read_mrz_chip_read_chip = Method(
            "read_mrz_chip_read_chip", passport=type_item
        )
        self.read_mrz_chip_read_chip.set_task(
            self.t_read_mrz_chip, self.read_mrz_chip_read_chip.passport
        )
        self.read_mrz_chip_read_chip.add_precondition(Not(self.chip_reader_used()))
        self.read_mrz_chip_read_chip.add_precondition(self.mrz_reader_used())
        self.read_mrz_chip_read_chip.add_precondition(
            self.current_arm_pose(self.over_chip)
        )
        self.read_mrz_chip_read_chip.add_precondition(
            self.holding(self.read_mrz_chip_read_chip.passport)
        )
        self.read_mrz_chip_read_chip.add_subtask(
            self.read_chip, self.read_mrz_chip_read_chip.passport
        )

        # already read mrz, move arm over chip and read chip
        self.read_mrz_chip_move_arm_2 = Method(
            "read_mrz_chip_move_arm_2", passport=type_item
        )
        self.read_mrz_chip_move_arm_2.set_task(
            self.t_read_mrz_chip, self.read_mrz_chip_move_arm_2.passport
        )
        self.read_mrz_chip_move_arm_2.add_precondition(self.mrz_reader_used())
        self.read_mrz_chip_move_arm_2.add_precondition(Not(self.chip_reader_used()))
        self.read_mrz_chip_move_arm_2.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.read_mrz_chip_move_arm_2.add_precondition(
            self.holding(self.read_mrz_chip_move_arm_2.passport)
        )
        s1 = self.read_mrz_chip_move_arm_2.add_subtask(self.move_arm, self.over_chip)
        s2 = self.read_mrz_chip_move_arm_2.add_subtask(
            self.read_chip, self.read_mrz_chip_move_arm_2.passport
        )
        self.read_mrz_chip_move_arm_2.set_ordered(s1, s2)

        # arm over mrz, read mrz and chip
        self.read_mrz_chip_read_mrz = Method(
            "read_mrz_chip_read_mrz", passport=type_item
        )
        self.read_mrz_chip_read_mrz.set_task(
            self.t_read_mrz_chip, self.read_mrz_chip_read_mrz.passport
        )
        self.read_mrz_chip_read_mrz.add_precondition(Not(self.mrz_reader_used()))
        self.read_mrz_chip_read_mrz.add_precondition(Not(self.chip_reader_used()))
        self.read_mrz_chip_read_mrz.add_precondition(
            self.current_arm_pose(self.over_mrz)
        )
        self.read_mrz_chip_read_mrz.add_precondition(
            self.holding(self.read_mrz_chip_read_mrz.passport)
        )
        s1 = self.read_mrz_chip_read_mrz.add_subtask(
            self.read_mrz, self.read_mrz_chip_read_mrz.passport
        )
        s2 = self.read_mrz_chip_read_mrz.add_subtask(self.move_arm, self.over_chip)
        s3 = self.read_mrz_chip_read_mrz.add_subtask(
            self.read_chip, self.read_mrz_chip_read_mrz.passport
        )
        self.read_mrz_chip_read_mrz.set_ordered(s1, s2, s3)

        # passport in hand, read mrz and chip
        self.read_mrz_chip_full = Method("read_mrz_chip_full", passport=type_item)
        self.read_mrz_chip_full.set_task(
            self.t_read_mrz_chip, self.read_mrz_chip_full.passport
        )
        self.read_mrz_chip_full.add_precondition(Not(self.mrz_reader_used()))
        self.read_mrz_chip_full.add_precondition(Not(self.chip_reader_used()))
        self.read_mrz_chip_full.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.read_mrz_chip_full.add_precondition(
            self.holding(self.read_mrz_chip_full.passport)
        )
        s1 = self.read_mrz_chip_full.add_subtask(self.move_arm, self.over_mrz)
        s2 = self.read_mrz_chip_full.add_subtask(
            self.read_mrz, self.read_mrz_chip_full.passport
        )
        s3 = self.read_mrz_chip_full.add_subtask(self.move_arm, self.over_chip)
        s4 = self.read_mrz_chip_full.add_subtask(
            self.read_chip, self.read_mrz_chip_full.passport
        )
        self.read_mrz_chip_full.set_ordered(s1, s2, s3, s4)

        # PLACE PASSPORT
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
        self.place_passport_place.add_subtask(
            self.place_passport_in_box,
            self.place_passport_place.passport,
            self.place_passport_place.status,
        )
        self.place_passport_place.set_ordered(s1)

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
        self.place_passport_full.add_precondition(
            self.current_arm_pose(self.unknown_pose)
        )
        self.place_passport_full.add_precondition(
            self.holding(self.place_passport_full.passport)
        )
        s1 = self.place_passport_full.add_subtask(self.move_arm, self.over_boxes)
        s2 = self.place_passport_full.add_subtask(
            self.place_passport_in_box,
            self.place_passport_full.passport,
            self.place_passport_full.status,
        )
        self.place_passport_full.set_ordered(s1, s2)

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
        self.scan_passport_get.add_precondition(Not(self.passport_status_known()))
        self.scan_passport_get.add_precondition(self.passports_available())
        s1 = self.scan_passport_get.add_subtask(
            self.t_get_passport, self.scan_passport_get.passport
        )
        s2 = self.scan_passport_get.add_subtask(
            self.t_read_mrz_chip, self.scan_passport_get.passport
        )
        s3 = self.scan_passport_get.add_subtask(
            self.inspect, self.scan_passport_get.passport
        )
        self.scan_passport_get.set_ordered(s1, s2, s3)

        # scan finished, place in box
        self.scan_passport_place = Method("scan_passport_place", passport=type_item)
        self.scan_passport_place.set_task(
            self.t_scan_passport, self.scan_passport_place.passport
        )
        self.scan_passport_place.add_subtask(
            self.t_place_passport,
            self.scan_passport_place.passport,
        )

        # no passports in passport supports refill_passports
        self.scan_passport_refill = Method("scan_passport_refill", passport=type_item)
        self.scan_passport_refill.set_task(self.t_scan_passport, self.scan_passport_refill.passport)
        self.scan_passport_refill.add_precondition(Not(self.passports_available()))
        self.scan_passport_refill.add_subtask(self.refill_passports)

        self.methods = (
            self.get_passport_noop,
            self.get_passport_detect_corner,
            self.get_passport_move_arm_1,
            self.get_passport_pick,
            self.get_passport_perceive,
            self.get_passport_move_arm_2,
            self.get_passport_full,
            self.read_mrz_chip_noop,
            self.read_mrz_chip_read_mrz,
            self.read_mrz_chip_move_arm_2,
            self.read_mrz_chip_read_chip,
            self.read_mrz_chip_full,
            self.place_passport_place,
            self.place_passport_full,
            self.place_passport_empty_box,
            self.scan_passport_get,
            self.scan_passport_place,
            self.scan_passport_refill,
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

            self.detect_passport_corner, _ = self.create_action(
                "detect_passport_corner", _callable=actions.detect_passport_corner
            )
            self.detect_passport_corner.add_precondition(
                Not(self.passport_corner_detected())
            )
            self.detect_passport_corner.add_effect(
                self.passport_corner_detected(), True
            )
            self.detect_passport_corner.add_effect(
                self.current_arm_pose(self.unknown_pose), True
            )
            self.detect_passport_corner.add_effect(
                self.current_arm_pose(self.over_mrz), False
            )
            self.detect_passport_corner.add_effect(
                self.current_arm_pose(self.arm_up), False
            )

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
            self.pick_passport.add_effect(
                self.current_arm_pose(self.unknown_pose), True
            )
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
                self.current_arm_pose(self.unknown_pose), True
            )
            self.place_passport_in_box.add_effect(self.passport_status_known(), False)
            self.place_passport_in_box.add_effect(self.mrz_reader_used(), False)
            self.place_passport_in_box.add_effect(self.chip_reader_used(), False)
            self.place_passport_in_box.add_effect(
                self.passport_corner_detected(), False
            )

            self.empty_box, [s] = self.create_action(
                "empty_box",
                status=Status,
                _callable=actions.empty_box,
            )
            self.empty_box.add_precondition(Not(self.space_in_box(s)))
            self.empty_box.add_precondition(self.passport_status_known())
            self.empty_box.add_effect(self.space_in_box(s), True)

            self.refill_passports, [] = self.create_action(
                "refill_passports",
                _callable=actions.refill_passports,
            )
            self.refill_passports.add_precondition(Not(self.passports_available()))
            self.refill_passports.add_effect(self.passports_available(), True)

    def set_state_and_goal(self, problem, goal=None) -> None:
        success = True
        if goal is None:
            problem.task_network.add_subtask(self.t_scan_passport(self.passport))
        else:
            logerr((f"Task ({goal}) is unknown! Please leave the goal empty."))
            success = False
        return success
