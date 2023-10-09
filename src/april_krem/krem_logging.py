import os
import rospkg
import rospy
import logging

from std_msgs.msg import String


class KREMLogging:
    def __init__(self):
        self.cycle_complete = False
        self.cycle_counter = 0
        self.cycle_start_time = None
        self.overall_cycle_time = 0.0

        # wait for human intervention counter
        self.wfhi_counter = 0
        self.error_replan_counter = 0

        formatter = logging.Formatter("%(asctime)s: %(message)s")

        filehandler = logging.FileHandler(
            os.path.join(rospkg.RosPack().get_path("april_krem"), "logs", "krem.log")
        )
        filehandler.setLevel(logging.ERROR)
        filehandler.setFormatter(formatter)
        self.krem_logger = logging.getLogger()
        self.krem_logger.addHandler(filehandler)
        self.krem_logger.setLevel(logging.ERROR)

        self._log_pub = rospy.Publisher("/krem/logs", String, queue_size=10)

    def log_info(self, msg):
        self.krem_logger.critical(msg)
        self._log_pub.publish(msg)
        print(msg)
