# april_krem

## Installation

ROS noetic and Python 3 are assumed to be available. [vcs](https://pypi.org/project/vcstool/) will be installed when needed.

```bash
./install-deps.sh
./build.sh
```

## Launch

Replace `uc?` with one of the use case numbers, see Launch Parameters below.
```bash
roslaunch april_krem krem.launch use_case:=uc? temporal_actions:=false enable_monitor:=false goal:="" non_robot_actions_timeout:=20.0 robot_actions_timeout:=120.0
```

### Launch Parameters

- **use_case**: specify the use case, possible values (uc1, uc2, uc3, uc5, uc5_2, uc5_3, uc5_4, uc4, uc6)
- **enable_monitor**: enables monitoring of pre- and postconditions of actions before and after executing them
- **goal**: empty string to run whole use case scenario
- **non_robot_actions_timeout**: timeout for actions that do not use the robotic arm e.g. perceive,
  conveyor actions, etc. Default is 20 seconds.
- **robot_actions_timeout**: timeout for actions that to use the robotic arm e.g. pick, place, etc. Default is 120 seconds.

## Logging

Logs are created and put into a **log file**, published on a **ros topic** and put into the **console output** of KREM.

- Log file location: ```april_krem/logs/krem.log```
- Rostopic: ```/krem/logs std_msgs/String```

</br>

The following metrics are logged:

- Planning time in seconds
- Number of replans due to errors
- Number of human intervention actions
- Execution time of performed actions in seconds
- Execution time of each cycle in seconds
- Number of cycles done
- Overall execution time of all cycles in seconds
