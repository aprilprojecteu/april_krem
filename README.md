# april_krem

## Launch

```bash
roslaunch april_krem krem.launch use_case:=uc? enable_monitor:=false goal:="" non_robot_actions_timeout:=20 robot_actions_timeout:=120
```

### Launch Parameters

- **use_case**: specify the use case, possible values (uc1, uc2, uc3, uc4, uc6)
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
- Overall exection time of all cycles in seconds
