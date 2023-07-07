cmake_minimum_required(VERSION 3.0.2)
project(april_krem)

find_package(catkin REQUIRED)

catkin_python_setup()


catkin_package(
)

include_directories(
)

catkin_install_python(PROGRAMS
  nodes/UP_executor_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
