#!/usr/bin/env bash
set -e

export ROS_DISTRO="noetic"

FILEPATH=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
cd "$FILEPATH"/.. || exit


sudo apt-get update -qq
sudo apt-get install -qq -y python3-vcstool python3-pip python3-rosdep python3-catkin-tools build-essential git

# Install demo dependencies.
vcs import --recursive --skip-existing --workers 1 < "$FILEPATH"/dependencies.repos
vcs pull --workers 1

sudo rosdep init > /dev/null 2>&1 || true
rosdep update
rosdep install --from-paths ../ --ignore-src -r -y --rosdistro ${ROS_DISTRO} --skip-keys=python-rospkg --skip-keys=python-yaml

# Install Unified Planning Embedded Systems Bridge
pip install up-esb==0.0.145
pip install up-aries==0.0.8
pip install 'unified-planning==0.5.0.438.dev1'
