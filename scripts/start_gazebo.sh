#! /bin/bash
cd ..
cp -r models/* ~/.gazebo/models/
cd world
W_FILE="${PWD}/multi_robots.world"
python create_multi_robots.py --robot_num=$1 --env_size=$2