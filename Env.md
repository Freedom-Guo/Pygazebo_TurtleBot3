# Multi-Robot Collision Avoidance 3D

---

## TODO List

1. add dynamic model (@david)
2. velocity smooth (bang-bang control)
3. improve performance of the collision detection function

---

## Environment

* Python 3.5
* Ubuntu 16.04
* Tensorflow==1.12
* Gazebo>=9.6

---

## Install

1. install python lib.

```shell
pip install numpy
```

2. install RVO2-3D

``` shell
git clone https://github.com/mtreml/Python-RVO2-3D
cd Python-RVO2-3D
pip install -e .
```

3. git clone this repo

``` shell
git clone git@github.com:wangdawei1996/3D_collisionavoidance.git
```

4. install pygazebo

``` shell
cd pygazebo
mkdir build
cd build
cmake ..
make
cd ..
pip install -e .
```

---

## Run

1. run training program

``` shell
python run.py # train ppo (on-policy)
python run_ddpg.py --train # train ddpg (off-policy)
```

2. run rvo

``` shell
python rvo_run.py
```

---

## Change Log

1. 0.1.0

* generate world file automatially
* simple gazebo_env.py which support 50 robots

2. 0.2.0

* add ppo algorithm

3. 0.3.0

* add simple noise
* add rviz visualizer
* increase the maximum number of drones
* some improvements on performances

4. 0.3.5

* add rvo2-3d

5. 0.4.0

* add running mean std
* add energy reward

6. 0.4.1

* rm energy reward
* scale approaching reward

7. 0.4.2

* add pygazebo
* rm ros

8. 0.4.3

* add ddpg training code