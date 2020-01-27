# SOLAR_GP_ROS

## Getting Started

### Dependencies
This package depends on the below packages in order to compile and operate:
- teleop_utils (<https://github.com/bpwilcox/teleop_utils>)
- baxter_interface (<https://github.com/RethinkRobotics/baxter_interface>)
  - May require additional baxter dependencies

Though the core SOLAR_GP algorithm does not depend on Baxter, currently the data buffer uses the `EndpointState` custom msg 
from the `baxter_core_msgs` package (a dependency of `baxter_interface`). This is may change in the future.

### Running on Baxter
This package is currently based around the Rethink Robotics Baxter robot implementation of the SOLAR_GP algorithm. In the
[Baxter launch](https://github.com/bpwilcox/SOLAR_GP_ROS/tree/master/launch/baxter) and
[Baxter scripts](https://github.com/bpwilcox/SOLAR_GP_ROS/tree/master/scripts/robot/baxter) directories are Baxter-specific
implementations of core classes.

#### Run a live teleoperation experiments using an Xbox Controller:
- launch Baxter simulator in Gazebo or real Baxter robot (see )
- set teleop_device in [run_baxter.launch](https://github.com/bpwilcox/SOLAR_GP_ROS/blob/master/launch/baxter/run_baxter.launch)
to "xbox"
- run `roslaunch SOLAR_GP_ROS run_baxter.launch`

#### Run a live trajectory playback of rosbag:
- launch Baxter simulator in Gazebo or real Baxter robot (see )
- set `teleop_device` in [run_baxter.launch](https://github.com/bpwilcox/SOLAR_GP_ROS/blob/master/launch/baxter/run_baxter.launch)
to "bag"
- set `bagfile` to path of trajectory rosbag
- run `roslaunch SOLAR_GP_ROS run_baxter.launch`

### Using your own robot
In order to the SOLAR_GP algorithm with your own robot, you need to implement derived classes of the following base classes:
- [SolarTrainer](https://github.com/bpwilcox/SOLAR_GP_ROS/blob/master/scripts/robot_controller.py)
- [SolarPredictor](https://github.com/bpwilcox/SOLAR_GP_ROS/blob/master/scripts/predictor.py)
- [RobotController](https://github.com/bpwilcox/SOLAR_GP_ROS/blob/master/scripts/trainer.py)

In these derived classes, you can set up robot-specific variables and API, and implement robot-specific overrides
of the base class functions in order to perform training, prediction, and control.

After creating the derived classes, you can create a `main` to initialize and run the derived classes. See the below Baxter
implementations for examples:
- [BaxterTrainer](https://github.com/bpwilcox/SOLAR_GP_ROS/blob/master/scripts/robot/baxter/baxter_train.py)
- [BaxterPredictor](https://github.com/bpwilcox/SOLAR_GP_ROS/blob/master/scripts/robot/baxter/baxter_predict_teleop.py)
- [BaxterController](https://github.com/bpwilcox/SOLAR_GP_ROS/blob/master/scripts/robot/baxter/baxter_control.py)
