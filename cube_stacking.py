import time
import numpy as np
import py_trees
import RobotDART as rd
from transitions import Machine
import dartpy  # OSX breaks if this is imported before RobotDART


from utils import create_grid, create_problems, damped_pseudoinverse

dt = 0.001  # you are NOT allowed to change this
simulation_time = 40.0  # you are allowed to change this
total_steps = int(simulation_time / dt)

#########################################################
# DO NOT CHANGE ANYTHING IN HERE
# Create robot
robot = rd.Franka(int(1. / dt))
init_position = [0., np.pi / 4., 0., -
                 np.pi / 4., 0., np.pi / 2., 0., 0.04, 0.04]

robot.set_positions(init_position)
robot.set_position_enforced(False)
robot_init = robot.clone_ghost()


max_force = 5.
robot.set_force_lower_limits(
    [-max_force, -max_force], ["panda_finger_joint1", "panda_finger_joint2"])
robot.set_force_upper_limits([max_force, max_force], [
                             "panda_finger_joint1", "panda_finger_joint2"])
#########################################################
robot.set_actuator_types("servo")  # you can use torque here

#########################################################
# DO NOT CHANGE ANYTHING IN HERE
# Create boxes
box_positions = create_grid()

box_size = [0.04, 0.04, 0.04]

# Red Box
# Random cube position
red_box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[red_box_pt][0],
            box_positions[red_box_pt][1], box_size[2] / 2.0]
red_box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [
                              0.9, 0.1, 0.1, 1.0], "red_box")


# Green Box
# Random cube position
green_box_pt = np.random.choice(len(box_positions))

while green_box_pt == red_box_pt:
    green_box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[green_box_pt][0],
            box_positions[green_box_pt][1], box_size[2] / 2.0]
green_box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [
                                0.1, 0.9, 0.1, 1.0], "green_box")

# Blue Box
# Random cube position
box_pt = np.random.choice(len(box_positions))

while box_pt == green_box_pt or box_pt == red_box_pt:
    box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[box_pt][0],
            box_positions[box_pt][1], box_size[2] / 2.0]
blue_box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [
                               0.1, 0.1, 0.9, 1.0], "blue_box")
#########################################################
# Goal Box
# Random cube position
box_pt = np.random.choice(len(box_positions))

while box_pt == green_box_pt or box_pt == red_box_pt:
    box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[box_pt][0],
            box_positions[box_pt][1], box_size[2] / 2.0]
goal_box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [
    0.1, 0.1, 0.9, 1.0], "goal_box")


# Random cube position
box_pt = np.random.choice(len(box_positions))

while box_pt == green_box_pt or box_pt == red_box_pt:
    box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[box_pt][0],
            box_positions[box_pt][1], box_size[2] / 2.0]
goal_box_1 = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [
                                 0.1, 0.1, 0.9, 1.0], "goal_box")


#########################################################
# PROBLEM DEFINITION
# Choose problem
problems = create_problems()
problem_id = np.random.choice(len(problems))
problem = problems[problem_id]

print('We want to put the', problem[2], 'cube on top of the', problem[1],
      'and the', problem[1], 'cube on top of the', problem[0], 'cube.')
#########################################################


# find the order of the boxes the robot has to move to solve the problem

if problem[1] == 'red':
    first_box_name = red_box

elif problem[1] == 'green':
    first_box_name = green_box
else:
    first_box_name = blue_box

if problem[0] == 'red':
    second_box_name = red_box
elif problem[0] == 'green':
    second_box_name = green_box
else:
    second_box_name = blue_box

if problem[2] == 'red':
    third_box_name = red_box
elif problem[2] == 'green':
    third_box_name = green_box
else:
    third_box_name = blue_box

# Create Graphics
gconfig = rd.gui.Graphics.default_configuration()
gconfig.width = 1280  # you can change the graphics resolution
gconfig.height = 960  # you can change the graphics resolution
graphics = rd.gui.Graphics(gconfig)

# Create simulator object
simu = rd.RobotDARTSimu(dt)
simu.set_collision_detector("fcl")  # you can use bullet here
simu.set_control_freq(100)
simu.set_graphics(graphics)
graphics.look_at((0., 4.5, 2.5), (0., 0., 0.25))
simu.add_checkerboard_floor()
simu.add_robot(robot)
simu.add_robot(red_box)
simu.add_robot(blue_box)
simu.add_robot(green_box)
#########################################################
tmp = 0
flag = None

class PITask:
    def __init__(self, target, dt, Kp=10., Ki=0.1, flag=None):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._sum_error = 0
        self._flag = flag

    def set_target(self, target):
        self._target = target

    # function to compute error
    def error(self, tf):
        
        #compute error directly in world frame
        if self._flag == True:
            rot_error = rd.math.logMap(
                self._target.rotation() @ tf.rotation().T)
        else:
            rot_error = rd.math.logMap(
                self._target.rotation() @ self._target.rotation())
            # rot_error = rd.math.logMap(
            #     robot_init.body_pose("panda_ee").rotation() @ tf.rotation().T)

        lin_error = self._target.translation() - tf.translation()

        return np.r_[rot_error, lin_error]

    def update(self, current):
        error_in_world_frame = self.error(current)
        self._sum_error = self._sum_error + error_in_world_frame * self._dt
        return self._Kp * error_in_world_frame + self._Ki * self._sum_error


class ReachTarget(py_trees.behaviour.Behaviour):
    def __init__(self, robot, tf_desired, dt, goal_box, name, flag):
        super(ReachTarget, self).__init__(name)
        # robot
        self.robot = robot
        # end-effector name
        self.eef_link_name = "panda_ee"
        # set target tf
        self.tf_desired = dartpy.math.Isometry3()
        self.tf_desired.set_translation(tf_desired.translation())
        self.tf_desired.set_rotation(tf_desired.rotation())
        # dt
        self.dt = dt
        self.flag = flag
        self.name = name

        # goal box
        self.goal_box = goal_box

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def setup(self):
        self.logger.debug("%s.setup()->does nothing" %
                          (self.__class__.__name__))

    def initialise(self):
        self.logger.debug("%s.initialise()->init controller" %
                          (self.__class__.__name__))
        self.Kp = 2.  # Kp could be an array of 6 values
        self.Ki = 0.01  # Ki could be an array of 6 values
        goal_box_pose = self.goal_box.body_pose(0)
        # if we want the robot to go to its starting position before moving the box
        if (self.flag == 1):
            # print(self.goal_box.body_pose(eef_link_name))
            goal_box_pose = self.goal_box.body_pose(eef_link_name)

            self.controller = PITask(
                goal_box_pose, self.dt, self.Kp, self.Ki, True)
        #if the robot completed all the tasks and now we want to end the simulation
        elif (self.flag == 2):
            simu.stop_sim()
            goal_box_pose = self.goal_box.body_pose(eef_link_name)

            self.controller = PITask(
                goal_box_pose, self.dt, self.Kp, self.Ki, True)
        #if we want the robot to go to the box position
        else:
            self.controller = PITask(
                goal_box_pose, self.dt, self.Kp, self.Ki, False)

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        # control the robot
        tf = self.robot.body_pose(self.eef_link_name)
        vel = self.controller.update(tf)
        jac = self.robot.jacobian(self.eef_link_name)  # this is in world frame
        # np.linalg.pinv(jac) # get pseudo-inverse
        jac_pinv = damped_pseudoinverse(jac)
        cmd = jac_pinv @ vel
        # alpha = 10.
        # cmd = alpha * (jac.T @ vel) # using jacobian transpose
        self.robot.set_commands(cmd)
        # if error too small, report success
        err = np.linalg.norm(self.controller.error(tf))
        if err < 1e-3:

            init_time = time.time()
            simu.step()

            delay = 0.8
            #set a small delay to make sure the error is small for a while 
            #(so if the gripper moves the box even a little bit, it will not be considered as a success)
            while (time.time()-init_time < delay):
                simu.step()
                simu.step_world()
                err = np.linalg.norm(self.controller.error(tf))
                #if the error is small after the delay continue to close the gripper
                if err < 1e-3:
                    
                    gripper_velocity_command = 5.0

                    # Set the commands for both finger joints to close the gripper
                    self.robot.set_commands([gripper_velocity_command], [
                                            "panda_finger_joint1"])
                    simu.step_world()
                    init_time2 = time.time()
                    #set a small delay to make sure the gripper is closed
                    delay2 = 1.5
                    while (time.time()-init_time2 < delay2):
                        simu.step()
                        # simu.step_world()

                    new_status = py_trees.common.Status.SUCCESS
        if new_status == py_trees.common.Status.SUCCESS:
            self.feedback_message = "Reached target"
            self.logger.debug("%s.update()[%s->%s][%s]" % (
                self.__class__.__name__, self.status, new_status, self.feedback_message))
        else:
            self.feedback_message = "Error: {0}".format(err)
            self.logger.debug("%s.update()[%s][%s]" % (
                self.__class__.__name__, self.status, self.feedback_message))
        return new_status

    def terminate(self, new_status):
        self.logger.debug(
            "%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


# get end-effector pose
eef_link_name = "panda_ee"
tf_desired = robot.body_pose(eef_link_name)
vec_desired = robot.body_pose_vec(eef_link_name)


# Behavior Tree
py_trees.logging.level = py_trees.logging.Level.DEBUG

# Create tree root
root = py_trees.composites.Parallel(
    name="Root", policy=py_trees.common.ParallelPolicy.SuccessOnOne())
# Create sequence node (for sequential targets)
blackboard = py_trees.blackboard.Blackboard()
sequence = py_trees.composites.Sequence(name="Sequence", memory=blackboard)
# Target A
trA = ReachTarget(robot, tf_desired, dt, first_box_name, "Reach Target A", 0)
# add target to sequence node
sequence.add_child(trA)

# Target B
tf = dartpy.math.Isometry3()
tf.set_translation([tf_desired.translation()[0], tf_desired.translation()[
                   1], tf_desired.translation()[2]])
tf.set_rotation(tf_desired.rotation())
# we want to place the second box of the stack on top of the first box.
# so we need to move the gripper up by a little bit
goal_box_pose = [0., 0., 0., second_box_name.body_pose(0).translation()[0], second_box_name.body_pose(
    0).translation()[1], second_box_name.body_pose(0).translation()[2] + 0.08]
goal_box.set_positions(goal_box_pose)
trB = ReachTarget(robot, tf, dt, goal_box, "Reach Target B", 0)
# add target to sequence node
sequence.add_child(trB)


# Target Initial
tf = dartpy.math.Isometry3()
tf.set_translation([tf_desired.translation()[0],
                   tf_desired.translation()[1], tf_desired.translation()[2]])
tf.set_rotation(tf_desired.rotation())
# we go to the initial position of the robot to avoid collisions with the box
trIn = ReachTarget(robot, tf, dt, robot_init, "Reach Target Initial", 1)
# add target to sequence node
sequence.add_child(trIn)

# Target C
tf = dartpy.math.Isometry3()
tf.set_translation([tf_desired.translation()[0],
                   tf_desired.translation()[1], tf_desired.translation()[2]])
tf.set_rotation(tf_desired.rotation())
# we go to pickup the third box of the stack
trC = ReachTarget(robot, tf, dt, third_box_name, "Reach Target C", 0)
# add target to sequence node
sequence.add_child(trC)


# Target D
tf = dartpy.math.Isometry3()
tf.set_translation([tf_desired.translation()[0],
                   tf_desired.translation()[1], tf_desired.translation()[2]])
tf.set_rotation(tf_desired.rotation())
goal_box_pose_1 = [0., 0., 0., second_box_name.body_pose(0).translation()[0], second_box_name.body_pose(
    0).translation()[1], second_box_name.body_pose(0).translation()[2] + 0.16]
goal_box_1.set_positions(goal_box_pose_1)
# we want to place the third box of the stack on top of the second box.
# so we need to move the gripper up by a little bit of the second box
trD = ReachTarget(robot, tf, dt, goal_box_1, "Reach Target D", 0)
# add target to sequence node
sequence.add_child(trD)

# Target End
tf = dartpy.math.Isometry3()
tf.set_translation([tf_desired.translation()[0],
                   tf_desired.translation()[1], tf_desired.translation()[2]])
tf.set_rotation(tf_desired.rotation())
trEnd = ReachTarget(robot, tf, dt, robot_init, "Reach Target End", 2)
# add target to sequence node
sequence.add_child(trEnd)


# Add sequence to tree
root.add_child(sequence)

# Render tree structure
# py_trees.display.render_dot_tree(root)

# tick once
root.tick_once()
cnt = 0

for step in range(total_steps):
    if (simu.schedule(simu.control_freq())):

        root.tick_once()

    if (simu.step_world()):
        break
