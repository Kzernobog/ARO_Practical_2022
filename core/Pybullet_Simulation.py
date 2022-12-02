from scipy.spatial.transform import Rotation as npRotation
from scipy.spatial.transform import Slerp as slerp
from scipy.special import comb
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import time
import yaml

from Pybullet_Simulation_base import Simulation_base

class Simulation(Simulation_base):
    """A Bullet simulation involving Nextage robot"""

    def __init__(self, pybulletConfigs, robotConfigs, refVect=None):
        """Constructor
        Creates a simulation instance with Nextage robot.
        For the keyword arguments, please see in the Pybullet_Simulation_base.py
        """
        super().__init__(pybulletConfigs, robotConfigs)
        if refVect:
            self.refVector = np.array(refVect)
        else:
            self.refVector = np.array([1,0,0])

        self.target_robot_configuration = {}
        self.current_joint_velocities = {}
        self.previous_robot_configuration = {}

        # initialise the integral in the PID controller
        self.integral = 0

        self.jointRotationAxis = {
            'base_to_dummy': np.zeros(3),  # Virtual joint
            'base_to_waist': np.zeros(3),  # Fixed joint
            'CHEST_JOINT0': np.array([0, 0, 1]),
            'HEAD_JOINT0': np.array([0, 0, 1]),
            'HEAD_JOINT1': np.array([0, 1, 0]),
            'LARM_JOINT0': np.array([0, 0, 1]),
            'LARM_JOINT1': np.array([0, 1, 0]),
            'LARM_JOINT2': np.array([0, 1, 0]),
            'LARM_JOINT3': np.array([1, 0, 0]),
            'LARM_JOINT4': np.array([0, 1, 0]),
            'LARM_JOINT5': np.array([0, 0, 1]),
            'RARM_JOINT0': np.array([0, 0, 1]),
            'RARM_JOINT1': np.array([0, 1, 0]),
            'RARM_JOINT2': np.array([0, 1, 0]),
            'RARM_JOINT3': np.array([1, 0, 0]),
            'RARM_JOINT4': np.array([0, 1, 0]),
            'RARM_JOINT5': np.array([0, 0, 1]),
            #'RHAND'      : np.array([0, 0, 0]),
            #'LHAND'      : np.array([0, 0, 0])
        }


        self.frameTranslationFromParent = {
            'base_to_dummy': np.zeros(3),  # Virtual joint
            'base_to_waist': np.zeros(3),  # Fixed joint
            'CHEST_JOINT0': np.array([0, 0, 0.267]),
            'HEAD_JOINT0': np.array([0, 0, .302]),
            'HEAD_JOINT1': np.array([0, 0, 0.066]),
            'LARM_JOINT0': np.array([0.04, 0.135, 0.1015]),
            'LARM_JOINT1': np.array([0, 0, 0.066]),
            'LARM_JOINT2': np.array([0, 0.095, -0.25]),
            'LARM_JOINT3': np.array([0.1805, 0, -0.03]),
            'LARM_JOINT4': np.array([0.1495, 0, 0]),
            'LARM_JOINT5': np.array([0, 0, -0.1335]),
            'RARM_JOINT0': np.array([0.04, -0.135, 0.1015]),
            'RARM_JOINT1': np.array([0, 0, 0.066]),
            'RARM_JOINT2': np.array([0, -0.095, -0.25]),
            'RARM_JOINT3': np.array([0.1805, 0, -0.03]),
            'RARM_JOINT4': np.array([0.1495, 0, 0]),
            'RARM_JOINT5': np.array([0, 0, -0.1335]),
            #'RHAND'      : np.array([0, 0, 0]), # optional
            #'LHAND'      : np.array([0, 0, 0]) # optional
        }

        self.headchain = {'base_to_waist': 1, 'CHEST_JOINT0': 2, 'HEAD_JOINT0':
                          3,
                          'HEAD_JOINT1': 4}

        # self.rightchain = ['base_to_waist', 'CHEST_JOINT0', 'RARM_JOINT0',
        #     'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4',
        #     'RARM_JOINT5', 'RHAND']
        self.rightchain = {'base_to_waist': 1, 'CHEST_JOINT0': 2, 'RARM_JOINT0': 11,
                           'RARM_JOINT1': 12, 'RARM_JOINT2': 13, 'RARM_JOINT3':
                           14, 'RARM_JOINT4':15,
                           'RARM_JOINT5': 16}

        # self.leftchain = ['base_to_waist', 'CHEST_JOINT0', 'LARM_JOINT0',
        #     'LARM_JOINT1',  'LARM_JOINT2',  'LARM_JOINT3',  'LARM_JOINT4',
        #     'LARM_JOINT5', 'LHAND']
        self.leftchain = {'base_to_waist': 1, 'CHEST_JOINT0': 2, 'LARM_JOINT0': 5,
                          'LARM_JOINT1': 6,  'LARM_JOINT2': 7,  'LARM_JOINT3':
                          8,  'LARM_JOINT4': 9,
                          'LARM_JOINT5': 10}


    ########## Task 1: Kinematics ##########
    # Task 1.1 Forward Kinematics
    def get_kinematic_chain(
        self,
        jointName,
        source_frame="world"):
        """
        retrieves the suitable kinematic chain that the endeffector is a part
        of. If the source frame is anything but world frame you have a
        different chain you need to return
        """

        # if jointName == "CHEST_JOINT0":
        #     return ["base_to_waist", "CHEST_JOINT0"]
        kinematic_chain = ["base_to_waist", "CHEST_JOINT0"]

        for chain in self.robotLimbs:
            if jointName in chain:
                kinematic_chain = kinematic_chain + chain[:chain.index(jointName)+1]
                break

        if source_frame != "world":
            kinematic_chain = kinematic_chain[
                kinematic_chain.index(source_frame)+1: ]

        return kinematic_chain

    def getJointRotationalMatrix(self, jointName=None, theta=None):
        """
            Returns the 3x3 rotation matrix for a joint from the axis-angle representation,
            where the axis is given by the revolution axis of the joint and the angle is theta.
        """
        if jointName == None:
            raise Exception("[getJointRotationalMatrix] \
                Must provide a joint in order to compute the rotational matrix!")
        if theta == None:
            raise Exception("[getJointRotationalMatrix] Must provide a valid \
                            theta in radians")
        # Hint: the output should be a 3x3 rotational matrix as a numpy array
        # Check the rotation axis of the joint.
        axis = self.jointRotationAxis[jointName]
        index = np.nonzero(axis)
        if len(index[0]) == 0:
            return np.identity(3)
        if (index[0] == 0):
            return np.matrix([[1, 0, 0],
                              [0, np.cos(theta), -1*np.sin(theta)],
                              [0, np.sin(theta), np.cos(theta)]])
        elif (index[0] == 1):
            return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                             [0, 1, 0],
                             [-1*np.sin(theta), 0, np.cos(theta)]])
        elif (index[0] == 2):
            return np.matrix([[np.cos(theta), -1*np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0],
                             [0, 0, 1]])
        # Since none of the joints have more than 1 rotation axis return the
        # rotation matrix corresponding to the theta passed into the function
        # if no rotation axis is mentioned in the dictionary then return an
        # identity matrix
        #return np.matrix()

    def getTransformationMatrices(self):
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
        """
        transformationMatrices = {}
        # Hint: the output should be a dictionary with joint names as keys and
        # their corresponding homogeneous transformation matrices as values.

        # RECIPE
        # iterate through joint names in one of the dictionaries
        for key in self.jointRotationAxis.keys():

            # retrieve the corresponding joint angle from getJointPos
            theta = self.getJointPos(key)

            # extract rotation matrix
            rotmat = self.getJointRotationalMatrix(key, theta)

            # add the position offset as a column
            temp = np.column_stack((rotmat, self.frameTranslationFromParent[key]))

            # add [0, 0, 0, 1] as a row
            transform_matrix = np.asarray(np.vstack((temp, np.array([0, 0, 0,
                                                                     1]))))


            assert transform_matrix.shape == (4, 4), "Incorrect transformation matrix assembled"

            # update the transformation matrix dictionary with the
            # corresponding key

            transformationMatrices[key] = transform_matrix


        return transformationMatrices

    def getJointLocationAndOrientation(self, jointName):
        """
            Returns the position and rotation matrix of a given joint using Forward Kinematics
            according to the topology of the Nextage robot.
        """
        # Remember to multiply the transformation matrices following the kinematic chain for each arm.
        # Hint: return two numpy arrays, a 3x1 array for the position vector,
        # and a 3x3 array for the rotation matrix
        #return pos, rotmat

        # TODO transformation matrices are computed everytime a joint's
        # position is estimated.
        transformation_matrices = self.getTransformationMatrices()

        transformation = np.identity(4)
        transformation[2, 3] = 0.85

        kinematic_chain = self.get_kinematic_chain(jointName)

        # check which kinematic tree the given joint exists in, accordingly
        # descend down that tree computing transformation 
        for joint in kinematic_chain:
            transformation = transformation @ transformation_matrices[joint]

        # if jointName in self.headchain.keys():
        #     for item in self.headchain.keys():
        #         print("multiplication order: {}".format(item))
        #         transformation = np.matmul(transformation,
        #                                    transformation_matrices[item])
        #         if item == jointName:
        #             break
        # elif jointName in self.rightchain.keys():
        #     for item in self.rightchain.keys():
        #         print("multiplication order: {}".format(item))
        #         transformation = np.matmul(transformation,
        #                                    transformation_matrices[item])
        #         if item == jointName:
        #             break
        # elif jointName in self.leftchain.keys():
        #     for item in self.leftchain.keys():
        #         print("multiplication order: {}".format(item))
        #         # transformation = np.matmul(transformation,
        #         #                            transformation_matrices[item])
        #         transformation = np.matmul(transformation_matrices[item],
        #                                    transformation)
        #         if item == jointName:
        #             break

        print("transformation: {}".format(transformation))
        rotmat = transformation[:3, :3]
        pos = transformation[:3, 3]



        return pos, rotmat

    def getJointPosition(self, jointName):
        """Get the position of a joint in the world frame, leave this unchanged please."""
        return self.getJointLocationAndOrientation(jointName)[0]

    def getJointOrientation(self, jointName, ref=None):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        if ref is None:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.refVector).squeeze()
        else:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ ref).squeeze()

    def getJointAxis(self, jointName):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.jointRotationAxis[jointName]).squeeze()

    def jacobianMatrix(self, endEffector):
        """Calculate the Jacobian Matrix for the Nextage Robot."""
        # You can implement the cross product yourself or use calculateJacobian().
        # Hint: you should return a numpy array for your Jacobian matrix. The
        # size of the matrix will depend on your chosen convention. You can have
        # a 3xn or a 6xn Jacobian matrix, where 'n' is the number of joints in
        # your kinematic chain.
        # return np.array()

        # estimate transformation matrices in all kinematic chains
        # iterate through joints
        # accumulate position Jacobian
        # accumulate orientation Jacobian
        # concatenate into matrix column
        position_jacobian = []
        orientation_jacobian = []

        kinematic_chain = self.get_kinematic_chain(endEffector)

        p_eff = self.getJointPosition(endEffector)
        a_eff = self.getJointAxis(endEffector)

        for joint in kinematic_chain:
            p_i = self.getJointPosition(joint)
            a_i = self.getJointAxis(joint)

            J_pos = np.cross(a_i, (p_eff - p_i))
            J_ori = np.cross(a_i, a_eff)

            position_jacobian.append(J_pos)
            orientation_jacobian.append(J_ori)

        # if endEffector in self.leftchain.keys():
        #     p_eff, end_rot = self.getJointLocationAndOrientation(endEffector)
        #     for item in self.leftchain.keys():
        #         p_i, rotmat = self.getJointLocationAndOrientation(item)
        #         a_i = rotmat @ self.jointRotationAxis[item]
        #         a_eff = end_rot @ self.jointRotationAxis[endEffector]
        #         J_pos = np.cross(a_i, (p_eff - p_i))
        #         J_ori = np.cross(a_i, a_eff)
        #         # jacobian.append(np.hstack((J_pos, J_ori)))
        #         position_jacobian.append(J_pos)
        #         orientation_jacobian.append(J_ori)
        # elif endEffector in self.rightchain.keys():
        #     p_eff, end_rot = self.getJointLocationAndOrientation(endEffector)
        #     for item in self.rightchain.keys():
        #         # pos, rotmat = self.getJointLocationAndOrientation(item)
        #         # J_pos = np.cross(self.jointRotationAxis[item], (end_pos - pos))
        #         # J_ori = np.cross(self.jointRotationAxis[item], end_rot @
        #         #                      self.jointRotationAxis[endEffector])
        #         p_i, rotmat = self.getJointLocationAndOrientation(item)
        #         a_i = rotmat @ self.jointRotationAxis[item]
        #         a_eff = end_rot @ self.jointRotationAxis[endEffector]
        #         J_pos = np.cross(a_i, (p_eff - p_i))
        #         J_ori = np.cross(a_i, a_eff)
        #         # jacobian.append(np.hstack((J_pos, J_ori)))
        #         position_jacobian.append(J_pos)
        #         orientation_jacobian.append(J_ori)


        return np.asarray(position_jacobian).T, np.asarray(orientation_jacobian).T
    # Task 1.2 Inverse Kinematics

    def inverseKinematics(
        self,
        endEffector,
        targetPosition,
        targetOrientation,
        interpolationSteps,
        maxIterPerStep,
        threshold,
        orientation_jacobian=False):
        """Your IK solver \\
        Arguments: \\
            endEffector: the jointName the end-effector \\
            targetPosition: final destination the the end-effector \\
            orientation: the desired orientation of the end-effector
                         together with its parent link \\
            interpolationSteps: number of interpolation steps
            maxIterPerStep: maximum iterations per step
            threshold: accuracy threshold
        Return: \\
            Vector of x_refs
        """
        # intermediate source frame


        # Hint: return a numpy array which includes the reference angular
        # positions for all joints after performing inverse kinematics.
        # takes in next step in interpolation and returns the dq required to
        # move the endeffector to the targetposition
        J_pos, J_ori = self.jacobianMatrix(endEffector)
        if orientation_jacobian:
            jacobian = np.hstack((J_pos, J_ori))
            trans_delta = targetPosition - self.getJointPosition(endEffector)
            ori_delta = targetOrientation - self.getJointOrientation(endEffector)
            dy = np.hstack((trans_delta, ori_delta))
        else:
            jacobian = J_pos
            dy = targetPosition - self.getJointPosition(endEffector)

        dq = np.linalg.pinv(jacobian) @ dy

        return dq

    def move_without_PD(
        self,
        endEffector,
        targetPosition,
        interpolationSteps,
        speed=0.01,
        orientation=None,
        threshold=1e-3,
        maxIter=3000,
        debug=False,
        verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """

        # iterate through joints and update joint states based on IK solver
        pltTime = []
        pltDistance = []

        # which kinematic chain does it belong to
        kinematic_chain = self.get_kinematic_chain(endEffector)

        # intialise the end effector position
        EF_init_pos = self.getJointPosition(endEffector)
        EF_init_ori = self.getJointOrientation(endEffector)

        # Linear interpolation in the end-effector space
        translation_steps = np.linspace(EF_init_pos,
                                        targetPosition,
                                        num=interpolationSteps)
        # if end-effector goal orientation is also provided then perform
        # spherical linear interpolation
        if orientation is not None:
            start_ori = npRotation.from_matrix(EF_init_ori)
            orientation_steps = slerp(np.linspace(0, 1, interpolationSteps),
                                      [start_ori, orientation])


        # iterate through the interpolated end effector positions and perform
        # IK on each interpolated point
        for i in range(interpolationSteps):
            # get translation and orientation goals
            translation_goal = translation_steps[i, :]
            if orientation is not None:
                orientation_goal = orientation_steps[i, :]

            # estimate the differential in configuration of the joints in the
            # robot
            if orientation is not None:
                dq = self.inverseKinematics(endEffector,
                                               translation_goal,
                                               orientation_goal,
                                               interpolationSteps,
                                               maxIterPerStep=maxIter,
                                               threshold=threshold,
                                               orientation_jacobian=True)
            else:
                dq = self.inverseKinematics(endEffector,
                                               translation_goal,
                                               targetOrientation=None,
                                               interpolationSteps=interpolationSteps,
                                               maxIterPerStep=maxIter,
                                               threshold=threshold,
                                               orientation_jacobian=False)


            # update the joint states 
            joint_states = {}
            for idx, joint in enumerate(kinematic_chain):
                joint_states[joint] =  self.getJointPos(joint) + dq[idx]

            updated_pose_values = np.array(list(joint_states.values()))

            # store the updated joint states in a global variable
            self.target_robot_configuration = {key: value for key, value in
                                               zip(kinematic_chain,
                                                   updated_pose_values)}
            # move the joints in the simulation
            self.tick_without_PD()

            # log for graphing and debug
            dist_to_target_position = np.linalg.norm(
                targetPosition - self.getJointPosition(endEffector))
            pltDistance.append(dist_to_target_position)

            if (dist_to_target_position < threshold):
                pltTime = np.linspace(0,
                                      self.dt*len(pltDistance),
                                      len(pltDistance))

                return pltTime, pltDistance

        pltTime = np.linspace(0,
                              self.dt*len(pltDistance),
                              len(pltDistance))

        return pltTime, pltDistance
        # raise Exception("Failed to reach target pose")

    def tick_without_PD(self):
        """Ticks one step of simulation without PD control. """
        # Iterate through all joints and update joint states.
        # For each joint, you can use the shared variable self.jointTargetPos.
        for jointName in self.target_robot_configuration.keys():
            self.p.resetJointState(self.robot,
                                   self.jointIds[jointName],
                                   self.target_robot_configuration[jointName])

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

    ########## Task 2: Dynamics ##########
    # Task 2.1 PD Controller
    def calculateTorque(self, x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd):
        """ This method implements the closed-loop control \\
        Arguments: \\
            x_ref - the target position \\
            x_real - current position \\
            dx_ref - target velocity \\
            dx_real - current velocity \\
            integral - integral term (set to 0 for PD control) \\
            kp - proportional gain \\
            kd - derivetive gain \\
            ki - integral gain \\
        Returns: \\
            u(t) - the manipulation signal
        """
        p = kp * (x_ref - x_real)
        i = ki * integral
        d = kd * (dx_ref - dx_real)

        return (p + d + i)

    # Task 2.2 Joint Manipulation
    def moveJoint(self, joint, targetPosition, targetVelocity, verbose=False):
        """ This method moves a joint with your PD controller. \\
        Arguments: \\
            joint - the name of the joint \\
            targetPos - target joint position \\
            targetVel - target joint velocity
        """
        def toy_tick(x_ref, x_real, dx_ref, dx_real, integral):
            # loads your PID gains
            jointController = self.jointControllers[joint]
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Start your code here: ###
            # Calculate the torque with the above method you've made
            torque = self.calculateTorque(x_ref, x_real, dx_ref, dx_real,
                                          integral, kp, ki, kd)
            ### To here ###

            pltTorque.append(torque)

            # send the manipulation signal to the joint
            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )
            # calculate the physics and update the world
            self.p.stepSimulation()
            time.sleep(self.dt)

        targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)

        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = (
            [], [], [], [], [], [])

        integral = 0
        q_old = 0 # this to compute instantaneous velocity using backward eulers

        for c in range(1000):
            # obtain current position
            q_joint = self.getJointPos(joint)

            # compute the integral error
            integral += self.dt * (targetPosition - q_joint)

            # compute joint velocity using backward eulers
            q_dot_joint = (q_joint - q_old) / self.dt

            # step the robot by a state
            toy_tick(
                x_ref=targetPosition,
                x_real=q_joint,
                dx_ref=targetVelocity,
                dx_real=q_dot_joint,
                integral=integral
            )

            # log for plotting
            pltTarget.append(targetPosition)
            pltPosition.append(q_joint)
            pltVelocity.append(q_dot_joint)
            pltTime.append(c * self.dt)

            # update variables
            q_old = q_joint

            # compute control command from target position and velocity

        pltTorqueTime = np.linspace(0, self.dt * len(pltTorque), len(pltTorque))

        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    def move_with_PD(
        self,
        endEffector,
        targetPosition,
        interpolationSteps,
        control_freq,
        speed=0.01,
        orientation=None,
        threshold=1e-3,
        maxIter=3000,
        debug=False,
        verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output
        from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        #TODO add your code here
        # Iterate through joints and use states from IK solver as reference states in PD controller.
        # Perform iterations to track reference states using PD controller until reaching
        # max iterations or position threshold.

        # Hint: here you can add extra steps if you want to allow your PD
        # controller to converge to the final target position after performing
        # all IK iterations (optional).

        #return pltTime, pltDistance

        # iterate through joints and update joint states based on IK solver
        pltTime = []
        pltDistance = []

        # which kinematic chain does it belong to
        kinematic_chain = self.get_kinematic_chain(endEffector)

        # intialise the end effector position
        EF_init_pos = self.getJointPosition(endEffector)
        EF_init_ori = self.getJointOrientation(endEffector)


        # Linear interpolation in the end-effector space
        translation_steps = np.linspace(EF_init_pos,
                                        targetPosition,
                                        num=interpolationSteps)
        # if end-effector goal orientation is also provided then perform
        # spherical linear interpolation
        if orientation is not None:
            start_ori = npRotation.from_matrix(EF_init_ori)
            orientation_steps = slerp(np.linspace(0, 1, interpolationSteps),
                                      [start_ori, orientation])


        # some interpolation loop initialisations
        self.target_robot_configuration = {}
        # initalise previous configuration
        for joint in kinematic_chain:
            self.previous_robot_configuration[joint] = self.getJointPos(joint)

        # iterate through the interpolated end effector positions and perform
        # IK on each interpolated point
        for i in range(interpolationSteps):
            # get translation and orientation goals
            translation_goal = translation_steps[i, :]
            if orientation is not None:
                orientation_goal = orientation_steps[i, :]

            # estimate the differential in configuration of the joints in the
            # robot
            if orientation is not None:
                dq = self.inverseKinematics(endEffector,
                                               translation_goal,
                                               orientation_goal,
                                               interpolationSteps,
                                               maxIterPerStep=maxIter,
                                               threshold=threshold,
                                               orientation_jacobian=True)
            else:
                dq = self.inverseKinematics(endEffector,
                                               translation_goal,
                                               targetOrientation=None,
                                               interpolationSteps=interpolationSteps,
                                               maxIterPerStep=maxIter,
                                               threshold=threshold,
                                               orientation_jacobian=False)


            # update the joint states 
            joint_states = {}
            for idx, joint in enumerate(kinematic_chain):
                joint_states[joint] =  self.getJointPos(joint) + dq[idx]

            updated_pose_values = np.array(list(joint_states.values()))

            # store the updated joint states in a global variable
            self.target_robot_configuration = {key: value for key, value in
                                               zip(kinematic_chain,
                                                   updated_pose_values)}

            # the outer loop assumes a straight line trajectory and
            # interpolates between initial position and target position. This
            # inner loop executes the robot control loop between two immediate
            # interpolated points.
            # TODO: implement cubic interpolation between current joint
            # configuration and control set point.
            self.integral = 0.0
            for step in range(control_freq):
                # move the joints in the simulation
                self.tick()



            # log for graphing and debug
            dist_to_target_position = np.linalg.norm(
                targetPosition - self.getJointPosition(endEffector))
            pltDistance.append(dist_to_target_position)

            if (dist_to_target_position < threshold):
                pltTime = np.linspace(0,
                                      self.dt*len(pltDistance),
                                      len(pltDistance))

                return pltTime, pltDistance

        pltTime = np.linspace(0,
                              self.dt*len(pltDistance),
                              len(pltDistance))

        return pltTime, pltDistance
        # raise Exception("Failed to reach target pose")


    def tick(self):
        """Ticks one step of simulation using PD control."""
        # Iterate through all joints and update joint states using PD control.
        for joint in self.joints:
            # skip dummy joints (world to base joint)
            jointController = self.jointControllers[joint]
            if jointController == 'SKIP_THIS_JOINT':
                continue

            # disable joint velocity controller before apply a torque
            self.disableVelocityController(joint)

            # loads your PID gains
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Implement your code from here ... ###

            # get target position
            targetPosition = self.target_robot_configuration[joint]
            # get current position
            q_joint = self.getJointPos(joint)
            # get target velocity
            targetVelocity = 0.0
            # get previous configuration
            # TODO: update previous robot configuration somewhere
            q_old = self.previous_robot_configuration[joint]
            # get current velocity
            q_dot_joint = (q_joint - q_old) / self.dt
            # get integral
            self.integral += self.dt * (targetPosition - q_joint)
            torque = self.calculateTorque(
                x_ref=targetPosition,
                x_real=q_joint,
                dx_ref=targetVelocity,
                dx_real=q_dot_joint,
                integral=self.integral,
                kp=kp,
                ki=ki,
                kd=kd)

            # update the previous robot configuration
            self.previous_robot_configuration[joint] = q_joint
            ### ... to here ###

            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )

            # Gravity compensation
            # A naive gravitiy compensation is provided for you
            # If you have embeded a better compensation, feel free to modify
            compensation = self.jointGravCompensation[joint]
            self.p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.jointIds[joint],
                forceObj=[0, 0, -compensation],
                posObj=self.getLinkCoM(joint),
                flags=self.p.WORLD_FRAME
            )
            # Gravity compensation ends here

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

    ########## Task 3: Robot Manipulation ##########
    def cubic_interpolation(self, points, nTimes=100):
        """
        Given a set of control points, return the
        cubic spline defined by the control points,
        sampled nTimes along the curve.
        """
        #TODO add your code here
        # Return 'nTimes' points per dimension in 'points' (typically a 2xN array),
        # sampled from a cubic spline defined by 'points' and a boundary condition.
        # You may use methods found in scipy.interpolate

        #return xpoints, ypoints
        pass

    # Task 3.1 Pushing
    def dockingToPosition(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005,
            threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

    # Task 3.2 Grasping & Docking
    def clamp(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005, threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

 ### END
