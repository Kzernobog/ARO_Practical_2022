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
import sys

from Pybullet_Simulation_base import Simulation_base
# FIX: controlling for orientation is absolutely shit! needs to be debugged
# TEST: is `move_without_PD` working with orientation control
# FAIL: getting wierd configurations
# TEST: tested with better target positions and target orientations
# PASS:
# TEST: `move_with_PD` still misbehaving
# FAIL:

# FIX: get_kinematic_chain is being called too many times, maybe make it a
# class variable
# FIX: maybe change the 'base_to_waist' frames translation from [0,0,0] to
# [0,0,0.85]. As of now this is hard coded in the forward kinematics
# QUE: Am i computing the transformation matrices to many times? keep in mind
# that every time i call `getJointLocationAndOrientation` i am computing the
# transformation matrices of all joints in the kinematic chain.


class Simulation(Simulation_base):
    """A Bullet simulation involving Nextage robot"""

    def __init__(self, pybulletConfigs, robotConfigs, refVect=None):
        """Constructor
        Creates a simulation instance with Nextage robot.
        For the keyword arguments, please see in the 
        Pybullet_Simulation_base.py
        """
        super().__init__(pybulletConfigs, robotConfigs)
        if refVect:
            self.refVector = np.array(refVect)
        else:
            self.refVector = np.array([1,0,0])

        self.target_robot_configuration = {}
        self.current_joint_velocities = {}
        self.previous_robot_configuration = {}
        self.rest_robot_configuration = {}
        self.initialised = False

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
        sourceFrame):
        """Returns kinematic chain

        Retrieves the suitable kinematic chain that the endeffector is a part
        of. If the source frame is anything but `world` frame you have a
        different chain you need to return. Specify source joint name according
        to naming in the coursework assignment document

        @param jointName: The joint whose kinematic chain needs to be known
        @type jointName: str
        @param sourceFrame: The frame w.r.t which the kinematic chain
        transformations are being estimated
        @type sourceFrame: str
        @returns kinematic_chain: The entire kinematic chain from source_frame
        @rtype: list(str)
        to jointName
        """

        # if jointName == "CHEST_JOINT0":
        #     return ["base_to_waist", "CHEST_JOINT0"]
        kinematic_chain = ["base_to_waist", "CHEST_JOINT0"]

        for chain in self.robotLimbs:
            if jointName in chain:
                kinematic_chain = kinematic_chain + chain[:chain.index(jointName)+1]
                break

        if sourceFrame != "world":
            kinematic_chain = kinematic_chain[
                kinematic_chain.index(sourceFrame)+1: ]

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

    def getJointLocationAndOrientation(self, jointName, sourceFrame):
        """Estimate position and Rotation matrix w.r.t source frame

        Returns the position and rotation matrix of a given joint using
        Forward Kinematics w.r.t the given source frame. This is accordance
        witht he topology of the Nextage robot

        Args:
            @param jointName (str): name of the joint whose transformation
            needs to be estimated
            @param sourceFrame (str): name of the source joint/frame w.r.t
            which the transformation needs to be estimated.
        Returns:
            position (np.ndarray): position of the joint named `jointName`
            orientation (np.ndarray): orientation of the joint named
            `jointName`
        """
        transformation_matrices = self.getTransformationMatrices()

        transformation = np.identity(4)

        # since we start with the `base_to_waist` frame this transformation
        # needs to be done
        if sourceFrame == 'world':
            transformation[2, 3] = 0.85

        kinematic_chain = self.get_kinematic_chain(jointName, sourceFrame)

        # check which kinematic tree the given joint exists in, accordingly
        # descend down that tree computing transformation 
        for joint in kinematic_chain:
            transformation = transformation @ transformation_matrices[joint]

        rotmat = npRotation.from_matrix(transformation[:3, :3]).as_matrix()
        pos = transformation[:3, 3]

        return pos, rotmat

    def getTransformationMatrix(self, jointName, sourceFrame):
        """Gets the transformation matrix of the specified joint

        Returns the transformatoin matrix of a given joint using
        Forward Kinematics w.r.t the given source frame. This is accordance
        witht he topology of the Nextage robot

        Args:
            @param jointName (str): name of the joint whose transformation
            needs to be estimated
            @param sourceFrame (str): name of the source joint/frame w.r.t
            which the transformation needs to be estimated.
        Returns:
            transformation (np.ndarray): transformation of the joint named `jointName`
        """
        transformation_matrices = self.getTransformationMatrices()

        transformation = np.identity(4)

        # since we start with the `base_to_waist` frame this transformation
        # needs to be done
        if sourceFrame == 'world':
            transformation[2, 3] = 0.85

        kinematic_chain = self.get_kinematic_chain(jointName, sourceFrame)

        # check which kinematic tree the given joint exists in, accordingly
        # descend down that tree computing transformation 
        for joint in kinematic_chain:
            transformation = transformation @ transformation_matrices[joint]

        return transformation

    def getJointPosition(self, jointName, sourceFrame):
        """Get the position of a joint in the source frame

        retrieves the 3D position in the world frame

        Args:
            @param jointName (str): name of the joint as a string
        Returns:
            @returns (np.ndarray)
        """
        return self.getJointLocationAndOrientation(jointName, sourceFrame=sourceFrame)[0]

    def getJointOrientation(self, jointName, sourceFrame, ref=None):
        """Get the orientation of a joint in the source frame

        retrieves the 3D rotation matrix in the world frame

        Args:
            @param jointName (str): name of the joint as a string
            @param sourceFrame (str): name of the frame w.r.t which the
            transformation needs to be computed
        Returns:
            @returns (np.ndarray)
        """
        if ref is None:
            return np.array(self.getJointLocationAndOrientation(jointName,
                                                                sourceFrame=sourceFrame)[1] @ self.refVector).squeeze()
        else:
            return np.array(self.getJointLocationAndOrientation(jointName,
                                                                sourceFrame=sourceFrame)[1] @ ref).squeeze()

    def getJointAxis(self, jointName, sourceFrame):
        """Get the orientation of a joint in the source frame

        retrieves the axis around which rotations happen w.r.t the source frame

        Args:
            @param jointName (str): name of the joint as a string
            @param sourceFrame (str): name of the frame w.r.t which the
            transformation needs to be computed
        Returns:
            @returns (np.ndarray)

        """
        return np.array(self.getJointLocationAndOrientation(jointName,
                                                            sourceFrame=sourceFrame)[1] @ self.jointRotationAxis[jointName]).squeeze()

    def jacobianMatrix(self,
                       endEffector,
                       sourceFrame):
        """Calculates the Jacobian of the specified chain

        Takes the end effector and source frame and computes the jacobian of
        the kinematic chain connecting them

        Args:
            endEffector (str): name of the end effector joint
            sourceFrame (str): name of the source joint
        Returns:
            jacobianMatrix (np.ndarray)
        """
        # NOTE: The jacobian (theoreticallty speaking) is a 1st order
        # partial differential of a vector valued function. Each column
        # represents a joint in the kinematic chain.

        # NOTE: difference between `getJointOrientation` and `getJointAxis` -
        # same except that `getJointOrientation` multiplies with a standard
        # reference vector and `getJointAxis` multiplies with the corresponding
        # joint axis
        position_jacobian = []
        orientation_jacobian = []

        kinematic_chain = self.get_kinematic_chain(endEffector,
                                                   sourceFrame=sourceFrame)

        p_eff = self.getJointPosition(endEffector,
                                      sourceFrame=sourceFrame)
        # DEBG: use `getJointOrientation` instead

        a_eff = self.getJointAxis(endEffector,
                                  sourceFrame=sourceFrame)
        # a_eff = self.getJointOrientation(endEffector,
        #                                  sourceFrame=sourceFrame)

        for joint in kinematic_chain:
            p_i = self.getJointPosition(joint,
                                        sourceFrame=sourceFrame)
            # a_i = self.getJointOrientation(joint,
            #                                sourceFrame=sourceFrame)
            a_i = self.getJointAxis(joint,
                                    sourceFrame=sourceFrame)

            J_pos = np.cross(a_i, (p_eff - p_i))

            # DEBG: peter's old code just takes the joint axis as the jacobian

            J_ori = np.cross(a_i, a_eff)
            # J_ori = a_i

            position_jacobian.append(J_pos)
            orientation_jacobian.append(J_ori)

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
        sourceFrame,
        oriJacob=False):
        """Inverse Kinematics solver


        Takes in the end effector joint name and its corresponding target
        position, constructs the corresponding position jacobian inverse and multiplies
        it with the change in position. If a target orientation is given, then
        it constructs the orientation vector jacobian
        NOTE: This solver assumes that the interpolation happens outside this
        function. Hence, the target position(and orientation) is the next
        point in a series of interpolation steps

        Returns a dq - delta in the configuration of the joints in the
        corresopnding kinematic chain

        Args:
            endEffector (str): name of the end effector joint
            targetPosition (list(float)): target position of given end-effector
            targetOrientation (list(float)): target orientation of end-effector
            oriJacob (bool): flag for considering orientation or not
        Returns:
            dq (list(floats)): change in configuration of the kinematic chain.
            The length should be equal to length of the kinematic chain from
            source frame to end effector
        """
        # intermediate source frame


        # Hint: return a numpy array which includes the reference angular
        # positions for all joints after performing inverse kinematics.
        # takes in next step in interpolation and returns the dq required to
        # move the endeffector to the targetposition
        J_pos, J_ori = self.jacobianMatrix(endEffector, sourceFrame=sourceFrame)
        if oriJacob:
            jacobian = np.vstack((J_pos, J_ori))
            trans_delta = targetPosition - self.getJointPosition(endEffector,
                                                                 sourceFrame=sourceFrame)
            current_ori = self.getJointOrientation(endEffector,
                                                   sourceFrame=sourceFrame)
            ori_delta = targetOrientation - current_ori
            dy = np.hstack((trans_delta, ori_delta))
        else:
            jacobian = J_pos
            dy = targetPosition - self.getJointPosition(endEffector,
                                                        sourceFrame=sourceFrame)

        # DEBG: 
        print("dy shape: {}".format(dy.shape))
        print("jacobian shape: {}".format(jacobian.shape))
        J_inv = np.linalg.pinv(jacobian)
        print("jacobian inverse:{}".format(J_inv.shape))
        dq = J_inv @ dy

        return dq

    def move_without_PD(
        self,
        endEffector,
        targetPosition,
        interpolationSteps,
        speed=0.01,
        sourceFrame='world',
        targetOrientation=None,
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

        # intialise for graphing
        pltTime = []
        pltDistance = []

        # which kinematic chain does the end effector belong to
        kinematic_chain = self.get_kinematic_chain(endEffector, sourceFrame)

        # intialise the end effector position
        p_ST_init_S, R_ST_init = self.getJointLocationAndOrientation(endEffector,
                                                                     sourceFrame)

        # NOTE: the target position and orientations are given in the
        # world frame `W`. In order for the jacobian to be applied on the right
        # `dy` the target position and orientation needs to be transformed to
        # the source frame. Change the target's source frame
        if sourceFrame == 'world':
            p_ST_S = targetPosition
            if targetOrientation is not None:
                R_ST = npRotation.from_quat(targetOrientation).as_matrix()
        else:
            # estimate transformation between `S` and `W`
            X_WS = self.getTransformationMatrix(sourceFrame, 'world')
            X_SW = np.linalg.inv(X_WS)
            if targetOrientation is not None:
                # DOUBT: is this a proper rotation?
                R_SW = npRotation.from_matrix(X_SW[:3, :3]).as_matrix()
                R_WT = npRotation.from_quat(targetOrientation).as_matrix()
                R_ST = R_SW @ R_WT # rotation between source and target
            p_ST_S = X_SW @ np.hstack((targetPosition, [1]))
            p_ST_S = p_ST_S[:3]


        # Linear interpolation in the end-effector space
        translation_steps = np.linspace(p_ST_init_S,
                                        p_ST_S,
                                        num=interpolationSteps)
        # if end-effector goal orientation is also provided then perform
        # spherical linear interpolation
        if targetOrientation is not None:
            tar_ori = npRotation.from_matrix(R_ST)
            start_ori = npRotation.from_matrix(R_ST_init)
            rotations = npRotation.concatenate([start_ori, tar_ori])
            orientation_interp = slerp([0, interpolationSteps-1],
                                      rotations)
            orientation_steps = orientation_interp((np.arange(interpolationSteps))).as_euler('xyz')


        # iterate through the interpolated end effector positions and perform
        # IK on each interpolated point
        for i in range(interpolationSteps):
            # get translation and orientation goals
            translation_goal = translation_steps[i, :]
            if targetOrientation is not None:
                orientation_goal = orientation_steps[i, :]

            # estimate the differential in configuration of the joints in the
            # robot
            if targetOrientation is not None:
                dq = self.inverseKinematics(endEffector,
                                               translation_goal,
                                               orientation_goal,
                                               interpolationSteps,
                                               maxIterPerStep=maxIter,
                                               threshold=threshold,
                                            sourceFrame=sourceFrame,
                                               oriJacob=True)
            else:
                dq = self.inverseKinematics(endEffector,
                                               translation_goal,
                                               targetOrientation=None,
                                               interpolationSteps=interpolationSteps,
                                               maxIterPerStep=maxIter,
                                               threshold=threshold,
                                            sourceFrame=sourceFrame,
                                               oriJacob=False)


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
                targetPosition - self.getJointPosition(endEffector, sourceFrame='world'))
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
        sourceFrame='world',
        speed=0.01,
        targetOrientation=None,
        threshold=1e-3,
        maxIter=3000,
        debug=False,
        verbose=False):
        """Move the given end-effector with PD control

        Move joints using inverse kinematics solver and using PD control. This
        method should update joint states using the torque output from the PD
        controller.

        @param endEffector: Name of the joint that needs to be moved,
        @type endEffector: str,
        @param targetPosition: target position of the end effector,
        @type targetPosition: np.ndarray(float),
        @param interpolationSteps: num of interpolation steps for IK,
        @type interpolationSteps: int,
        @param control_freq: frequency of the control loop, also used for cubic
        interpolation,
        @type control_freq: int,
        @param sourceFrame: base frame of the end effector. The Jacobian needs
        to be estimated w.r.t this frame
        @type sourceFrame: str
        @param speed: ---
        @type speed: float
        @param orientation: target orientation of the end effector, if none
        then orientation is not controlled. Assume a quaternion
        @type orientation: np.ndarray(float)
        @param threshold: threshold for distance between given target
        position/orientation and the IK - used to check for IK convergence
        @type threshold: float
        @param maxIter: maximum number of IK iterations
        @type maxIter: int


        @returns pltTime, pltDistance: returns time series distance between end
        effector and target position
        @rtype x, y: np.ndarray(), np.ndarray
        """
        # TEST: test by moving arm to known place
        # PASS: works when you fix the control set points

        # TEST: add kinematic chain prescription
        # PASS:

        # TEST: cubic interpolation
        # FAIL: wobbling at every trajectory interpolation step 
        # TEST: control for a specific eef orientation
        # FAIL: Very erratic joint space behaviour

        # TODO: change position and orientation variables to reflect tedrake
        # notation
        # DONE:

        # TODO: raise exception/warning when inverse kinematics cannot reach
        # the target position

        # TODO: figure out what default representation the code base uses and
        # then navigate that

        # NOTE: interpolates and executes control between initial and target 
        # position/orientation. Only works on the kinematic chain responsible
        # for the current end effector. Does this by updating a global variable
        # `self.target_robot_configuration` after `inverseKinematics`. The tick
        # function then reads this configuration and estimates torque from the
        # tuned PID controller. It iterates through the joints only within the
        # current kinematic chain and sets the motor torques.

        # QUE: Am i sure that I have transformed all the positions and
        # orientations w.r.t the source frame
        # ANS: initial position and orientations - yes, target position
        # FIX: target orientations 

        # QUE: what does `getJointOrientation` return? what is stored in the
        # jointAxis dictionary? what representations should I be using?



        # intialise for graphing
        pltTime = []
        pltDistance = []

        # which kinematic chain does the end effector belong to
        kinematic_chain = self.get_kinematic_chain(endEffector, sourceFrame)

        # intialise the end effector position
        p_ST_init_S, R_ST_init = self.getJointLocationAndOrientation(endEffector,
                                                                     sourceFrame)

        # NOTE: the target position and orientations are given in the
        # world frame `W`. In order for the jacobian to be applied on the right
        # `dy` the target position and orientation needs to be transformed to
        # the source frame. Change the target's source frame
        if sourceFrame == 'world':
            p_ST_S = targetPosition
            if targetOrientation is not None:
                R_ST = npRotation.from_quat(targetOrientation).as_matrix()
        else:
            # estimate transformation between `S` and `W`
            # TODO: need to change the reference frame of the target
            # orientation as well
            # DONE:
            X_WS = self.getTransformationMatrix(sourceFrame, 'world')
            X_SW = np.linalg.inv(X_WS)
            if targetOrientation is not None:
                R_SW = X_SW[:3, :3] # DOUBT: is this a proper rotation?
                R_WT = npRotation.from_quat(targetOrientation).as_matrix()
                R_ST = R_SW @ R_WT # rotation between source and target
            p_ST_S = X_SW @ np.hstack((targetPosition, [1]))
            p_ST_S = p_ST_S[:3]

        # p_TS_S = EF_init_pos
        # p_TS_S += [0.2, 0.0, 0.0]
        # Linear interpolation in the end-effector space
        translation_steps = np.linspace(p_ST_init_S,
                                        p_ST_S,
                                        num=interpolationSteps)
        # translation_steps = np.linspace(EF_init_pos,
        #                                 p_TS_S,
        #                                 num=interpolationSteps)
        # if end-effector goal orientation is also provided then perform
        # spherical linear interpolation
        # TODO: change orientations to a scipy.Rotations object
        # FIX: fix the orientation slerp
        if targetOrientation is not None:
            tar_ori = npRotation.from_matrix(R_ST)
            start_ori = npRotation.from_matrix(R_ST_init)
            rotations = npRotation.concatenate([start_ori, tar_ori])
            orientation_interp = slerp([0, interpolationSteps-1],
                                      rotations)
            orientation_steps = orientation_interp((np.arange(interpolationSteps))).as_euler('xyz')



        # some interpolation loop initialisations
        self.target_robot_configuration = {}
        # initalise previous configuration
        for joint in self.joints:
            self.previous_robot_configuration[joint] = self.getJointPos(joint)
            self.rest_robot_configuration[joint] = self.getJointPos(joint)

        # iterate through the interpolated end effector positions and perform
        # IK on each interpolated point
        for i in range(interpolationSteps):
            # get translation and orientation goals
            translation_goal = translation_steps[i, :]
            if targetOrientation is not None:
                orientation_goal = orientation_steps[i, :]

            # TODO: break after threshold reached, check for threshiold reached


            # TODO: break after maxIter reached


            # estimate the differential in configuration of the joints in the
            # robot
            if targetOrientation is not None:
                dq = self.inverseKinematics(endEffector,
                                               translation_goal,
                                               orientation_goal,
                                               interpolationSteps,
                                               maxIterPerStep=maxIter,
                                               threshold=threshold,
                                            sourceFrame=sourceFrame,
                                               oriJacob=True)
            else:
                dq = self.inverseKinematics(endEffector,
                                               translation_goal,
                                               targetOrientation=None,
                                               interpolationSteps=interpolationSteps,
                                               maxIterPerStep=maxIter,
                                               threshold=threshold,
                                            sourceFrame=sourceFrame,
                                               oriJacob=False)




            # DEBG: updation of joint states now happens inside the control
            # loop

            # # update the joint states 
            # joint_states = {}

            # # list to maintain change in configuration - mainly for control
            # # interpolation
            # delta_joint_state = []
            # for idx, joint in enumerate(kinematic_chain):
            #     joint_states[joint] =  self.getJointPos(joint) + dq[idx]
            #     delta_joint_state.append(dq[idx])

            # updated_pose_values = np.array(list(joint_states.values()))

            # # store the updated joint states in a global variable
            # self.target_robot_configuration = {key: value for key, value in
            #                                    zip(kinematic_chain,
            #                                        updated_pose_values)}

            temp = np.zeros((1, len(dq)))
            points = np.vstack((temp, dq))


            # NOTE: The outer loop assumes a straight line trajectory and
            # interpolates between initial position and target position. 
            # This inner loop executes the robot control loop between 
            # two immediate interpolated points. As one can imagine, each outer
            # loop interpolated end effector position gives you a discrete
            # configuration that the PD control loop controls for. The target
            # configuration `self.target_robot_configuration` in the tick 
            # function becomes a step function. 

            # NOTE: To smoothen the target configuration provided to the PD
            # controller, the next step is to employ a cubic spline and fit the
            # two consecutive target configuration control set points.


            # TEST: cubic interpolation implmentation

            # cubic interpolation between control set-points
            _, set_points = self.cubic_interpolation(points,
                                                     control_freq)
            # DEBG: changing control set points to repeat instead of cubic
            # interpolation
            # set_points = np.repeat(np.expand_dims(dq, axis=1),
            #                        control_freq, axis=1).transpose()


            # initialise the integral error
            self.integral = 0.0

            # local variable to hold the current configuration
            current_robot_configuration = {}
            for joint in kinematic_chain:
                current_robot_configuration[joint] = self.getJointPos(joint)


            # iterate through control points
            for point in set_points:

                # iterate through the kinematic chain and update the control
                # setpoint
                joint_states = {}

                for idx, joint in enumerate(kinematic_chain):
                    joint_states[joint] =  current_robot_configuration[joint] + point[idx]

                # organise the updated values
                updated_pose_values = np.array(list(joint_states.values()))

                # store the updated control set points in a global variable
                self.target_robot_configuration = {key: value for key, value in
                                                   zip(kinematic_chain,
                                                       updated_pose_values)}

                self.tick()

                # check threshold



                # log for graphing and debug
                dist_to_target_position = np.linalg.norm(
                    targetPosition - self.getJointPosition(endEffector,
                                                           sourceFrame='world'))
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

            # get target position, if the joint belongs to an inactive
            # kinematic chain, then take rest configuration of the joint and
            # set that as the control set point
            if joint in self.target_robot_configuration.keys():
                targetPosition = self.target_robot_configuration[joint]
            elif joint in self.rest_robot_configuration.keys():
                targetPosition = self.rest_robot_configuration[joint]
            else:
                continue
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
            # HACK: is this where this needs to be updated?
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
        """Cubic interpolation of Control Points

        Given a set of control `points` - typically there will be two points
        for every joint in the configuration that we are trying to control, the
        starting configuration and the ending configuration for a particular
        interpolation time step - return a list of `nTimes` `points` sampled
        along a cubic spline, fitted to the control points.


        @param points: control points that needs to be interpolated - shape (2,
        N) where N is the number of joints in the kinematic chain
        @type points: np.ndarray
        @param nTimes: The number of points that need to be returned along the
        cubic spline
        @type nTimes: int
        @returns x, y: 2 lists of sampled points from the cubic spline
        @rtype: np.array, np.array
        """

        # TEST: needs to be tested

        boundary = 'natural'

        x_temp = np.array([0, nTimes])

        # build interpolation
        spline = CubicSpline(x_temp, points, bc_type=boundary)

        # sample interpolation
        x = np.arange(nTimes)

        y = spline(x)

        return x, y


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
