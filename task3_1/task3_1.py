import subprocess
import math
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pybullet as bullet_simulation
import pybullet_data

# setup paths and load the core
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path + '/..'
core_path = root_path + '/core'
sys.path.append(core_path)
from Pybullet_Simulation import Simulation
# from core.Pybullet_Simulation import Simulation

# specific settings for this task

taskId = 3.1

try:
    if sys.argv[1] == 'nogui':
        gui = False
    else:
        gui = True
except:
    gui = True

pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": gui,
    "panels": False,
    "realTime": False,
    "controlFrequency": 1000,
    "updateFrequency": 250,
    "gravity": -9.81,
    "gravityCompensation": 1.,
    "floor": True,
    "cameraSettings": (1.07, 90.0, -52.8, (0.07, 0.01, 0.76))
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains.yaml",
    "robotStartPos": [0, 0, 0.85],
    "robotStartOrientation": [0, 0, 0, 1],
    "fixedBase": True,
    "colored": False
}

sim = Simulation(pybulletConfigs, robotConfigs)


def getReadyForTask():
    global finalTargetPos
    # compile urdfs
    finalTargetPos = np.array([0.7, 0.00, 0.91])
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
                     abs_path+"/lib/task_urdfs/task3_1_target.urdf"])

    sim.p.resetJointState(bodyUniqueId=1, jointIndex=12, targetValue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)
    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName=abs_path+"/lib/task_urdfs/table/table_taller.urdf",
        basePosition=[0.8, 0, 0],
        baseOrientation=sim.p.getQuaternionFromEuler([0, 0, math.pi/2]),
        useFixedBase=True,
        globalScaling=1.4
    )
    cubeId = sim.p.loadURDF(
        fileName=abs_path+"/lib/task_urdfs/cubes/cube_small.urdf",
        basePosition=[0.33, 0, 1.0],
        baseOrientation=sim.p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=False,
        globalScaling=1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1, 1, 0, 1])

    targetId = sim.p.loadURDF(
        fileName=abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
        basePosition=finalTargetPos,
        baseOrientation=sim.p.getQuaternionFromEuler([0, 0, math.pi]),
        useFixedBase=True,
        globalScaling=1
    )
    for _ in range(200):
        sim.tick()
        time.sleep(1./1000)

    return tableId, cubeId, targetId


def solution():
    # TODO: Move the LHAND5 .
    # STATUS: `move_without_PD` seems to be moving the end-effector.
    # `move_with_PD` behaves erratically.

    # targetPosition = [0.33, 0, 1.20]
    endEffector = "RARM_JOINT5"
    temp = sim.getJointPosition(
            endEffector,
            sourceFrame='world') + np.array(
                    [0.0, 0.1, 0.2])
    targetPosition = temp
    # targetPosition = finalTargetPos
    targetOrientation = sim.p.getQuaternionFromEuler([math.pi/36, 0, 0])
    waypointID = sim.p.loadURDF(
        fileName=abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
        basePosition=targetPosition,
        baseOrientation=targetOrientation,
        useFixedBase=True,
        globalScaling=0.5
    )
    sim.p.resetVisualShapeData(waypointID, -1, rgbaColor=[1, 0, 0, 0.5])
    # targetOrientation = sim.p.getQuaternionFromEuler([5.0, 0, 0])
    # targetOrientation = sim.p.getQuaternionFromEuler(
    #         sim.getJointOrientation(
    #             endEffector,
    #             sourceFrame='world'))
    # baseFrame = 'world'
    baseFrame = 'CHEST_JOINT0'
    interpSteps = 10
    controlFrequency = 1000
    sim.move_with_PD(endEffector,
                     targetPosition,
                     interpSteps,
                     controlFrequency,
                     baseFrame,
                     targetOrientation=targetOrientation)
    # sim.move_with_PD(endEffector,
    #                  targetPosition,
    #                  interpSteps,
    #                  controlFrequency,
    #                  sourceFrame=baseFrame)
    return


tableId, cubeId, targetId = getReadyForTask()
solution()
