# System related imports
import sys
import os
import time
sys.path.append('/opt/drake/lib/python3.8/site-packages')
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path + '/..'
core_path = root_path + '/core'
sys.path.append(core_path)

# Simulation related imports
import numpy as np
import pydot
from pydrake.common import temp_directory
from pydrake.geometry import (
        MeshcatVisualizer,
        MeshcatVisualizerParams,
        Role,
        StartMeshcat
        )
from pydrake.visualization import ModelVisualizer
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

class Simulation:
    def __init__(self):
        temp_dir = temp_directory()

        # Create a table top SDFormat model.
        self.table_top_sdf_file = os.path.join(temp_dir, "table_top.sdf")
        table_top_sdf = """<?xml version="1.0"?>
        <sdf version="1.7">
          <model name="table_top">
            <link name="table_top_link">
              <visual name="visual">
                <pose>0 0 0.445 0 0 0</pose>
                <geometry>
                  <box>
                    <size>0.55 1.1 0.05</size>
                  </box>
                </geometry>
                <material>
                 <diffuse>0.9 0.8 0.7 1.0</diffuse>
                </material>
              </visual>
              <collision name="collision">
                <pose>0 0 0.445  0 0 0</pose>
                <geometry>
                  <box>
                    <size>0.55 1.1 0.05</size>
                  </box>
                </geometry>
              </collision>
            </link>
            <frame name="table_top_center">
              <pose relative_to="table_top_link">0 0 0.47 0 0 0</pose>
            </frame>
          </model>
        </sdf>

        """

        with open(self.table_top_sdf_file, "w") as f:
            f.write(table_top_sdf)

        # Define a simple cylinder model.
        self.cylinder_sdf = """<?xml version="1.0"?>
        <sdf version="1.7">
          <model name="cylinder">
            <pose>0 0 0 0 0 0</pose>
            <link name="cylinder_link">
              <inertial>
                <mass>1.0</mass>
                <inertia>
                  <ixx>0.005833</ixx>
                  <ixy>0.0</ixy>
                  <ixz>0.0</ixz>
                  <iyy>0.005833</iyy>
                  <iyz>0.0</iyz>
                  <izz>0.005</izz>
                </inertia>
              </inertial>
              <collision name="collision">
                <geometry>
                  <cylinder>
                    <radius>0.1</radius>
                    <length>0.2</length>
                  </cylinder>
                </geometry>
              </collision>
              <visual name="visual">
                <geometry>
                  <cylinder>
                    <radius>0.1</radius>
                    <length>0.2</length>
                  </cylinder>
                </geometry>
                <material>
                  <diffuse>1.0 1.0 1.0 1.0</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """
        return

    def initialise(self):
        self.meshcat = StartMeshcat()
        # self.visualizer = ModelVisualizer(meshcat=self.meshcat)
        diagram, visualizer = self.create_nextage_scene()
        simulator = Simulator(diagram)
        simulator.Initialize()
        simulator.set_target_realtime_rate(1.)
        return simulator, visualizer
        # self.visualizer.parser().AddModels(nextage_model_url)

    def run(self):
        self.sim, self.visualizer = self.initialise()
        self.visualizer.StartRecording()
        self.sim.AdvanceTo(5.0)
        self.visualizer.PublishRecording()

    def create_nextage_scene(self, sim_time_step=0.0001):
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
                builder, time_step=sim_time_step)
        # plant, scene_graph = AddMultibodyPlantSceneGraph(
        #         builder, time_step=0.0)
        parser = Parser(plant, 'robot_fixed')
        (nextage_robot, ) = parser.AddModels(
                core_path +
                '/nextagea_description/urdf/NextageaOpenObj.urdf')

        # add models to the scene

        # parser.AddModels(root_path +
        #                  "/task3_1"+"/lib/task_urdfs/table/table_taller.urdf")
        (table, ) = parser.AddModels(self.table_top_sdf_file)
        parser.AddModels(root_path +
                         "/task3_1"+"/lib/task_urdfs/cubes/cube_small.urdf")
        # parser.AddModels(url="package://drake/manipulation/models/ycb/sdf/003_cracker_box.sdf")
        # parser.AddModels(root_path +
        #                  "/task3_1"+"/lib/task_urdfs/task3_1_target_compiled.urdf")

        # arrange the models in the scene
        robot_frame = plant.GetFrameByName("base_link", nextage_robot)
        table_frame = plant.GetFrameByName("table_top_center")
        plant.WeldFrames(
                frame_on_parent_F=plant.world_frame(),
                frame_on_child_M=robot_frame,
                X_FM=xyz_rpy_deg([0, 0, 0.85], [0, 0, 0]))
        plant.WeldFrames(
                frame_on_parent_F=plant.world_frame(),
                frame_on_child_M=table_frame,
                X_FM=xyz_rpy_deg([0.5, 0, 0.75], [0, 0, 0]))

        plant.Finalize()

        plant_context = plant.CreateDefaultContext()
        cube = plant.GetBodyByName("baseLink")
        X_WorldTable = plant.GetFrameByName(
                "table_top_center").CalcPoseInWorld(
                        plant_context)
        X_TableCube = RigidTransform(RollPitchYaw(
            np.asarray([0, 0, 0]) * np.pi / 180),
            p=[0, 0, 0.5])
        X_WorldCube = X_WorldTable.multiply(X_TableCube)
        plant.SetDefaultFreeBodyPose(cube, X_WorldCube)

        # plant_context = plant.CreateDefaultContext()

        visualizer = MeshcatVisualizer.AddToBuilder(
                builder, scene_graph, self.meshcat,
                MeshcatVisualizerParams(role=Role.kPerception,
                                        prefix="visual"))

        diagram = builder.Build()
        return diagram, visualizer

    def create_sample_scene(self, sim_time_step=0.0001):

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
                builder, time_step=sim_time_step)
        parser = Parser(plant)

        # Loading models.
        # Load a cracker box from Drake. 
        parser.AddModels(
                url="package://drake/manipulation/models/ycb/sdf/003_cracker_box.sdf")
        # Load the table top and the cylinder we created.
        parser.AddModelsFromString(self.cylinder_sdf, "sdf")
        parser.AddModels(self.table_top_sdf_file)

        # Weld the table to the world so that it's fixed during the simulation.
        table_frame = plant.GetFrameByName("table_top_center")
        plant.WeldFrames(plant.world_frame(), table_frame)
        # Finalize the plant after loading the scene.
        plant.Finalize()
        # We use the default context to calculate the transformation of the table
        # in world frame but this is NOT the context the Diagram consumes.
        plant_context = plant.CreateDefaultContext()

        # Set the initial pose for the free bodies, i.e., the custom box and the
        # cracker box.
        cylinder = plant.GetBodyByName("cylinder_link")
        X_WorldTable = table_frame.CalcPoseInWorld(plant_context)
        X_TableCylinder = RigidTransform(
                RollPitchYaw(np.asarray([90, 0, 0]) * np.pi / 180), p=[0, 0, 0.5])
        X_WorldCylinder = X_WorldTable.multiply(X_TableCylinder)
        plant.SetDefaultFreeBodyPose(cylinder, X_WorldCylinder)

        cracker_box = plant.GetBodyByName("base_link_cracker")
        X_TableCracker = RigidTransform(
                RollPitchYaw(np.asarray([45, 30, 0]) * np.pi / 180), p=[0,0,0.8])
        X_WorldCracker = X_WorldTable.multiply(X_TableCracker)
        plant.SetDefaultFreeBodyPose(cracker_box, X_WorldCracker)

        # Add visualizer to visualize the geometries.
        visualizer = MeshcatVisualizer.AddToBuilder(
                builder, scene_graph, self.meshcat,
                MeshcatVisualizerParams(role=Role.kPerception, prefix="visual"))

        diagram = builder.Build()
        return diagram, visualizer


def xyz_rpy_deg(xyz, rpy_deg):
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)


if __name__ == "__main__":
    sim = Simulation()
    try:
        sim.run()
    except KeyboardInterrupt:
        sys.exit()



