# System related imports
import numpy as np
import sys
import os
import pydot
import time
sys.path.append('/opt/drake/lib/python3.8/site-packages')
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path + '/..'
core_path = root_path + '/core'
sys.path.append(core_path)
from pydrake.common import temp_directory
from pydrake.geometry import (
        MeshcatVisualizerParams,
        Role
        )
from pydrake.visualization import ModelVisualizer

from pydrake.all import (AddMultibodyPlantSceneGraph, AngleAxis,
                         DiagramBuilder, FindResourceOrThrow, Integrator,
                         JacobianWrtVariable, LeafSystem, MeshcatVisualizer,
                         MultibodyPlant, MultibodyPositionToGeometryPose,
                         Parser, PiecewisePolynomial, PiecewisePose,
                         Quaternion, Rgba, RollPitchYaw, RigidTransform, RotationMatrix,
                         SceneGraph, Simulator, StartMeshcat, TrajectorySource)


class Simulation:
    def __init__(self):

        self.table_top_sdf_file = root_path+"/assets/table_top.sdf"

        return

    def initialise(self):
        self.meshcat = StartMeshcat()
        # self.visualizer = ModelVisualizer(meshcat=self.meshcat)
        scene_diagram, visualizer = self.create_nextage_scene()
        simulator = Simulator(scene_diagram)
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
        parser = Parser(plant, 'robot')
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
                X_FM=xyz_rpy_deg([1, 0, 0.85], [0, 0, 0]))
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
        pydot.graph_from_dot_data(
            diagram.GetGraphvizString(
                max_depth=2))[0].write_svg('./output/nextage.svg')
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
                RollPitchYaw(
                    np.asarray([90, 0, 0]) * np.pi / 180),
                p=[0, 0, 0.5])
        X_WorldCylinder = X_WorldTable.multiply(X_TableCylinder)
        plant.SetDefaultFreeBodyPose(cylinder, X_WorldCylinder)

        cracker_box = plant.GetBodyByName("base_link_cracker")
        X_TableCracker = RigidTransform(
                RollPitchYaw(
                    np.asarray([45, 30, 0]) * np.pi / 180),
                p=[0, 0, 0.8])
        X_WorldCracker = X_WorldTable.multiply(X_TableCracker)
        plant.SetDefaultFreeBodyPose(cracker_box, X_WorldCracker)

        # Add visualizer to visualize the geometries.
        visualizer = MeshcatVisualizer.AddToBuilder(
                builder, scene_graph, self.meshcat,
                MeshcatVisualizerParams(
                    role=Role.kPerception,
                    prefix="visual"))

        diagram = builder.Build()
        return diagram, visualizer


def xyz_rpy_deg(xyz, rpy_deg):
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)


class PseudoInverseController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.DeclareVectorInputPort("iiwa_position", 7)
        self.DeclareVectorOutputPort("iiwa_velocity", 7,
                                     self.CalcOutput)

    def CalcOutput(self, context, output):
        q = self.get_input_port().Eval(context)
        self._plant.SetPositions(
                self._plant_context,
                self._iiwa,
                q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
                self._plant_context, JacobianWrtVariable.kQDot,
                self._G, [0, 0, 0], self._W, self._W)
        J_G = J_G[:, 0:7]  # ignore gripper terms

        V_G_desired = np.array([0,
                                -0.1,
                                0,
                                0,
                                -0.5,
                                -0.1])
        v = np.linalg.pinv(J_G).dot(V_G_desired)
        output.SetFromVector(v)


class PseudoInverseVelocityController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa_position", 7)
        self.DeclareVectorOutputPort("iiwa_velocity", 7,
                                     self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context, output):
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(
                self._plant_context,
                self._iiwa,
                q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
                self._plant_context, JacobianWrtVariable.kV,
                self._G, [0, 0, 0], self._W, self._W)

        # ignore gripper terms
        J_G = J_G[:, self.iiwa_start:self.iiwa_end+1]

        v = np.linalg.pinv(J_G).dot(V_G)
        output.SetFromVector(v)


def MakeNextageStation():

    # some necessary paths
    table_top_sdf_file = root_path+"/assets/table_top.sdf"

    # get a system builder
    builder = DiagramBuilder()

    # Add physics and geometry engines
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0001)

    # Add elements into the simulation scene
    parser = Parser(plant, 'robot')
    (nextage_robot, ) = parser.AddModels(
            core_path +
            '/nextagea_description/urdf/NextageaOpenObj.urdf')
    (table, ) = parser.AddModels(table_top_sdf_file)
    parser.AddModels(root_path +
                     "/task3_1"+"/lib/task_urdfs/cubes/cube_nextage.urdf")

    # arrange the models in the scene
    robot_frame = plant.GetFrameByName("base_link", nextage_robot)
    table_frame = plant.GetFrameByName("table_top_center")
    plant.WeldFrames(
            frame_on_parent_F=plant.world_frame(),
            frame_on_child_M=robot_frame,
            X_FM=xyz_rpy_deg([1, 0, 0.85], [0, 0, 0]))
    plant.WeldFrames(
            frame_on_parent_F=plant.world_frame(),
            frame_on_child_M=table_frame,
            X_FM=xyz_rpy_deg([0.5, 0, 0.75], [0, 0, 0]))

    plant.Finalize()

    # create basic placements of entities in the scene
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

    diagram = builder.Build()
    diagram.set_name("NextageStation")

    return diagram


def MakeGripperPoseTrajectory(X_G, times):
    """
    Constructs a gripper position trajectory from the plan "sketch".
    """

    sample_times = []
    poses = []
    for name in ["initial", "prepick", "pick_start", "pick_end", "postpick",
                 "clearance", "preplace", "place_start", "place_end",
                 "postplace"]:
        sample_times.append(times[name])
        poses.append(X_G[name])

    return PiecewisePose.MakeLinear(sample_times, poses)


def MakeGripperFrames(X_G, X_O):
    """
    Takes a partial specification with X_G["initial"] and X_O["initial"] and
    X_0["goal"], and returns a X_G and times with all of the pick and place
    frames populated.
    """
    # Define (again) the gripper pose relative to the object when in grasp.
    p_GgraspO = [0, 0.11, 0]
    R_GgraspO = RotationMatrix.MakeXRotation(
        np.pi / 2.0) @ RotationMatrix.MakeZRotation(np.pi / 2.0)
    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
    X_OGgrasp = X_GgraspO.inverse()
    # pregrasp is negative y in the gripper frame (see the figure!).
    X_GgraspGpregrasp = RigidTransform([0, -0.08, 0])

    X_G["pick"] = X_O["initial"] @ X_OGgrasp
    X_G["prepick"] = X_G["pick"] @ X_GgraspGpregrasp
    X_G["place"] = X_O["goal"] @ X_OGgrasp
    X_G["preplace"] = X_G["place"] @ X_GgraspGpregrasp

    # I'll interpolate a halfway orientation by converting to axis angle and halving the angle.
    X_GprepickGpreplace = X_G["prepick"].inverse() @ X_G["preplace"]
    angle_axis = X_GprepickGpreplace.rotation().ToAngleAxis()
    X_GprepickGclearance = RigidTransform(
        AngleAxis(angle=angle_axis.angle() / 2.0, axis=angle_axis.axis()),
        X_GprepickGpreplace.translation() / 2.0 + np.array([0, -0.3, 0]))
    X_G["clearance"] = X_G["prepick"] @ X_GprepickGclearance

    # Now let's set the timing
    times = {"initial": 0}
    X_GinitialGprepick = X_G["initial"].inverse() @ X_G["prepick"]
    times["prepick"] = times["initial"] + 10.0 * np.linalg.norm(
        X_GinitialGprepick.translation())
    # Allow some time for the gripper to close.
    times["pick_start"] = times["prepick"] + 2.0
    times["pick_end"] = times["pick_start"] + 2.0
    X_G["pick_start"] = X_G["pick"]
    X_G["pick_end"] = X_G["pick"]
    times["postpick"] = times["pick_end"] + 2.0
    X_G["postpick"] = X_G["prepick"]
    time_to_from_clearance = 10.0 * np.linalg.norm(
        X_GprepickGclearance.translation())
    times["clearance"] = times["postpick"] + time_to_from_clearance
    times["preplace"] = times["clearance"] + time_to_from_clearance
    times["place_start"] = times["preplace"] + 2.0
    times["place_end"] = times["place_start"] + 2.0
    X_G["place_start"] = X_G["place"]
    X_G["place_end"] = X_G["place"]
    times["postplace"] = times["place_end"] + 2.0
    X_G["postplace"] = X_G["preplace"]

    return X_G, times

def nextage_control():

    # some initialisation
    builder = DiagramBuilder()
    meshcat = StartMeshcat()

    # build manipulation station
    # TODO: add the nextage manipulation station
    station = builder.AddSystem(
            MakeNextageStation())

    # TODO: setup initial scene context
    plant = station.GetSubsystemByName("plant")
    initial_context = station.CreateDefaultContext()
    initial_plant_context = plant.GetMyContextFromRoot(
            initial_context)
    X_O = {
            'initial': plant.EvalBodyPoseInWorld(
                initial_plant_context,
                plant.GetBodyByName("baseLink"))
            }

    X_G = {
            'initial': plant.EvalBodyPoseInWorld(
                initial_plant_context,
                plant.GetBodyByName("RARM_JOINT5_Link"))
            }

    pydot.graph_from_dot_data(
            station.GetGraphvizString(
                max_depth=1))[0].write_svg('./output/debug.svg')

    # TODO: setup pseudo-inverse controller

    return


if __name__ == "__main__":
    nextage_control()
