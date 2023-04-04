import sys
import os
import numpy as np
import time
sys.path.append('/opt/drake/lib/python3.8/site-packages')
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path + '/..'
utils_path = root_path + "/utils"
asset_path = root_path + "/assets"
sys.path.append(utils_path)
sys.path.append(asset_path)

from pydrake.all import (AddMultibodyPlantSceneGraph, AngleAxis,
                         DiagramBuilder, FindResourceOrThrow, Integrator,
                         JacobianWrtVariable, LeafSystem, MeshcatVisualizer,
                         MultibodyPlant, MultibodyPositionToGeometryPose,
                         Parser, PiecewisePolynomial, PiecewisePose,
                         Quaternion, Rgba, RigidTransform, RotationMatrix,
                         SceneGraph, Simulator, StartMeshcat, TrajectorySource)
import pydot
from utils import FindResource
from scenarios import MakeManipulationStation


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
        J_G = J_G[:, self.iiwa_start:self.iiwa_end+1] # ignore gripper terms

        v = np.linalg.pinv(J_G).dot(V_G)
        output.SetFromVector(v)


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
        J_G = J_G[:, 0:7] # ignore gripper terms

        V_G_desired = np.array([0,
                                -0.1,
                                0,
                                0,
                                -0.5,
                                -0.1])
        v = np.linalg.pinv(J_G).dot(V_G_desired)
        output.SetFromVector(v)


def jacobian_controller_example():
    builder = DiagramBuilder()

    station = builder.AddSystem(
            MakeManipulationStation(
                filename=FindResource("models/iiwa_and_wsg.dmd.yaml")))
    plant = station.GetSubsystemByName('plant')

    controller = builder.AddSystem(PseudoInverseController(plant))
    integrator = builder.AddSystem(Integrator(7))

    builder.Connect(controller.get_output_port(),
                    integrator.get_input_port())
    builder.Connect(integrator.get_output_port(),
                    station.GetInputPort("iiwa_position"))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                    controller.get_input_port())

    meshcat = StartMeshcat()
    visualizer = MeshcatVisualizer.AddToBuilder(
            builder, station.GetOutputPort("query_object"), meshcat)

    diagram = builder.Build()
    pydot.graph_from_dot_data(
        diagram.GetGraphvizString(
            max_depth=2))[0].write_svg('./output/iiwa2.svg')

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    station_context = station.GetMyContextFromRoot(context)
    station.GetInputPort("iiwa_feedforward_torque").FixValue(
            station_context, np.zeros((7, 1)))
    station.GetInputPort("wsg_position").FixValue(station_context, [0.1])
    integrator.set_integral_value(
            integrator.GetMyContextFromRoot(context),
            plant.GetPositions(plant.GetMyContextFromRoot(context),
                               plant.GetModelInstanceByName("iiwa")))

    visualizer.StartRecording()
    simulator.AdvanceTo(5)
    visualizer.PublishRecording()


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


def MakeGripperCommandTrajectory(times):
    opened = np.array([0.107]);
    closed = np.array([0.0]);

    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
        [times["initial"], times["pick_start"]], np.hstack([[opened], [opened]]))
    traj_wsg_command.AppendFirstOrderSegment(times["pick_end"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_start"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_end"], opened)
    traj_wsg_command.AppendFirstOrderSegment(times["postplace"], opened)
    return traj_wsg_command


def iiwa_control():
    X_O = {"initial": RigidTransform(RotationMatrix.MakeZRotation(np.pi/2.0),
                                     [-.2, -.65, 0.0]),
           "goal": RigidTransform(RotationMatrix.MakeZRotation(np.pi),
                                  [.5, 0, 0.0])}

    builder = DiagramBuilder()
    meshcat = StartMeshcat()

    running_as_notebook = False

    model_directives = """
    directives:
    - add_directives:
        file: package://manipulation/clutter.dmd.yaml
    - add_model:
        name: foam_brick
        file: package://drake/examples/manipulation_station/models/061_foam_brick.sdf
    """
    station = builder.AddSystem(
        MakeManipulationStation(model_directives=model_directives))

    pydot.graph_from_dot_data(
        station.GetGraphvizString(
            max_depth=1))[0].write_svg('./output/scenes/iiwa_manipulation.svg')

    plant = station.GetSubsystemByName("plant")
    plant.SetDefaultFreeBodyPose(
            plant.GetBodyByName("base_link"),
            X_O['initial'])

    # Find the initial pose of the gripper (as set in the default Context)
    temp_context = station.CreateDefaultContext()
    temp_plant_context = plant.GetMyContextFromRoot(
            temp_context)
    X_G = {
        "initial":
            plant.EvalBodyPoseInWorld(temp_plant_context,
                                      plant.GetBodyByName("body"))
    }
    X_G, times = MakeGripperFrames(
            X_G,
            X_O)
    print(f"Sanity check: The entire maneuver will \
          take {times['postplace']} seconds to execute.")

    # Make the trajectories
    traj = MakeGripperPoseTrajectory(X_G, times)
    traj_p_G = traj.get_position_trajectory()
    traj_V_G = traj.MakeDerivative()

    V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
    V_G_source.set_name("v_WG")
    controller = builder.AddSystem(PseudoInverseVelocityController(plant))
    controller.set_name("PseudoInverseController")
    builder.Connect(V_G_source.get_output_port(), controller.GetInputPort("V_WG"))

    integrator = builder.AddSystem(Integrator(7))
    integrator.set_name("integrator")
    builder.Connect(controller.get_output_port(),
                    integrator.get_input_port())
    builder.Connect(integrator.get_output_port(),
                    station.GetInputPort("iiwa_position"))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                    controller.GetInputPort("iiwa_position"))

    traj_wsg_command = MakeGripperCommandTrajectory(times)
    wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))
    wsg_source.set_name("wsg_command")
    builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg_position"))

    meshcat.Delete()
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)

    diagram = builder.Build()
    diagram.set_name("pick_and_place")
    pydot.graph_from_dot_data(
        diagram.GetGraphvizString(
            max_depth=1))[0].write_svg('./output/scenes/iiwa_depth1.svg')

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    station_context = station.GetMyContextFromRoot(context)
    integrator.set_integral_value(
        integrator.GetMyContextFromRoot(context),
        plant.GetPositions(plant.GetMyContextFromRoot(context),
                           plant.GetModelInstanceByName("iiwa")))

    # meshcat.SetProperty('/Background', 'visible', False)
    visualizer.StartRecording(False)
    # simulator.AdvanceTo(traj_p_G.end_time() if running_as_notebook else 0.1)
    simulator.AdvanceTo(traj_p_G.end_time())
    visualizer.PublishRecording()
    f = open("./output/iiwa_picknplace.html", "w")
    f.write(meshcat.StaticHtml())
    f.close()


if __name__ == "__main__":
    iiwa_control()
    # jacobian_controller_example()
