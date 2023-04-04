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
                         Quaternion, Rgba, RollPitchYaw, RigidTransform,
                         RotationMatrix, SceneGraph, Simulator, StartMeshcat,
                         TrajectorySource, PidController, ModelInstanceIndex,
                         PassThrough, StateInterpolatorWithDiscreteDerivative,
                         Demultiplexer, PidControlledSystem, JointIndex,
                         InverseDynamicsController)



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


# TODO: customize to nextage robot or generalize
class PseudoInverseVelocityController(LeafSystem):
    def __init__(self, plant, config):
        LeafSystem.__init__(self)
        self._config = config
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._robot = plant.GetModelInstanceByName(config['robot_name'])

        # QUE: following args that go into calcjacobian. what are they?

        # Gripper/eef frame w.r.t world
        self._G = plant.GetBodyByName(config['end_effector']).body_frame()
        # world frame
        self._W = plant.world_frame()

        # FIX: change input ports to reflect generic robot.
        # Class also needs to take runtime arguments w.r.t joints
        # to be controlled
        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("nextage_position", 7)
        self.DeclareVectorOutputPort("nextage_velocity", 7,
                                     self.CalcOutput)
        self.nextage_start = plant.GetJointByName(
                config['kinematic_chain_start']).velocity_start()
        self.nextage_end = plant.GetJointByName(
                config['kinematic_chain_end']).velocity_start()

    def CalcOutput(self, context, output):
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(
                self._plant_context,
                self._nextage,
                q)
        # FIX: figure put the jacobian args
        J_G = self._plant.CalcJacobianSpatialVelocity(
                self._plant_context, JacobianWrtVariable.kV,
                self._G, [0, 0, 0], self._W, self._W)

        # ignore gripper terms
        J_G = J_G[:, self.nextage_start:self.nextage_end+1]

        v = np.linalg.pinv(J_G).dot(V_G)
        output.SetFromVector(v)


def MakeNextageStation(
        time_step=0.002):

    # some necessary paths
    table_top_sdf_file = root_path+"/assets/table_top.sdf"
    nextage_description = "/home/aditya/Documents/projects/avro22/ARO_Practical_2022/core/nextagea_description/"

    # a few robot related config
    nextage_prefix = "NextageA"
    num_nextage_positions = 5

    # get a system builder
    builder = DiagramBuilder()

    # Add physics and geometry engines
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0001)

    # SECTION: Simulation elements

    # QUE: add the following in another function
    # DOUBT: better design to setup a separate multibody plant for robot and
    # add it in another function?

    # Add elements into the simulation scene
    parser = Parser(plant)

    parser.package_map().PopulateFromFolder(nextage_description)
    nextage_robot = parser.AddModelFromFile(
            'core/nextagea_description/urdf/NextageaOpenObj.urdf')
    # nextage_robot = parser.AddModels(
    #         'core/nextagea_description/urdf/NextageaOpenObj.urdf')
    table = parser.AddModelFromFile(table_top_sdf_file)
    parser.AddModelFromFile(
            root_path+"/task3_1"+"/lib/task_urdfs/cubes/cube_nextage.urdf")

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

    # SECTION: PID controller and corresponding connections

    # first a few plant related number initialisations
    # HACK: Is there a better way to get nextage instance?
    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)
        if model_instance_name.startswith(nextage_prefix):
            num_nextage_positions = plant.num_positions(model_instance)
            break

    # TODO: add PID controller
    # QUE: What is the interface?
    # ANS: input: desired state, iiwa eg uses result from interpolation
    # ANS: input: estimated state
    # ANS: output: force
    # nextage_controller = builder.AddSystem(
    #         PidController(
    #             kp=np.array([100] * num_nextage_positions, dtype=float),
    #             ki=np.array([1] * num_nextage_positions, dtype=float),
    #             kd=np.array([20] * num_nextage_positions), dtype=float))

    # setup a controller plant
    controller_plant = MultibodyPlant(time_step)
    controller_parser = Parser(controller_plant)
    # control_only_nextage = controller_parser.AddModelFromFile(
    #         core_path +
    #         '/nextagea_description/urdf/NextageaOpenObj.urdf')
    control_only_nextage = controller_parser.AddModelFromFile(
            'core/nextagea_description/urdf/NextageaOpenObj.urdf')
    control_robot_frame = controller_plant.GetFrameByName(
            "base_link",
            control_only_nextage)
    controller_plant.WeldFrames(
            frame_on_parent_F=controller_plant.world_frame(),
            frame_on_child_M=control_robot_frame,
            X_FM=xyz_rpy_deg([1, 0, 0.85], [0, 0, 0]))
    controller_plant.Finalize()

    # DEBG: print out joint names and acutated joint names
    # joint_names = [joint.name()
    #                for joint in [controller_plant.get_joint(
    #                    JointIndex(i))
    #                              for i in range(controller_plant.num_joints())]]
    # print("joint names: {}".format(joint_names))
    # print("number of actuators: {}".format(plant.num_actuators()))
    # actuated_joint_names = [
    #         plant.get_joint_actuator(ind).joint().name()
    #         for ind in plant.GetJointActuatorIndices(model_instance)]
    # print("actuated joint names: {}".format(actuated_joint_names))

    # CODE: delete after testing
    # nextage_controller = builder.AddSystem(
    #         PidControlledSystem(
    #             controller_plant,
    #             kp=[100] * num_nextage_positions,
    #             ki=[1] * num_nextage_positions,
    #             kd=[20] * num_nextage_positions))

    nextage_controller = builder.AddSystem(
            InverseDynamicsController(
                controller_plant,
                kp=[100] * num_nextage_positions,
                ki=[1] * num_nextage_positions,
                kd=[20] * num_nextage_positions,
                has_reference_acceleration=False))
    nextage_controller.set_name('Inverse Dynamics Controller')

    # export input  and output ports
    nextage_input_position = builder.AddSystem(
            PassThrough(num_nextage_positions))
    nextage_input_position.set_name('Input passthrough')
    nextage_output_position = builder.AddSystem(
            PassThrough(num_nextage_positions))
    builder.ExportInput(nextage_input_position.get_input_port(),
                        model_instance_name + "_position")
    builder.ExportOutput(nextage_output_position.get_output_port(),
                         model_instance_name + "_position_measured")
    builder.ExportOutput(nextage_controller.get_output_port_control(),
                         model_instance_name + "_torque_measured")

    # connect controller ports
    # controller's out put needs to be connected to robot's actuation
    # controller's input needs to be connected to current state and
    # interpolated desired state
    pydot.graph_from_dot_data(
            plant.GetGraphvizString(
                max_depth=3))[0].write_svg('./output/debug.svg')

    # DEBG:
    print("number of actuations: {}".format(plant.num_actuators()))

    builder.Connect(
            nextage_controller.get_output_port_control(),
            plant.get_actuation_input_port(model_instance))

    # CODE:
    # builder.Connect(
    #         nextage_controller.get_state_output_port(),
    #         plant.get_actuation_input_port(model_instance_name))

    # the plant recieves the robots desired configuration from the planner.
    # The following class appends the velocity of robot joints to the positions
    # and constructs the total state. The desired state is a tuple given by
    # (q_d, v_d)
    desired_state_from_position = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                num_nextage_positions,
                time_step,
                suppress_initial_transient=True))
    desired_state_from_position.set_name(
            model_instance_name + '_desired_state_from_position')
    builder.Connect(
            nextage_input_position.get_output_port(),
            desired_state_from_position.get_input_port())
    builder.Connect(
            desired_state_from_position.get_output_port(),
            nextage_controller.get_input_port_desired_state()
            )
    builder.Connect(
            plant.get_state_output_port(model_instance),
            nextage_controller.get_input_port_estimated_state()
            )

    # add demuxifier to return position
    demux = builder.AddSystem(
            Demultiplexer(2 * num_nextage_positions,
                          num_nextage_positions))
    builder.Connect(plant.get_state_output_port(model_instance),
                    demux.get_input_port())
    builder.Connect(demux.get_output_port(0),
                    nextage_output_position.get_input_port())
    builder.ExportOutput(demux.get_output_port(1),
                         model_instance_name + "_velocity_measured")

    # SECTION: Finalize the simulation setup


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

    builder.ExportOutput(scene_graph.get_query_output_port(),
                         "query_object")
    builder.ExportOutput(plant.get_contact_results_output_port(),
                         "contact_results")
    builder.ExportOutput(plant.get_body_poses_output_port(),
                         "body_poses")
    diagram = builder.Build()
    diagram.set_name("Nextage Station")

    return diagram


def MakeGripperPoseTrajectory(X_G, times, names):
    """
    Constructs a gripper position trajectory from the plan "sketch".
    """

    sample_times = []
    poses = []
    for name in names:
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


def nextage_control(config, robot_config):

    # some initialisation
    builder = DiagramBuilder()
    meshcat = StartMeshcat()

    # build manipulation station
    # TODO: add the nextage manipulation station
    # FIX: fix port design for nextage manipulation station
    station = builder.AddSystem(
            MakeNextageStation())

    # TODO: setup initial scene context
    plant = station.GetSubsystemByName("plant")
    initial_context = station.CreateDefaultContext()
    initial_plant_context = plant.GetMyContextFromRoot(
            initial_context)

    # reading object and eef poses from the scene context
    X_O = {
            'initial': plant.EvalBodyPoseInWorld(
                initial_plant_context,
                plant.GetBodyByName(config['object_initial_pose']))
            }

    X_G = {
            'initial': plant.EvalBodyPoseInWorld(
                initial_plant_context,
                plant.GetBodyByName(config['end_effector_initial_pose']))
            }

    X_G, times = MakeGripperFrames(X_G, X_O)

    # filter out only the following frames
    names = config['waypoint_frames']

    # construct trajectory
    traj = MakeGripperPoseTrajectory(X_G, times, names)
    traj_p_G = traj.get_position_trajectory()
    traj_V_G = traj.MakeDerivative()

    # create a trajectory system
    V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
    V_G_source.set_name("v_WG")

    # TODO: setup pseudo-inverse Motion Planner (called controller for
    # traditional reasons)
    planner = builder.AddSystem(PseudoInverseVelocityController(
        plant,
        robot_config))
    planner.set_name("PseudoInverseController")
    integrator = builder.AddSystem(Integrator(7))
    integrator.set_name("integrator")
    builder.Connect(planner.get_output_port(),
                    integrator.get_input_port())
    builder.Connect(integrator.get_output_port(),
                    station.GetInputPort("NextageAOpen_position"))
    builder.Connect(V_G_source.get_output_port(),
                    planner.GetInputPort("V_WG"))

    # FIX: change planner to reflect general robot
    builder.Connect(station.GetOutputPort("NextageAOpen_position_measured"),
                    planner.GetInputPort("nextage_position"))

    # TODO: Add a visualizer
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)

    # build the station
    diagram = builder.Build()
    diagram.set_name("Pick and Place")

    # TODO: plot the system
    pydot.graph_from_dot_data(
            station.GetGraphvizString(
                max_depth=1))[0].write_svg('./output/debug.svg')

    # get a simulation
    simulator = Simulator(diagram)

    # save and watch the sim
    visualizer.StartRecording(False)
    simulator.AdvanceTo(traj_p_G.end_time())
    visualizer.PublishRecording()
    f = open("./output/visuals/nextage_manipulation.html", "w")
    f.write(meshcat.StaticHtml())
    f.close()

    return

def reexecute_if_unbuffered():
    """Ensures that output is immediately flushed (e.g. for segfaults).
    ONLY use this at your entrypoint. Otherwise, you may have code be
    re-executed that will clutter your console."""
    import os
    import shlex
    import sys
    if os.environ.get("PYTHONUNBUFFERED") in (None, ""):
        os.environ["PYTHONUNBUFFERED"] = "1"
        argv = list(sys.argv)
        if argv[0] != sys.executable:
            argv.insert(0, sys.executable)
        cmd = " ".join([shlex.quote(arg) for arg in argv])
        sys.stdout.flush()
        os.execv(argv[0], argv)


def traced(func, ignoredirs=None):
    """Decorates func such that its execution is traced, but filters out any
     Python code outside of the system prefix."""
    import functools
    import sys
    import trace
    if ignoredirs is None:
        ignoredirs = ["/usr", sys.prefix]
    tracer = trace.Trace(trace=1, count=0, ignoredirs=ignoredirs)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return tracer.runfunc(func, *args, **kwargs)

    return wrapped


# NOTE: You don't have to trace all of your code. If you can identify a
# single function, then you can just decorate it with this. If you're
# decorating a class method, then be sure to declare these functions above
# it.
# @traced
def main():
    config = {
            "object_initial_pose": "baseLink",
            "end_effector_inital_pose": "RARM_JOINT5_Link",
            "waypoint_frames": ['initial', 'prepick'],
            }

    robot_config = {
            "robot_name": "NextageAOpen",
            "kinematic_chain_start": "CHEST_JOINT0",
            "kinematic_chain_end": "RARM_JOINT5",
            "end_effector":  "RARM_JOINT5_Link",
            }
    nextage_control(config, robot_config)


if __name__ == "__main__":
    reexecute_if_unbuffered()
    main()
