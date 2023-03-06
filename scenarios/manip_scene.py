import sys
sys.append('/home/aditya/Documents/projects/manipulation')
import numpy as np
import pydot
from pydrake.all import (
        AddMultibodyPlantSceneGraph, AngleAxis,
        DiagramBuilder, FindResourceOrThrow, Integrator,
        JacobianWrtVariable, LeafSystem, MeshcatVisualizer,
        MultibodyPlant, MultibodyPositionToGeometryPose,
        Parser, PiecewisePolynomial, PiecewisePose,
        Quaternion, Rgba, RigidTransform, RotationMatrix,
        SceneGraph, Simulator, StartMeshcat, TrajectorySource)


