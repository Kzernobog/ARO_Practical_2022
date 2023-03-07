# System related imports
import sys
import os
sys.path.append('/home/aditya/Documents/projects/manipulation')
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

class Simulation:
    def __init__(self):
        pass

    def initialise(self):
        self.meshcat = StartMeshcat()
        self.visualizer = ModelVisualizer(meshcat=self.meshcat)
        nextage_model_url = core_path+'/nextagea_description/urdf/NextageaOpenObj.urdf'
        self.visualizer.parser().AddModels(nextage_model_url)


if __name__ == "__main__":
    sim = Simulation()
    sim.initialise()



