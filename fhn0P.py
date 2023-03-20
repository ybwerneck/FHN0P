from modulus.models.fully_connected import FullyConnectedArch
from modulus.models.fourier_net import FourierNetArch
from modulus.models.siren import SirenArch
from modulus.models.modified_fourier_net import ModifiedFourierNetArch
from modulus.models.dgm import DGMArch

from sympy import Symbol, Eq
from sympy import Symbol, Function, Number
from modulus.eq.pde import PDE
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_1d import Point1D
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.domain.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node
from modulus.eq.pde import PDE
from modulus.geometry import Parameterization
from sympy import Symbol, Eq, Abs, tanh, Or, And
from modulus.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)
from modulus.solver import SequentialSolver

from modulus.models.deeponet import DeepONetArch
from modulus.domain.constraint.continuous import DeepONetConstraint
from modulus.models.moving_time_window import MovingTimeWindowArch
from modulus.domain.monitor import Monitor
from modulus.domain.constraint import Constraint
from modulus.graph import Graph
from modulus.key import Key
from modulus.constants import TF_SUMMARY
from modulus.distributed import DistributedManager
from modulus.utils.io import dict_to_csv, csv_to_dict
from modulus.domain.inferencer.pointwise import PointwiseInferencer as PointwiseInferencer
from modulus.loss.loss import CausalLossNorm
import torch as pt

class SpringMass(PDE):
    name = "SpringMass"

    def __init__(self):

        t = Symbol("t")
       
        input_variables = {"t": t}

        x = Function("x1")(*input_variables)
        w= Function("w")(*input_variables)
        self.equations = {}
        self.equations["ode_x1"] =10*(x*(x-0.4)*(1-x)-w) -x.diff(t)
        self.equations["ode_w"]  =0.5*(x*0.2-0.8*w) -w.diff(t)

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    
    # make list of nodes to unroll graph on
    sm = SpringMass()
    sm.pprint()
    #sm_net = FullyConnectedArch(
    #    input_keys=[Key("t"), Key("K")],
    #    output_keys=[Key("x1")],
    #)
    #nodes = sm.make_nodes() + [
    #    sm_net.make_node(name="network")
    #]


    
    # make list of nodes to unroll graph on
    sm = SpringMass()
    sm.pprint()
    #sm_net = FullyConnectedArch(
    #    input_keys=[Key("t"), Key("K")],
    #    output_keys=[Key("x1")],
    #)
    #nodes = sm.make_nodes() + [
    #    sm_net.make_node(name="network")
    #]

    
    
    flow_net = FullyConnectedArch(
            input_keys=[Key("t")],
            output_keys=[Key("x1"),Key("w")],
            layer_size=300,
            nr_layers=10,
        )

    

    time_window_net = MovingTimeWindowArch(flow_net, 1.0)

    nodes = sm.make_nodes() +[time_window_net.make_node(name="network")]

    time_window_net.load_model("window1")
    
    model=time_window_net
    
    model.eval()
    
    t = 1.0
    my2dspace = pt.tensor(np.linspace([0], [t], num=100), requires_grad=False)
#for i in range(1,628):
 #   t = t + 0.01
  #  my2dspace = pt.cat((my2dspace, pt.tensor(np.linspace([0.0, t], num=313), requires_grad=False)), 0)

#print(my2dspace)

    myOutput = model(my2dspace.float().cuda())
    myCPUOutput = myOutput.cpu()
    
run()
