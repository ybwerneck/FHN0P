
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

  
import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.cuda.profiler as profiler
import torch.distributed as dist
from termcolor import colored, cprint
from copy import copy
from operator import add
from omegaconf import DictConfig, OmegaConf
import hydra
import itertools
from collections import Counter
from typing import Dict, List, Optional
import logging
from contextlib import ExitStack
from typing import List, Union, Tuple, Callable


from modulus.domain.constraint import Constraint
from modulus.domain import Domain
from modulus.loss.aggregator import Sum
from modulus.utils.training.stop_criterion import StopCriterion
from modulus.constants import TF_SUMMARY, JIT_PYTORCH_VERSION
from modulus.hydra import (
    instantiate_optim,
    instantiate_sched,
    instantiate_agg,
    add_hydra_run_path,
)
from modulus.distributed.manager import DistributedManager
    
class SSolver(SequentialSolver):
    def __init__(  self,
        cfg: DictConfig,
        domains: List[Tuple[int, Domain]],
        custom_update_operation: Union[Callable, None] = None,
    ):
        SequentialSolver.__init__(self,cfg,domains,custom_update_operation)
  

    
    @property
    def network_dir(self):
        dir_name ="/home/jovyan/final/outputs/fhn0P/"+ self.domain.name
        if self.domains[self.domain_index][0] > 1:
            dir_name += "_" + str(self.iteration_index).zfill(4)
        return dir_name

    def setnetwork_dir(self,dir):
        self.network_dir=dir
    def eval(
        self,
    ):
        # check the directory exists


        # create global model for restoring and saving

    
        #print(self.domains)
        for domain_index in range(0, len(self.domains)):
                # solve for number of iterations in train_domain
            for iteration_index in range(0, self.domains[domain_index][0]   ):

                    # set internal domain index and iteration index
                    self.domain_index = domain_index
                    self.iteration_index=iteration_index 
                    self.log.info(
                        "Predicting for Domain "
                        + str(self.domain.name)
                        + ", iteration "
                        + str(self.iteration_index)
                    )
                    
                    if not os.path.exists(self.network_dir):
                        print(os.getcwd()+self.network_dir)
                        raise RuntimeError("Network checkpoint is required for eval mode.")
                    self.saveable_models = self.get_saveable_models()

        # set device
                    if self.device is None:
                        self.device = self.manager.device

        # load model
                    self.step = self.load_step()
                    self.step = self.load_model()
                    self.step_str = f"[step: {self.step:10d}]"

                    # make summary writer
                    self.writer = SummaryWriter(
                        log_dir=self.network_dir, purge_step=self.summary_freq + 1
                    )
                    self.summary_histograms = self.cfg["summary_histograms"]

                    if self.manager.cuda:
                        torch.cuda.synchronize(self.device)

                    # write inference / validation datasets to tensorboard and file
                    if self.has_validators:
                        self._record_validators(self.step)
                    if self.has_inferencers:
                        self._record_inferencers(self.step)
                    if self.has_monitors:
                        self._record_monitors(self.step)

t_max = 10.0
n_w=10
t_w= t_max/n_w

def generateValidator(w_i,nodes):

    
   
    ASD=np.linspace(0,1,1000)
    ASD=np.expand_dims(ASD,axis=-1)
    print(np.shape(ASD))
    invar_numpy = {"t": ASD}
    outvar = {"x1","w"}
   
    validator = PointwiseInferencer(
        nodes=nodes, invar=invar_numpy,
        output_names=outvar, batch_size=100,plotter=None
    )
    return validator




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
    
    sm = SpringMass()
    sm.pprint()


    
    
    flow_net = FullyConnectedArch(
            input_keys=[Key("t")],
            output_keys=[Key("x1"),Key("w")],
            layer_size=300,
            nr_layers=10,
        )

    

    time_window_net = MovingTimeWindowArch(flow_net, t_w)

    nodes = sm.make_nodes() +[time_window_net.make_node(name="network")]


    for node in nodes:
        print(node.__str__())
   
    # add constraints to solver
    # make geometry
    geo = Point1D(0)
    
    t_symbol = Symbol("t")
    x_symbol = Symbol("x1")


    time_range = {t_symbol: (0,t_w )}
  

    tr = {t_symbol: (0, t_w)}

    # make domain
        # make initial condition domain
    ic_domain =  Domain("initial_conditions")

    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"x1": 0.6,"w":0},
        batch_size=3000,
        parameterization={**{t_symbol:0}},
        lambda_weighting={
            "x1": 100.0,#* ((t_symbol/t_max)) + 10,
            "w": 100.0,
        },  
        
        quasirandom=True,
    )
    ic_domain.add_constraint(IC, name="IC")
    
    interior = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_x1": 0.0,"ode_w":0.0},
        batch_size=500,
        parameterization={**time_range},
        #criteria=And(t_symbol > 0, t_symbol < 3),
        lambda_weighting={
            "ode_x1":10,#* ((t_symbol/t_max)) + 10,
            "ode_w": 10  #* ((t_symbol/t_max)) + 1
        },
        quasirandom=True,

    )
    ic_domain.add_constraint(interior, name="interior")
    
    
       
    
    domains=[]
    for i in range(1,n_w):

        # make moving window domain
        window_domain = Domain("window"+str(i))

        # solve over given time period
        interior1 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_x1": 0.0,"ode_w":0.0},
        batch_size=3000,
        parameterization={**tr},
        #criteria=And(t_symbol > 0, t_symbol < 3),
        lambda_weighting={
            "ode_x1": 1,# + 1000*x_symbol.diff(t_symbol)*x_symbol.diff(t_symbol),
            "ode_w": 1 #+ 1000*x_symbol.diff(t_symbol)*x_symbol.diff(t_symbol)
        },
        quasirandom=True,
        )      
        
        IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"x1_prev_step_diff":0,"w_prev_step_diff":0},
        batch_size=500,
        parameterization={**{t_symbol:0}},
        lambda_weighting={
            "x1_prev_step_diff": 100,
            "w_prev_step_diff": 100
        },
        
        quasirandom=True,
        
    )
        
        
        window_domain.add_constraint(IC, name="IC")
        window_domain.add_constraint(interior1, "interior")


        domains.append(window_domain)
    
  
    
    

    
    dom=[]
    dom.append((1,ic_domain))

    for domain in domains:
        dom.append((1,domain))
    print(cfg)
    # make solver
    #slv = Solver(cfg, domain)
    #print(domains)
    i=0
    for a,d in dom:
        print(d)
        print(d.name)
        d.add_inferencer(generateValidator(i,nodes))

        
        i=i+1
    
      
    slv = SSolver(
        cfg,
        dom,
        custom_update_operation=time_window_net.move_window,

    )



    slv.eval()

        
        
        
if __name__ == "__main__":
    run()
