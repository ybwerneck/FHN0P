
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

    
t_max = 10.0
n_w=10
t_w= t_max/n_w





def generateExactSolution(t,dt,x0,w0,rate,P,begin,end):
    
    
    n2=int(t/(dt))+2
    n = int((end-begin)/(dt*rate))
    Sol=np.zeros((n,3))
  
    Sol2=np.zeros((n2,2))
    Sol2[0]=0.6,0
    T=0
    k=0
    while(k<n2-1):
        x,w=Sol2[k]
        Sol2[k+1]=10*(x*(x-0.4)*(1-x)-w)*dt+  x, 0.5*(x*0.2-0.8*w)*dt +w
 
        if ((k*dt==begin or ((k+1)%rate == 0 and k*dt>=begin and k*dt<=end))and T<n):
          
           
            Sol[T] = Sol2[k][0],Sol2[k][1] , k*dt
            T=T+1
        
        k=k+1
        if(k*dt > end):
            break
    return Sol

def generateValidator(w_i,nodes):

    
    
    T=np.empty([0])
    K=np.empty([0])
    SOLs=np.empty([0])
    SOLw=np.empty([0])
    V=np.empty([0])
    krange= [(0 + 0.05*i*1) for i in range(0,20)]
    vrange= [(0 + 0.05*i*1) for i in range(0,20)]

    deltaT = 0.01
    
    rate=5
    for KR in krange:
        for VR in vrange:
            begin=w_i* t_w
            end=begin + t_w
            sol=generateExactSolution(t_max,deltaT,KR,VR,rate,KR,begin,end)

            T=sol.T[2] - begin
            SOLs=sol.T[0]
            SOLw=sol.T[1]
    
    
    t = np.expand_dims(T, axis=-1)


    k = np.expand_dims(K, axis=-1)

    
    Solx = np.expand_dims(SOLs, axis=-1)

    
    Solw = np.expand_dims(SOLw, axis=-1)
   
    v= np.expand_dims(V,axis=-1)
    print(t,"val set de ",begin,"a ", end)

    
    invar_numpy = {"t": t}
    outvar_numpy = {
        "x1": Solx,
        "w":Solw
    }
   
    validator = PointwiseValidator(
        nodes=nodes, invar=invar_numpy, true_outvar=outvar_numpy, batch_size=100,plotter=None
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
    ic_domain = Domain("initial_conditions")

  
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
        d.add_validator(generateValidator(i,nodes))

        
        i=i+1
    
      
    slv = SequentialSolver(
        cfg,
        dom,
        custom_update_operation=time_window_net.move_window,

    )

    # start solver
    slv.solve()

        
        
        
if __name__ == "__main__":
    run()
