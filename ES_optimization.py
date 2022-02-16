# LS Optimization model for load curtailment
import pyomo
from matplotlib import pyplot
from pyomo.gdp import Disjunct
from sympy.integrals.rubi.utility_function import Inequality

import DR_aggregation
from DR_aggregation import DR_aggregator
import numpy as np
import pandas as pd
import pyomo.environ as pe
from pyomo.environ import ConstraintList, NonNegativeReals, Reals, Var, NegativeReals, Binary, Objective, maximize, \
    ConcreteModel, value, Set, Param, Integers, PercentFraction, PositiveIntegers, NonNegativeIntegers, PositiveReals

from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import seaborn as sns

model = ConcreteModel()
model.constraints = ConstraintList()

smes = list(range(1, 3))
time = list(range(0, 25))

model.k = Set(initialize=smes)
model.t = Set(initialize=time)


def C_init():
    Contract_price = dict()
    for k in model.k:
        for t in model.t:
            Contract_price[(k, t)] = DR_aggregator.SMEs_dict[k]['es']['es_price']
    return Contract_price

def param_init(j):
    list_param = []
    for k in model.k:
        list_param.append(DR_aggregator.SMEs_dict[k]['es']['power_rating'])
        list_param.append(DR_aggregator.SMEs_dict[k]['es']['discharge_ramp'])
        list_param.append(DR_aggregator.SMEs_dict[k]['es']['discharge_ramp'])
        list_param.append(DR_aggregator.SMEs_dict[k]['es']['Energy_capacity'])
        list_param.append(DR_aggregator.SMEs_dict[k]['es']['discharge_eff'])
        list_param.append(DR_aggregator.SMEs_dict[k]['es']['MN'])
        list_param.append(DR_aggregator.SMEs_dict[k]['es']['ERT'])
        return list_param[j]

id=2
model.C_price = Param(model.k, model.t, domain=PositiveIntegers, initialize=C_init())
model.ro_price = Param(model.t, initialize=DR_aggregator.Market_signal[id])
model.MaxPower = Param(model.k, domain=NonNegativeReals, initialize=param_init(0))
model.MinPower = Param(model.k, domain=NonNegativeReals, default=0)
model.Ramp_up = Param(model.k, domain=NonNegativeReals, initialize=param_init(1))
model.Ramp_down = Param(model.k, domain=NonNegativeReals, initialize=param_init(2))
model.E_max = Param(model.k, domain=NonNegativeReals, initialize=param_init(3))
model.mue = Param(model.k, domain=Reals, initialize=param_init(4))
model.MN = Param(model.k, domain=Integers, initialize=param_init(5))
model.ERT = Param(model.k, domain=Reals, initialize=param_init(6))




# Variables declaration
def power_limit(model, k,t):
   return (0, model.MaxPower[k])

model.Power_es = Var(model.k, model.t, domain=Integers)
model.LR_es = Var(model.t, domain=Reals)
model.LR1_es = Var(model.t, domain=Reals)
model.LR2_es = Var(model.t, domain=Reals)
model.CLR_es = Var(model.t, domain=Reals)
model.u = Var(model.k, model.t, within=Binary)
model.y = Var(model.k, model.t, within=Binary)
model.z = Var(model.k, model.t, within=Binary)

#  Objective Function
expr_es = sum(model.ro_price[t] * model.LR_es[t] for t in time) - sum(model.CLR_es[t] for t in time)
model.objective = Objective(sense=maximize, expr=expr_es)

# constrains
for t in model.t:  # constrain 28
    model.constraints.add(model.LR_es[t] == sum(model.Power_es[k, t] for k in model.k))
    model.constraints.add(model.LR1_es[t] == sum(model.Power_es[k, t] for k in model.k if k == 1))
    model.constraints.add(model.LR2_es[t] == sum(model.Power_es[k, t] for k in model.k if k == 2))

for t in model.t:  # constrain29
    model.constraints.add(
        model.CLR_es[t] == sum(model.C_price[k, t] * model.Power_es[k, t] for k in model.k))

for k in model.k:   # constrain 30
    for t in list(range(0, 25)):
        model.constraints.add(model.Power_es[k, t] >= 0)
        model.constraints.add(model.Power_es[k, t] <= model.u[k,t] * model.MaxPower[k])


for k in model.k:  # constrain 31-32
    for t in list(range(1, 25)):
        model.constraints.add(model.Power_es[k, t] - model.Power_es[k, t - 1] <= model.Ramp_up[k])
        model.constraints.add(model.Power_es[k, t - 1] - model.Power_es[k, t] <= model.Ramp_down[k])

for k in model.k:  # constrain 33
    model.constraints.add(model.mue[k] * model.E_max[k] >= sum(model.Power_es[k, t] for t in list(range(0, 25))))

for k in model.k:  # constrain 34
    model.constraints.add(sum(model.y[k, t] for t in model.t) == model.MN[k])

'''for k in model.k:  # constrain 35
    for t in list(range(0, 25)):
        t1 = list(range(t, t + model.ERT[k] + 1))
        model.constraints.add(model.y[k,t] <= sum(model.z[k, j] for j in t1 if j <= 24))'''

for k in model.k:  # constrains 36
    for t in list(range(1, 25)):
        model.constraints.add(model.y[k, t] - model.z[k, t] == model.u[k, t] - model.u[k, t - 1])
for k in model.k:      # constrain 37
    for t in list(range(0, 25)):
        model.constraints.add(model.y[k,t] + model.z[k, t] <= 1)

opt = SolverFactory("gurobi", solver_io="python")
results = opt.solve(model, tee=True)

res_tup = []
# Appending the results in the list
for k in model.k:
    for t in model.t:
        res_tup.append((t, model.LR_es[t].value, model.LR1_es[t].value, model.LR2_es[t].value, model.CLR_es[t].value))

# Initiating a df and append the list of results
result_df = pd.DataFrame(res_tup)
del res_tup
result_df.columns = ['t', 'LR_es', 'LR1_es', 'LR2_es', 'CLR_es']

LR = []
LR1 = []
LR2 = []
for i in list(range(0, 25)):
    lr = result_df['LR_es'][i]
    lr1 = result_df['LR1_es'][i]
    lr2 = result_df['LR2_es'][i]
    LR.append(lr)
    LR1.append(lr1)
    LR2.append(lr2)

x = list(DR_aggregator.Market_signal[id].keys())
y1 = list(DR_aggregator.Market_signal[id].values())

fig, ax = plt.subplots()
ax.plot(x, y1, color="b", alpha=0.5, drawstyle='steps', label="Market Price")
plt.ylim(0, 65)
plt.xlim(0, 24)
plt.xticks(np.arange(0, 24, 2))
ax.set_xlabel("time in hours", fontsize=11)
ax.set_ylabel('Energy_price ($/MWh)', color="Black", fontsize=11)

# make a plot with different y-axis using second axis object
ax2 = ax.twinx()
ax2.plot(x, LR, color="r", drawstyle='steps', label="contract 2")
ax2.plot(x, LR1, color="b", drawstyle='steps', label="contract 1")
ax2.legend(loc='upper right', bbox_to_anchor=(0.3, 1))
ax2.legend(loc='upper right', bbox_to_anchor=(0.3, 1))
ax.legend(loc='upper right', bbox_to_anchor=(0.3, 0.87))
plt.ylim(0, 35)
ax2.set_ylabel("Scheduled ES(MW)", color="Black", fontsize=11)
# fig.savefig('Optimal ES Scheduling for SME1 and SME2', format='jpeg', dpi=100, bbox_inches='tight')
plt.show()
