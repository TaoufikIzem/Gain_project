# ABMS model
# Create SMEs object
from collections import defaultdict
import pandas as pd
import numpy as np




def main():  # run it as main file
    pass


class DR_aggregator:
    SMEs_dict = dict()
    Market_signal = dict()

    def __init__(self, C_dict=None, SME2_input=None, SME1_out=None, SME2_out=None):
        self.SME1_input = dict()
        self.SME2_input = dict()
        self.SME1_out = defaultdict()
        self.SME2_out = defaultdict()
        self.df = None
        self.C_dict = dict()
        self.C_dict1= dict()
        self.df1= None
        self.df2= None


    #  Market difinition and extracting data from xlsx file for 9th July 2012
    def get_whole_sale_price_dict(self):
        self.df = pd.read_excel('hourly1.xlsx')
        self.df1 = self.df['energy_price'].values
        for n in range(len(list(self.df1))):
            self.C_dict[n] = self.df1[n]
        return self.C_dict

    def get_whole_price_dict1(self):
         self.df2 = pd.read_excel('hourly_Jan.xlsx')
         self.df3 = self.df2['Energy_pr'].values
         for n in range(len(list(self.df3))):
             self.C_dict1[n] = self.df3[n]
         return self.C_dict1


    def addMarket(self, id, name):
        self.Market_signal[id] = name

    @classmethod
    def getMarket(cls, id):
        return cls.Market_signal[id]

    #  SMEs dictionary creating
    def addSME_dict(self, index, sme_name):  # add elements to a dictionary
            self.SMEs_dict[index] = sme_name

    @classmethod
    def getSMEx(cls, id):  # call object from dictionary using ID
        return cls.SMEs_dict[id]




    # def create_SME2_out(self, id, LC2_q_out, LS2_q_out, OG2_q_out, ES2_q_out, LC2_t_on, LC2_t_off,
    #           LS2_t_on, LS2_t_off, OG2_t_on, OG2_t_off, ES2_t_on, ES2_t_off):


class SME():
    num_smes= 0
    def __init__(self, id, lc=None, ls=None, og=None, es=None):
        self.id = id
        self.lc = lc
        self.ls = ls
        self.og = og
        self.es = es
        SME.num_smes += 1

# Load reduction strategies:

class load_curtailment:
    def __init__(self, lc_capacity, lc_price, lc_init, lc_min_duration, lc_max_duration, lc_daily_cur):
        self.lc_capacity = lc_capacity
        self.lc_price = lc_price
        self.lc_init = lc_init
        self.lc_min_duration = lc_min_duration
        self.lc_max_duration = lc_max_duration
        self.lc_daily_cur = lc_daily_cur

    def __str__(self):
        return f"quantity= {self.lc_capacity} price= {self.lc_price} LRinit= {self.lc_init} Min_duration= " \
               f"{self.lc_min_duration} Max_duration= {self.lc_max_duration} Daily_curt= {self.lc_daily_cur}"


class load_shifting:
    def __init__(self, ls_capacity, ls_price, ls_init, ls_min_duration, ls_max_duration, ls_daily_shift, tls_from,
                 tls_to, tsh_from, tsh_to, alpha):
        self.ls_capacity = ls_capacity
        self.ls_price = ls_price
        self.ls_init = ls_init
        self.ls_min_duration = ls_min_duration
        self.ls_max_duration = ls_max_duration
        self.ls_daily_shift = ls_daily_shift
        self.tls_from = tls_from
        self.tls_to = tls_to
        self.tsh_from = tsh_from
        self.tsh_tp = tsh_to
        self.alpha = alpha

    def __str__(self):
        return f"Hi i am load shifting"


class onsite_generation:
    def __init__(self, min_power, max_power, og_price, startup_cost, t_on, t_off, ramp_up, ramp_down, stratup_fuel,
                 fuel_limit):
        self.min_power = min_power
        self.max_power = max_power
        self.og_price = og_price
        self.startup_cost = startup_cost
        self.t_on = t_on
        self.t_off = t_off
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.stratup_fuel = stratup_fuel
        self.fuel_limit = fuel_limit

    def __str__(self):
        return f"Hi i am onsite generation"


class energy_storage:
    def __init__(self, power_rating, es_price, Energy_capacity, discharge_eff, discharge_ramp, ERT, MN):
        self.power_rating = power_rating
        self.es_price = es_price
        self.Energy_capacity = Energy_capacity
        self.discharge_eff = discharge_eff
        self.discharge_ramp = discharge_ramp
        self.ERT = ERT
        self.MN = MN

    def __str__(self):
        return f"Hi i am energy storage"


if __name__ == "__main__":
    main()

# instantiating the objects
lc1 = load_curtailment(3, 40, 100, 3, 6, 1).__dict__
ls1 = load_shifting(10, 40, 100, 3, 6, 1, 10, 16, 4, 10, 100).__dict__
og1 = onsite_generation(1, 10, 40, 100, 1, 1, 10, 10, 20, 100).__dict__
es1 = energy_storage(10, 40, 60, 0.9, 20, 12, 1).__dict__
SME1 = SME(1, lc1, ls1, og1, es1).__dict__

lc2 = load_curtailment(4, 45, 100, 3, 6, 1).__dict__
ls2 = load_shifting(10, 45, 100, 3, 6, 1, 14, 20, 8, 14, 100).__dict__
og2 = onsite_generation(1, 10, 45, 100, 1, 1, 10, 10, 20, 100).__dict__
es2 = energy_storage(10, 45, 60, 0.9, 20, 12, 1).__dict__
SME2 = SME(2, lc2, ls2, og2, es2).__dict__

lc3 = load_curtailment(5, 50, 100, 3, 6, 1).__dict__
ls3 = load_shifting(10, 50, 100, 3, 6, 1, 14, 20, 8, 14, 100).__dict__
og3 = onsite_generation(1, 10, 50, 100, 1, 1, 10, 10, 20, 100).__dict__
es3 = energy_storage(10, 50, 60, 0.9, 20, 12, 1).__dict__
SME3 = SME(3, lc3, ls3, og3, es3).__dict__
# print(SME21.lc)

gr = DR_aggregator()
# add SME objects to a dictionary
gr.addSME_dict(1, SME1)
gr.addSME_dict(2, SME2)
gr.addSME_dict(3, SME3)

# create SMEs inputs
SME1_input = gr.getSMEx(1)
SME2_input = gr.getSMEx(2)
SME2_input = gr.getSMEx(3)

# create Market_price input
Market1 = gr.get_whole_sale_price_dict()
Market2 = gr.get_whole_price_dict1()
gr.addMarket(1, Market1)
gr.addMarket(2, Market2)
#Market1 = gr.getMarket(1)
#Market2 = gr.getMarket(2)



# Market_price.append(C_dict)
# print(Market_signal)
#print(SME.num_smes)
#print(help(load_curtailment))

