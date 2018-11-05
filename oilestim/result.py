#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:49:08 2018

@author: evgeniyamiller
"""

#from . import loading_data
from loading_data import load_file_into_dict
from optimizers import BasicEstimator

file = "/Users/evgeniyamiller/Documents/GitHub/tppy/oilestim/data_for_est_oil.csv"

d = load_file_into_dict(file)

estim = BasicEstimator(4,4, name='lukoil_2014', **d[2014])
estim.constraint_prod_capacity()
estim.constraint_q_oil()
estim.constraint_ref_capacity()
estim.constraint_ref_yield()
estim.constraint_non_negative()
estim.constraint_oil_exp()
# estim.constraint_oil_ref()
estim.constraint_petr_dom_demand()
estim.constraint_petr_exp_demand()

estim.solve(estim.calculate_price())
    