#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:17:24 2018

функция зачитываеия из св переменных для одного года в словарик (НАЗВАНИЕ)
для каждого года создаю словарь и эстиматор
@author: evgeniyamiller
"""
import pandas as pd 
import numpy as np

file = "/Users/evgeniyamiller/Documents/GitHub/tppy/oilestim/data_for_est_oil.csv"

alpha = np.matrix([[0.402054292, 0.241746148, 0.35619956, 0],
               [0, 0, 0, 1],
               [0, 0.213396562, 0.786603438, 0],
               [0, 0.966403162, 0.033596838, 0]],
              dtype=np.float32)


def load_file_into_dict(file):
    df = pd.read_csv(file, delimiter=';', index_col="year", decimal=",")
    d = df.to_dict('index')
    cost_oil_prod = ["cost_oil_prod_%d" % i for i in range(1,5)]
    oil_base = ["oil_base_%d" % i for i in range(1,5)]
    prod_capacity = ["prod_capacity_%d" % i for i in range(1,5)]
    q_oil = ["q_oil_%d" % i for i in range(1,5)]
    q_oil_ref = ["q_oil_ref_%d" % i for i in range(1,5)]
    q_petr = ["q_petr_%d" % i for i in range(1,5)]
    ref_capacity = ["ref_capacity_%d" % i for i in range(1,5)]
    ref_yield = ["ref_yield_%d" % i for i in range(1,5)]
    year = [2014, 2015, 2016, 2017]
    varr = [(cost_oil_prod, "cost_oil_prod"), (oil_base, "oil_base"), (prod_capacity, "prod_capacity"), (q_oil, "q_oil"), (q_oil_ref, "q_oil_ref"), (q_petr, "q_petr"), (ref_capacity, "ref_capacity"), (ref_yield, "ref_yield")]
    for n in year:
        for m, m_name in varr:
            d[n][m_name] = [d[n][k] for k in m]
            for k in m:
                del d[n][k]
            d[n]['alpha'] = alpha
                
    return d
