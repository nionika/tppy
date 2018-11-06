#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:49:08 2018

@author: evgeniyamiller
"""

# Загружаем функции считывания данных из файла и решения линейной задачи оптимизации
from loading_data import load_file_into_dict
from optimizers import BasicEstimator
from loading_data import store_res_into_file

# Путь файла и его считка
file = "/Users/evgeniyamiller/Documents/GitHub/tppy/oilestim/data_for_est_oil.csv"
d = load_file_into_dict(file)

# Создаем пустой словарь, куда будут сохраняться результаты
results = {}

# Решение задачи оптимизации для каждого года
estim = BasicEstimator(4,4, name='lukoil_2014', **d[2014])
estim.constraint_prod_capacity()
estim.constraint_q_oil()
estim.constraint_ref_capacity()
estim.constraint_ref_yield()
estim.constraint_non_negative()
estim.constraint_oil_exp()
estim.constraint_oil_ref()
estim.constraint_petr_dom_demand()
estim.constraint_petr_exp_demand()
estim.solve(estim.calculate_price())
results[2014] = estim.res._asdict()

estim = BasicEstimator(4,4, name='lukoil_2015', **d[2015])
estim.constraint_prod_capacity()
estim.constraint_q_oil()
estim.constraint_ref_capacity()
estim.constraint_ref_yield()
estim.constraint_non_negative()
estim.constraint_oil_exp()
estim.constraint_oil_ref()
estim.constraint_petr_dom_demand()
estim.constraint_petr_exp_demand()
estim.solve(estim.calculate_price())
results[2015] = estim.res._asdict()

estim = BasicEstimator(4,4, name='lukoil_2016', **d[2016])
estim.constraint_prod_capacity()
estim.constraint_q_oil()
estim.constraint_ref_capacity()
estim.constraint_ref_yield()
estim.constraint_non_negative()
estim.constraint_oil_exp()
estim.constraint_oil_ref()
estim.constraint_petr_dom_demand()
estim.constraint_petr_exp_demand()
estim.solve(estim.calculate_price())
results[2016] = estim.res._asdict()

estim = BasicEstimator(4,4, name='lukoil_2017', **d[2017])
estim.constraint_prod_capacity()
estim.constraint_q_oil()
estim.constraint_ref_capacity()
estim.constraint_ref_yield()
estim.constraint_non_negative()
estim.constraint_oil_exp()
estim.constraint_oil_ref()
estim.constraint_petr_dom_demand()
estim.constraint_petr_exp_demand()
estim.solve(estim.calculate_price())
results[2017] = estim.res._asdict()

file_res = "/Users/evgeniyamiller/Documents/GitHub/tppy/oilestim/data_res_oil.csv"
store_res_into_file(file_res, results)