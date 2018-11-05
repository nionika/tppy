#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python2.7 ocmpatibility
from __future__ import print_function, division, unicode_literals
from collections import namedtuple
import numpy as np
from scipy.optimize import linprog


class BasicEstimator:
    b_t = 0.1364 # Коэффициент перевода баррелей в тонны

    def __init__(self, N, M, **kwargs):
        '''param N: количество нефтеперерабатывающих заводов
           param M: котичество месторождений
        '''
        self.N = N
        self.M = M
        self.xdim = 2 * (N + M) # размерность целевой переменной
        # Соберем необходимые названия переменных.
        self.names = []
        self.names += ['q_oil_e_%d' % (m + 1) for m in range(M)]
        self.names += ['q_oil_d_%d' % (m + 1) for m in range(M)]
        self.names += ['q_petr_e_%d' % (n + 1) for n in range(N)]
        self.names += ['q_petr_d_%d' % (n + 1) for n in range(N)]
        # Тип для переменной-результата.
        self.ResultType = namedtuple('Quantities', self.names)

        # Имя оптимизационной задачи (например 'lukoil_2014')
        self.name = None
        if 'name' in kwargs.keys():
            self.name = kwargs['name']

        # Здесь будут ограничения
        self.A_ub = None
        self.b_ub = None
        self.A_eq = None
        self.b_eq = None
        # Здесь будет вектор цен
        self.p = None

        # Пока оставим параметры в виде словаря 
        self.param_dict = kwargs
        # for k,v in kwargs.iteritems():

    def solve(self, p=None):
        if p is not None:
            self.param_dict['p'] = p
        self.p = self.param_dict['p']

        self.opt_res = linprog(-self.p, A_ub=self.A_ub, b_ub=self.b_ub,
                               A_eq=self.A_eq, b_eq=self.b_eq)
        print(self.opt_res.message)
        self.res = self.ResultType(*self.opt_res.x.tolist())
        print(self.name, ':', self.res)
        return self.res

    # Ограничение 1(1): сумма объемов добытой нефти на месторождении m,
# идущая на экспорт и внутреннее потребление не превышает мощность добычи месторождения m
    def constraint_prod_capacity(self, prod_capacity=None):
        N = self.N
        M = self.M
        if prod_capacity is not None:
            self.param_dict['prod_capacity'] = prod_capacity
        prod_capacity = self.param_dict['prod_capacity']
        assert prod_capacity is not None, \
               'prod_capacity is None'
        assert len(prod_capacity) == M, \
               'incorrect prod_capacity shape. expected %d, got %d' \
                % (M, len(prod_capacity))
        
        # Составляем ограничения
        A = np.zeros(shape=(M, self.xdim), dtype=np.float32)
        b = np.zeros(shape=(M,), dtype=np.float32)
        for m in range(M):
            A[m,m] = 1
            A[m,M + m] = 1
            b[m] = prod_capacity[m] * 365 * self.b_t
        
        self._stack_ub(A, b)

# Ограничение 1(2): сумма объемов добытой нефти на месторождении m,
# идущая на экспорт и внутреннее потребление не превышает объем добычи месторождения m
    def constraint_q_oil(self, q_oil=None):
        N = self.N
        M = self.M

        if q_oil is not None:
            self.param_dict['q_oil'] = q_oil
        q_oil = self.param_dict['q_oil']
        assert q_oil is not None, \
               'q_oil is None'
        assert len(q_oil) == M, \
               'incorrect q_oil shape. expected %d, got %d' \
                % (M, len(q_oil))
        
        # Составляем ограничения
        A = np.zeros(shape=(M, self.xdim), dtype=np.float32)
        b = np.zeros(shape=(M,), dtype=np.float32)
        for m in range(M):
            A[m,m] = 1
            A[m,M + m] = 1
            b[m] = q_oil[m]
        
        self._stack_ub(A, b)

# Ограничение 2: сумма объемов нефти добытой на месторождении m и доставленной на НПЗ n
# не превышает установленную мощность НПЗ n
    def constraint_ref_capacity(self, alpha=None, ref_capacity=None):
        N = self.N
        M = self.M

        if alpha is not None:
            self.param_dict['alpha'] = alpha
        alpha = self.param_dict['alpha']

        assert alpha is not None, \
               'alpha is None'
        assert alpha.shape == (M, N), \
               'incorrect alpha shape. expected (%d, %d), got %s' \
                % (M, N, alpha.shape)

        if ref_capacity is not None:
            self.param_dict['ref_capacity'] = ref_capacity
        ref_capacity = self.param_dict['ref_capacity']
        assert ref_capacity is not None, \
               'ref_capacity is None'
        assert len(ref_capacity) == N, \
               'incorrect ref_capacity shape. expected %d, got %d' \
                % (N, len(ref_capacity))

        # Составляем ограничения
        A = np.zeros(shape=(N, self.xdim), dtype=np.float32)
        b = np.zeros(shape=(N,), dtype=np.float32)
        for n in range (N):
            for m in range(M):
                A[n, M + m] = alpha[m, n]
            b[n] = ref_capacity[n]
        
        self._stack_ub(A, b)

# Ограничение 3: сумма объемов нефти добытой на месторождении m и доставленной на НПЗ n
# больше, чем объем производства на НПЗ n нефтепродуктов (бензина и мазута)
    def constraint_ref_yield(self, alpha=None, ref_yield=None):
        N = self.N
        M = self.M

        if alpha is not None:
            self.param_dict['alpha'] = alpha
        alpha = self.param_dict['alpha']

        if ref_yield is not None:
            self.param_dict['ref_yield'] = ref_yield
        ref_yield = self.param_dict['ref_yield']

        A = np.zeros(shape=(N, self.xdim), dtype=np.float32)
        b = np.zeros(shape=(N,), dtype=np.float32)

        for n in range (N):
            for m in range(M):
                A[n, m + M] = -alpha[m,n]
            A[n, 2 * M + n ] = 100 / (ref_yield[n])
            A[n, 2 * M + N + n] = 100 / (ref_yield[n])
        self._stack_ub(A, b)

# Ограничение 4: объемы реализации нефти и нп на экспорт и на внутренний рынок положительные
    def constraint_non_negative(self):
        self._stack_ub(-np.eye(self.xdim, dtype=np.float32), 
                       np.zeros(self.xdim, dtype=np.float32))

# Ограничение 5: сумма объемов экспорта нефти с месторождений не превышает спрос
    def constraint_oil_exp(self, q_oil_exp=None):
        if q_oil_exp is not None:
            self.param_dict['q_oil_exp'] = q_oil_exp
        q_oil_exp = self.param_dict['q_oil_exp']

        A = np.zeros((1, self.xdim))
        A[0, : self.M] = 1
        b = np.array([q_oil_exp])

        self._stack_ub(A, b)

# Ограничение 6: объемы внутреннего сбыта нефти не превышают сумму переработанной нефти на всех НПЗ
    def constraint_oil_ref(self, q_oil_ref=None):
        if q_oil_ref is not None:
            self.param_dict['q_oil_ref'] = q_oil_exp
        q_oil_ref = self.param_dict['q_oil_ref']

        M = self.M
        A = np.zeros(shape=(1, self.xdim), dtype=np.float32)
        for m in range(M):
            A[0, M + m] = 1 # -1?
        b = np.array([np.array([q_oil_ref]).sum()]) # -1?
        self._stack_ub(A, b)

# Ограничение 7: сумма объемов внутреннего сбыта нефтепродуктов не превышают спрос
    def constraint_petr_dom_demand(self, q_petr_dom=None):
        if q_petr_dom is not None:
            self.param_dict['q_petr_dom'] = q_petr_dom
        b = np.array([self.param_dict['q_petr_dom']])

        A = np.zeros(shape=(1, self.xdim))
        M = self.M
        N = self.N
        for n in range(N):
            A[0, 2 * M + N + n] = 1
        self._stack_ub(A, b)

# Ограничение 8: объемы сбыта нефтепродуктов на экспорт не превышают спрос
    def constraint_petr_exp_demand(self, q_petr_exp=None):
        if q_petr_exp is not None:
            self.param_dict['q_petr_exp'] = q_petr_exp
        b = np.array([self.param_dict['q_petr_exp']])

        A = np.zeros(shape=(1, self.xdim))
        M = self.M
        N = self.N
        for n in range(N):
            A[0, 2 * M + n] = 1
        self._stack_ub(A, b)


    def update_params(self, **kwargs):
        self.param_dict.update(kwargs)

#  Заполняем столбец p
    def calculate_price(self, **kwargs):
        self.param_dict.update(kwargs)
        p_oil_exp = self.param_dict['p_oil_exp']
        p_petr_exp = self.param_dict['p_petr_exp']
        p_petr_dom = self.param_dict['p_petr_dom']
        r = self.param_dict['r']
        ndpi = self.param_dict['ndpi']
        exp_oil_tax = self.param_dict['exp_oil_tax']
        exp_petr_tax = self.param_dict['exp_petr_tax']
        excise_petr = self.param_dict['excise_petr']
        cost_oil_prod = self.param_dict['cost_oil_prod']
        cost_oil_tran = self.param_dict['cost_oil_tran']
        cost_oil_ref = self.param_dict['cost_oil_ref']
        cost_petr_tran = self.param_dict['cost_petr_tran']

        p = np.zeros(shape=(self.xdim,), dtype=np.float32)
        M = self.M
        N = self.N
        #  Заполняем столбец p
        for m in range(M):
            p[m] = (p_oil_exp - ndpi - exp_oil_tax) * r - cost_oil_prod[m] - cost_oil_tran
        for m in range(M):
            p[M + m] = 0
        for n in range(N):
            p[2 * M + n ] = (p_petr_exp - exp_petr_tax) * r - cost_oil_ref - cost_petr_tran
        for n in range(N):
            p[2 * M + N + n] = p_petr_dom * r - excise_petr - cost_oil_ref - cost_petr_tran

        self.p = p
        self.param_dict['p'] = self.p
        return p

    def _stack_constraints(self, A, b, A_old, b_old):
        '''Добавить ограничения в задачу
        '''
        assert A.shape[1] == self.xdim, \
               'incorrect dimentions of inequeality constraints'
        assert A.shape[0] == b.shape[0], \
               'inconsistent A and b shapes for inequality constraints'

        if A_old is None:
            return A.copy(), b.copy()
        else:
            return np.vstack([A_old, A]), np.hstack([b_old, b])

    def _stack_ub(self, A, b):
        '''Добавить ограничения-неравенства в задачу
        '''
        self.A_ub, self.b_ub = \
            self._stack_constraints(A, b, self.A_ub, self.b_ub)

    def _stack_eq(self, A, b):
        '''Добавить ограничения-равенства в задачу
        '''
        self.A_eq, self.b_eq = \
            self._stack_constraints(A, b, self.A_eq, self.b_eq)

# Пример
if __name__ == '__main__':
    # defines for test case
    prod_capacity = [970000, 313000, 308000, 136000] # барр. н. э./сут, мощность добычи
    oil_base = [7304000000, 2410000000, 2163000000, 912000000] # млн барр., запасы
    q_oil = [46840000, 15814000, 14585000, 6659000] # тонн, объем добычи нефти в 2014
    q_oil_exp = 11725000 # тонн, объем экспорта нефти в 2014
    q_oil_dom = 29833000 # тонн, объем внутреннего потребления в 2014
    q_oil_imp = 1034000 # тонн, объем импорта нефти в 2014

    ref_yield = [92.5, 82.9, 64.8, 64.3] # %, глубина переработки нефти
    ref_capacity = [11300000, 13100000, 17000000, 4000000] # тонн в год, установленная мощность НПЗ
    q_oil_ref = [11413000, 12685000, 17021000, 3993000] # тонн, объем переработки нефти в 2014
    q_petr = [10932000, 12430000, 16294000, 3835000] # тонн, объем произведенных нефтепродуктов в 2014
    q_petr_exp = 23377000 # тонн, объем экспорта нефтепродуктов в 2014
    q_petr_dom = 22337000 # тонн, объем внутреннего потребления нефтепродуктов в 2014
    q_petr_imp = 2041000 # тонн, объем импорта нефтепродуктов в 2014

    p_oil_exp = 718.79 # $/тонну, цена на нефть марки "Юралс" (СИФ Средиземномрск регион) в 2014
    p_petr_exp = 918.87 # $/тонну, высокооктановый бензин (ФОБ Роттердам) в 2014
    p_petr_dom = 834.17 # $/тонну, бензин (Аи-95) в 2014

    ndpi = 151.67 # $/тонну, НДПИ в 2014
    exp_oil_tax = 366.53 # $/тонну, экспортная пошлина на нефть в 2014
    exp_petr_tax = 329.83  # $/тонну, экспортная пошлина на нефтепродукты в 2014
    excise_petr = 6450 # руб/ тонну, акциз на бензин класса Евро-5 в 2014

    r = 38.42 # руб/ $, обменный курс в 2014
    b_t = 0.1364 # 1 баррель = 0.1364 тоннны

    cost_oil_prod = [2677, 5474, 2750, 8938] # руб/ тонну, удельные издержки на добычу нефти в 2014
    cost_oil_ref = 1632 # руб/ тонну, удельные издержки на нефтепереработку в 2014
    cost_oil_tran = 696 # руб/ тонну, удельные издержки на транспортировку нефти в 2014
    cost_petr_tran = 6332 # руб/ тонну, удельные издержки на транспортировку нефтепродуктов в 2014
    alpha = np.matrix([[0.402054292, 0.241746148, 0.35619956, 0],
                       [0, 0, 0, 1],
                       [0, 0.213396562, 0.786603438, 0],
                       [0, 0.966403162, 0.033596838, 0]],
                      dtype=np.float32)

    # tesing:
    estim = BasicEstimator(4,4, name='lukoil_2014')
    estim.constraint_prod_capacity(prod_capacity=prod_capacity)
    estim.constraint_q_oil(q_oil=q_oil)
    estim.constraint_ref_capacity(alpha=alpha, ref_capacity=ref_capacity)
    estim.constraint_ref_yield(alpha=alpha, ref_yield=ref_yield)
    estim.constraint_non_negative()
    estim.constraint_oil_exp(q_oil_exp=q_oil_exp)
    # estim.constraint_oil_ref(q_oil_exp=q_oil_ref)
    estim.constraint_petr_dom_demand(q_petr_dom=q_petr_dom)
    estim.constraint_petr_exp_demand(q_petr_exp=q_petr_exp)

    p = estim.calculate_price(p_oil_exp=p_oil_exp, p_petr_exp=p_petr_exp,
                              p_petr_dom=p_petr_dom, r=r, ndpi=ndpi, 
                              exp_oil_tax=exp_oil_tax, exp_petr_tax=exp_petr_tax, 
                              excise_petr=excise_petr, cost_oil_prod=cost_oil_prod, 
                              cost_oil_tran=cost_oil_tran, cost_oil_ref=cost_oil_ref, 
                              cost_petr_tran=cost_petr_tran)
    estim.solve()