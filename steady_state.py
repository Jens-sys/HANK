# find steady state

import time
import numpy as np
from scipy import optimize

from consav import elapsed

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ##################################
    # 1. grids and transition matrix #
    ##################################

    # b. a
    par.a_grid[:] = equilogspace(par.a_min,par.a_max,par.Na)

    # c. z
    par.z_grid[:],ss.z_trans[:,:,:],e_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,n=par.Nz)

    ###########################
    # 2. initial distribution #
    ###########################
    
    for i_fix in range(par.Nfix):
        ss.Dz[i_fix,:] = e_ergodic/par.Nfix
        ss.Dbeg[i_fix,:,0] = ss.Dz[i_fix,:]
        ss.Dbeg[i_fix,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_a = np.zeros((par.Nfix,par.Nz,par.Na))
    
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            z = par.z_grid[i_z]
            income = (1-ss.tau)*ss.w*ss.L*z ++ ss.chi

            c = (1+ss.ra)*par.a_grid + income
            v_a[i_fix,i_z,:] = c**(-par.sigma)

            ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a[i_fix]
        
def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # step 2
    ss.Gamma = 1.0
    ss.r = ss.i = par.r_target_ss #due to equation 13
    ss.G = par.G_target_ss
    ss.chi = par.chi_target_ss
    ss.q = 1/(1+ss.r-par.delta) #calculate q already now
    ss.B = par.qB_target_ss/ss.q
    
    # step 3
    ss.L = 1.0
    
    # step 4
    ss.pi = ss.pi_w = 0.0
    
    # step 5
    ss.w = ss.Gamma
    ss.Y = ss.Gamma*ss.L
    ss.tau = (ss.G + ss.chi - (ss.q*(1-par.delta)-1)*ss.B)/ss.Y
    
    ss.ra = 1/ss.q + par.delta - 1
    
    # step 6
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)
    
    print(np.sum(ss.D))
    
    # step 7
    par.varphi = (1/par.mu)*(1-ss.tau)*ss.w*(np.sum(ss.c*ss.D)**(-par.sigma))/(ss.L**(-par.nu))
    
    # step 8
    #ss.A = np.sum(ss.a*ss.D)
    ss.clearing_A = ss.q*ss.B - np.sum(ss.a*ss.D)
    ss.clearing_Y = ss.Y - np.sum(ss.c*ss.D) - ss.G
    
def obj_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    par.beta = x
    evaluate_ss(model,do_print=do_print)
    
    return ss.clearing_A

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    t0 = time.time()

    beta_min = 0.85**(1/4)
    beta_max = 1/(1+par.r_target_ss)-1e-4

    res = optimize.root_scalar(obj_ss,bracket=(beta_min,beta_max),method='brentq',args=(model,))

    # final evaluation
    obj_ss(res.root,model)

    # b. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f' r    = {ss.r:8.4f}')
        print(f' q    = {ss.q:8.4f}')
        print(f' qB   = {ss.q*ss.B:8.4f}')
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}')
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')
