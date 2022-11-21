import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def block_pre(par,ini,ss,path,ncols=1):

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        chi = path.chi[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        G = path.G[ncol,:]
        Gamma = path.Gamma[ncol,:]
        i = path.i[ncol,:]
        L = path.L[ncol,:]
        NKWC_res = path.NKWC_res[ncol,:]
        pi_w = path.pi_w[ncol,:]
        pi = path.pi[ncol,:]
        r = path.r[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        q = path.q[ncol,:]
        ra = path.ra[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]

        #################
        # implied paths #
        #################
        
        #Firms
        w[:] = Gamma #Easiest one
        
        Gamma_lag = lag(ini.Gamma, Gamma)
        
        pi[:] = (1+pi_w)/(Gamma/Gamma_lag) - 1
        
        Y[:] = Gamma*L
        
        for t in range(par.T):
            i_lag = ss.i if t == 0 else i[t-1]
            i[t] = ((1+i_lag)**par.rho_i)*((1+ss.r)*((1+pi[t])**par.phi_pi))**(1-par.rho_i) - 1

        pi_plus = lead(pi,ss.pi)
        
        r[:] = (1+i)/(1+pi_plus) - 1
        
        
        i_lag_ra = lag(ini.i, i)
        
        ra[:] = (1+i_lag_ra)/(1+pi) - 1
        
        
        # If additional B endo:
        B_lag = lag(ini.B, B)
        
        tau[:] = ss.tau + par.omega*ss.q*(B_lag - ss.B)/ss.Y
        
        q[:] = (B_lag + G + chi - tau*Y)/(B-par.delta*B_lag)                      
        
        
        #for t in range(par.T):
                        
        #    q_lag = ini.q if t == 0 else q[t-1]
            
        #    q[t] = ((1+ra[t])*q_lag - 1)/par.delta
            
        #    B_lag = ini.B if t == 0 else B[t-1]
            
        #    tau[t] = ss.tau + par.omega*ss.q*(B_lag - ss.B)/ss.Y
            
        #    B[t] = (B_lag + G[t] + chi[t] - tau[t]*Y[t])/q[t] + par.delta*B_lag
        
        
        
        # firms
        #w[:] = Gamma #Easiest one
        #L[:] = Y/(Gamma)
        
        #Monetary policy
        #i_lag = lag(ini.i, i)
        
        
        #pi[:] = ((((1+i)/((1+i_lag)**(par.rho_i)))**(1/(1-par.rho_i)))*(1/(1+ss.r)))**(1/par.phi_pi) - 1
        
        #Gamma_lag = lag(ini.Gamma, Gamma)
        
        #pi_w[:] = (pi+1)*Gamma/Gamma_lag - 1
        
        #pi_plus = lead(pi,ss.pi)
        
        #r[:] = (1+i)/(1+pi_plus)-1 
 
        #Government
        #B_lag = lag(ini.B, B)
        #tau[:] = ss.tau + par.omega * ss.q * (B_lag - ss.B) / ss.Y
        
        #q[:] = (B_lag + G + chi - tau*Y)/(B - par.delta*B_lag) #Without chi for now
        
        #q_lag = lag(ini.q,q)
        
        #r_plus = lead(r, ss.r)
        #ra[:] = r_plus[:] = (1+par.delta*q)/q_lag - 1
        
        
        
        
@nb.njit
def block_post(par,ini,ss,path,ncols=1):

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        chi = path.chi[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        G = path.G[ncol,:]
        Gamma = path.Gamma[ncol,:]
        i = path.i[ncol,:]
        L = path.L[ncol,:]
        NKWC_res = path.NKWC_res[ncol,:]
        pi_w = path.pi_w[ncol,:]
        pi = path.pi[ncol,:]
        r = path.r[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        q = path.q[ncol,:]
        ra = path.ra[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]
        
        #################
        # check targets #
        #################
        
        pi_w_plus = lead(pi_w, ss.pi_w)
        
        NKWC_res[:] = pi_w - par.kappa*(par.varphi*(L**par.nu)-(1/par.mu)*(1-tau)*w*(C_hh**(-1*par.sigma))) - par.beta*pi_w_plus

        # b. market clearing
        clearing_A[:] = A_hh - q*B
        clearing_Y[:] = Y - C_hh - G
