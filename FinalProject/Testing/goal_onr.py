"""
© 2024 This work is licensed under a CC-BY-NC-SA license.
Title:
"""

import numpy as np
import utils as ut

#import _pickle as cPickle
import traceback


def sigm ( x, dv ):

	if dv < 1 / 30:
		return x > 0
	y = x / dv
    #y = x / 10.

	out = 1.5*(1. / (1. + np.exp (-y*3. )) - .5)

	return out

def gaussian(x, mu, sig):
	return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

class RESERVOIRE_SIMPLE_NL_MULT:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape']
        net_shape = (self.N, self.T)

        self.dt = par['dt']#1. / self.T
        #self.itau_m = self.dt / par['tau_m_f']
        #self.itau_m = np.logspace( np.log(self.dt / par['tau_m_f']) ,np.log(self.dt / par['tau_m_s']),self.N)
        self.tau_m = np.linspace(  par['tau_m_f'],  par['tau_m_s'] ,self.N)
        #self.tau_m = np.logspace(  np.log(par['tau_m_f']) , np.log( par['tau_m_s']) ,self.N)
        #self.itau_m = np.linspace( self.dt / par['tau_m_f'], self.dt / par['tau_m_s'] ,self.N)

        self.itau_s = np.exp (-self.dt / par['tau_s'])
        self.itau_ro = np.exp (-self.dt / par['tau_ro'])

        self.dv = par['dv']

        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec'], size = (self.N, self.N))#np.zeros ((self.N, self.N))

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I))
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O))
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N))#np.zeros ((self.O, self.N))
        self.h_Jout = np.zeros((self.O,))
        
        self.y = np.zeros((self.O,))

        # Remove self-connections
        self.name = 'model'

        # Impose reset after spike
        self.s_inh = -par['s_inh']
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h']

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h

        # Membrane potential
        self.H = np.ones (self.N,) * par['Vo']*0.

        self.Vo = par['Vo']

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N,)
        self.S_hat = np.zeros (self.N,)
        self.S_ro = np.zeros (self.N,)
        self.dH = np.zeros (self.N,)
        self.Vda = np.zeros (self.N,)

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N,)
        self.state_out_p = np.zeros (self.N,)

        # Here we save the params dictionary
        self.par = par

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv

        out = np.zeros (x.shape)
        mask = x > 0
        out [mask] = 1. / (1. + np.exp (-y [mask]))
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]))
        out = out+1.
        return out

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv))

    def step_rate (self, inp, inp_modulation,sigma_S, if_tanh=True):
        #itau_m = self.itau_m
        itau_s = self.itau_s
        itau_ro = self.itau_ro

        self.S_hat   = np.copy(self.S)#(self.S_hat   * itau_s + self.S   * (1. - itau_s))
        #(self.S_ro   * itau_ro + self.S   * (1. - itau_ro))
        #self.dH   = self.dH  * (1. - itau_m) + itau_m * self.S_hat
        #self.H   = self.H   * (1. - itau_m) + itau_m * np.tanh ( self.J @ self.S_hat   + self.Jin @ inp )
        #self.H   = self.H   * (1. - itau_m) + itau_m * (self.J @ self.S_hat   + self.Jin @ inp )
        #self.H   = self.H   * np.exp(-self.dt/self.tau_m) + (1-np.exp(-self.dt/self.tau_m) ) * (self.J @ self.S_hat   + self.Jin @ inp )
        #self.H   = self.H   * (1-self.dt/self.tau_m) + (self.dt/self.tau_m)  * (self.J @ self.S_hat   + self.Jin @ inp )

        self.H   = self.H   * np.exp(-self.dt/self.tau_m) + (1-np.exp(-self.dt/self.tau_m) ) * np.tanh( ( self.J @ self.S_hat   + self.Jin @ inp )* (inp_modulation) )

        self.S  = np.copy(self.H + np.random.normal (0., sigma_S, size = np.shape(self.H) ))#(np.tanh(self.H)+1)/2
        self.S_ro   = np.copy(self.S)
        #self.y = np.tanh(self.Jout@ self.S_ro + self.h_Jout)# np.tanh(self.Jout@ self.S_ro + self.h_Jout)#np.tanh(self.Jout@ self.S_ro + self.h_Jout)
        if if_tanh:
            self.y = np.tanh(self.Jout@ self.S_ro + self.h_Jout)# np.tanh(self.Jout@ self.S_ro + self.h_Jout)#np.tanh(self.Jout@ self.S_ro + self.h_Jout)
        else:
            self.y = (self.Jout@ self.S_ro + self.h_Jout)# np.tanh(self.Jout@ self.S_ro + self.h_Jout)#np.tanh(self.Jout@ self.S_ro + self.h_Jout)


        return self.H


    def reset (self, init = None):
        self.S   = np.zeros ((self.N,))#init if init else np.zeros (self.N,)
        self.S_hat   = np.zeros ((self.N,)) #  * self.itau_s if init else np.zeros (self.N,)
        self.S_ro   = np.zeros ((self.N,))#   * self.itau_s if init else np.zeros (self.N,)
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  = np.zeros ((self.N,))
        #self.H  += self.Vo
        self.y = np.zeros((self.O,))
        
    def init_clock (self):
        n_steps = self.I
        T = self.T

        I_clock = np.zeros((n_steps,T))
        for t in range(T):
            k = int(np.floor(t/T*n_steps))
            I_clock[k,t] = 1
            self.I_clock = I_clock