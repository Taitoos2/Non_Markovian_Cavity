import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.interpolate import interp1d
from scipy.linalg import expm
from numpy.fft import fft, fftfreq
from math import comb, factorial

# ---------------- Heisenberg Langevin integrator ---------------------------- # 

from scipy.integrate import DOP853
s_0 = np.asarray([[0,0],[1,0]])

@dataclass

class new_cav_model:
    """ This class is used as an integrator for an Two Level System (TLS) coupled to a " Long cavity" . The TLS  interacts with itself in 
    a non-Markovian regime, and is coherently driven by a classical coherent field resonant with the frequency of the excited state. 
    
    Arguments
    ---------
    gamma: float
        Effective coupling between the emitters and the EM field
    phi: float
        Phase acquired by the photon when travelling the entire waveguide and back 
    Omega: float 
        Rabi frequency, fixes the intensity of the coherent field. 
    eta: float 
        The fraction of field that stays in the waveguide when bouncing back at the mirror. 
        
    buffer_size: int = 100
        Dimension of buffer for saving operators

    """

    gamma: float = 1.0
    phi: float = np.pi
    tau: float = 10
    Omega: float = 0.0 
    eta: float =1.0

    buffer_size: int = 100
    next_index: int = 0

    s_array: np.ndarray = field(init=False)  	# list that stores values of sigma
    a_out_array: np.ndarray = field(init=False)  # list that stores values of a_out(t)
    t_array: np.ndarray = field(init=False)  	# list that stores values of t
    
    a_out_delayed: np.ndarray = field(init=False)  	# local variable to store the value of a_out(t-\tau)

    a_dalayed : callable = field(init=False, default=None)  # interpolator 
    interpolador_low_index : int = 0 
    interpolador_high_index: int = 0 

    def __post_init__(self):

        self.Delta = self.phi / self.tau
        self.next_index = 1
        self.a_out_delayed=np.zeros((2,2),dtype=np.complex128).reshape(-1)
        self.a_delayed = None

        self.s_array = np.ones((self.buffer_size, 1),dtype=np.complex128) * s_0.reshape(-1)
        self.a_out_array = np.ones((self.buffer_size,1), dtype=np.complex128) *np.sqrt(self.gamma)*s_0.reshape(-1)
        self.t_array = np.zeros(self.buffer_size)


    def derivative(self,t,y):
        " To be Done: implement a mobile window of interpolation"
        s=y.reshape((2,2))
        sz = 2*(np.conjugate(s.T))@s - np.eye(2)
        dsdt = sz@(0.5*self.gamma*s -1j*self.Omega*0.5*np.eye(2))
        if t > self.tau + 1e-12 :
            interpolator=interp1d(self.t_array[:self.next_index],
            self.a_out_array[:self.next_index],
            assume_sorted=True,
            axis=0,
            copy=False,
            )
            self.a_out_delayed = interpolator(t-self.tau)
        return (dsdt - np.sqrt(self.gamma)*np.exp(1j*(self.phi+np.pi))*sz@self.a_out_delayed.reshape((2,2))).reshape(-1)
    
    def evolve(self,t_max,dt):
        integrator = DOP853(self.derivative,self.t_array[0],self.s_array[0],t_bound=t_max,max_step=dt)
        
        while integrator.t<t_max:
            
            integrator.step()

            if integrator.status == 'failed':
                print(f"⚠️ Integrador falló en t = {integrator.t:.3e}")
                i = min(self.next_index, self.buffer_size)
                self.t_array = self.t_array[:i]
                self.s_array = self.s_array[:i]
                self.a_out_array = self.a_out_array[:i]
                return 
            i = self.next_index
            if i >= self.buffer_size:
                ''' resize arrays when necessary '''
                self.buffer_size = int(self.buffer_size * 1.5)
                self.t_array = np.resize(self.t_array, self.buffer_size)
                self.s_array = np.resize(self.s_array, (self.buffer_size, 4))
                self.a_out_array = np.resize(self.a_out_array, (self.buffer_size, 4))

            self.t_array[i] = integrator.t
            self.s_array[i] = integrator.y
            self.a_out_array[i] = np.sqrt(self.gamma)*integrator.y +self.eta*np.exp(1j*self.phi)*self.a_out_delayed 
            self.next_index = i + 1	


        self.t_array = self.t_array[:self.next_index] # trim the array 
        self.s_array = self.s_array[:self.next_index]
        self.a_out_array = self.a_out_array[:self.next_index]

    def excited_state(self,initial):
        s = self.s_array.reshape(-1,2,2)
        s_dag = np.conjugate(np.transpose(s,axes=(0,2,1)))
        return self.t_array,np.abs(np.einsum('i,tik,tkj,j->t',np.conjugate(np.asarray(initial).T),s_dag,s,np.asarray(initial)))
    def current(self,initial):
        a=self.a_out_array.reshape(-1,2,2)
        a_dag = np.conjugate(np.transpose(a,axes=(0,2,1)))
        return self.t_array,np.abs(np.einsum('i,tik,tkj,j->t',np.conjugate(np.asarray(initial).T),a_dag,a,np.asarray(initial)))

# ----------------- analytical solution single emitter ----------------------- 

from numpy.polynomial import Polynomial

def J_analytical_new(gamma,phi,tau,t):
    
    alpha = 1j*phi/tau + 0.5*gamma 
    result =  np.exp(-alpha*t)*np.ones(len(t),dtype=complex)
    poli = Polynomial([1])
    N = int(t[-1]/tau)
    
    for n in range(1,N+1):
        dummie = poli.integ() 
        result += np.exp(-alpha*t)*np.exp(n*alpha*tau)*dummie(-gamma*(t-n*tau))*np.heaviside(t-n*tau,1)
        poli += dummie 

        
    return result

# --------------------- fourier transform --------------------------

def fast_f_t(x : np.ndarray,y:np.ndarray, M:int = 500):

        t_interp = np.linspace(0, x[-1], M)  
        dt = t_interp[1] - t_interp[0]
        y = np.interp(t_interp,np.real(x), np.real(y)) # i am forcing real values, this might not be correct, but currents should be real 
        y -= np.mean(y)
        k = np.linspace(0, 1 / dt, M + 1)[:-1]
        yk = np.abs(np.fft.fft(y))
        u = yk[: M // 2]
        if np.sum(np.abs(u)) != 0:
            return 2*np.pi*k[: M // 2], u    # normalization removed
        else:
            return 2*np.pi*k[: M // 2], u 

def fourier_transform_matrix(t,m,M):
    N = m.shape[-1]
    w,u = fast_f_t(t,m[:,0],M)
    result =np.zeros((w.shape[0],N),dtype=complex)
    result[:,0] = u
    for n in range(1,N):
        _,result[:,n] = fast_f_t(t,m[:,n],M)
    return w,result 

def spectrum_1(aw,initial):
    '''' This is The observable corresponding to the 'intensity' of the fourier transform 
    of a_out. I don't think this is the output current in a discrete-momentum scenario. '''
    state = np.asarray(initial)
    a = aw.reshape(-1,2,2)
    a_dag = np.conjugate(np.transpose(a,axes=(0,2,1)))
    return np.einsum('i,tik,tkj,j -> t',state,a_dag,a,state)
