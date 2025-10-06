import numpy as np 
from dataclasses import dataclass, field
from scipy.linalg import expm

# -------------------------- exact one-mode cavity integrator ------------------------------ 

@dataclass
class Driven_cavity_class: 
	omega_qubit : float = 1 
	omega_laser : float = 1			# laser initialy resonant with the qubit
	omega_cavity : float= 1	  		# cavity initialy resonant with the qubit
	Driving_intensity : float = 0.0	
	g  : float= 0.5 
	N_photons : int = 20 # Max number of excitations in the cavity - 1 
	start_ground: bool = False 

	sigma_minus : np.ndarray = field(init=False) 
	sigma_plus : np.ndarray = field(init=False) 
	b : np.ndarray = field(init=False)
	b_dag : np.ndarray = field(init=False) 
	Hamiltonian_0: np.ndarray = field(init=False) 
	psi0: np.ndarray = field(init=False) 
	
	 
	def __post_init__(self):
		B = np.zeros((self.N_photons,self.N_photons))
		B_dag = np.zeros((self.N_photons,self.N_photons))
		for n in range(self.N_photons-1):
			B[n,n+1] = np.sqrt(n+1)
			B_dag[n+1,n] = np.sqrt(n+1)
			
		self.sigma_plus = np.kron(np.array([[0,1],[0,0]]),np.eye(self.N_photons))
		self.sigma_minus = np.kron(np.array([[0,0],[1,0]]),np.eye(self.N_photons))
		self.b = np.kron(np.eye(2),B)
		self.b_dag = np.kron(np.eye(2),B_dag)

		self.Hamiltonian_0 = (self.omega_qubit-self.omega_laser)*self.sigma_plus@self.sigma_minus + (self.omega_cavity-self.omega_laser)*self.b_dag@self.b 
		self.Hamiltonian_0 +=  self.g*(self.sigma_plus@self.b + self.sigma_minus@self.b_dag) 
		self.Hamiltonian_0 += self.Driving_intensity/2 * (self.sigma_minus + self.sigma_plus)

		initial_cav = np.zeros(self.N_photons,dtype=complex)   
		initial_cav[0]=1 # zero photons in the initial state 
		if self.start_ground:
			initial_emitter = np.array([0,1],dtype=complex) # one excitation in the initial state  
		else:
			initial_emitter = np.array([1,0],dtype=complex) # one excitation in the initial state  
		self.psi0 = np.kron(initial_emitter,initial_cav)



	def evolve(self,times):
		vec = []
		for t in times:
			U = expm(-1j*self.Hamiltonian_0*t)
			psi_t = U @ self.psi0
			vec.append(psi_t)
		vec = np.array(vec)
		pe = np.real(np.einsum('ti,ij,tj ->t ',np.conjugate(vec),self.sigma_plus@self.sigma_minus,vec))
		exc =  np.real(np.einsum('ti,ij,tj ->t ',np.conjugate(vec),self.b_dag@self.b,vec))
		return pe,exc
	
# --------------------- exact two-mode cavity integrator ----------------------------------------- 

@dataclass

class two_modes_cavity: 
	''' To be done: this class can substitute the previous one (Driven_cavity_class),
	but there is no need to delete anything so far.  '''
	omega_q : float = 1 
	omega_c1:float = 1.2 
	omega_c2 : float = 1.8 
	g1:float = 0.2 
	g2: float = 0.2
	Rabi_freq : float  = 0.0 
	driving_freq: float = 1 
	N_photons : int = 10 # Max number of excitations in the cavity - 1 

	sigma_minus : np.ndarray = field(init=False) 
	sigma_plus : np.ndarray = field(init=False) 
	b1 : np.ndarray = field(init=False)
	b1_dag : np.ndarray = field(init=False) 
	b2 : np.ndarray = field(init=False)
	b2_dag : np.ndarray = field(init=False) 
	Hamiltonian_0: np.ndarray = field(init=False) 
	psi0: np.ndarray = field(init=False) 
	
	 
	def __post_init__(self):
		B = np.zeros((self.N_photons,self.N_photons))
		B_dag = np.zeros((self.N_photons,self.N_photons))
		for n in range(self.N_photons-1):
			B[n,n+1] = np.sqrt(n+1)
			B_dag[n+1,n] = np.sqrt(n+1)
			
		self.sigma_plus = np.kron(np.kron(np.array([[0,1],[0,0]]),np.eye(self.N_photons)),np.eye(self.N_photons))
		self.sigma_minus = np.kron(np.kron(np.array([[0,0],[1,0]]),np.eye(self.N_photons)),np.eye(self.N_photons))
		self.b1 = np.kron(np.kron(np.eye(2),B),np.eye(self.N_photons))
		self.b1_dag = np.kron(np.kron(np.eye(2),B_dag),np.eye(self.N_photons))
		self.b2 = np.kron(np.kron(np.eye(2),np.eye(self.N_photons)),B)
		self.b2_dag = np.kron(np.kron(np.eye(2),np.eye(self.N_photons)),B_dag)

		self.Hamiltonian_0 = (self.omega_q-self.driving_freq)*self.sigma_plus@self.sigma_minus 
		self.Hamiltonian_0 += (self.omega_c1-self.driving_freq)*self.b1_dag@self.b1
		self.Hamiltonian_0 +=  (self.omega_c2-self.driving_freq)*self.b2_dag@self.b2
		self.Hamiltonian_0 +=  self.g1*(self.sigma_plus@self.b1 + self.sigma_minus@self.b1_dag) 
		self.Hamiltonian_0 +=  self.g2*(self.sigma_plus@self.b2 + self.sigma_minus@self.b2_dag) 
		self.Hamiltonian_0 += 0.5*self.Rabi_freq * (self.sigma_minus + self.sigma_plus)

		initial_c1 = np.zeros(self.N_photons,dtype=complex)   
		initial_c1[0]=1 # zero photons in the initial state 
		initial_emitter = np.array([1,0],dtype=complex) # one excitation in the initial state  
		self.psi0 = np.kron(initial_emitter,initial_c1)
		self.psi0 = np.kron(self.psi0,initial_c1)



	def evolve(self,times):
		vec = []
		for t in times:
			U = expm(-1j*self.Hamiltonian_0*t)
			psi_t = U @ self.psi0
			vec.append(psi_t)
		vec = np.array(vec)
		pe = np.real(np.einsum('ti,ij,tj ->t ',np.conjugate(vec),self.sigma_plus@self.sigma_minus,vec))
		exc1 =  np.real(np.einsum('ti,ij,tj ->t ',np.conjugate(vec),self.b1_dag@self.b1,vec))
		exc2 =  np.real(np.einsum('ti,ij,tj ->t ',np.conjugate(vec),self.b2_dag@self.b2,vec))
		return pe,exc1,exc2
	