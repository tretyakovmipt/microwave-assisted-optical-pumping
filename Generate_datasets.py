# -*- coding: utf-8 -*-
"""

Generated Simulated MAOP signals
"""

#%%Imports
import numpy as np
import matplotlib.pyplot as plt
from qutip import*

#%%Parameters and constants

'''Initiate parameters'''
gOP=1*2*np.pi   #Optical pumping rate
W=0.01*gOP          # coupling strength in terms of the Rabi frequency 
w_L=1*gOP*2e2         #ethalon Larmor frequency
gL=2*w_L          #cavity line width
gT=1.0*gOP          # thermal relaxation rate
delta=np.linspace(-4*w_L,4*w_L,320)# microwave detuning

#%%Initialize quantum system
states=[]# list with the basis states
for i in np.arange(8):
    states.append(basis(8,i))

psi1_1=states[0]#(1,-1) ground state
psi10=states[1]#(1,0) ground state
psi11=states[2]#(1,+1)ground state 
psi2_2=states[3]#(2,-2)ground state 
psi2_1=states[4]#(2,-1)ground state 
psi20=states[5]#(2,0)ground state 
psi21=states[6]#(2,+1)ground state 
psi22=states[7]#(2,+2)ground state 

'''projectors onto the basis states'''
P=[]
for psi in states:
    P.append(ket2dm(psi))



#%%Collasple operators
c_ops=[]#list with collapse operators

'''Optical pumping relaxation'''
for psi1 in states[0:3]:
    for psi2 in states[3:]:
        c_ops.append(np.sqrt(gOP)*psi1*psi2.dag())

'''Thernmal relaxation between the Zeeman states'''
for i in np.arange(8):
    for j in np.arange(8):
        if i != j:
            c_ops.append(np.sqrt(gT)*states[i]*states[j].dag())



#%%Hamiltonian
'''Interaction Hamiltonian'''

S_o=W*np.array([0.5*np.sqrt(3),  0.25*np.sqrt(2), 0.25*np.sqrt(6), 0.25*np.sqrt(6), 0.25*np.sqrt(2),  0.5*np.sqrt(3)])# Rabi frequencies for orthogonal transition
S_l=W*np.array([0.5*np.sqrt(3), 1, 0.5*np.sqrt(3)])# Rabi frequencies for longitudinal transition

#add cavity line-width
def L(f,g):
    if g==0:
        return 1
    return g**2/(g**2+f**2)

def Sl(W_par,W_ort):    
    wL=np.sqrt(W_par**2+W_ort**2)
    if wL==0:
        return S_l*np.sqrt(L(np.array([-2,0.0,2])*wL,gL))   
    return S_l*W_par/wL*np.sqrt(L(np.array([-2,0.0,2])*wL,gL))

def So(W_par,W_ort):
    wL=np.sqrt(W_par**2+W_ort**2)
    if wL==0:
        return 0*S_o*np.sqrt(L(np.array([-3,-1,-1,1,1,3])*wL,gL))
    return S_o*W_ort/wL*np.sqrt(L(np.array([-3,-1,-1,1,1,3])*wL,gL))



"interaction part of the Hamiltonian as a function of Larmor frequency"
def V(W_par,W_ort):
    V_l=0*qeye(len(states))
    V_o=0*qeye(len(states)) 
    
    for i in range(3):
        V_l+=0.5*Sl(W_par,W_ort)[i]*(states[i]*states[i+4].dag())+0.5*Sl(W_par,W_ort)[i]*(states[i]*states[i+4].dag()).dag()#pi transition
        V_o+=0.5*So(W_par,W_ort)[2*i]*(states[i]*states[i+3].dag())+0.5*So(W_par,W_ort)[2*i]*(states[i]*states[i+3].dag()).dag()#sigma-
        V_o+=0.5*So(W_par,W_ort)[2*i+1]*(states[i]*states[i+5].dag())+0.5*So(W_par,W_ort)[2*i+1]*(states[i]*states[i+5].dag()).dag()#sigma+
    return V_o+V_l


'''Diagonal part of the Hamiltonian as a function of detuning d form the clock transition and the Larmor frequency'''
def H_d(d, wL):
    H=0*qeye(len(states))
    H+=(wL+0.5*d)*P[0]
    H+=0.5*d*P[1]
    H+=(-wL+0.5*d)*P[2]
    H+=-(0.5*d+2*wL)*P[3]
    H+=-(0.5*d+wL)*P[4]
    H+=-0.5*d*P[5]
    H+=-(0.5*d-wL)*P[6]    
    H+=-(0.5*d-2*wL)*P[7]    
    return H

'''Total Hamiltonian'''
def H(d,W_par,W_ort):
    wL=np.sqrt(W_par**2+W_ort**2) 
    H=H_d(d,wL)+V(W_par,W_ort)
    return H


#%% Generate data

#function that generates datasets with inputs and corresponding targets
def generate_dataset(N1,N2):
    #N1 - number of scan values
    #N2 - number of random values
    R=np.random.rand(N2)
    W_par=np.concatenate((np.zeros(N1),np.linspace(0,1,N1),R))
    W_ort=np.concatenate((np.linspace(0,1,N1),np.zeros(N1),np.sqrt(1-R**2)))
    targets=np.stack((W_par, W_ort), axis=1)# array with target (label) values of the magnetic field
    Population=[]
    for i in range(len(W_par)):
        ss_1=[]#
        ss0=[]
        ss1=[]
        for d in delta:
            ss=steadystate(H(d,W_par[i]*w_L,W_ort[i]*w_L), c_ops,method='power', drop_tol=1e-10)#steady state
            ss_1.append(expect(P[0],ss))#steady-state populations of psi1
            ss0.append(expect(P[1],ss))#steady-state populations of psi2
            ss1.append(expect(P[2],ss))#steady-state populations of psi3
        ##convert to arrays
        ss_1=np.array(ss_1)
        ss0=np.array(ss0)
        ss1=np.array(ss1)
        rho=ss_1+ss0+ss1 # Total population of in F=1
        Population.append(rho)# add total population to list
        inputs = np.array(Population)
        print(i)
    
    
    return inputs, targets
#%%%Training set
N1=60
N2=120

X_train1,Y_train1 = generate_dataset(N1,N2)

#%%%% Save data
np.save("X_train_C.npy", X_train1)
np.save("Y_train_C.npy", Y_train1)

#%%%%
'''plot results'''

'''Population in F=1 substates'''
fig, ax = plt.subplots()
ax.plot(delta/w_L, X_train1[2])
ax.set_xlabel(r'$\Delta_{\mu}/\omega_{L}$',fontsize=12)
ax.set_ylabel('Signal',fontsize=12)
ax.legend(loc=0)


#%%%Validation set
"""Validation set"""

N1=60# number of scan values points
N2=120#number of ransom values

R=np.random.rand(N2)

W_par=np.concatenate((np.zeros(N1),np.linspace(0,1,N1),R))
W_ort=np.concatenate((np.linspace(0,1,N1),np.zeros(N1),np.sqrt(1-R**2)))

targets=np.stack((W_par, W_ort), axis=1)



#%%%%Find the steady state

Population=[]
for i in range(len(W_par)):
#for w in wL:
    ss_1=[]#
    ss0=[]
    ss1=[]
    for d in delta:
        ss=steadystate(H(d,W_par[i]*w_L,W_ort[i]*w_L), c_ops,method='power', drop_tol=1e-10)#steady state
        ss_1.append(expect(P[0],ss))#steady-state populations of psi1
        ss0.append(expect(P[1],ss))#steady-state populations of psi2
        ss1.append(expect(P[2],ss))#steady-state populations of psi3
    ##convert to arrays
    ss_1=np.array(ss_1)
    ss0=np.array(ss0)
    ss1=np.array(ss1)
    rho=ss_1+ss0+ss1
    Population.append(rho)# total population in F=1
    print(i)

training_set=np.array(Population)

np.save("X_val_C.npy",training_set)
np.save("Y_val_C.npy",targets)

"""Testing set"""

#%%%Testing set
"""Generate testing set"""
N1=5# number of scan values
N2=10#number of random values points


R=np.random.rand(N2)

W_par=np.concatenate((np.zeros(N1),np.linspace(0,1,N1),R))
W_ort=np.concatenate((np.linspace(0,1,N1),np.zeros(N1),np.sqrt(1-R**2)))

targets=np.stack((W_par, W_ort), axis=1)
np.save("Y_test_C.npy",targets)


Population=[]
for i in range(len(W_par)):
#for w in wL:
    ss_1=[]#
    ss0=[]
    ss1=[]
    for d in delta:
        ss=steadystate(H(d,W_par[i]*w_L,W_ort[i]*w_L), c_ops,method='power', drop_tol=1e-10)#steady state
        ss_1.append(expect(P[0],ss))#steady-state populations of psi1
        ss0.append(expect(P[1],ss))#steady-state populations of psi2
        ss1.append(expect(P[2],ss))#steady-state populations of psi3
    ##convert to arrays
    ss_1=np.array(ss_1)
    ss0=np.array(ss0)
    ss1=np.array(ss1)
    rho=ss_1+ss0+ss1
    Population.append(rho)# total population in F=1
    print(i)

test_set=np.array(Population)
np.save("X_test_C.npy",test_set)