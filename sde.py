# come va implementato di base secondo il paper:
# il treno parte da vcruise e da un determinato punto x0




import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
import time

class DMR(object):
    '''
    This function will simulate speed of a train and relative space between it and a train in front of it.
    '''
    #changed the parameters
#                            
    def __init__(self, b:float, vcruise:float, a: float, vmax:float, x0:float, sigma:float) ->None:
        '''
        - Parameters:
        beta: strength of mean reversion to the mean level vcruise
        vcruise: target velocity on the line
        alfa: strength of distance to the train in front of it
        vmax: maximum speed of the train
        x0: initial position of the train with respect of the train in front of it
        sigma: noise strength
        '''
        self.a=a
        self.b = b
        self.vcruise = vcruise
        self.vmax = vmax
        self.x0 = x0
        self.sigma = sigma
        if(a<=0 or b<=0 or vcruise<=0 or vmax<=0):
            raise RuntimeError("One of the given parameters is 0 or negative")
        
        return

    def __checkInputs(self,T: float,N: int)->None:
        '''
        Given the inputs for a trajectory, this method will
        check if they are correct
        '''
        #T checks:
        if T<0:
            raise RuntimeError("Time interval must have a positive lenght")

        #N checks:
        if N<=1:
            raise RuntimeError("The simulation must have at least two steps")

        return

    def __RK4(self,t_n: float, v_t:float, s_t:float, h: float)->float:
        '''
        Given a point in the trajectory, the time instant and the step lenght,
        this method will compute the variation for y using the RK4 for the 
        deterministic part of the PLS
        '''
        #Perform the check of the inputs. For N we hard code a good 
        #self.__checkInputs(y_n,t_n,10)
        
        if h<=0:
            raise RuntimeError("Given h is negative or 0")

        v = lambda tn,vt,st: self.b*(self.vcruise-vt)+self.a*(self.vcruise*tn-st) ### <- changed the function to the deterministic part of the PLS
        s = lambda tn,vt:   vt*h
        

        Vk_1 = v(t_n,       v_t,          s_t)
        Sk_1 = s(t_n,       v_t)

        Vk_2 = v(t_n+(h/2), v_t+h*Vk_1/2, v_t+h*Sk_1/2)
        Sk_2 = s(t_n+(h/2), v_t+h*Vk_1/2)
        
        Vk_3 = v(t_n+(h/2), v_t+h*Vk_2/2, v_t+h*Sk_2/2)
        Sk_3 = s(t_n+(h/2), v_t+h*Vk_2/2)
        
        Vk_4 = v(t_n+h,     v_t+h*Vk_3,   v_t+h*Sk_3)
        Sk_4 = s(t_n+h,     v_t+h*Vk_3)

        return (h*(Vk_1+2*Vk_2+2*Vk_3+Vk_4)/6, h*(Sk_1+2*Sk_2+2*Sk_3+Sk_4)/6)
        
    def __euler_maruyama(self,tn: float, vt:float, st:float, h: float)->float:
        #ito for
        #h=dt
        dW= np.random.normal()*np.sqrt(h)
        sigmahat=np.sqrt((vt*(self.vmax-vt))/(self.vcruise*(self.vmax-self.vcruise)))*self.sigma
        
        dv = (self.b*(self.vcruise-vt)+ self.a*(self.vcruise*tn-st))*h + sigmahat*dW
        ds = vt*h
        #print(vt," ", h, " ", self.vcruise*tn-st)
        return dv,ds 
    
    def simulateTraj(self,T: float,N: int): 
        '''
        Given the initial population value x0, the considered interval lenght T
        and the number of step in the computation N, this method will return a
        trajectory for the PLS.
        '''
        #Check the inputs
        self.__checkInputs(T,N)

        if T==0:
            return np.array([self.vcruise])
        

        #Setup step lenght and traj array
        h = T/N
        v = np.zeros(N+1,dtype=float)
        v[0] = self.vcruise
        s = np.zeros(N+1,dtype=float)
        s[0] = 0 #distance from the position if speed was constant
        time = np.zeros(N+1,dtype=float)
        time[0] = 0
        cs= np.zeros(N+1,dtype=float)
        cs[0] = 0
        #Setup random generator

        rng = Generator(PCG64())
        for i in range(1,N+1):
            time[i] = i*h
            dv, ds = self.__euler_maruyama(time[i-1],v[i-1],s[i-1],h)
            #dv, ds = self.__RK4(time[i-1],v[i-1],s[i-1],h)
            #dv+=(v[i-1]*(self.vmax-v[i-1]))/(self.vcruise*(self.vmax-self.vcruise))*rng.normal()*np.sqrt(h)
            v[i] = v[i-1] + dv
            s[i] = s[i-1] + ds
            cs[i]= self.vcruise*(time[i])
        return v, s, cs, time
    

omega = [0.0,1.0] 
Nbins = 150 
Nsim = 50
m = (omega[1]-omega[0])/Nbins

system = DMR(0.02,35,0.0005,40,3200,0.1)

rng_0 = Generator(PCG64())

bins = np.zeros(Nbins,dtype=float)

# Crea una figura con due subplot affiancati per velocità e distanza

print("starting")
start_time = time.time()

# Lista delle posizioni di un treno a velocità costante 35 con 500 sample

fig, (ax_vel, ax_spacegap, ax_dist) = plt.subplots(1, 3, figsize=(12, 5))
ax_vel.set_ylim(15, 40)
ax_dist.set_ylim(3000, 3400)

all_gap = []
all_vgap = []
all_acc = []
all_d_gap = []
all_d_vgap = []

for _ in range(Nsim):
    vtraj, straj, cstraj, ttraj = system.simulateTraj(1000,1000)
    ax_vel.plot(ttraj, vtraj, color='gray', linewidth=0.5)
    ax_spacegap.plot(ttraj, cstraj-straj, color='gray', linewidth=0.5)
    ax_dist.plot(ttraj, 3200+cstraj-straj, color='gray', linewidth=0.5)

    gap =  straj - cstraj
    vgap = vtraj - system.vcruise
    acc = np.gradient(vtraj, ttraj[1] - ttraj[0])
    dt = ttraj[1] - ttraj[0]
    d_gap = np.gradient(gap, dt)
    d_vgap = np.gradient(vgap, dt)

    #step = 50  # aumenta per meno frecce
    all_gap.append(gap[:500])
    all_vgap.append(vgap[:500])
    all_acc.append(acc[:500])
    all_d_gap.append(d_gap[:500])
    all_d_vgap.append(d_vgap[:500])
    
gap_q = np.concatenate(all_gap)
vgap_q = np.concatenate(all_vgap)
acc_q = np.concatenate(all_acc)
d_gap_q = np.concatenate(all_d_gap)
d_vgap_q = np.concatenate(all_d_vgap)

end_time = time.time()
print(f"Simulation time: {end_time - start_time:.2f} seconds")

plt.figure(figsize=(8,6))
q = plt.quiver(gap_q, vgap_q, d_gap_q, d_vgap_q, acc_q, cmap='coolwarm', scale=20)
plt.xlabel("space gap wrt constant speed (m)")
plt.ylabel("speed gap wrt constant speed (m/s)")
cbar = plt.colorbar(q, label="acceleration (m/s²)")
plt.title("Phase space vector field (aggregated)")
plt.show()
plt.show()

#bins = bins/(Nsim*m)


