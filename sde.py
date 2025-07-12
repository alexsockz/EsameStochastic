# come va implementato di base secondo il paper:
# il treno parte da vcruise e da un determinato punto x0




import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import multiprocessing as mp

class trainModel(object):
    '''
    This function will simulate speed of a train and relative space between it and a train in front of it.
    '''
    #changed the parameters
#                            
    def __init__(self, b:float, vcruise:float, a: float, vmax:float, x0:float, sigma:float, breaking:float, select_model:str) ->None:
        '''
        - Parameters:
        beta: strength of mean reversion to the mean level vcruise
        vcruise: target velocity on the line
        alfa: strength of distance to the train in front of it
        vmax: maximum speed of the train
        x0: initial position of the train with respect of the train in front of it
        sigma: noise strength
        '''

        self.b = b
        self.vcruise = vcruise
        self.vmax = vmax
        self.x0 = x0
        self.sigma = sigma
        self.breaking= -abs(breaking)
        
        if(a<=0 or b<=0 or vcruise<=0 or vmax<=0):
            raise RuntimeError("One of the given parameters is 0 or negative")
        
        if select_model not in ["CIR", "DMR"]:
            raise RuntimeError("The selected model is not supported, please choose between 'CIR' and 'DMR'")
        elif select_model == "CIR":
            self.a = 0.0
        elif select_model == "DMR":
            self.a = a
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
      
    def __euler_maruyama(self,tn: float, vt:float, st:float, h: float)->float:
        #ito for
        #h=dt
        dW= np.random.normal()*np.sqrt(h)
        sigmahat=np.sqrt((vt*(self.vmax-vt))/(self.vcruise*(self.vmax-self.vcruise)))*self.sigma
        
        #TODO: need to change ito to stratonovich
        dv = (self.b*(self.vcruise-vt)+ self.a*(self.vcruise*tn-st))*h + sigmahat*dW
        ds = vt*h
        #print(vt," ", h, " ", self.vcruise*tn-st)
        return dv,ds 
    
    def __determinsticChange(self, acc: float, vt: float, h: float) -> bool:
        dv = acc * h
        ds = vt * h
        return dv, ds
    
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
        
        tobreak = False
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
        dist= np.zeros(N+1,dtype=float)
        dist[0] = self.x0
        #Setup random generator

        rng = Generator(PCG64())
        for i in range(1,N+1):
            time[i] = i*h
            if tobreak:
                dv, ds = self.__determinsticChange(self.breaking, v[i-1], h)
            else:    
                dv, ds = self.__euler_maruyama(time[i-1],v[i-1],s[i-1],h)
    
            v[i] = v[i-1] + dv
            s[i] = s[i-1] + ds
            cs[i]= self.vcruise*(time[i])
            dist[i]= self.x0+cs[i]-s[i] 

            if dist[i] < 3000:
                tobreak =True
            elif dist[i] >= self.x0:
                tobreak = False
        return v, s, cs, dist, time
    
    def create_plots(self, plot, vtraj, distraj, ttraj):
        fig, (ax_vel, ax_dist) = plot
        ax_vel.set_ylim(15, 40) #da cambiare se si vuole
        ax_dist.set_ylim(3000, 4500)  #da cambiare se si vuole
        ax_vel.plot(ttraj, vtraj, color='gray', linewidth=0.5)
        ax_dist.plot(ttraj, distraj, color='gray', linewidth=0.5)

        return fig, (ax_vel, ax_dist)

omega = [0.0,1.0] 
Nbins = 150 
Nsim = 2000
m = (omega[1]-omega[0])/Nbins
                    #b, vcruise, a, vmax, x0, sigma, breaking, select_model
system = trainModel(0.02,35,0.0005,40,3200,0.1, 0.55, "CIR")

rng_0 = Generator(PCG64())

bins = np.zeros(Nbins,dtype=float)

    # Crea una figura con due subplot affiancati per velocità e distanza

print("starting")
start_time = time.time()

    # Lista delle posizioni di un treno a velocità costante 35 con 500 sample

all_headway = []
all_speed = []

# Create the first figure with 3 subplots for time series plots
fig1 = plt.subplots(1, 2, figsize=(12, 5))
fig2 = plt.figure(figsize=(8, 5))
ax2 = fig2.add_subplot(111)

for i in range(Nsim):
    vtraj, straj, cstraj, distraj, ttraj = system.simulateTraj(1000, 1000)
    if(i%25==0):
        system.create_plots(fig1, vtraj, distraj, ttraj)
    all_speed.append(vtraj)
    all_headway.append(distraj)

headway_q = np.concatenate(all_headway) / 1000
speed_q = np.concatenate(all_speed)

end_time = time.time()
print(f"Simulation time: {end_time - start_time:.2f} seconds")

# Create the second figure for the 2D histogram
h = ax2.hist2d(headway_q, speed_q, bins=50, norm=mpl.colors.LogNorm(), cmap=mpl.cm.Blues)
plt.xlabel("headway (km)")
plt.ylabel("speed follower (m/s)")
ax2.set_xlim(3, 4.5)
ax2.set_ylim(18,42)
cb = plt.colorbar(h[3])
cb.set_label("Bin Counts")
plt.tight_layout()

fig, axs = fig1

fig.savefig("CIR_ito_speed_and_distance.svg",format='svg')
fig2.savefig("CIR_ito_speed_to_distance.svg",format='svg')
plt.show()

# Now, plot contains the first figure with subplots, and fig2 is the histogram figure



#bins = bins/(Nsim*m)



        # all_gap = []
        # all_vgap = []
        # all_acc = []
        # all_d_gap = []
        # all_d_vgap = []



        # gap =  straj - cstraj
        # vgap = vtraj - system.vcruise
        # acc = np.gradient(vtraj, ttraj[1] - ttraj[0])
        # dt = ttraj[1] - ttraj[0]
        # d_gap = np.gradient(gap, dt)
        # d_vgap = np.gradient(vgap, dt)


        # #step = 50  # aumenta per meno frecce
        # all_gap.append(gap[:500])
        # all_vgap.append(vgap[:500])
        # all_acc.append(acc[:500])
        # all_d_gap.append(d_gap[:500])
        # all_d_vgap.append(d_vgap[:500])
            
        # gap_q = np.concatenate(all_gap)
        # vgap_q = np.concatenate(all_vgap)
        # acc_q = np.concatenate(all_acc)
        # d_gap_q = np.concatenate(all_d_gap)
        # d_vgap_q = np.concatenate(all_d_vgap)

        # plt.figure(figsize=(8,6))
        # plt.quiver(gap_q, vgap_q, d_gap_q, d_vgap_q, acc_q, cmap='coolwarm', scale=20)
        # plt.xlabel("space gap wrt constant speed (m)")
        # plt.ylabel("speed gap wrt constant speed (m/s)")
        # plt.colorbar(q, label="acceleration (m/s²)")
        # plt.title("Phase space vector field (aggregated)")
        # plt.show()
