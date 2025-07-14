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
    def __init__(self, Ntrains:float, b:float, vcruise:float, a: float, vmax:float, x0:float, dmin:float, sigma:float, breaking:float, select_model:str) ->None:
        '''
        - Parameters:
        beta: strength of mean reversion to the mean level vcruise
        vcruise: target velocity on the line
        alfa: strength of distance to the train in front of it
        vmax: maximum speed of the train
        x0: initial position of the train with respect of the train in front of it
        sigma: noise strength
        '''
        self.dmin=dmin
        self.Ntrains=Ntrains
        self.b = b
        self.vcruise = vcruise
        self.vmax = vmax
        self.x0 = x0
        self.sigma = sigma
        self.breaking= -abs(breaking)
        
        if(a<=0 or b<=0 or vcruise<=0 or vmax<=0):
            raise RuntimeError("One of the given parameters is 0 or negative")
        if(dmin>x0):
            raise RuntimeError("the minimum distance must be less then the starting distance")
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
      
    def __euler_maruyama(self,tn: float, vt:float, st:float, dv_prec:float, h: float)->float:
        #ito for
        #h=dt
        dW= np.random.normal()*np.sqrt(h)
        sigmahat=np.sqrt((vt*(self.vmax-vt))/(self.vcruise*(self.vmax-self.vcruise)))*self.sigma
        
        dsigmahat=(self.sigma*dv_prec*(self.vmax-(2*vt)))/np.sqrt(4*self.vcruise*(self.vmax-self.vcruise)*vt*(self.vmax-vt))
        
        #TODO: need to change ito to stratonovich
        
        # ito
        # dv = (self.b*(self.vcruise-vt)+ self.a*(self.vcruise*tn-st))*h + sigmahat*dW
        # ds = vt*h
        
        a=(self.b*(self.vcruise-vt)+ self.a*(self.vcruise*tn-st))
        
        dv=(a+0.5*sigmahat*dsigmahat)*h+sigmahat*dW
        ds=vt*h
        
        #print(vt," ", h, " ", self.vcruise*tn-st)
        return dv,ds 
    
    def __determinsticChange(self, acc: float, vt: float, h: float) -> bool:
        dv = acc * h
        ds = vt * h
        return dv, ds
    
    def simulateTraj(self,T: float,N: int): 
        '''
        Given the initial populatibreakedAton value x0, the considered interval lenght T
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
        dv=0
        ds=0
        breakedAt=[]

        #Setup random generator
        rng = Generator(PCG64())
        for i in range(1,N+1):
            time[i] = i*h
            if tobreak:
                dv, ds = self.__determ3insticChange(self.breaking, v[i-1], h)
            else:    
                dv, ds = self.__euler_maruyama(time[i-1],v[i-1],s[i-1], dv, h)
    
            v[i] = v[i-1] + dv
            s[i] = s[i-1] + ds
            cs[i]= self.vcruise*(time[i])
            dist[i]= self.x0+cs[i]-s[i] 

            if dist[i] < self.dmin:
                if tobreak==False:
                    breakedAt.append(i)
                tobreak =True
            elif dist[i] >= self.x0:
                tobreak = False
        return v, s, cs, dist, time, breakedAt
    
    def create_plots(self, plot, vtraj, distraj, ttraj):
        fig, (ax_vel, ax_dist) = plot
        ax_vel.set_ylim(15, 40) #da cambiare se si vuole
        ax_dist.set_ylim(3000, 4500)  #da cambiare se si vuole
        ax_vel.plot(ttraj, vtraj, color='gray', linewidth=0.5)
        ax_dist.plot(ttraj, distraj, color='gray', linewidth=0.5)

        return fig, (ax_vel, ax_dist)

#---------------------------------------------------------------------------------------------------------------    
for s in ["DMR","CIR"]:
    all_headway = []
    all_speed = []

    # Create the first figure with 2 subplots for time series plots
    fig1 = plt.subplots(1, 2, figsize=(12, 5))
    
    #relationship between speed and distance from front train
    fig2 = plt.figure(figsize=(8, 5))
    ax2 = fig2.add_subplot(111)
    
    #pdf of first break
    fig3 = plt.figure(figsize=(8, 5))
    ax3 = fig3.add_subplot(111)
    
    Nsim = 500
    #Ntreni, b, vcruise, a, vmax, x0, min dist, sigma, breaking, select_model
    args= (1,0.02,35,0.0005,40,3200,3000, 0.1, 0.55, s)
    period=(5000, 5000)
    
    system = trainModel(*args)

    print("starting")
    start_time = time.time()

            #T    #N
    distr_of_breaks=np.zeros(period[1]+1)

    for i in range(Nsim):
        vtraj, straj, cstraj, distraj, ttraj, breakedAt = system.simulateTraj(*period)
        for point in breakedAt:
            distr_of_breaks[point]+=1
            break
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
    ax2.set_xlabel("headway (km)")
    ax2.set_ylabel("speed follower (m/s)")
    ax2.set_xlim(3, 4.5)
    ax2.set_ylim(18,42)
    cb = fig2.colorbar(h[3],ax=ax2)
    cb.set_label("Bin Counts")

    # Bin brake events into groups of 10 time steps
    bin_size = 50
    binned_breaks = np.add.reduceat(distr_of_breaks, np.arange(0, period[1]+1, bin_size))
    binned_breaks=binned_breaks/np.sum(binned_breaks)


    bar=ax3.bar(np.arange(len(binned_breaks)), binned_breaks, color='gray', align='edge')
    ax3.set_xlabel("Time step of brake event (binned every 10 steps)")
    ax3.set_ylabel("Number of brake events")
    ax3.set_title("Histogram of Brake Events Over Time (10-step bins)")
    #plt.xlim(0, period[1])

    plt.tight_layout()

    filedata= '_'.join([str(s) for s in args])
    fig1[0].savefig("plots/"+filedata+"_5000_stratonovich_speed_and_distance.svg",format='svg')
    fig2.savefig("plots/"+filedata+"_5000_stratonovich_speed_to_distance.svg",format='svg')
    fig3.savefig("plots/"+filedata+"_5000_stratonovich_pdf_first_break.svg",format='svg')
