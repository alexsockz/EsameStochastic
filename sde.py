# come va implementato di base secondo il paper:
# il treno parte da vcruise e da un determinato punto x0




import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import perf_counter
import time
import multiprocessing as mp
import warnings
from itertools import chain
from numba import njit

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

        self.stratonovich_constant=0.5*(self.sigma*self.sigma)/(4*(self.vcruise*(self.vmax-self.vcruise)))
        self.sigmahat_constant=(self.sigma*self.sigma)/(self.vcruise*(self.vmax-self.vcruise))
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
    
    @staticmethod
    @njit 
    def __euler_maruyama(a:float,  b: float, stratonovich_constant:float, sigmahat_constant:float, vcruise:float, vmax:float, vt:float, st:float, dv_prec:float, sPrec:float, h: float)->float:
        #ito for
        #h=dt

        #PROBLEMA se la distanza tra sPrec e st è troppo elevata sovrasta la tendenza alla media e quindi devo limitare a,
        #SOLUZIONE min(vt, 40) se 40 allora a=0
        #ALTERNATIVA PIU FICA, LIMITARE sPrec-st con una sigmoide
        
        #stochastic part
        sigmahat=np.sqrt((vt*(vmax-vt))*sigmahat_constant)
        dW= np.random.normal()*np.sqrt(h)
        
        #the limiter must be smooth and be o in vmax
        #limiter=1 / (1 + np.exp(-self.vmax+vt))vt/self.vmax
        limiter=1-np.power(vt/vmax,vcruise)
        a=(b*(vcruise-vt)+ a*(sPrec-st)*limiter)

        
        # ito

        # dv = a*h + sigmahat*dW
        # ds = vt*h
        
        #stratonovich
        #b=(1/2)f'(x)f(x)
        b=stratonovich_constant*dv_prec*(vmax-(2*vt))
        dv=(a+b)*h+sigmahat*dW
        ds=vt*h
        
        #print(vt," ", h, " ", self.vcruise*tn-st)
        return dv,ds 
    
    def __determinsticChange(self, acc: float, vt: float, h: float) -> bool:
        dv = acc * h
        ds = vt * h
        return dv, ds
    
    def simulateTraj(self, period: tuple[float, int]): 
        '''
        Given the initial populatibreakedAton value x0, the considered interval lenght T
        and the number of step in the computation N, this method will return a
        trajectory for the PLS.
        '''
        T,N=period
        #Check the inputs
        self.__checkInputs(T,N)

        if T==0:
            return np.array([self.vcruise])
        
        tobreak = np.full(self.Ntrains+1, False)
        #Setup step lenght and traj array
        h = T/N
        v = np.zeros((self.Ntrains+1,N+1),dtype=float)
        v[:,0].fill(self.vcruise)

        s = np.zeros((self.Ntrains+1,N+1),dtype=float)

        #one for everyone
        time = np.zeros(N+1,dtype=float)
        time[0] = 0

        dist= np.zeros((self.Ntrains+1,N+1),dtype=float)
        dist[:,0].fill(self.x0)               #TODO: in the future maybe transform x0 in an array with a specific distance for every train
        dist[0,0]=0
        
        dv=np.zeros(self.Ntrains+1,dtype=float)
        ds=np.zeros(self.Ntrains+1,dtype=float)

        breakedAt=np.empty((self.Ntrains+1,), dtype=object)
        breakedAt.fill([])
        
        #Setup random generator
        rng = Generator(PCG64())
        for i in range(1,N+1):
            time[i] = i*h
            v[0,i]= self.vcruise
            s[0,i]= self.vcruise*(time[i])
            
            for train in range(1, self.Ntrains+1):
                if tobreak[train]:
                    dv[train], ds[train] = self.__determinsticChange(self.breaking, v[train,i-1], h)
                else:    
                    dv[train], ds[train] = self.__euler_maruyama(
                        self.a,
                        self.b,
                        self.stratonovich_constant,
                        self.sigmahat_constant,
                        self.vcruise,
                        self.vmax,
                        v[train, i-1],
                        s[train, i-1],
                        dv[train],
                        s[train-1,i-1],
                        h
                        )
                v[train, i] = v[train, i-1] + dv[train]
                if v[train, i]<=0:
                    v[train, i]=0
                    dv[train]=-v[train, i-1]
                s[train, i] = s[train, i-1] + ds[train]
                dist[train, i]= self.x0+s[train-1,i]-s[train,i] 
                
                if dist[train, i] < self.dmin and train!=1:
                    if tobreak[train] == False:
                        breakedAt[train].append(i)
                    tobreak[train] = True
                elif dist[train, i] >= self.x0:
                    tobreak[train] = False

        return v, s, dist, time, breakedAt
    
    @staticmethod
    def create_plots(plot, vtraj, distraj, ttraj):
        fig, (ax_vel, ax_dist) = plot
        #ax_vel.set_ylim(10, 40) #da cambiare se si vuole
        #ax_dist.set_ylim(3000, 4500)  #da cambiare se si vuole
        ax_vel.plot(ttraj, vtraj, color='gray', linewidth=0.5)
        ax_dist.plot(ttraj, distraj, color='gray', linewidth=0.5)

        return fig, (ax_vel, ax_dist)
    
def pool_wrapper(period, args):
    system = trainModel(*args)
    res= system.simulateTraj(period)
    return res
#---------------------------------------------------------------------------------------------------------------    
if __name__ == '__main__':
    pool=mp.Pool()
    start=perf_counter()
    for trainToFollow in range(1,10):
        for s in ["DMR","CIR"]:
            a=perf_counter()
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

            args= (trainToFollow,0.02,35,0.0005,40,3200,3000, 0.1, 0.55, s)
            
            b=perf_counter()
            print("assegnazione ", b-a)

            period =(5000,5000)
            distr_of_breaks=np.zeros(period[1]+1)
            try:
                jobs_args=[(period,args)] * Nsim
                chunksize= int(np.ceil(len(jobs_args)/mp.cpu_count()))
                print("inizio parallel")
                results_handler=pool.starmap_async(pool_wrapper,jobs_args,chunksize)
                print("continuo parallel")
                #vtraj, straj, distraj, ttraj, breakedAt =zip(*results_handler.get())
                results= results_handler.get()
                print("fine parallel")
            except Exception as e:
                pool.terminate()
                raise e
            c=perf_counter()
            print("esecuzione ", c-b, " total ",c-a)

            i=0
            for vtraj, straj, distraj, ttraj, breakedAt in results:
                i+=1
                if(breakedAt[trainToFollow]!=[]):
                    distr_of_breaks[breakedAt[trainToFollow][0]]+=1
            # for point in breakedAt[trainToFollow]:
            #     #print(point)
            #     distr_of_breaks[point]+=1
                if(i%4==0):
                    trainModel.create_plots(fig1, vtraj[trainToFollow], distraj[trainToFollow], ttraj)
                all_speed.append(vtraj[trainToFollow])
                all_headway.append(distraj[trainToFollow])

            # for i in range(Nsim):
                
            #     vtraj, straj, distraj, ttraj, breakedAt = system.simulateTraj(period)

            #     if(breakedAt[trainToFollow]!=[]):
            #         distr_of_breaks[breakedAt[trainToFollow][0]]+=1
            #     # for point in breakedAt[trainToFollow]:
            #     #     #print(point)
            #     #     distr_of_breaks[point]+=1

            #     if(i%14==0):
            #         system.create_plots(fig1, vtraj[trainToFollow], distraj[trainToFollow], ttraj)
            #     all_speed.append(vtraj[trainToFollow])
            #     all_headway.append(distraj[trainToFollow])
            headway_q=np.concatenate(all_headway)/1000
            speed_q=np.concatenate(all_speed)
            # headway_q =np.stack([arr[trainToFollow] for arr in distraj]).flatten()
            # speed_q = np.stack([arr[trainToFollow] for arr in vtraj]).flatten()
            # print(type(headway_q), np.shape(headway_q))
            # time_q=np.fromiter(chain.from_iterable(ttraj),float)
            # trainModel.create_plots(fig1,speed_q,headway_q,time_q)

            d=perf_counter()
            print("assegnazione risultati ",d-c,"total ",d-a)

            # Create the second figure for the 2D histogram
            h = ax2.hist2d(headway_q, speed_q, bins=50, norm=mpl.colors.LogNorm(), cmap=mpl.cm.Blues)
            ax2.set_xlabel("headway (km)")
            ax2.set_ylabel("speed follower (m/s)")
            #ax2.set_xlim(3, 4.5)
            #ax2.set_ylim(10,42)
            cb = fig2.colorbar(h[3],ax=ax2)
            cb.set_label("Bin Counts")

            # Bin brake events into groups of 10 time steps
            bin_size = 50
            binned_breaks = np.add.reduceat(distr_of_breaks, np.arange(0, period[1]+1, bin_size))
            binned_breaks_perc=np.nan_to_num(binned_breaks/np.sum(binned_breaks),nan=0, posinf=0,neginf=0)
            print(np.sum(binned_breaks))
            if(np.sum(binned_breaks)!=0):    
                bar=ax3.bar(np.arange(len(binned_breaks_perc)), binned_breaks_perc, color='gray', align='edge')
                ax3.set_xlabel("Time step of brake event (binned every 10 steps)")
                ax3.set_ylabel("Number of brake events, total "+ str(np.sum(binned_breaks)))
                ax3.set_title("Histogram of Brake Events Over Time (10-step bins)")
            #plt.xlim(0, period[1])

            plt.tight_layout()

            filedata= '_'.join([str(s) for s in args])
            fig1[0].savefig("plots_fixed_model/"+filedata+"_5000_with_limiter_both_sides_stratonovich_speed_and_distance.svg",format='svg')
            fig2.savefig("plots_fixed_model/"+filedata+"_5000_with_limiter_both_sides_tratonovich_speed_to_distance.svg",format='svg')
            if(np.sum(binned_breaks)!=0):
                fig3.savefig("plots_fixed_model/"+filedata+"_5000_with_limiter_both_sides_tratonovich_pdf_first_break.svg",format='svg')
            e=perf_counter()
            print("plotting ",e-d,"total ",e-a)
            print("total till start ", e-start)
        #plt.show()
    pool.close()
    pool.join()
