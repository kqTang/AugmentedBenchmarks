from distfit import distfit
import matplotlib.pyplot as plt
from scipy.stats import loggamma
from scipy.stats import dweibull
from scipy.stats import lognorm
from scipy.stats import beta
from scipy.stats import t
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import genextreme
from scipy.stats import uniform
import numpy as np
import scipy.stats as st
import pdb
from scipy.stats import gaussian_kde

def read(filename):
    with open(filename,'r') as f:
        lines = int(f.readlines()[5].split()[1])
    hvrp = np.genfromtxt(filename, dtype=[np.float32, np.float32, np.float32],skip_header=5+lines+2)
    return np.array([list(data) for data in hvrp])


def main():
    data = np.concatenate((
        [read('./vfmpf{:02d}.txt'.format(i)) for i in (list(range(3,7))+list(range(13,21)))]),axis=0)
    x = np.concatenate((data[:,0],data[:,1]),axis=0)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 15
    plt.legend(loc="upper right")

    plt.subplots_adjust(top=0.98)

    y = np.hstack(
    [  
beta.rvs(1.076779254000189, 0.9377605447846655, 4.9163859819341305, 59.08361401806588,size=400*20),
beta.rvs(1.393697724942604, 1.816461177458021, 6.588164480499485, 70.82424026202536,size=200*50),
 uniform.rvs(5,64,size=200*50), 
 beta.rvs(1.312417739939613, 1.5063247304744996, 3.469388425874576, 73.05816995825282,size=200*75), 
 beta.rvs(1.310260789623817, 1.7394175479946559, 1.6739245288181805, 76.89302111091536,size=200*100),
]
    )    

    density1 = gaussian_kde(x)
    density2 = gaussian_kde(y)    
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot(np.linspace(min(x), max(x), 1000), density1(np.linspace(min(x), max(x), 1000)),color='red', label='HFVRP instances')

    plt.plot(np.linspace(min(y), max(y), 1000), density2(np.linspace(min(y), max(y), 1000)),color='lime',  label='Augmented HFVRP instances')
    plt.hist(x, density=True, bins=np.linspace(min_val, max_val, 12),color='orangered', alpha=0.5,label="HFVRP instances",
              edgecolor="black")
    plt.hist(y, density=True, bins=np.linspace(min_val, max_val, 12),color='dodgerblue', alpha=0.5, label="Augmented HFVRP instances",
              edgecolor="blue")


    plt.grid()  
    plt.legend(loc='best', frameon=False)
    plt.xlabel("Customer coordinates of instances")
    plt.ylabel("Frequency")
    plt.subplots_adjust(left=0.15,right=0.95)
    plt.savefig('HFVRP.png',dpi=600)


if __name__ == "__main__":
    main()
