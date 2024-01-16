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
    x = data[:,2]
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 15
    plt.legend(loc="upper right")
    plt.subplots_adjust(top=0.98)

    y = np.hstack(
    [  
beta.rvs(0.8642633536799385, 1.5220056442338135, 2.9999999999999996, 40.226883299771465,size=400*20),
beta.rvs(2.7714143761917542, 3.723128095560626, 2.4521713683032433, 39.86288068538075,size=200*50),
 dweibull.rvs(1.3147468576869024, 13.556192615423416, 7.000870677427514,size=200*50), 
 dweibull.rvs(1.3613673370372674, 17.43848604336276, 7.033052957053348,size=200*75), 
 beta.rvs(1.3588688498588615, 3.204528250365446, 0.7675159630755668, 46.303417759071905,size=200*100)
]
    )    

    y=np.clip(y,a_min=1,a_max=41)

    density1 = gaussian_kde(x)
    density2 = gaussian_kde(y)    
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    print("min", x.min(),x.max())
    plt.plot(np.linspace(min(x), max(x), 1000), density1(np.linspace(min(x), max(x), 1000)),color='red', label='HFVRP instances')

    plt.plot(np.linspace(min(y), max(y), 1000), density2(np.linspace(min(y), max(y), 1000)),color='lime',  label='Augmented HFVRP instances')
    plt.hist(x, density=True, bins=np.linspace(min_val, max_val, 15),color='orangered', alpha=0.5,label="HFVRP instances",
              edgecolor="black")
    plt.hist(y, density=True, bins=np.linspace(min_val, max_val, 15),color='dodgerblue', alpha=0.5, label="Augmented HFVRP instances",
              edgecolor="blue")

    plt.grid()  
    plt.legend(loc='center', frameon=False)
    plt.xlabel("Customer demands of instances")
    plt.ylabel("Frequency")
    # plt.show()
    plt.savefig('HFVRP.png',dpi=600)


if __name__ == "__main__":
    main()
