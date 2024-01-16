from distfit import distfit
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import pdb

def read(filename):
    with open(filename,'r') as f:
        lines = int(f.readlines()[5].split()[1])
    hvrp = np.genfromtxt(filename, dtype=[np.float32, np.float32, np.float32],skip_header=5+lines+2)
    return np.array([list(data) for data in hvrp])


def main():

    data = read('./vfmpf{:02d}.txt'.format(19))# Can be 3, 13, 15, 17, 19

    x_y = np.concatenate((data[:,0],data[:,1]),axis=0)
    demands = data[:,2]
    
    dist = distfit()
    fig, ax = plt.subplots(1, 1)
    dist.fit_transform(demands)
    print(dist.model)
    dist.plot(ax=ax)
    
    plt.show()

# 3-6：beta(0.8642633536799385, 1.5220056442338135, 2.9999999999999996, 40.226883299771465)
# 13-14：beta(2.7714143761917542, 3.723128095560626, 2.4521713683032433, 39.86288068538075)
# 15-16： dweibull(1.3147468576869024, 13.556192615423416, 7.000870677427514) 
# 17-18： dweibull(1.3613673370372674, 17.43848604336276, 7.033052957053348) 
# 19-20： beta(1.3588688498588615, 3.204528250365446, 0.7675159630755668, 46.303417759071905)
if __name__ == "__main__":
    main()
