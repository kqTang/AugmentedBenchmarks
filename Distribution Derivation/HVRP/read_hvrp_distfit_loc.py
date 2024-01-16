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

    data = read('./vfmpf{:02d}.txt'.format(19)) # Can be 3, 13, 15, 17, 19
    x_y = np.concatenate((data[:,0],data[:,1]),axis=0)
    dist = distfit()
    fig, ax = plt.subplots(1, 1)
    dist.fit_transform(x_y)
    print(dist.model)
    dist.plot(ax=ax)
    
    plt.show()

# 3-6：beta(1.076779254000189, 0.9377605447846655, 4.9163859819341305, 59.08361401806588)
# 13-14：beta(1.393697724942604, 1.816461177458021, 6.588164480499485, 70.82424026202536)
# 15-16： uniform(5,64) 
# 17-18： beta(1.312417739939613, 1.5063247304744996, 3.469388425874576, 73.05816995825282) 
# 19-20： beta(1.310260789623817, 1.7394175479946559, 1.6739245288181805, 76.89302111091536)
if __name__ == "__main__":
    main()
