from distfit import distfit
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import pdb

def read(filename):
    solomon = np.genfromtxt(filename, dtype=[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32],skip_header=9)
    enter_time = np.array([x[4] for x in solomon])
    leave_time = np.array([x[5] for x in solomon])
    return (leave_time[1:]-enter_time[1:])/2



def main():
    with open('./Solomon/output.txt','w') as output:
        
        # for j in [5]:
        # for j in [1]:
                
        filename='./Solomon/C1{:02}.txt'.format(1)# 1,2,3,4,5,6,7,8,9,10,11 varying in different series
        
        
        x = read(filename)
        # X.append(x)
        # x=np.hstack(X)
        
        # pdb.set_trace()
        # x = np.array(x)
        dist = distfit(todf=True)
        # dist = distfit(distr='full')
        # dist = distfit(distr='norm')

        fig, ax = plt.subplots(1, 1)
        dist.fit_transform(x)
        # dist.predict(y)
        output.write(filename+'\n')
        
        output.write(dist.model['name']+'(')
        for params_i in dist.model['params']:
            output.write('{:.5g}'.format(params_i) +',') 
        output.write('); [{:.2f},{:.2f}]'.format(dist.model['CII_min_alpha'],dist.model['CII_max_alpha'])) 
        output.write('\n') 
        pdb.set_trace()
        # print(dist.model)
        dist.plot(ax=ax)
        plt.show()


if __name__ == "__main__":
    main()
