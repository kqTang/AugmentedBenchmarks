import os
import re
f = os.popen('./LKH eil30.par','r')
lines=f.read()
cost = re.findall("Cost.min = (\d+(?:.\d+)?)",lines)
time = re.findall("Time.total = (\d+(?:.\d+)?)",lines)


print(float(cost[0]), float(time[0]))

