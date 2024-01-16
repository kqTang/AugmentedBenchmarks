import numpy as np
import torch
import scipy.stats as st

def read(filename, capacity, sub_num_samples, size):

    solomon = np.genfromtxt(filename, dtype=[
        np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32])

    x = [x[1] for x in solomon]
    x = torch.from_numpy(np.array(x))

    y = [x[2] for x in solomon]
    y = torch.from_numpy(np.array(y))

    enter_time = [x[4] for x in solomon]
    enter_time = torch.from_numpy(np.array(enter_time))

    leave_time = [x[5] for x in solomon]
    leave_time = torch.from_numpy(np.array(leave_time))

    service_duration = [x[6] for x in solomon]
    service_duration = torch.from_numpy(np.array(service_duration))

    demand = [x[3] for x in solomon]
    demand = torch.from_numpy(np.array(demand))/capacity

    loc = torch.cat((x[1:, None], y[1:, None]), dim=-1)
    depot = torch.cat((x[0:1], y[0:1]), dim=-1)

    low = (loc - depot.unsqueeze(-2)).norm(p=2, dim=-1).floor()

    high = (leave_time[0] - (loc - depot.unsqueeze(-2)
                             ).norm(p=2, dim=-1) - service_duration[1]).floor()
    if filename == "./SolomonBenchmark/C1/Solomon1.txt":
        solomon = np.genfromtxt(
            "./SolomonBenchmark/C1/center_time.txt", dtype=[np.float32])
        center = [x[0] for x in solomon]
        center = torch.from_numpy(np.array(center))[
            None, :].expand(sub_num_samples, -1)
    elif filename == "./SolomonBenchmark/C2/Solomon1.txt":
        solomon = np.genfromtxt(
            "./SolomonBenchmark/C2/center_time.txt", dtype=[np.float32])
        print("./SolomonBenchmark/C2/center_time.txt")
        center = [x[0] for x in solomon]
        center = torch.from_numpy(np.array(center))[
            None, :].expand(sub_num_samples, -1)
    else:
        center = torch.rand(sub_num_samples, size)*((high-low)[None, :]).expand(sub_num_samples, -1) + \
            low[None, :].expand(sub_num_samples, -1)
# pdb.set_trace()
    return x[:size+1], y[:size+1], demand[:size+1], enter_time[:size+1], leave_time[:size+1],\
        center[:, :size], low[:size], high[:size], service_duration[:size+1]


def split_percentage(center,
                     percentage_size,
                     width_half,
                     width_half_percentage,
                     low,
                     high,
                     percentage_select=[25, 50, 75, 100]):
    low = low[None, :].expand_as(center)
    high = high[None, :].expand_as(center)
    size = high.size(-1)

    width_half = width_half.clamp_(min=5)
    width_half_percentage = width_half_percentage.clone().clamp_(min=5)
    if center[percentage_size:].shape[0] > width_half.shape[0]:
        width_half = torch.cat((width_half[0:center[percentage_size:].shape[0]-width_half.shape[0],...],width_half))
    # print("size:", size)

    enter_time_all = (center[percentage_size:] -
                      width_half).clamp_(min=0).floor()
    leave_time_all = (center[percentage_size:]+width_half).floor()

    # 优先保证时间窗宽度，然后再保证时间窗中间值，即当中间值-半宽度<low的时候，保证enter>=low和半宽度不变，使中间值随之变化
    '''
    Prioritize ensuring the width of the time window, and then ensure the middle value of the time window. 
    When the middle value - half width<low, ensure that the middle>=low and half width remain unchanged, 
    so that the middle value changes accordingly
    '''
    enter_time_all = torch.where(
        leave_time_all > high[percentage_size:],
        high[percentage_size:] - (2*width_half),
        enter_time_all
    )
    leave_time_all = torch.where(
        enter_time_all < low[percentage_size:],
        low[percentage_size:] + (2*width_half),
        leave_time_all
    )
    enter_time_all = torch.where(
        enter_time_all > low[percentage_size:], enter_time_all, low[percentage_size:])
    leave_time_all = torch.where(
        leave_time_all < high[percentage_size:], leave_time_all, high[percentage_size:])

    enter_time_percentage = (
        center[:percentage_size]-width_half_percentage).clamp_(min=0).floor()
    leave_time_percentage = (
        center[:percentage_size]+width_half_percentage).floor()

    enter_time_percentage = torch.where(
        leave_time_percentage > high[:percentage_size],
        high[:percentage_size] - (2*width_half_percentage),
        enter_time_percentage
    )
    leave_time_percentage = torch.where(
        enter_time_percentage < low[:percentage_size],
        low[:percentage_size] + (2*width_half_percentage),
        leave_time_percentage
    )
    enter_time_percentage = torch.where(
        enter_time_percentage > low[:percentage_size], enter_time_percentage, low[:percentage_size])
    leave_time_percentage = torch.where(
        leave_time_percentage < high[:percentage_size], leave_time_percentage, high[:percentage_size])

    percentage = torch.cat(
        [enter_time_percentage[:, None, :], leave_time_percentage[:, None, :]], dim=1)
    enter_time_low = torch.zeros_like(enter_time_percentage)
    # enter_time_low = low[:percentage_size]
    leave_time_high = high[:percentage_size]
    low_high = torch.cat(
        [enter_time_low[:, None, :], leave_time_high[:, None, :]], dim=1)

    # The next 4 lines generate percentage
    per_size = torch.FloatTensor(np.array([np.random.choice(
        percentage_select, percentage_size)/100.])).squeeze(0)[:, None, None].expand(-1, 2, size)
    rand = torch.rand(percentage[:, 0:1, :].size()).expand(-1, 2, -1)
    percentage = torch.where(rand < per_size, percentage, low_high)

    # perceBool is a boolean array show whether a customer is in the time window constraint. If in, then = 1, otherwise = 0
    perceBool = torch.where(rand < per_size, torch.ones_like(
        percentage), torch.zeros_like(percentage))
    assert (perceBool[:, 0, :] == perceBool[:, 1, :]
            ).all(), 'perceBool out of limit'
    perceBool = torch.cat(
        [perceBool[:, 0, :], torch.ones_like(enter_time_all)], dim=0)

    enter_time_percentage = percentage[:, 0, :].floor()
    leave_time_percentage = percentage[:, 1, :].floor()

    # cat percentage with others
    enter_time_all = torch.cat((enter_time_percentage, enter_time_all), dim=0)
    leave_time_all = torch.cat((leave_time_percentage, leave_time_all), dim=0)
    # pdb.set_trace()
    # assert (enter_time_all >= low).all(), 'enter_time out of limit'
    assert (enter_time_all < high).all(), 'enter_time out of limit'
    assert (leave_time_all > enter_time_all).all() & (
        leave_time_all <= high).all(), 'leave_time out of limit'
    return enter_time_all, leave_time_all, perceBool


def generate_solomon_train_data(train_type, num_samples, size):
    print(train_type)
    if train_type == "c1":
        print("generate similar data of cluster_1")
        capacity = 200.
        x, y, demand, enter_time, leave_time, center, low, high, service_duration = \
            read("./SolomonBenchmark/C1/Solomon1.txt", capacity, num_samples,
                    size)

        # 3/8 beta
        percentage_size = int(3*num_samples/8)
        a, b, loc, scale = 4.06, 5.95, 16.05, 35.34
        width_half_percentage = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                                size=[percentage_size, size])).to(torch.float)
        # 1/8 beta
        a, b, loc, scale = 3.66, 5.33, 33.51, 67.03
        width_half_1 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                    size=[int(num_samples/8), size])).to(torch.float)
        # 1/8 gamma
        a, loc, scale = 1.52, 12.49, 43.03
        width_half_2 = torch.from_numpy(st.gamma.rvs(a=a, loc=loc, scale=scale,
                                                        size=[int(num_samples/8), size])).to(torch.float)
        # 1/8 beta
        a, b, loc, scale = 3.73, 5.23, 66.20, 133.38
        width_half_3 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                    size=[int(num_samples/8), size])).to(torch.float)

        # 2/8 90,180
        width_half_4 = torch.FloatTensor(np.array([np.random.choice(
            [90, 180], int(num_samples/4))])).squeeze(0)[:, None].expand(-1, size)

        # pdb.set_trace()
        width_half = torch.cat(
            (width_half_1, width_half_2, width_half_3, width_half_4), dim=0)
        percentage_select = [25, 50, 75, 100]
        enter_time_all, leave_time_all, perceBool = split_percentage(
            center, percentage_size, width_half, width_half_percentage, low, high, percentage_select)

    elif train_type == "c2":
        print("generate similar data of Solomon_cluster_2")
        capacity = 700.
        x, y, demand, enter_time, leave_time, center, low, high, service_duration = \
            read("./SolomonBenchmark/C2/Solomon1.txt", capacity, num_samples,
                    size)
        # 3/8 constant width 80, but 25-100%
        percentage_size = int(num_samples/2)
        width_half_percentage = torch.FloatTensor(
            [80])[None, :].expand(percentage_size, size)

        # 3/8 constant width 80 & 160 & 320
        width_half_1 = torch.FloatTensor(np.array([np.random.choice(
            [160, 320], int(num_samples/4))])).squeeze(0)[:, None].expand(-1, size)
        # 1/8 Beta
        a, b, loc, scale = 3.67, 5.20, 133.29, 266.06
        width_half_2 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                    size=[int(num_samples/8), size])).to(torch.float)
        # 1/8 Beta
        a, b, loc, scale = 0.86, 1.41, 88.50, 547.94
        width_half_3 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                    size=[int(num_samples/8), size])).to(torch.float)

        width_half = torch.cat(
            (width_half_1, width_half_2, width_half_3), dim=0)

        enter_time_all, leave_time_all, perceBool = split_percentage(
            center, percentage_size, width_half, width_half_percentage, low, high)

    elif train_type == "r1":
        print("generate similar data of Solomon_random_1")
        capacity = 200.
        x, y, demand, enter_time, leave_time, center, low, high, service_duration = \
            read("./SolomonBenchmark/R1/Solomon1.txt", capacity, num_samples,
                    size)
        # 1/2 5,15
        percentage_size = int(num_samples/2)
        width_half_percentage = torch.FloatTensor(np.array([np.random.choice(
            [5, 15], percentage_size)])).squeeze(0)[:, None].expand(-1, size)
        # 1/8 genextreme
        c, loc, scale = 0.23, 27.77, 4.35
        width_half_1 = torch.from_numpy(st.genextreme.rvs(c=c, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)
        # 1/8 beta
        a, b, loc, scale = 1.23, 1.82, 11.32, 79.54
        width_half_2 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                    size=[int(num_samples/8), size])).to(torch.float)
        # 1/8 beta
        a, b, loc, scale = 0.77, 1.25, 9.50, 88.05
        width_half_3 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                    size=[int(num_samples/8), size])).to(torch.float)
        # 1/8 genextreme
        c, loc, scale = 0.24, 55.60, 8.57
        width_half_4 = torch.from_numpy(st.genextreme.rvs(c=c, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)
        width_half = torch.cat((
            width_half_1, width_half_2, width_half_3, width_half_4), dim=0)

        enter_time_all, leave_time_all, perceBool = split_percentage(
            center, percentage_size, width_half, width_half_percentage, low, high)

    elif train_type == "r2":
        print("generate similar data of Solomon_random_2")
        capacity = 1000.
        x, y, demand, enter_time, leave_time, center, low, high, service_duration = \
            read("./SolomonBenchmark/R2/Solomon1.txt", capacity, num_samples,
                    size)

        # 3/8 genextreme
        c, loc, scale = 0.22, 51.24, 17.33
        width_half_percentage_1 = torch.from_numpy(st.genextreme.rvs(c=c, loc=loc, scale=scale,
                                                                        size=[int(3*num_samples/8), size])).to(torch.float)

        # 1/4 constant width 120, 25%-100%
        width_half_percentage_2 = torch.FloatTensor(
            [120])[None, :].expand(int(num_samples/4), size)

        # 1/8 beta
        a, b, loc, scale = 1.30, 2.27, 44.52, 359.15
        width_half_1 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                    size=[int(num_samples/8), size])).to(torch.float)
        # 1/8 beta
        a, b, loc, scale = 0.90, 1.76, 36.50, 457.83
        width_half_2 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                    size=[int(num_samples/8), size])).to(torch.float)
        # 1/8 genextreme
        c, loc, scale = 0.22, 222.49, 34.74
        width_half_3 = torch.from_numpy(st.genextreme.rvs(c=c, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)

        width_half = torch.cat(
            (width_half_1, width_half_2, width_half_3), dim=0)

        width_half_percentage = torch.cat(
            (width_half_percentage_1, width_half_percentage_2), dim=0)
        percentage_size = int(3*num_samples/8) + int(num_samples/4)

        enter_time_all, leave_time_all, perceBool = split_percentage(
            center, percentage_size, width_half, width_half_percentage, low, high)

    elif train_type == "rc1":
        print("generate similar data of Solomon_random_cluster_1")
        capacity = 200.
        x, y, demand, enter_time, leave_time, center, low, high, service_duration = \
            read("./SolomonBenchmark/RC1/Solomon1.txt", capacity, num_samples,
                    size)
        # 1/2 constant width 15, but 25-100%
        percentage_size = int(num_samples/2)
        width_half_percentage = torch.FloatTensor(
            [15])[None, :].expand(percentage_size, size)

        # 1/8 mix beta, constant 5, and constant 60
        a, b, loc, scale = 1.94, 87.21, 8.89, 663.77
        width_half_1_1 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                        size=[int(num_samples/16), size])).to(torch.float)
        width_half_1_2 = torch.FloatTensor(
            [5])[None, :].expand(int(num_samples/32), size)
        width_half_1_3 = torch.FloatTensor(
            [60])[None, :].expand(int(num_samples/32), size)
        width_half_1 = torch.cat(
            (width_half_1_1, width_half_1_2, width_half_1_3), dim=0)  # cat all 1 to 3
        np.random.shuffle(width_half_1.view(-1).numpy())

        # 1/8 constant width 30
        width_half_2 = torch.FloatTensor(
            [30])[None, :].expand(int(num_samples/8), size)

        # 1/8 mix beta and beta
        a, b, loc, scale = 2.88, 8.24, 19.28, 40.81
        width_half_3_1 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                        size=[int(num_samples/16), size])).to(torch.float)
        a, b, loc, scale = 12.26, 10.26, 16.42, 78.39
        width_half_3_2 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                        size=[int(num_samples/16), size])).to(torch.float)
        width_half_3 = torch.cat(
            (width_half_3_1, width_half_3_2), dim=0)
        np.random.shuffle(width_half_3.view(-1).numpy())

        # 1/8 beta
        a, b, loc, scale = 9.90, 5.49, -27.18, 129.57
        width_half_4 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                    size=[int(num_samples/8), size])).to(torch.float)

        width_half = torch.cat(
            (width_half_1, width_half_2, width_half_3, width_half_4), dim=0)

        enter_time_all, leave_time_all, perceBool = split_percentage(
            center, percentage_size, width_half, width_half_percentage, low, high)

    elif train_type == "rc2":
        print("generate similar data of Solomon_random_cluster_2")
        capacity = 1000.
        x, y, demand, enter_time, leave_time, center, low, high, service_duration = \
            read("./SolomonBenchmark/RC2/Solomon1.txt", capacity, num_samples,
                    size)
        # 1/2 constant width 60, 25%-100%
        percentage_size = int(num_samples/2)
        width_half_percentage = torch.FloatTensor(
            [60])[None, :].expand(percentage_size, size)

        # 1/8 mix 30, 240 and dweibull
        width_half_1_1 = torch.FloatTensor(np.array([np.random.choice(
            [30, 240], int(num_samples/16))])).squeeze(0)[:, None].expand(-1, size)
        c, loc, scale = 2.05, 92.65, 31.63
        width_half_1_2 = torch.from_numpy(st.dweibull.rvs(c=c, loc=loc, scale=scale,
                                                            size=[int(num_samples/16), size])).to(torch.float)
        width_half_1 = torch.cat(
            (width_half_1_1, width_half_1_2), dim=0)
        np.random.shuffle(width_half_1.view(-1).numpy())

        # 1/8
        width_half_2 = torch.FloatTensor(
            [120])[None, :].expand(int(num_samples/8), size)

        # 1/8 beta
        a, b, loc, scale = 1.30, 2.27, 44.52, 359.15
        width_half_3 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                    size=[int(num_samples/8), size])).to(torch.float)

        # 1/8 genextreme
        c, loc, scale = 0.22, 222.48, 34.73
        width_half_4 = torch.from_numpy(st.genextreme.rvs(c=c, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)

        width_half = torch.cat(
            (width_half_1, width_half_2, width_half_3, width_half_4), dim=0)

        enter_time_all, leave_time_all, perceBool = split_percentage(
            center, percentage_size, width_half, width_half_percentage, low, high)
    x = x[None, :].expand(num_samples,-1)
    y = y[None, :].expand(num_samples,-1)
    d = demand[None, :].expand(num_samples,-1)
    e = torch.cat((enter_time[None, 0:1].expand(num_samples,-1),enter_time_all),dim=-1)
    l = torch.cat((leave_time[None, 0:1].expand(num_samples,-1),leave_time_all),dim=-1)
    dur = service_duration[None, :].expand(num_samples,-1)
    return x, y, d, e, l, dur
