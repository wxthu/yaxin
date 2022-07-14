
import os
import matplotlib.pyplot as plt
import numpy as np

def ncu_rep_parser(filename: str) -> None :
    cmd = "ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed --page raw --csv --import "
    cmd += filename
    result = os.popen(cmd)
    res = result.read().splitlines()

    compute_throughput = []
    memory_throughput = []
    kernel_names = []
    for i in range(2, len(res)):
        output = res[i].split(',')
        compute_throughput.append(float(eval(output[-1])))
        memory_throughput.append(float(eval(output[-2])))
        kernel_names.append(output[4])
    
    plt.figure(figsize=(20, 10), dpi=800)
    plt.subplot(2, 1, 1)

    length = len(compute_throughput)
    index = np.arange(4*length)
    sm_index = index[0:4*length:4]
    mem_index = sm_index
    xticks_loc = index[0:4*length:12]
    xticks_lab = np.arange(length)[0:length:3]

    width = 2
    p1 = plt.bar(sm_index, compute_throughput, width, label='Compute Throughtput', color="#87CEFA")
    plt.xlabel('kernel indices', fontdict={'fontsize': 7})
    plt.ylabel('pct_of peak_sustained_elapsed', fontdict={'fontsize': 7})
    plt.xticks(xticks_loc, xticks_lab, fontsize=5)
    plt.yticks(np.arange(0, 100, 10), fontsize=8)
    plt.legend(loc='upper right', fontsize=8)

    plt.subplot(2, 1, 2)
    p2 = plt.bar(mem_index, memory_throughput, width, label='Memory Throughput', color="red")
    plt.xlabel('kernel indices', fontdict={'fontsize':7})
    plt.ylabel('pct_of peak_sustained_elapsed', fontdict={'fontsize':7})
    plt.xticks(xticks_loc, xticks_lab, fontsize=5)
    plt.yticks(np.arange(0, 100, 10), fontsize=8)

    # ax = plt.gca()
    # ax.spines['top'].set_linewidth(2.5)
    # ax.spines['bottom'].set_linewidth(2.5)
    # ax.spines['left'].set_linewidth(2.5)
    # ax.spines['right'].set_linewidth(2.5)    
        
    plt.legend(loc='upper right', fontsize=8)
    plt.savefig('GPU Throughput.jpg')
