
'''
This file is aimed at collecting some samples
to train a model for predicting the latency
'''

import csv
import multiprocessing as mp

from multiprocessing import Barrier, Process, Queue
import random
import pickle

import torch
from torchvision.models.resnet import resnet152, resnet101, resnet18, resnet34, resnet50
from torchvision.models import vgg16, vgg13, vgg11, alexnet, squeezenet1_0, squeezenet1_1
import time

from worker import Worker
from control import Controller
    
def runWorker(ind, models, c2wQueue, w2cQueue, br):
    worker = Worker(ind, models, c2wQueue, w2cQueue, br)
    
    worker.run()

def runController(models, c2wQueues, w2cQueues, sample_times):
    ctrl = Controller(models=models, c2wQueues=c2wQueues, w2cQueues=w2cQueues, sample_times=sample_times)
    ctrl.run()
    
if __name__ == '__main__':

    models = {'vgg11':vgg11(), 'vgg13':vgg13(), 'vgg16':vgg16(), 'resnet18':resnet18(), 'resnet34':resnet34(),
            'sqnt11':squeezenet1_1(), 'sqnt10':squeezenet1_0(),'alexnet':alexnet(), 'resnet101':resnet101(), 
            'resnet50':resnet50(), 'resnet152':resnet152()}
              
    barrier = Barrier(2)
    c2wQueues = [Queue(1000) for _ in range(2)]
    w2cQueues = [Queue(1000) for _ in range(2)]
    sample_times = 100
    
    controller = Process(target=runController, args=(models, c2wQueues, w2cQueues, sample_times))
    workers = [Process(target=runWorker, args=(i, models, c2wQueues[i], w2cQueues[i], barrier)) for i in range(2)]
    
    controller.start()
    for w in workers:
        w.start()
        
    for w in workers:
        w.join()
    controller.join()

    barrier.abort()
         
    print('sampling finished !!!')
    




