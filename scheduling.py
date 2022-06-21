"""
We often have following configs for testing:
(i)  rnt50, rnt152, alexnet, vgg11, yolov5s, yolov5x 
(ii) sqt10, rnt18, rnt34, rnt101, yolov5s, yolov5x
(iii)sqt11, rnt50, rnt152, vgg13, yolo5s, yolov5x
"""
import torch.cuda.profiler as profiler
import argparse
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import random
import torch

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import alexnet, vgg16, vgg13, vgg11, squeezenet1_0, squeezenet1_1
from utils.general import check_requirements, print_args
from controller import Controller
from worker import Worker
from analysis import Analyser

def runWorker(i, modules, c2wQueue, w2cQueue, imgQueue, time_stamp, model):
    worker = Worker(i,
                    modules, 
                    c2wQueue=c2wQueue,
                    w2cQueue=w2cQueue,
                    imgQueue=imgQueue, 
                    time_stamp=time_stamp,
                    model=model
                    )
    worker.run()

def runController(image_num, video_num, c2wQueues, w2cQueues, imgQueues, configs, models, visualize):
    controller = Controller(img_num=image_num*video_num, 
                            configs=configs,
                            models=models,
                            c2wQueues=c2wQueues, 
                            w2cQueues=w2cQueues, 
                            imgQueues=imgQueues,
                            if_plotted=visualize)
    controller.run()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', type=int, default=0, help='specify the hardware platform, Tesla 4 by default , rtx 3090 otherwise')
    parser.add_argument('--img_num', type=int, default=200, help='the number of img sent by each client')
    parser.add_argument('--videos', type=int, default=4, help='the number of video stream')
    parser.add_argument('--memory', type=int, default=1500, help='available gpu memory in the server')
    parser.add_argument('--visualize', action='store_true', help='visualize the timeline for each worker')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    mp.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    args_dict = vars(opt)

    platform = args_dict['platform']
    models = models1 if platform == 0 else models2
    
    workers_num = len(models)
    image_num = args_dict['img_num']
    video_num = args_dict['videos']
    total_memory = args_dict['memory']
    visualize = args_dict['visualize']

    # record all workers timestamp
    time_stamp = Manager().list()
    MAX_Q_NUM = 1000
    
    c2wQueues = [Queue(MAX_Q_NUM) for _ in range(workers_num)]
    w2cQueues = [Queue(MAX_Q_NUM) for _ in range(workers_num)]
    imgQueues = [Queue(MAX_Q_NUM) for _ in range(workers_num)]
    
    fake_img = torch.zeros(1, 3, 384, 640).cuda()
    analysers = []
    configs = []

    for i, mod in enumerate(models):
        analysers.append(Analyser(mod[1], i))
        configs.append(mod[2])
    
    print('*** Scheduler begin to collect models properties and wait for a while .. ***')
    
    for m in analysers:
        m(fake_img)
       
    torch.cuda.empty_cache() 
    print('*** Online profiling finished and begin to launch processes .. ***')
        
    controller = Process(target=runController, args=(image_num,
                                                     video_num,
                                                     c2wQueues,
                                                     w2cQueues,
                                                     imgQueues,
                                                     configs,
                                                     analysers,
                                                     visualize
                                                     )
                         )
    
    detectors = [
                Process(target=runWorker, 
                        args=(i,
                                analysers[i].modules,
                                c2wQueues[i],
                                w2cQueues[i],
                                imgQueues[i], 
                                time_stamp,
                                models[i]
                            )
                        ,name="YAXIN-%d"%(i)
                        )
                            
                for i in range(workers_num)
                ]
    
    profiler.start()
    controller.start()
    for detector in detectors:
        detector.start()

    controller.join()
    for detector in detectors:
        detector.join()
    
    profiler.stop()
    duration = max(time_stamp) - min(time_stamp)
    print("***** The system end-to-end latency is : {:.3f}s *****".format(duration))


if __name__ == "__main__":
    models1 = []
    models2 = []
    # profiling : memory && inference time
    # models1.append(['sqt10', squeezenet1_0(), {1:[58, {'infer_time':3.96}],
    #                                           2:[120, {'infer_time':8.58}],
    #                                           3:[186, {'infer_time':13.05}],
    #                                           4:[238, {'infer_time':17.01}]}])
    # models1.append(['sqt11', squeezenet1_1(), {1:[46, {'infer_time':3.52}],
    #                                           2:[70, {'infer_time':4.45}],
    #                                           3:[106, {'infer_time':6.84}],
    #                                           4:[130, {'infer_time':9.03}]}])
    # models1.append(['rnt18', resnet18(), {1:[124, {'infer_time':4.39}],
    #                                      2:[126, {'infer_time':8.29}],
    #                                      3:[154, {'infer_time':11.99}],
    #                                      4:[196, {'infer_time':15.3}]}])
    # models1.append(['rnt34', resnet34(), {1:[166, {'infer_time':7.54}],
    #                                      2:[168, {'infer_time':14.48}],
    #                                      3:[198, {'infer_time':21.12}],
    #                                      4:[238, {'infer_time':27.37}]}])
    models1.append(['rnt50', resnet50(), {1:[192, {'infer_time':13.44}],
                                         2:[238, {'infer_time':24.62}],
                                         3:[274, {'infer_time':37.04}],
                                         4:[192, {'infer_time':48.22}]}])
    models1.append(['rnt101', resnet101(), {1:[266, {'infer_time':22.75}],
                                           2:[312, {'infer_time':40.81}],
                                           3:[376, {'infer_time':63.33}],
                                           4:[432, {'infer_time':83.93}]}])
    models1.append(['rnt152', resnet152(), {1:[334, {'infer_time':32.86}],
                                           2:[354, {'infer_time':59.28}],
                                           3:[400, {'infer_time':94.24}],
                                           4:[494, {'infer_time':124.54}]}])
    # models1.append(['alexnet', alexnet(), {1:[264, {'infer_time':4.24}],
    #                                       2:[264, {'infer_time':4.71}],
    #                                       3:[276, {'infer_time':6.11}],
    #                                       4:[292, {'infer_time':6.85}]}])
    models1.append(['vgg11', vgg11(), {1:[618, {'infer_time':13.39}],
                                      2:[742, {'infer_time':24.3}],
                                      3:[848, {'infer_time':35.53}],
                                      4:[944, {'infer_time':47.06}]}])
    models1.append(['vgg13', vgg13(), {1:[634, {'infer_time':18.59}],
                                      2:[774, {'infer_time':35.35}],
                                      3:[894, {'infer_time':53.64}],
                                      4:[1006, {'infer_time':73.81}]}])
    models1.append(['vgg16', vgg16(), {1:[674, {'infer_time':23.03}],
                                      2:[794, {'infer_time':44.92}],
                                      3:[914, {'infer_time':71.11}],
                                      4:[1026, {'infer_time':99.23}]}])
    
    # RTX-3090 configurations    
    # models2.append(['sqt10', squeezenet1_0(), {1:[58, {'infer_time':2.30}],
    #                                           2:[120, {'infer_time':2.74}],
    #                                           3:[186, {'infer_time':3.99}],
    #                                           4:[238, {'infer_time':5.06}]}])
    # models2.append(['sqt11', squeezenet1_1(), {1:[46, {'infer_time':2.21}],
    #                                           2:[70, {'infer_time':2.31}],
    #                                           3:[106, {'infer_time':2.48}],
    #                                           4:[130, {'infer_time':2.89}]}])
    # models2.append(['rnt18', resnet18(), {1:[124, {'infer_time':2.31}],
    #                                      2:[126, {'infer_time':2.43}],
    #                                      3:[154, {'infer_time':3.79}],
    #                                      4:[196, {'infer_time':4.70}]}])
    # models2.append(['rnt34', resnet34(), {1:[166, {'infer_time':4.12}],
    #                                      2:[168, {'infer_time':4.29}],
    #                                      3:[198, {'infer_time':6.77}],
    #                                      4:[238, {'infer_time':8.4}]}])
    models2.append(['rnt50', resnet50(), {1:[192, {'infer_time':5.8}],
                                         2:[238, {'infer_time':7.84}],
                                         3:[274, {'infer_time':11.91}],
                                         4:[192, {'infer_time':14.66}]}])
    models2.append(['rnt101', resnet101(), {1:[266, {'infer_time':11.3}],
                                           2:[312, {'infer_time':13.56}],
                                           3:[376, {'infer_time':20.44}],
                                           4:[432, {'infer_time':24.55}]}])
    models2.append(['rnt152', resnet152(), {1:[334, {'infer_time':15.67}],
                                           2:[354, {'infer_time':19.55}],
                                           3:[400, {'infer_time':29.4}],
                                           4:[494, {'infer_time':35.31}]}])
    # models2.append(['alexnet', alexnet(), {1:[264, {'infer_time':0.9}],
    #                                       2:[264, {'infer_time':1.15}],
    #                                       3:[276, {'infer_time':1.49}],
    #                                       4:[292, {'infer_time':1.78}]}])
    models2.append(['vgg11', vgg11(), {1:[618, {'infer_time':4.3}],
                                      2:[742, {'infer_time':7.26}],
                                      3:[848, {'infer_time':10.73}],
                                      4:[944, {'infer_time':13.37}]}])
    models2.append(['vgg13', vgg13(), {1:[634, {'infer_time':5.91}],
                                      2:[774, {'infer_time':10.45}],
                                      3:[894, {'infer_time':15.53}],
                                      4:[1006, {'infer_time':19.77}]}])
    models2.append(['vgg16', vgg16(), {1:[674, {'infer_time':7.45}],
                                      2:[794, {'infer_time':13.08}],
                                      3:[914, {'infer_time':19.60}],
                                      4:[1026, {'infer_time':24.78}]}])

    opt = parse_opt()
    main(opt)