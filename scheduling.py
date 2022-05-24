"""
We often have following configs for testing:
(i)  rnt50, rnt152, alexnet, vgg11, yolov5s, yolov5x 
(ii) sqt10, rnt18, rnt34, rnt101, yolov5s, yolov5x
(iii)sqt11, rnt50, rnt152, vgg13, yolo5s, yolov5x
"""

import argparse
import multiprocessing
from multiprocessing import Process, Queue, Manager
import os
import numpy as np
import time
import torch

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import alexnet, vgg16, vgg13, vgg11, squeezenet1_0, squeezenet1_1
from utils.general import check_requirements, print_args
from utils.torch_utils import time_sync
from controller import Controller

class Detect:
    def __init__(self, model):
        self.model = model[1].eval()
        self.name = model[0]
        self.device = 'cpu'
        self.match_pair=dict()
        self.model._apply(self.pin_memory)
        # self.stms = [torch.cuda.Stream() for _ in range(2)]
        
        # torch.nn.modules.module.register_module_forward_pre_hook(self.pre_hook(self.stms[0]))
    
    def pre_hook(self, strm):
        def forward_pre_hook(module, data_input):
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                with torch.cuda.stream(strm):
                    module._apply(self.to_cuda)
                torch.cuda.synchronize()
        
        return forward_pre_hook
            
    def pin_memory(self, tensor):
        return tensor.pin_memory()
    
    def to_cuda(self, param):
        new = param.cuda(non_blocking=True)
        self.match_pair[new] = param.data
        self.match_pair[param] = param.data
        return new

    # This function is suitable for all the models in torchvision, except yolov5s, yolov5x
    def to_cpu(self, param):
        if param in self.match_pair:
            return self.match_pair[param]
        # assert False
        return param.cpu()
    
    def load_model(self):
        self.device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.model._apply(self.to_cuda)
        
    def release_model(self):
        self.device = 'cpu'
        self.model._apply(self.to_cpu)
        self.match_pair.clear()
        
        torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self, w_id, im):
        
        print(f'worker {w_id} : data shape is {im.shape} ')  
        
        # with torch.cuda.stream(self.stms[1]):
        dt = []
        t1 = time_sync()
        # Inference
        self.model(im)
        t2 = time_sync()
        dt.append(t2 - t1)
        
        # Print results
        t = tuple(x * 1E3 for x in dt)
        print(f'Speed: %.1fms inference, worker id :%d' % (t + (w_id,)))
    

class Worker:
    
    def __init__(self, id, batchsize, c2wQueue, w2cQueue, imgQueue, time_stamp, model):
        self.id = id
        self.batchsize = batchsize
        self.engine = Detect(model)
        self.name = model[0]
        self.c2wQueue = c2wQueue
        self.w2cQueue = w2cQueue
        self.imgQueue = imgQueue
        self.time_stamp = time_stamp
        self.img_count = 0
        self.imgs = []
        self.strms = [torch.cuda.Stream() for _ in range(2)]
        
    def run(self):
        self.time_stamp.append(time.time())
        
        while True:
            command = None if self.c2wQueue.empty() else self.c2wQueue.get()
            if command == 'loading':
                with torch.cuda.stream(self.strms[0]):
                    self.engine.load_model() 
                    
                    while self.c2wQueue.empty():
                        pass
                    
                    bs = self.c2wQueue.get()
                    for _ in range(int(bs)):
                        if self.imgQueue.qsize() > 0:
                            self.img_count = self.imgQueue.get()
                            self.imgs.append(np.zeros(shape=(384, 640, 3)))
                    assert len(self.imgs) > 0
                    
                    self.imgs = np.stack(self.imgs).transpose((0, 3, 1, 2))
                    self.imgs = np.ascontiguousarray(self.imgs)
                    self.imgs = torch.from_numpy(self.imgs).type(torch.FloatTensor).cuda()

                torch.cuda.synchronize()
                self.w2cQueue.put('load_over')
                
            if command == 'inference':

                pred = self.engine.run(self.id, self.imgs)
                self.w2cQueue.put('task_over')
                with torch.cuda.stream(self.strms[1]):
                    self.engine.release_model()
                    self.imgs = []
                        
            if command == 'exit':
                print(f'worker {self.id} processed image number : {self.img_count} and to exit ...')
                break
                
        self.time_stamp.append(time.time())
        return              

def runWorker(i,
            batchsize, 
            c2wQueue,
            w2cQueue,
            imgQueue, 
            time_stamp,
            model):
    worker = Worker(i,
                    batchsize, 
                    c2wQueue=c2wQueue,
                    w2cQueue=w2cQueue,
                    imgQueue=imgQueue, 
                    time_stamp=time_stamp,
                    model=model
                    )
    worker.run()

def runController(workers_num, 
                  image_num,
                  video_num,
                  c2wQueues,
                  w2cQueues, 
                  imgQueues,
                  config):
    controller = Controller(detector_num=workers_num, img_num=image_num*video_num, config=config,
                            c2wQueues=c2wQueues, w2cQueues=w2cQueues, imgQueues=imgQueues)
    controller.run()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1, help='batch size of img')
    parser.add_argument('--img_num', type=int, default=200, help='the number of img sent by each client')
    parser.add_argument('--videos', type=int, default=4, help='the number of video stream')
    parser.add_argument('--workers', type=int, default=6, help='the number of detector')
    parser.add_argument('--memory', type=int, default=1500, help='available gpu memory in the server')
    parser.add_argument('--strategy', type=str, default='infer_time', help='scheduling strategy')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    args_dict = vars(opt)

    strategies = ['infer_time', 'total_time', 'gpu_util']
    workers_num = len(models)
    image_num = args_dict['img_num']
    batchsize = args_dict['bs']
    video_num = args_dict['videos']
    total_memory = args_dict['memory']
    strategy = args_dict['strategy']

    if strategy not in strategies:
        strategy = 'infer_time'
        print('*****  Expected strategy is non-existent and assign it by default *****')
    
    config = [x[2] for x in models]
    # record all workers timestamp
    time_stamp = Manager().list()
    MAX_Q_NUM = 1000
    
    c2wQueues = [Queue(MAX_Q_NUM) for _ in range(workers_num)]
    w2cQueues = [Queue(MAX_Q_NUM) for _ in range(workers_num)]
    imgQueues = [Queue(MAX_Q_NUM) for _ in range(workers_num)]
    
    controller = Process(target=runController, args=(workers_num, 
                                                     image_num,
                                                     video_num,
                                                     c2wQueues,
                                                     w2cQueues,
                                                     imgQueues,
                                                     config
                                                     )
                         )
    
    detectors = [
                Process(target=runWorker, 
                        args=(i,
                                batchsize,
                                c2wQueues[i],
                                w2cQueues[i],
                                imgQueues[i], 
                                time_stamp,
                                models[i]
                            )
                        )
                            
                for i in range(workers_num)
                ]

    controller.start()
    
    for detector in detectors:
        detector.start()

    controller.join()
    for detector in detectors:
        detector.join()
    
    warmup_overhead = 2.5
    duration = max(time_stamp) - min(time_stamp)
    print("***** The system end-to-end latency is : {:.3f}s *****".format(duration - warmup_overhead))


if __name__ == "__main__":
    models = []
    # yolov5x = DetectMultiBackend('yolov5x.pt', device=torch.device('cpu'))
    # yolov5s = DetectMultiBackend('yolov5s.pt', device=torch.device('cpu'))

    # profiling : memory, inference, total_time, gpu_utilization
    # models.append(['sqt10', squeezenet1_0(), {1:[58, {'infer_time':3.96, 'total_time':10.0, 'gpu_util':47}],
    #                                           2:[120, {'infer_time':8.58, 'total_time':14.62, 'gpu_util':65}],
    #                                           3:[186, {'infer_time':13.05, 'total_time':19.09, 'gpu_util':74}],
    #                                           4:[238, {'infer_time':17.01, 'total_time':23.05, 'gpu_util':78}]}])
    # models.append(['sqt11', squeezenet1_1(), {1:[46, {'infer_time':3.52, 'total_time':9.62, 'gpu_util':40}],
    #                                           2:[70, {'infer_time':4.45, 'total_time':10.55, 'gpu_util':51}],
    #                                           3:[106, {'infer_time':6.84, 'total_time':12.94, 'gpu_util':60}],
    #                                           4:[130, {'infer_time':9.03, 'total_time':15.13, 'gpu_util':67}]}])
    # models.append(['rnt18', resnet18(), {1:[124, {'infer_time':4.39, 'total_time':24.2, 'gpu_util':54}],
    #                                      2:[126, {'infer_time':8.29, 'total_time':28.1, 'gpu_util':60}],
    #                                      3:[154, {'infer_time':11.99, 'total_time':31.8, 'gpu_util':67}],
    #                                      4:[196, {'infer_time':15.3, 'total_time':35.11, 'gpu_util':69}]}])
    # models.append(['rnt34', resnet34(), {1:[166, {'infer_time':7.54, 'total_time':43.22, 'gpu_util':54}],
    #                                      2:[168, {'infer_time':14.48, 'total_time':50.16, 'gpu_util':61}],
    #                                      3:[198, {'infer_time':21.12, 'total_time':56.8, 'gpu_util':67}],
    #                                      4:[238, {'infer_time':27.37, 'total_time':63.05, 'gpu_util':71}]}])
    # models.append(['rnt50', resnet50(), {1:[192, {'infer_time':13.44, 'total_time':59.84, 'gpu_util':56}],
    #                                      2:[238, {'infer_time':24.62, 'total_time':71.02, 'gpu_util':61}],
    #                                      3:[274, {'infer_time':37.04, 'total_time':83.44, 'gpu_util':71}],
    #                                      4:[192, {'infer_time':48.22, 'total_time':94.62, 'gpu_util':71}]}])
    models.append(['rnt101', resnet101(), {1:[266, {'infer_time':22.75, 'total_time':132.62, 'gpu_util':50}],
                                           2:[312, {'infer_time':40.81, 'total_time':150.68, 'gpu_util':61}],
                                           3:[376, {'infer_time':63.33, 'total_time':173.2, 'gpu_util':68}],
                                           4:[432, {'infer_time':83.93, 'total_time':193.8, 'gpu_util':72}]}])
    models.append(['rnt152', resnet152(), {1:[334, {'infer_time':32.86, 'total_time':179.28, 'gpu_util':55}],
                                           2:[354, {'infer_time':59.28, 'total_time':205.7, 'gpu_util':59}],
                                           3:[400, {'infer_time':94.24, 'total_time':240.66, 'gpu_util':64}],
                                           4:[494, {'infer_time':124.54, 'total_time':270.96, 'gpu_util':75}]}])
    models.append(['alexnet', alexnet(), {1:[264, {'infer_time':4.24, 'total_time':136.29, 'gpu_util':39}],
                                          2:[264, {'infer_time':4.71, 'total_time':136.76, 'gpu_util':39}],
                                          3:[276, {'infer_time':6.11, 'total_time':138.16, 'gpu_util':42}],
                                          4:[292, {'infer_time':6.85, 'total_time':138.9, 'gpu_util':42}]}])
    models.append(['vgg11', vgg11(), {1:[618, {'infer_time':13.39, 'total_time':304.11, 'gpu_util':43}],
                                      2:[742, {'infer_time':24.3, 'total_time':315.02, 'gpu_util':45}],
                                      3:[848, {'infer_time':35.53, 'total_time':326.25, 'gpu_util':50}],
                                      4:[944, {'infer_time':47.06, 'total_time':337.78, 'gpu_util':55}]}])
    models.append(['vgg13', vgg13(), {1:[634, {'infer_time':18.59, 'total_time':306.89, 'gpu_util':44}],
                                      2:[774, {'infer_time':35.35, 'total_time':323.65, 'gpu_util':50}],
                                      3:[894, {'infer_time':53.64, 'total_time':341.94, 'gpu_util':58}],
                                      4:[1006, {'infer_time':73.81, 'total_time':362.11, 'gpu_util':66}]}])
    models.append(['vgg16', vgg16(), {1:[674, {'infer_time':23.03, 'total_time':319.4, 'gpu_util':46}],
                                      2:[794, {'infer_time':44.92, 'total_time':341.29, 'gpu_util':55}],
                                      3:[914, {'infer_time':71.11, 'total_time':367.48, 'gpu_util':66}],
                                      4:[1026, {'infer_time':99.23, 'total_time':395.6, 'gpu_util':78}]}])
    # models.append(['yolov5x', yolov5x, {1:[422, {'infer_time':31.91, 'total_time':140.8, 'gpu_util':70}],
    #                                     2:[458, {'infer_time':58.69, 'total_time':167.58, 'gpu_util':78}],
    #                                     3:[506, {'infer_time':90.23, 'total_time':199.12, 'gpu_util':78}],
    #                                     4:[566, {'infer_time':118.76, 'total_time':227.65, 'gpu_util':78}]}])
    # models.append(['yolov5s', yolov5s, {1:[60, {'infer_time':7.46, 'total_time':27.29, 'gpu_util':40}],
    #                                     2:[96, {'infer_time':7.51, 'total_time':27.34, 'gpu_util':46}],
    #                                     3:[110, {'infer_time':11.22, 'total_time':31.05, 'gpu_util':54}],
    #                                     4:[132, {'infer_time':13.75, 'total_time':33.58, 'gpu_util':57}]}])

    opt = parse_opt()
    main(opt)