
import pickle
from threading import Thread
import torch
import time 
import numpy as np
import os
from utils.torch_utils import time_sync

class Detect:
    def __init__(self, model, modules, id):
        self.model = model[1].eval()
        self.name = model[0]
        self.id = id
        self.device = 'cpu'
        self.match_pair = dict()
        self.modules = modules
        self.module_num = len(modules)
        self.start_imod = -1
        self.stop_imod = -1     # inference cursor
        self.start_lmod = 0
        self.stop_lmod = 0      # loading cursor
        self.finish_partial = False
        self.finish_complete = False
        self.model._apply(self.pin_memory)
                    
    def pre_hook(self):
        def forward_pre_hook(module, data_input):
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                while True:
                    if self.start_imod < self.stop_imod:
                        break
                self.start_imod += 1
                if self.start_imod == self.stop_imod:
                    self.finish_partial = True
                                
        return forward_pre_hook
            
    def pin_memory(self, tensor):
        return tensor.pin_memory()
    
    def to_cuda(self, param):
        new = param.cuda(non_blocking=True)
        self.match_pair[new] = param.data
        self.match_pair[param] = param.data
        return new

    def to_cpu(self, param):
        if param in self.match_pair:
            return self.match_pair[param]
        # assert False
        return param.cpu()
    
    def load_model(self):
        self.device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # self.model._apply(self.to_cuda)
        for i in range(self.start_lmod, self.stop_lmod):
            self.modules[i]._apply(self.to_cuda)
        
    def release_model(self):
        self.device = 'cpu'
        self.model._apply(self.to_cpu)
        self.match_pair.clear()
        
        torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self, w_id, im):
        
        # print(f'worker {w_id} : data shape is {im.shape} ')  
        handler = torch.nn.modules.module.register_module_forward_pre_hook(self.pre_hook())

        dt = []
        t1 = time_sync()
        # Inference
        self.model(im)
        t2 = time_sync()
        dt.append(t2 - t1)
        
        # Print results
        t = tuple(x * 1E3 for x in dt)
        # print(f'Speed: %.1fms inference, worker id :%d' % (t + (w_id,)))
        handler.remove()
        self.finish_complete = True

class Worker:
    
    def __init__(self, id, modules, c2wQueue, w2cQueue, imgQueue, time_stamp, model):
        self.id = id
        self.engine = Detect(model, modules, id)
        self.name = model[0]
        self.c2wQueue = c2wQueue
        self.w2cQueue = w2cQueue
        self.imgQueue = imgQueue
        self.time_stamp = time_stamp
        self.img_count = 0
        self.imgs = []
        self.ithread = None
        self.has_started = False
        
    def run(self):
        
        while True:
            command = None if self.c2wQueue.empty() else pickle.loads(self.c2wQueue.get())
            if not self.has_started and command is not None:
                self.time_stamp.append(time.time())
                self.has_started = True
            
            if command is not None:    
                if command[0] == 'loading':
                    self.engine.stop_lmod = command[1]
                    self.engine.load_model() 
                    
                    if self.engine.start_lmod == 0:
                        for _ in range(command[2]):
                            if self.imgQueue.qsize() > 0:
                                self.img_count = self.imgQueue.get()
                                self.imgs.append(np.zeros(shape=(384, 640, 3)))
                        assert len(self.imgs) > 0
                        
                        self.imgs = np.stack(self.imgs).transpose((0, 3, 1, 2))
                        self.imgs = np.ascontiguousarray(self.imgs)
                        self.imgs = torch.from_numpy(self.imgs).type(torch.FloatTensor).cuda()
                    
                    self.engine.start_lmod = command[1]
                    torch.cuda.synchronize()
                    signal = 'load_over' if self.engine.stop_lmod == self.engine.module_num else 'partial_load_over'
                    self.w2cQueue.put(signal)
                    self.ithread = Thread(target=self.engine.run, args=(self.id, self.imgs))
                    self.ithread.start()
                    
                if command[0] == 'inference':
                    self.engine.stop_imod = self.engine.stop_lmod
                    if self.engine.stop_imod == self.engine.module_num:
                        while True:
                            if self.engine.finish_complete:
                                self.w2cQueue.put('task_over')
                                self.engine.release_model()    
                                self.engine.finish_complete = False
                                break
                    else:
                        while True:
                            if self.engine.finish_partial:
                                self.w2cQueue.put('partial_task_over')
                                self.engine.finish_partial = False
                                break
                    
                    self.imgs = []

                if command == 'exit':
                    if self.ithread is not None:
                        self.ithread.join()
                    print(f'worker {self.id} processed image number : {self.img_count} and to exit ...')
                    break
                
        self.time_stamp.append(time.time())
        return              