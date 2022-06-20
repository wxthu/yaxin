
import torch
import time 
import numpy as np
import os
from utils.torch_utils import time_sync
import pickle

class Detect:
    def __init__(self, model, id):
        self.model = model[1].eval()
        self.name = model[0]
        self.id = id
        self.device = 'cpu'
        self.match_pair = dict()
        self.model._apply(self.pin_memory)
        
        self.warm_up()
    
    def warm_up(self):
        self.model.cuda()
        fake_img = torch.zeros(1, 3, 224, 224).cuda()
        for _ in range(5):
            self.model(fake_img)
        self.model.cpu()
        print(f'@@@@@@ worker {self.id} finished warm up @@@@@')
            
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
        self.engine = Detect(model, id)
        self.name = model[0]
        self.c2wQueue = c2wQueue
        self.w2cQueue = w2cQueue
        self.imgQueue = imgQueue
        self.time_stamp = time_stamp
        self.img_count = 0
        self.imgs = []
        
    def run(self):
        self.time_stamp.append(time.time())
        
        while True:
            command = pickle.loads(self.c2wQueue.get())
            if command[0] == 'loading':
                self.engine.load_model() 
                bs = command[1]
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
                
            if command[0] == 'inference':
                pred = self.engine.run(self.id, self.imgs)
                self.w2cQueue.put('task_over')
                self.engine.release_model()
                self.imgs = []

            if command[0] == 'exit':
                print(f'worker {self.id} processed image number : {self.img_count} and to exit ...')
                break
                
        self.time_stamp.append(time.time())
        return              