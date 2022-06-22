import numpy as np
import time
import os
import torch
import pickle
from utils.torch_utils import time_sync

class Detect:
    def __init__(self, model, id):
        self.model = model[1].eval()
        self.name = model[0]
        self.id = id
        self.device = 'cpu'
        self.model._apply(self.pin_memory)  
        
        self.warm_up()
    
    def warm_up(self):
        self.model.cuda()
        fake_img = torch.zeros(1, 3, 224, 224).cuda()
        for _ in range(5):
            self.model(fake_img)
        self.model.cpu()
    
    def load_model(self):
        self.device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.model.to(self.device)
        
    def release_model(self):
        self.device = 'cpu'
        self.model.to(self.device)
        torch.cuda.empty_cache()
        
    def pin_memory(self, tensor):
        return tensor.pin_memory()

    @torch.no_grad()
    def run(self, w_id, im):   
        # if len(im.shape) == 3:
        #     im = im[None]  # expand for batch dim
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
    
    def __init__(self, id, c2wQueue, w2cQueue, imgQueue, time_stamp, model):
        self.id = id
        self.engine = Detect(model, id)
        self.name = model[0]
        self.c2wQueue = c2wQueue
        self.w2cQueue = w2cQueue
        self.imgQueue = imgQueue
        self.time_stamp = time_stamp
        self.img_count = 0
        self.fake_imgs = []
        self.has_started = False 
    
    def run(self):
        self.w2cQueue.put('warm_up_over')
        while True:
            command = pickle.loads(self.c2wQueue.get())
            if not self.has_started and command is not None:
                self.time_stamp.append(time.time())
                self.has_started = True
                
            if command[0] == 'to_load':
                self.engine.load_model()
                assert not self.imgQueue.empty()
                 
                bs = command[1]
                for _ in range(bs):
                    if self.imgQueue.qsize() > 0:
                        self.img_count = self.imgQueue.get()
                        self.fake_imgs.append(np.zeros(shape=(384, 640, 3)))
                        
                self.fake_imgs = np.stack(self.fake_imgs)
                self.fake_imgs = self.fake_imgs.transpose((0, 3, 1, 2))
                self.fake_imgs = np.ascontiguousarray(self.fake_imgs)      
                self.fake_imgs = torch.from_numpy(self.fake_imgs).type(torch.FloatTensor).cuda()
                torch.cuda.synchronize()
                
                self.w2cQueue.put(pickle.dumps(['load_over']))
                assert len(self.fake_imgs) > 0
                    
            if command[0] == 'to_infer':
                pred = self.engine.run(self.id, self.fake_imgs)
                self.w2cQueue.put(pickle.dumps(['task_over', time.time()]))
                self.fake_imgs = []
                self.engine.release_model()
                        
            if command[0] == 'exit':
                print(f'worker {self.id} processed image number : {self.img_count} and to exit ...')
                break
                
        self.time_stamp.append(time.time())
        return 