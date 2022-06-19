import numpy as np
import time
import os
import torch
from utils.torch_utils import time_sync

class Detect:
    def __init__(self, model, id):
        self.model = model[1].eval()
        self.name = model[0]
        self.id = id
        self.device = 'cpu'
        # self.model._apply(self.pin_memory)  
        
        self.warm_up()
    
    def warm_up(self):
        self.model.cuda()
        fake_img = torch.zeros(1, 3, 224, 224).cuda()
        for _ in range(5):
            self.model(fake_img)
        self.model.cpu()
        print(f'@@@@@@ worker {self.id} finished warm up @@@@@')  
    
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
    def run(self, w_id, img):
        img = img.transpose((0, 3, 1, 2))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).type(torch.FloatTensor).to(self.device)
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
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
        self.imgs = 0
        self.has_started = False 
    
    def run(self):
        
        while True:
            command = self.c2wQueue.get()
            if not self.has_started and command is not None:
                self.time_stamp.append(time.time())
                self.has_started = True
                
            if command == 'to_load':
                self.engine.load_model()
                torch.cuda.synchronize()
                    
            if command == 'to_infer':
                assert not self.imgQueue.empty()
                
                # auto-blocking
                batch_cmd = self.c2wQueue.get()
                frames = []
                for _ in range(int(batch_cmd)):
                    if self.imgQueue.qsize() > 0:
                        self.imgs = self.imgQueue.get()
                        frames.append(np.zeros(shape=(384, 640, 3)))

                assert len(frames) > 0

                pred = self.engine.run(self.id, np.stack(frames))
                # send signal to controller
                self.w2cQueue.put('batch_done')
                self.engine.release_model()
                        
            if command == 'exit':
                print(f'worker {self.id} processed image number : {self.imgs} and to exit ...')
                break
                
        self.time_stamp.append(time.time())
        return 