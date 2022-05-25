import numpy as np
import time
import os
import torch
from utils.torch_utils import time_sync

class Detect:
    def __init__(self, model):
        self.model = model[1].eval()
        self.name = model[0]
        self.device = 'cpu'
        self.match_pair=dict()
    
    def load_model(self):
        self.device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.model.to(self.device)
        
    def release_model(self):
        self.device = 'cpu'
        self.model.to(self.device)
        self.match_pair.clear()
        torch.cuda.empty_cache()

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
    
    def __init__(self, id, batchsize, c2wQueue, w2cQueue, imgQueue, time_stamp, model):
        self.id = id
        self.batchsize = batchsize
        self.engine = Detect(model)
        self.name = model[0]
        self.c2wQueue = c2wQueue
        self.w2cQueue = w2cQueue
        self.imgQueue = imgQueue
        self.time_stamp = time_stamp
        self.imgs = 0
        self.hasInfered = False # Set a flag to avoid the case where controller hasn't put img into imgQueue
        self.loaded = False     # To indicate whether model has loaded into GPU
    
    def run(self):
        self.time_stamp.append(time.time())
        
        while True:
            command = None if self.c2wQueue.empty() else self.c2wQueue.get()
            if command == 'to_load':
                self.engine.load_model()
                self.loaded = True
                    
            if command == 'to_infer':
                assert self.loaded == True
                self.hasInfered = True
                assert not self.imgQueue.empty()
                
                while self.c2wQueue.empty():
                    pass
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
                self.loaded = False
                        
            if command == 'exit':
                print(f'worker {self.id} processed image number : {self.imgs} and to exit ...')
                break
                
        self.time_stamp.append(time.time())
        return 