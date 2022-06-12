
import pickle
from load_mock_data import load_mock_data
from analysis import Analyser
import torch
import joblib
import numpy as np
import time 

class Controller:
    """
    作为Controller, 监视所有的Worker
    """
    def __init__(self, img_num=800, configs=None, models=None, c2wQueues=None, w2cQueues=None, imgQueues=None):
        """ 
        two-way communiaction to prevent message contention
        """
        self.detector_num = len(configs)
        self.load_state = [False for _ in range(self.detector_num)]
        self.infer_state = [False for _ in range(self.detector_num)]
        self.if_full = [False for _ in range(self.detector_num)]
        self.candidate = []
        self.act_id = -1
        self.aux_id = -1
        self.next_act_id = -1
        self.next_aux_id = -1
        self.tCursor = 0
        self.lCursor = 0
        self.not_first_infer = False
        self.not_first_load = False
        self.last_completed = []
        self.aux_stop_mod = 0
        self.img_num = img_num
        self.c2wQueues = c2wQueues
        self.w2cQueues = w2cQueues
        self.imgQueues = imgQueues
        self.cmds = []  # batch number 
        self.deadlines = []
        self.round = 0 
        self.task_num = 0 # record the number of tasks at the moment
        self.total_requests = 0
        self.config = configs  # list, each element is dictionary
        self.models = models
        self.load_time_limit = 10000000
        self.both_full = False # denote whether parallel op groups have reached the end of models
        self.regressor = joblib.load('./train/lat_pred.pkl')
        
        # for i, m in enumerate(model_configs):
        #     self.config.append(m.pop())
        #     self.models.append(Analyser(m.pop(), i))
            
        # print('*** Controller needs to collect models properties and wait for a while .. ***')
        # for m in self.models:
        #     fake_img = torch.randn(1, 3, 384, 640).cuda()
        #     m(fake_img)
            
        # torch.cuda.empty_cache()
        print('##### Controller init success #####')
        
    def initialization(self):
        for q in range(len(self.imgQueues)):
            for i in range(self.img_num):
                # Too large data will block process 
                # q.put(np.zeros(shape=(1920, 1080, 3)))
                self.imgQueues[q].put(i+1)

        data, col = load_mock_data()
        for i in range(col):
            self.cmds.append(list(data[:, i, :][:, 0]))
            self.deadlines.append(list(data[:, i, :][:, 1]))

        for x in self.cmds:
            self.total_requests += sum(x)
            
        self.compute_candidates()
    
    def compute_candidates(self):
        '''
        This function is mainly responsible for computing candidates
        at the start if each round and preloading the first model to 
        infer
        '''
        if self.round >= len(self.cmds):
            return 
        
        cmd = self.cmds[self.round]
        ddls = self.deadlines[self.round]
        self.candidate.clear()
        
        for i, (batch, ddl) in enumerate(zip(cmd, ddls)):
            if batch:
                self.models[i].bs = batch
                self.models[i].ddl = sum(ddl)
                self.models[i].lat = self.config[i][batch][1]['infer_time']
                self.models[i].srt = sum(ddl) - self.models[i].lat
                self.candidate.append(self.models[i])
        
        self.candidate.sort(key=lambda obj: obj.srt)
        assert len(self.candidate) > 0, 'no image input, please re-generate mocked data'
        self.task_num = len(self.candidate)

        self.act_id = self.candidate[0].id
        self.candidate[0].cur_mod = self.candidate[0].module_num
        self.aux_id, self.aux_stop_mod, self.both_full = self.determine_parallel_ops()
        self.notify_to_load([self.act_id, self.aux_id])
            
        self.load_time_limit = self.get_predicted_latency(self.candidate[0].module_num, self.aux_stop_mod)
        self.compute_next_workers()

    def find_index(self, id):
        for i, c in enumerate(self.candidate):
            if c.id == id:
                return i
        return -1 
                
    def notify_to_load(self, ids):
        if self.lCursor >= len(self.candidate):
            print(f'***** To load worker  {ids}*****')
            print(f'***** no Data need to be loaded  {self.lCursor}*****')
            return

        for id in ids:
            if id >= 0:
                if not self.load_state[id] or not self.if_full[id]:
                    index = self.find_index(id)
                    assert index >= 0, 'Fail to find candiate to load...'
                    self.c2wQueues[id].put(
                        pickle.dumps(['loading', self.candidate[index].cur_mod, self.candidate[index].bs]))
                    self.if_full[id] = self.candidate[index].cur_mod == self.candidate[index].module_num
                    self.lCursor += 1
        
    def notify_to_infer(self, ids):
        # if self.tCursor >= len(self.candidate):
        #     print(f'***** no TASK needs to be processed {self.tCursor}/{len(self.candidate)}*****')
        #     return
        
        for id in ids:
            if id >= 0 and not self.infer_state[id] and self.load_state[id]:
                self.c2wQueues[id].put(pickle.dumps(['inference']))
                self.tCursor += 1 
                self.infer_state[id] = True 
                print(f'---------------to notify w{id} to infer  and tCursor {self.tCursor} ------')
    
    def get_module_flops(self, model_id, start, stop):
        if stop <= start:
            return 0
        return sum(self.models[model_id].flops[start:stop])
    
    def get_module_loading_time(self, model_id, start, stop):
        if stop <= start:
            return 0
        return sum(self.models[model_id].mlt[start:stop])  
                       
    # To check prediction model , TODO in future
    def get_predicted_latency(self, stop1, stop2):
        flops1 = self.get_module_flops(self.candidate[0].id, self.candidate[0].cur_mod, stop1)
        if stop2 > 0:
            flops2 = self.get_module_flops(self.candidate[1].id, self.candidate[1].cur_mod, stop2)
        else:
            return self.regressor.predict(
                np.array([self.candidate[0].bs, 0, flops1, 0]).reshape(1,4))
            
        return self.regressor.predict(
            np.array([self.candidate[0].bs, self.candidate[1].bs, flops1, flops2]).reshape(1,4))
        
    def swap_candidate(self, u, v):
        temp = self.candidate[u]
        self.candidate[u] = self.candidate[v]
        self.candidate[v] = temp
            
    def determine_parallel_ops(self):
        if len(self.candidate) <= 1:
            return -1, -1, False
        
        start1 = self.candidate[0].cur_mod
        stop1 = self.candidate[0].module_num 
        
        if self.get_module_loading_time(self.candidate[0].id, start1, stop1) >= self.load_time_limit:
            print(f'*** current model {self.candidate[0].id} loading too long and no parallel modules ***')
            return -1, -1, False
        
        for i in range(1, len(self.candidate)):
            if self.candidate[i].ddl <= self.candidate[0].lat:
                print('### urgent task need to be executed immediately ###')
                if i != 1:
                    self.swap_candidate(i, 1)
                self.candidate[1].cur_mod = self.candidate[1].module_num
                return self.candidate[1].id, self.candidate[1].module_num, True
        
        start2 = self.candidate[1].cur_mod
        stop2 = self.candidate[1].module_num
        flops1 = self.get_module_flops(self.candidate[0].id, start1, stop1)
        
        # binary search to find maximal op range
        rs = start2
        rt = stop2
        index = stop2
        
        while rt > rs:
            flops2 = self.get_module_flops(self.candidate[1].id, start2, index)
            pred_lat = self.regressor.predict(
                np.array([self.candidate[0].bs, self.candidate[1].bs, flops1, flops2]).reshape(1,4))
            if pred_lat <= self.candidate[0].ddl:
                rs = index + 1
            else:
                rt = index - 1
        
            index = int((rs + rt) / 2)
        
        selected = index
        ltime1 = self.get_module_loading_time(self.candidate[0].id, start1, stop1)            
        while start2 < selected:
            ltime2 = self.get_module_loading_time(self.candidate[1].id, start2, selected)
            if ltime1 + ltime2 <= self.load_time_limit:
                break
            selected -= 1
        
        if selected == start2:
            return -1, -1, False
        
        self.candidate[1].cur_mod = selected
        flag = selected == stop2   # indicate whether both models will have finished tasks
        
        return self.candidate[1].id, selected, flag
    
    def update_candidates_state(self):
        if len(self.candidate) == 0:
            return False 
        
        self.candidate.pop(0)
        if self.both_full:
            self.candidate.pop(0)
            
        if len(self.candidate) > 0:
            for i in range(len(self.candidate)):
                if i != 0 or self.both_full:
                    self.candidate[i].srt -= self.load_time_limit
        
            self.candidate.sort(key=lambda obj: obj.srt)
            return True
        
        return False
    
    def compute_next_workers(self):
        notOver = self.update_candidates_state()
        self.next_act_id = self.candidate[0].id if notOver else -1
        if self.next_act_id >= 0:
            self.candidate[0].cur_mod = self.candidate[0].module_num
            self.next_aux_id, self.aux_stop_mod, self.both_full = self.determine_parallel_ops()
        else:
            self.next_aux_id = -1
            self.aux_stop_mod = -1
            self.both_full = False
            
    # Shortest Remaining Time scheduling strategy
    def update_cmd_queue(self):    
            
        for id in [self.act_id, self.aux_id, self.next_act_id, self.next_aux_id]:  
            if id >= 0:
                command = None if self.w2cQueues[id].empty() else self.w2cQueues[id].get()
                if command is not None:
                    if command.endswith('task_over'):
                        self.last_completed.append(id)
                        self.tCursor -= 1
                        self.not_first_infer = True
                        print(f'&&&&& worker {id} finished and tCurosr : {self.tCursor} &&&&&&&&&')
                        if command == 'task_over':
                            self.task_num -= 1
                    elif command == 'load_over':
                        self.load_state[id] = True
                        self.not_first_load = True
                        self.lCursor -= 1
                        print(f'RRRRRRRRRRRRRRRRReceive w_{id} load and lcursor : {self.lCursor}')
                    else:
                        print(f'=== Unknown command: {command} ===')
                        assert False
                                       
        # Pipeline: loading next model while current inference
        if self.lCursor <= 0 and self.not_first_load:
            self.notify_to_infer([self.act_id, self.aux_id])
            print(f'++++++++++++++++++ {self.act_id}/{self.aux_id}/{self.next_act_id}/{self.next_aux_id}')
            print(f'nnnnn system load state : {self.load_state}')
            print(f'nnnnn system if_full state : {self.if_full}')
            self.notify_to_load([self.next_act_id, self.next_aux_id])       
            # time.sleep(1)
        if self.tCursor <= 0 and self.not_first_infer:
            for w_id in self.last_completed:
                self.infer_state[w_id] = False
                # self.if_full[w_id] = False
            self.last_completed.clear()
            
            self.notify_to_infer([self.next_act_id, self.next_aux_id])
            self.act_id = self.next_act_id
            self.aux_id = self.next_aux_id
            if len(self.candidate) > 0:
                self.load_time_limit = self.get_predicted_latency(self.candidate[0].module_num, self.aux_stop_mod)
            self.compute_next_workers()
            print(f'iiiiii Current infer img and load next {self.next_act_id}/{self.next_aux_id} and candi length : {len(self.candidate)}')
            # if self.lCursor <= 0:
            self.notify_to_load([self.next_act_id, self.next_aux_id]) 
            # time.sleep(1)
        
        if self.task_num <= 0: 
            self.load_time_limit = 10000000
            self.reset()
            self.round += 1
            self.compute_candidates()
    
    def reset(self):
        self.not_first_load = False
        self.not_first_infer = False    
        for i in range(self.detector_num):
            self.load_state[i] = False
            self.if_full[i] = False    
            
    def run(self):
        self.initialization()
        while True:
            self.update_cmd_queue()
            if self.round >= len(self.cmds):
                break

        for i in range(self.detector_num):
            self.c2wQueues[i].put(pickle.dumps('exit'))
        
        print(f"all workers has finished jobs, processing {self.total_requests} images in total!")
        return