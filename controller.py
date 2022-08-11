
import pickle
from load_mock_data import load_mock_data
from analysis import Analyser
import torch
import joblib
import numpy as np
import time
from utils.visualize import plot

class Controller:
    """
    作为Controller, 监视所有的Worker
    """
    def __init__(self, img_num=800, configs=None, models=None, c2wQueues=None, w2cQueues=None, imgQueues=None, if_plotted=False):
        """ 
        two-way communiaction to prevent message contention
        """
        self.detector_num = len(configs)
        self.load_state = [False for _ in range(self.detector_num)]
        self.infer_state = [False for _ in range(self.detector_num)]
        self.candidate = []
        self.act_id = -1
        self.aux_id = -1
        self.next_act_id = -1
        self.next_aux_id = -1
        self.tCursor = 0  # indicate whether current workers finished inference
        self.lCursor = 0  # indicate whether next workers loading complete
        self.both = False
        self.last_completed = []
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
        self.qos_violation_count = 0
        self.avg_violation_ratio = []
        self.completion_time = None  # recording the completion time for each worker
        self.regressor = joblib.load('./train/lat_pred.pkl')
        self.timestamps_for_each_round = [[] for _ in range(self.detector_num)]
        self.timestamps_for_all_round = [[] for _ in range(self.detector_num)]
        self.if_plotted = if_plotted
        
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
        
        for i in range(self.detector_num):
            signal = self.w2cQueues[i].get()
            assert signal == 'warm_up_over'
            print(f'@@@@@@ worker {i} finished warm up @@@@@')
            
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
        self.completion_time = [0.0 for _ in range(self.detector_num)]
        for i, (batch, ddl) in enumerate(zip(cmd, ddls)):
            if batch:
                self.models[i].bs = batch
                self.models[i].ddl = sum(ddl)
                self.models[i].lat = self.config[i][batch][1]['infer_time']
                self.models[i].srt = sum(ddl) - self.models[i].lat
                self.candidate.append(self.models[i])
                self.completion_time[i] = time.time()
        
        self.candidate.sort(key=lambda obj: obj.srt)
        if len(self.candidate) == 0:
            return
        # assert len(self.candidate) > 0, 'no image input, please re-generate mocked data'
        self.task_num = len(self.candidate)

        self.next_act_id = self.candidate[0].id
        self.next_aux_id, self.both = self.determine_parallel_model()
        self.notify_to_load([self.next_act_id, self.next_aux_id]) 
        self.load_time_limit = self.get_predicted_latency(self.next_aux_id)
    
    def find_index(self, ind):
        for i, c in enumerate(self.candidate):
            if c.id == ind:
                return i
        return -1 
                
    def notify_to_load(self, ids):
        if self.lCursor >= len(self.candidate):
            return

        for id in ids:
            if id >= 0 and not self.load_state[id]:
                index = self.find_index(id)
                assert index >= 0, 'Fail to find candiate to load...'
                self.c2wQueues[id].put(pickle.dumps(['loading', self.candidate[index].bs]))
                self.timestamps_for_each_round[id].append(time.time())
                self.lCursor += 1
        
    def notify_to_infer(self, ids):
        for id in ids:
            if id >= 0 and not self.infer_state[id] and self.load_state[id]:
                self.c2wQueues[id].put(pickle.dumps(['inference']))
                self.tCursor += 1 
                self.infer_state[id] = True 
    
    def get_model_flops(self, model_id):
        return sum(self.models[model_id].flops)
    
    def get_model_loading_time(self, model_id):
        return sum(self.models[model_id].mlt)  
                       
    # To check prediction model , TODO in future
    def get_predicted_latency(self, second_id):
        if second_id < 0:
            id = self.candidate[0].id
            bs = self.candidate[0].bs
            return self.config[id][bs][1]['infer_time']
        
        flops1 = self.get_model_flops(self.candidate[0].id)
        flops2 = self.get_model_flops(self.candidate[1].id)
        
        return self.regressor.predict(
            np.array([self.candidate[0].bs, self.candidate[1].bs, flops1, flops2]).reshape(1,4))
        
    def swap_candidate(self, u, v):
        temp = self.candidate[u]
        self.candidate[u] = self.candidate[v]
        self.candidate[v] = temp
            
    def determine_parallel_model(self):
        if len(self.candidate) <= 1:
            return -1, False
        
        if self.get_model_loading_time(self.candidate[0].id) >= self.load_time_limit:
            print(f'*** current model {self.candidate[0].id} loading too long and no parallel model ***')
            return -1, False
        
        for i in range(1, len(self.candidate)):
            if self.candidate[i].ddl <= self.candidate[0].lat:
                print("\033[1;31m [WARN] urgent task need to be executed immediately \033[0m")
                if i != 1:
                    self.swap_candidate(i, 1)
                return self.candidate[1].id, True
        
        flops1 = self.get_model_flops(self.candidate[0].id)
        flops2 = self.get_model_flops(self.candidate[1].id)
        pred_lat = self.regressor.predict(
            np.array([self.candidate[0].bs, self.candidate[1].bs, flops1, flops2]).reshape(1,4))
        
        lt = self.get_model_loading_time(self.candidate[0].id) + self.get_model_loading_time(self.candidate[1].id)
        if pred_lat <= self.candidate[0].ddl and lt <= self.load_time_limit:
            return self.candidate[1].id, True
            
        return -1, False
    
    def update_candidates_state(self):
        if len(self.candidate) == 0:
            return False 
        
        self.candidate.pop(0)
        if self.both:
            self.candidate.pop(0)
        
        if len(self.candidate) > 0:    
            for i in range(len(self.candidate)):
                self.candidate[i].srt -= self.load_time_limit
            self.candidate.sort(key=lambda obj: obj.srt)
            
            return True
        return False
        
    def compute_next_workers(self):
        notOver = self.update_candidates_state()
        self.next_act_id = self.candidate[0].id if notOver else -1
        self.next_aux_id, self.both = self.determine_parallel_model() if self.next_act_id >= 0 else (-1, False)
          
    def compute_qos_violation(self):
        for i, real in enumerate(self.completion_time):
            if real > self.models[i].ddl:
                batch = self.models[i].bs
                self.qos_violation_count += batch
                vio_ratio = [(real - self.models[i].ddl) / self.models[i].ddl * 100 for _ in range(batch)]
                self.avg_violation_ratio.extend(vio_ratio)
    
    def record_timestamps_and_clear_cache(self):
        for i, ts in enumerate(self.timestamps_for_each_round):
            if len(ts) == 2:
                self.timestamps_for_all_round[i].append(ts)
            self.timestamps_for_each_round[i] = []
    
    def timeline_view(self):
        # data alignment
        start_up = []
        for i in range(self.detector_num):
            start_up.append(self.timestamps_for_all_round[i][0][0]) 
        start_up = min(start_up)
        
        for i in range(self.detector_num):
            number = len(self.timestamps_for_all_round[i])
            for j in range(number):
                for k in range(2):
                    self.timestamps_for_all_round[i][j][k] -= start_up
                    # self.timestamps_for_all_round[i][j][k] *= 1000    # ms
        
        plot(self.timestamps_for_all_round)  
        return              
               
    # Shortest Remaining Time scheduling strategy
    def update_cmd_queue(self):    
        
        for id in [self.act_id, self.aux_id, self.next_act_id, self.next_aux_id]:  
            if id >= 0:
                command = None if self.w2cQueues[id].empty() else self.w2cQueues[id].get()
                if command is not None:
                    if command == 'task_over':
                        self.last_completed.append(id)
                        self.tCursor -= 1
                        self.task_num -= 1
                        self.completion_time[id] = 1000 * (time.time() - self.completion_time[id])
                        self.timestamps_for_each_round[id].append(time.time())
                    elif command == 'load_over':
                        self.load_state[id] = True
                        self.lCursor -= 1
                    else:
                        print(f'=== Unknown command: {command} ===')
                        assert False
                                       
        # Pipeline: loading next model while current inference
        if self.lCursor <= 0 and self.tCursor <= 0:
            for w_id in self.last_completed:
                self.infer_state[w_id] = False
                self.load_state[w_id] = False
            self.last_completed.clear()
            
            self.notify_to_infer([self.next_act_id, self.next_aux_id])
            self.act_id = self.next_act_id
            self.aux_id = self.next_aux_id
            if len(self.candidate) > 0:
                self.load_time_limit = self.get_predicted_latency(self.aux_id)
            self.compute_next_workers()
            self.notify_to_load([self.next_act_id, self.next_aux_id]) 
        
        if self.task_num <= 0: 
            self.load_time_limit = 10000000
            self.compute_qos_violation()
            self.record_timestamps_and_clear_cache()
            self.round += 1
            print(f'\033[1;34m [ROUND] : {self.round}\033[0m')
            self.compute_candidates()
      
            
    def run(self):
        self.initialization()
        while True:
            self.update_cmd_queue()
            if self.round >= len(self.cmds):
                break

        for i in range(self.detector_num):
            self.c2wQueues[i].put(pickle.dumps('exit'))
        
        print(f"All workers has finished jobs, processing {self.total_requests} images in total!")
        print(f'Total number of overtime task is : {self.qos_violation_count}')
        print(f'Average QoS violation ratio is : {sum(self.avg_violation_ratio)/self.qos_violation_count}')
        if self.if_plotted:
            self.timeline_view()
        return