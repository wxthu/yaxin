import torch
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import alexnet, vgg16, vgg13, vgg11, squeezenet1_0, squeezenet1_1

import time
from tqdm import tqdm

import pandas as pd
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

match_pair=dict()
def to_cuda(tensor):
    new=tensor.cuda()
    match_pair[new]=tensor.data
    match_pair[tensor]=tensor.data
    return new

# This function is suitable for all the models in torchvision, except yolov5s, yolov5x
def to_cpu(tensor):
    if tensor in match_pair:
        return match_pair[tensor]
    # assert False  
    return tensor.cpu()

def pin_memory(tensor):
    return tensor.pin_memory()

if __name__ == '__main__':
    warm = resnet50().eval().cuda()
    rdm_input = torch.randn(1, 3, 384, 640).to('cuda')
    t1 = time_sync()
    for _ in range(50):
        y = warm(rdm_input)
    duration = time_sync() - t1
    print(f'*** warm up finished, duration : {1000 * duration / 2000}ms ***')
    rdm_input = rdm_input.cpu()
    warm.cpu()
    torch.cuda.empty_cache()
    
    NUM = 500
    loading = [0 for _ in range(NUM)]
    inference = [0 for _ in range(NUM)]
    unloading = [0 for _ in range(NUM)]

    models = {}
    models.update({'sqt10': squeezenet1_0().eval()})
    models.update({'sqt11': squeezenet1_1().eval()})
    models.update({'rnt18': resnet18().eval()})
    models.update({'rnt34': resnet34().eval()})
    models.update({'rnt50': resnet50().eval()})
    models.update({'rnt101': resnet101().eval()})
    models.update({'rnt152': resnet152().eval()})
    models.update({'alexnet': alexnet().eval()})
    models.update({'vgg11': vgg11().eval()})
    models.update({'vgg13': vgg13().eval()})
    models.update({'vgg16': vgg16().eval()})
    
    for k, v in models.items():
        models[k]._apply(pin_memory)

    statis = []
    for name, model in models.items():
        params = count_parameters(model)
        for batch in range(1, 5):
            rdm_input = torch.randn(batch, 3, 384, 640).to('cuda')
            for i in tqdm(range(NUM)):
                t1 = time_sync()
                # model.to('cuda')
                model._apply(to_cuda)
                t2 = time_sync()
                y = model(rdm_input)
                t3 = time_sync()
                # model.to('cpu')
                model._apply(to_cpu)
                match_pair.clear()
                # torch.cuda.empty_cache()
                t4 = time_sync()
                loading[i] = 1000 * (t2 - t1)
                inference[i] = 1000 * (t3 - t2)
                unloading[i] = 1000 * (t4 - t3)
                # print(f'single round: loading {loading[i]}ms, infer {inference[i]}ms, unloading {unloading[i]}ms')
            rdm_input = rdm_input.cpu()
            torch.cuda.empty_cache()
            
            ldt = float('%.2f' % (sum(loading) / NUM))
            ift = float('%.2f' % (sum(inference) / NUM))
            uldt = float('%.2f' % (sum(unloading) / NUM))
            sumt = float('%.2f' % (ldt + ift + uldt))
            infer_pcent = float('%.2f' % (ift / sumt * 100))
            statis.append([name, ldt, ift, uldt, sumt, infer_pcent, params, batch, 0, 0, 0])
        
            print(f'{name} loading avg {ldt}ms, infer avg {ift}ms, unloading avg : {uldt}ms')
            print(f'***** sleep 120s to cold down *****')
            time.sleep(120)
            
    df = pd.DataFrame(np.asarray(statis),
                    columns=('model', 'load(ms)', 'inference(ms)', 'unload(ms)', 'sum(ms)', 'infer percent(%)', 
                             'params', 'batchsize', 'memory(MB)', 'model&data(MB)', 'gpu_utilization(%)'))
    df.to_csv('profiling_results.csv', index=False, float_format='%.3f')
        

    
    