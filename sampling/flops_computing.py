
'''
This file is aimed at layerwise analysis of computing complexity
'''

import torch
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet152, resnet101
from torchvision.models import vgg16, vgg13, vgg11, alexnet, squeezenet1_0, squeezenet1_1
from nnutils.cnn_complexity_analyzer.profile import Profiler

class ForwardModule(torch.nn.Module):
    def __init__(self,module):
        super(ForwardModule,self).__init__()
        self.module=module

    def forward(self,x):
        return self.module(x)

class Model:
    def __init__(self, model) -> None:
        self.model = model.eval()
        self.img = torch.randn(1, 3, 384, 640)
        self.module_pair = list()
        self.flops = []
        
    def run(self):
        handler = torch.nn.modules.module.register_module_forward_pre_hook(self.measure_flops_hook())
        self.model(self.img)
        handler.remove()
        for module,shape in self.module_pair:
            self.flops.append(2 * Profiler(ForwardModule(module), {'x':torch.randn(shape)}).profile().macs)
    
    def measure_flops_hook(self):      
        def forward_pre_hook(module, data_input):
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                assert len(data_input) == 1
                self.module_pair.append([module, data_input[0].shape])
        return forward_pre_hook


if __name__ == '__main__':
    models = [resnet18(), resnet34(), resnet50(), resnet152(), resnet101(), 
              vgg16(), vgg13(), vgg11(), alexnet(), squeezenet1_0(), squeezenet1_1()]
    
    mods = [Model(m) for m in models]
    for m in mods:
        m.run()
        print(sum(m.flops))
        
    
    




