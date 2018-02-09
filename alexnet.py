import torch
import torch.nn as nn
import torch.nn.functional as F


#vgg definition that conveniently let's you grab the outputs from any layer
class Alexnet(nn.Module):
    def __init__(self, num_classes=1000, pad = 2):
        super(Alexnet, self).__init__()
        #modules
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.style_layers = ['r11','r21','r31'] 
        self.content_layers = ['r32']
        self.style_weights = [1e3/n**2 for n in [64,128,256]]
        self.content_weights = [1e0]
    
    def forward(self, x, out_keys):
        out = {}
        out['r11'] = self.features[1](self.features[0](x))
        out['r21'] = self.features[4](self.features[3](self.features[2](out['r11'])))
        out['r31'] = self.features[7](self.features[6](self.features[5](out['r21'])))
        out['r32'] = self.features[9](self.features[8](out['r31']))
        #out['r33'] = self.features[11](self.features[10](out['r32']))
        return [out[key] for key in out_keys]
