import torch
import torch.nn as nn
import torch.nn.functional as F

#vgg definition that conveniently let's you grab the outputs from any layer
class Vgg16_4x(nn.Module):
    def __init__(self, model_path, num_classes=1000, pad=4):
        super(Vgg16_4x, self).__init__()
        #modules
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=3, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.style_layers = ['r11','r21','r31','r41', 'r51'] 
        self.content_layers = ['r42']
        self.content_weights = [1e0]
        #these are good initial weights settings:
        self.style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['state_dict']
        
        state_dict['features.0.weight'] = state_dict['features.module.0.weight']
        del state_dict['features.module.0.weight']
        state_dict['features.0.bias'] = state_dict['features.module.0.bias']
        del state_dict['features.module.0.bias']
        state_dict['features.2.weight'] = state_dict['features.module.2.weight']
        del state_dict['features.module.2.weight']
        state_dict['features.2.bias'] = state_dict['features.module.2.bias']
        del state_dict['features.module.2.bias']
        state_dict['features.5.weight'] = state_dict['features.module.5.weight']
        del state_dict['features.module.5.weight']
        state_dict['features.5.bias'] = state_dict['features.module.5.bias']
        del state_dict['features.module.5.bias']
        state_dict['features.7.weight'] = state_dict['features.module.7.weight']
        del state_dict['features.module.7.weight']
        state_dict['features.7.bias'] = state_dict['features.module.7.bias']
        del state_dict['features.module.7.bias']
        state_dict['features.10.weight'] = state_dict['features.module.10.weight']
        del state_dict['features.module.10.weight']
        state_dict['features.10.bias'] = state_dict['features.module.10.bias']
        del state_dict['features.module.10.bias']
        state_dict['features.12.weight'] = state_dict['features.module.12.weight']
        del state_dict['features.module.12.weight']
        state_dict['features.12.bias'] = state_dict['features.module.12.bias']
        del state_dict['features.module.12.bias']
        state_dict['features.14.weight'] = state_dict['features.module.14.weight']
        del state_dict['features.module.14.weight']
        state_dict['features.14.bias'] = state_dict['features.module.14.bias']
        del state_dict['features.module.14.bias']
        state_dict['features.17.weight'] = state_dict['features.module.17.weight']
        del state_dict['features.module.17.weight']
        state_dict['features.17.bias'] = state_dict['features.module.17.bias']
        del state_dict['features.module.17.bias']
        state_dict['features.19.weight'] = state_dict['features.module.19.weight']
        del state_dict['features.module.19.weight']
        state_dict['features.19.bias'] = state_dict['features.module.19.bias']
        del state_dict['features.module.19.bias']
        state_dict['features.21.weight'] = state_dict['features.module.21.weight']
        del state_dict['features.module.21.weight']
        state_dict['features.21.bias'] = state_dict['features.module.21.bias']
        del state_dict['features.module.21.bias']
        state_dict['features.24.weight'] = state_dict['features.module.24.weight']
        del state_dict['features.module.24.weight']
        state_dict['features.24.bias'] = state_dict['features.module.24.bias']
        del state_dict['features.module.24.bias']
        state_dict['features.26.weight'] = state_dict['features.module.26.weight']
        del state_dict['features.module.26.weight']
        state_dict['features.26.bias'] = state_dict['features.module.26.bias']
        del state_dict['features.module.26.bias']
        state_dict['features.28.weight'] = state_dict['features.module.28.weight']
        del state_dict['features.module.28.weight']
        state_dict['features.28.bias'] = state_dict['features.module.28.bias']
        del state_dict['features.module.28.bias']
        del state_dict['classifier.0.weight']
        del state_dict['classifier.0.bias']
        del state_dict['classifier.3.weight']
        del state_dict['classifier.3.bias']
        del state_dict['classifier.6.weight']
        del state_dict['classifier.6.bias']
        self.load_state_dict(state_dict)

        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
            
    def forward(self, x, out_keys):
        out = {}
        out['r11'] = self.features[1](self.features[0](x))
        out['r21'] = self.features[6](self.features[5](self.features[4](self.features[3](self.features[2](out['r11'])))))
        out['r31'] = self.features[11](self.features[10](self.features[9](self.features[8](self.features[7](out['r21'])))))
        out['r41'] = self.features[18](self.features[17](self.features[16](self.features[15](self.features[14](self.features[13](self.features[12](out['r31'])))))))
        out['r42'] = self.features[20](self.features[19](out['r41']))
        out['r51'] = self.features[25](self.features[24](self.features[23](self.features[22](self.features[21](out['r42'])))))
        
        return [out[key] for key in out_keys]
