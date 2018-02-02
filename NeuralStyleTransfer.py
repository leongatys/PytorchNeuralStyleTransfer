import time
import os
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from collections import OrderedDict
import argparse

from vgg11 import Vgg11
from vgg16 import Vgg16
from vgg19 import Vgg19


#TODO: parameterize
style_layers = ['r11','r21','r31','r41', 'r51'] 
content_layers = ['r42']
loss_layers = style_layers + content_layers


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename, format='JPEG', subsampling=0, quality=100)

def create_parser():
    arg_parser = argparse.ArgumentParser(description="parser for neural-style")
    arg_parser.add_argument("--iterations", type=int, default=200,
                                  help="number of iterations, default is 2")
    arg_parser.add_argument("--batch-size", type=int, default=1,
                                  help="batch size for training, default is 1")
    arg_parser.add_argument("--style-image", type=str, required=True,
                            help="style-image")
    arg_parser.add_argument("--content-image", type=str, required=True,
                            help="path to content image you want to stylize")
    arg_parser.add_argument("--output-image", type=str, default='out.jpg',
                            help="path for saving the output image")
    arg_parser.add_argument("--model-dir", type=str, default=os.getcwd() + '/Models/',
                                  help="directory for model, if model is not present in the directory it is downloaded")
    arg_parser.add_argument("--model-name", type=str, default='vgg11',
                                  help="directory for model, if model is not present in the directory it is downloaded")
    arg_parser.add_argument("--image-size", type=int, default=None,
                                  help="size of training images, default is 256 X 256")
    arg_parser.add_argument("--pyramid-levels", type=int, default=1,
                                  help="number of pyramid levels, default=1")
    arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
    arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    arg_parser.add_argument("--style-weight", type=float, default=1.0,
                                  help="weight for style-loss, default is 1.0")
    arg_parser.add_argument("--lr", type=float, default=0.5,
                                  help="learning rate, default is 0.5")
    arg_parser.add_argument("--log-interval", type=int, default=50,
                                  help="number of iterations after which the training loss is logged, default is 50")
    arg_parser.add_argument("--save-interval", type=int, default=100,
                                  help="number of iterations after which the image  is saved, default is 100")
    arg_parser.add_argument("--half", action='store_true',
                            help="if set, use fp16 (on gpu)")
    return arg_parser

#pre and post processing for images

def prep(img):

    prep_image_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                             std=[1,1,1]),
        transforms.Lambda(lambda x: x.mul_(255)),
    ])


    img = prep_image_fn(img).unsqueeze(0)
    if args.cuda > 0 and torch.cuda.is_available():
        img = img.cuda()
        if args.half:
            img = img.half()
    return Variable(img, requires_grad=True)
    
def postp(tensor): # to clip results in the range [0,1]
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                                 transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                      std=[1,1,1]),
                                 transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
    ])
    postpb = transforms.Compose([transforms.ToPILImage()])
    
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    return  postpb(t)
    
# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):      
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out * style_weight)

def compute_targets(vgg, style_image, content_image):
    #compute optimization targets
    style_targets = [GramMatrix()(A.float()).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.float().detach() for A in vgg(content_image, content_layers)]
    return style_targets + content_targets

#get network
def load_network():
    def net(x):
        return {
            'vgg11': Vgg11(),
            'vgg16': Vgg16(),
            'vgg19': Vgg19(),
        }[x]

    vgg = net(args.model_name)
    vgg.load_state_dict(torch.load(args.model_dir + args.model_name + '.pth'))
    for param in vgg.parameters():
        param.requires_grad = False
    if args.cuda and torch.cuda.is_available():
        vgg.cuda()
    if args.half:
        vgg.half()
    return vgg

# Run style transfer, complete with pre- and post- processing
def style(model, style_image, content_image, iterations):                    
    n_iter=[0]
    t0 = time.time()
    style_image = prep(style_image)
    content_image = prep(content_image)
    targets = compute_targets(model, style_image,  content_image)
    # optimizer = optim.LBFGS([content_image], args.lr)
    optimizer = optim.Adam([content_image],lr=args.lr,eps=1e-04)
    
    def closure():
        optimizer.zero_grad()
        out = model(content_image, loss_layers)
        layer_losses = [weights[a] * loss_fns[a](A.float(), targets[a]) for a,A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter[0]+=1
        if n_iter[0]%args.log_interval == 1:
            print('Iteration: %d, loss: %f time : %s'%(n_iter[0], loss.data[0], time.time()-t0))
            print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
        return loss
    while n_iter[0] <= iterations:
        optimizer.step(closure)
    return postp(content_image.data[0].float().cpu().squeeze())

# Use high quality filter for upsample
sampler = Image.BICUBIC
# Resize image so that the largest side is 'size'
def large_side(img, size):
   h = img.size[0]
   w =  img.size[1]
   if h > w:
       ls = h
   else:
       ls =  w
   if size == None or size == 0 or ls == size :
       return img
   else:
       if h > w:
           return img.resize([size, int((size*w)/h)], sampler)
       else:
           return img.resize([int((size*h)/w), size], sampler)

def pyramid_step(model, orig_style_image, content_image, step):
    global next_style_size, next_content_size
    step_decay = args.pyramid_levels-step+1
    step_iterations = int(float(args.iterations)/step_decay)
    if step == args.pyramid_levels:
        content_image = large_side(content_image, args.image_size)
        style_image = large_side(orig_style_image, args.style_size)
    else:
        content_image = content_image.resize(next_content_size, sampler)
        style_image = orig_style_image.resize(next_style_size, sampler)
    
    print('Step # %d iterations: %d style resolution: %dx%d content resolution: %dx%d'%(args.pyramid_levels - step+1, step_iterations,
                                                                            style_image.size[0], style_image.size[1],
                                                                            content_image.size[0], content_image.size[1]))
    content_image = style(model, style_image, content_image, step_iterations)

    if step > 1:
        # todo: count pyramid level weight 
        next_content_size = [int(content_image.size[0] * 1.5), int(content_image.size[1] * 1.5)]
        next_style_size = [int(style_image.size[0] * 1.5), int(style_image.size[1] * 1.5)]

        content_image = pyramid_step(model, orig_style_image, content_image, step-1)
    return content_image


def main():
    global args, cur_image_size, weights
    parser = create_parser()
    args =  parser.parse_args()
    
    model = load_network()
    
    global loss_fns, weights, style_weight
    style_weight = args.style_weight
    
    #define layers, loss functions, weights and compute optimization targets
    global loss_fns
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)   
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        
    #these are good initial weights settings:
    style_weights = [1e3/n**2 for n in [64,128,256,512,1024]]
    content_weights = [1e0]
    weights = style_weights + content_weights
            
    orig_style_image =  Image.open(args.style_image)
    orig_content_image = Image.open(args.content_image)   

    # run scale iteration here
    out_img = pyramid_step(model, orig_style_image, orig_content_image, args.pyramid_levels)  

    #display result
    out_img.save(args.output_image, format='JPEG', subsampling=0, quality=100)
    

main()
