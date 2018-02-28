import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from loss_scaler import DynamicLossScaler, LossScaler

FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)

def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val` is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn =  [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn

def fp32_to_fp16(val):
    """Convert fp32 `val` to fp16"""
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, FLOAT_TYPES):
            val = val.half()
        return val
    return conversion_helper(val, half_conversion)

def fp16_to_fp32(val):
    """Convert fp16 `val` to fp32"""
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, HALF_TYPES):
            val = val.float()
        return val
    return conversion_helper(val, float_conversion)

class FP16_Module(nn.Module):
    def __init__(self, module):
        super(FP16_Module, self).__init__()
        self.add_module('module', module.half())

    def forward(self, *inputs, **kwargs):
        return fp16_to_fp32(self.module(*(fp32_to_fp16(inputs)), **kwargs))

class FP16_Optimizer(object):
    """Wrapper for PyTorch optimizers that enables
       FP16 training with FP32 weights"""
    def __init__(self, optimizer, *args, **kwargs):
        if not torch.cuda.is_available:
            raise SystemError('Cannot use fp16 without CUDA')

        self.optimizer = optimizer
        self.state = optimizer.state
        self.param_groups = optimizer.param_groups

        self.fp16_params = []
        self.fp32_params = []
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                self.fp16_params.append(param)
                fp32_param = param
                if isinstance(fp32_param.data, HALF_TYPES):
                    fp32_param = param.clone().float().detach()
                fp32_param.requires_grad = param.requires_grad
                self.fp32_params.append(fp32_param)
                group['params'][i] = fp32_param

        if 'loss_scaler' in kwargs and kwargs['loss_scaler'] is not None:
            self.loss_scaler = kwargs['loss_scaler']
        elif 'dynamic_scale' in kwargs and kwargs['dynamic_scale']: 
            self.loss_scaler = DynamicLossScaler()
        else:
            scale = kwargs['scale'] if 'scale' in kwargs else 1
            self.loss_scaler = LossScaler(scale)

        self.overflow = False

    def zero_grad(self):
        # clear fp32 parameter grads
        self.optimizer.zero_grad()
        # clear fp16 parameter grads
        for p in self.fp16_params:
            if p.grad is not None:
                p.grad.detach_() # This does appear in torch.optim.optimizer.zero_grad(), but I'm not
                                 # sure why it's needed.
                p.grad.zero_()

    def check_overflow(self, fp16_params=None):
        if fp16_params is None:
            fp16_params = self.fp16_params

        if isinstance(fp16_params, list):
            fp16_params = list(fp16_params)

        has_overflow = self.loss_scaler.has_overflow(fp16_params)
        return has_overflow

    def update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    def copy_grads_fp16_to_fp32(self, fp16_params=None, fp32_params=None):
        if fp16_params is None:
            fp16_params = self.fp16_params

        if fp32_params is None:
            fp32_params = self.fp32_params

        if isinstance(fp16_params, list):
            assert isinstance(fp32_params, list) and len(fp16_params) == len(fp32_params)
        else:
            fp16_params = list(fp16_params)
            if not isinstance(fp32_params, list):
                fp32_params = list(fp32_params)

        for i, param in enumerate(fp16_params):
            if param.grad is None:
                continue
            fp32_param = fp32_params[i]
            fp32_param._grad = param.grad.clone().type_as(fp32_param).detach()
            
        return fp32_params

    def downscale_fp32(self, fp32_params=None):
        if fp32_params is None:
            fp32_params = self.fp32_params
        
        if not isinstance(fp32_params, list):
            fp32_params = list(p32_params)

        for param in fp32_params:
            param._grad.mul_(1./self.loss_scale)

        return fp32_params

    def clip_fp32_grads(self, fp32_params=None, clip=-1):
        if fp32_params is None:
            fp32_params = self.fp32_params

        if not isinstance(fp32_params, list):
            fp32_params = list(p32_params)

        if clip > 0:
            torch.nn.utils.clip_grad_norm(fp32_params, clip)

        return fp32_params

    def copy_params_fp32_to_fp16(self, fp16_params=None, fp32_params=None):
        if fp16_params is None:
            fp16_params = self.fp16_params

        if fp32_params is None:
            fp32_params = self.fp32_params

        if isinstance(fp16_params, list):
            assert isinstance(fp32_params, list) and len(fp16_params) == len(fp32_params)
        else:
            fp16_params = list(fp16_params)
            if not isinstance(fp32_params, list):
                fp32_params = list(fp32_params)

        for i, param in enumerate(fp32_params):
            fp16_param = fp16_params[i]
            fp16_param.data.copy_(param.data)
            
        return fp16_params

    def state_dict(self):
        sd = self.optimizer.state_dict()
#        sd['loss_scaler'] = self.loss_scaler
        return sd

    def load_state_dict(self, state_dict):
#        if 'loss_scaler' in state_dict:
#            self.loss_scaler = state_dict['loss_scaler']
#            state_dict = dict(state_dict)
#            del state_dict['loss_scaler']
        self.optimizer.load_state_dict(state_dict)

    def step(self, closure=None): # could add clip option.
        """
        If no closure is supplied, step() should be called after self.backward(loss).
        If no overflow, optionally clips fp32 gradients, updates fp32 weights 
        with normal optimizer, then copies updated weights to fp16.

        If a closure is supplied, step() 
        """
        has_overflow = self.check_overflow()
        self.overflow = has_overflow
        self.update_scale(has_overflow)

        if has_overflow:
            scale = self.loss_scaler.loss_scale
            print("OVERFLOW! Not taking step. loss scale: {}".format(scale))
            return
        
        if closure is not None:
            self.step_with_closure(closure)
        else:
            # fp32_params = self.clip_fp32(fp32_params, clip=clip)
            self.optimizer.step()

        self.copy_params_fp32_to_fp16()

        return

    def step_with_closure(self, closure):
        def wrapped_closure():
            # This memcpy is redundant the first time wrapped_closure is called within
            # self.optimizer.step() below, but is necessary for subsequent calls.
            self.copy_params_fp32_to_fp16()
            temp_loss = closure() 
            # Our backward() call is already set up to copy fp16 gradients
            # into fp32 gradients.
            return temp_loss
        self.optimizer.step(wrapped_closure)

    def backward(self, loss):
        # Convert to float to provide additional safety against overflow
        # when we multiply by the scale.  Ideally, the loss criterion should 
        # already be computed in float, but I think we have to leave that to the user.
        self.loss_scaler.backward(loss.float())
        fp32_params = self.copy_grads_fp16_to_fp32()
        fp32_params = self.downscale_fp32(fp32_params)

    @property
    def loss_scale(self):
        return self.loss_scaler.loss_scale
