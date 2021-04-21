import torch
import torch.nn as nn
# from torch.autograd import Variable
from utils.quaternion_layers import QuaternionConv, QuaternionLinear#, QuaternionTransposeConv #, QuaternionLinear
# from torch.nn import Parameter
from utils.QBN_Vecchi2 import QuaternionBatchNorm2d as QBatchNorm


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample,self).__init__()    
        
    def forward(self,input):
        value = input.permute(0,2,3,1)
        sh = [int(x) for x in value.shape]
        dim = len(sh[1:-1])
        
        out = (torch.reshape(value, [-1] + sh[-dim:]))
        
        for i in range(dim, 0, -1):
        
          out = torch.cat([out, torch.zeros(out.shape, device=(out.device))], i)
        
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = torch.reshape(out, out_size)

        return out.permute(0,3,1,2)
        

# def l2normalize(v, eps=1e-12):
#     return v / (v.norm() + eps)

# class github_SpectralNorm(nn.Module):
#     def __init__(self, module, name='weight', power_iterations=1):
#         super(SpectralNorm, self).__init__()
#         self.module = module
#         self.name = name
#         self.power_iterations = power_iterations
#         if not self._made_params():
#             self._make_params()

#     def _update_u_v(self):
#         u = getattr(self.module, self.name + "_u")
#         v = getattr(self.module, self.name + "_v")
#         w = getattr(self.module, self.name + "_bar")

#         height = w.data.shape[0]
#         for _ in range(self.power_iterations):
#             v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
#             u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

#         # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
#         sigma = u.dot(w.view(height, -1).mv(v))
#         setattr(self.module, self.name, w / sigma.expand_as(w))

#     def _made_params(self):
#         try:
#             u = getattr(self.module, self.name + "_u")
#             v = getattr(self.module, self.name + "_v")
#             w = getattr(self.module, self.name + "_bar")
#             return True
#         except AttributeError:
#             return False


#     def _make_params(self):
#         w = getattr(self.module, self.name)

#         height = w.data.shape[0]
#         width = w.view(height, -1).data.shape[1]

#         u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
#         v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
#         u.data = l2normalize(u.data)
#         v.data = l2normalize(v.data)
#         w_bar = Parameter(w.data)

#         del self.module._parameters[self.name]

#         self.module.register_parameter(self.name + "_u", u)
#         self.module.register_parameter(self.name + "_v", v)
#         self.module.register_parameter(self.name + "_bar", w_bar)


#     def forward(self, *args):
#         self._update_u_v()
#         return self.module.forward(*args)
    
    
    
# class SpectralNorm:
#     def __init__(self, name):
#         self.name = name

#     def compute_weight(self, module):
#         weight = getattr(module, self.name + '_orig')
#         u = getattr(module, self.name + '_u')
#         size = weight.size()
#         weight_mat = weight.contiguous().view(size[0], -1)
#         if weight_mat.is_cuda:
#             u = u.cuda()
#         v = weight_mat.t() @ u
#         v = v / v.norm()
#         u = weight_mat @ v
#         u = u / u.norm()
#         weight_sn = weight_mat / (u.t() @ weight_mat @ v)
#         weight_sn = weight_sn.view(*size)
#         #x = Variable(u.data)
#         #print('x',x)
#         return weight_sn, Variable(u.data)

#     @staticmethod
#     def apply(module, name):
#         fn = SpectralNorm(name)

#         weight = getattr(module, name)
#         del module._parameters[name]
#         module.register_parameter(name + '_orig', nn.Parameter(weight.data))
#         input_size = weight.size(0)
#         u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
#         setattr(module, name + '_u', u)
#         setattr(module, name, fn.compute_weight(module)[0])

#         module.register_forward_pre_hook(fn)

#         return fn

#     def __call__(self, module, input):
#         weight_sn, u = self.compute_weight(module)
#         setattr(module, self.name, weight_sn)
#         setattr(module, self.name + '_u', u)

def Qspectral_norm(module):

    # apply SN to each quaternion weight independently
    module = torch.nn.utils.spectral_norm(module, name='r_weight')
    module = torch.nn.utils.spectral_norm(module, name='i_weight')
    module = torch.nn.utils.spectral_norm(module, name='j_weight')
    module = torch.nn.utils.spectral_norm(module, name='k_weight')
    return module

def spectral_norm(module, name='weight'):
    module = torch.nn.utils.spectral_norm(module, name=name)
    return module

# class Spectral_Norm():
#     def __init__(self, name='weight'):
#         self.name = name
#         return
#     def __call__(self, module):
#         module = torch.nn.utils.spectral_norm(module, name=self.name)

#         return module


# class QConvSpectral_Norm():
#     def __init__(self):
#         return
#     def __call__(self, module):
#         # apply SN to each quaternion weight independently
#         module = torch.nn.utils.spectral_norm(module, name='r_weight')
#         module = torch.nn.utils.spectral_norm(module, name='i_weight')
#         module = torch.nn.utils.spectral_norm(module, name='j_weight')
#         module = torch.nn.utils.spectral_norm(module, name='k_weight')

#         return module


# def log_sum_exp(x, axis = 1):
#     m = torch.max(x, keepdim = True)
#     return m + torch.logsumexp(x - m, dim = 1, keepdim = True)

    
class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding, kernel_size = 4, stride = 2,
                spectral_normed = False):
        super(conv2d, self).__init__()

        self.conv = QuaternionConv(in_channels, out_channels, kernel_size, stride, padding = padding)
        torch.nn.init.normal_(self.conv.r_weight.data, std=0.02)
        torch.nn.init.normal_(self.conv.i_weight.data, std=0.02)
        torch.nn.init.normal_(self.conv.j_weight.data, std=0.02)
        torch.nn.init.normal_(self.conv.k_weight.data, std=0.02)
        
        if spectral_normed:
            self.conv = Qspectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out


# class deconv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, padding, kernel_size = 4, stride = 2,
#                 spectral_normed = False, iter = 1):
#         super(deconv2d, self).__init__()

#         self.deconv = QuaternionTransposeConv(in_channels, out_channels, kernel_size, stride, padding = padding)
#         torch.nn.init.normal_(self.deconv.r_weight.data, std=0.02)
#         torch.nn.init.normal_(self.deconv.i_weight.data, std=0.02)
#         torch.nn.init.normal_(self.deconv.j_weight.data, std=0.02)
#         torch.nn.init.normal_(self.deconv.k_weight.data, std=0.02)
        
#         if spectral_normed:
#             self.deconv = QConvspectral_norm(self.deconv)

#     def forward(self, input):
#         out = self.deconv(input)
#         return out    


# def conv_cond_concat(x, y):
#     x_shapes = list(x.size())
#     y_shapes = list(y.size())
#     return torch.cat((x,y*torch.ones(x_shapes[0],x_shapes[1],x_shapes[2],y_shapes[3])))


class Residual_G(nn.Module):
    def __init__(self, in_channels, out_channels = 256, kernel_size = 3, stride = 1, 
                batch_normed=False, up_sampling = False, n_classes=0):
        super(Residual_G, self).__init__()
        
        self.batch_normed = batch_normed
        self.up_sampling = up_sampling
        self.diff_dims = in_channels != out_channels or self.up_sampling

        
        self.relu = nn.ReLU()
        
        self.batch_norm1 = QBatchNorm(in_channels)
        self.batch_norm2 = QBatchNorm(out_channels)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        # self.upsample = Upsample()
        
        self.conv1 = conv2d(in_channels, out_channels,
                            kernel_size = kernel_size, stride = stride, padding = 1)
        self.conv2 = conv2d(out_channels, out_channels, 
                            kernel_size = kernel_size, stride = stride, padding = 1)

        # if classes > 0 : # implementation needed for CategoricalConditionalBatchNormalization
        #     self.b1 = CategoricalConditionalBatchNormalization(in_channels, n_cat=n_classes)
        #     self.b2 = CategoricalConditionalBatchNormalization(hidden_channels, n_cat=n_classes)
        # else:
        #     self.b1 = L.BatchNormalization(in_channels)
        #     self.b2 = L.BatchNormalization(hidden_channels)
        
        # if residual block has output features different from input features
        if self.diff_dims:    
            self.short_conv = conv2d(in_channels, out_channels,
                            kernel_size = 1, stride = stride, padding = 0)



            
    def residual(self, x):
        # input = x
        if self.batch_normed:
            x = self.relu(self.batch_norm1(x))
        else:
            x = self.relu(x)

        if self.up_sampling:
            x = self.upsample(x)
        x = self.conv1(x)
        
        if self.batch_normed:
            x = self.batch_norm2(x)
        y = self.conv2(self.relu(x))
        return y
    
    
    def shortcut(self, x):
        if self.diff_dims:
            if self.up_sampling:
                x = self.upsample(x)
            x = self.short_conv(x)
            return x
        else:
            return x
        
    def forward(self, input):
        
        return self.residual(input) + self.shortcut(input)
        



class Residual_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3, stride = 1,
                spectral_normed = False, down_sampling = False):
        super(Residual_D, self).__init__()
        
        self.down_sampling = down_sampling
        self.diff_dims = (in_channels!=out_channels) or self.down_sampling

        self.conv1 = conv2d(in_channels, in_channels, spectral_normed = spectral_normed,
                            kernel_size = kernel, stride = stride, padding = 1)

        self.conv2 = conv2d(in_channels, out_channels, spectral_normed = spectral_normed,
                            kernel_size = kernel, stride = stride, padding = 1)
    
        
        if self.diff_dims:
            self.conv_short = conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0,
                                spectral_normed = spectral_normed)
            
           
        self.avgpool_short = nn.AvgPool2d(2, 2, padding = 1)
        self.avgpool2 = nn.AvgPool2d(2, 2, padding = 1)
        self.relu = nn.ReLU()


    def shortcut(self, x):
        short = x
        if self.diff_dims:
            short = self.conv_short(short)
    
            if self.down_sampling:
                return self.avgpool_short(short)
            else:
                return short
                
        else:
            return short
        
        
    def residual(self, x):
        conv1 = self.conv1(self.relu(x))
        conv2 = self.conv2(self.relu(conv1))
        if self.down_sampling:
            conv2 = self.avgpool2(conv2)
        return conv2
        
    
    def forward(self, x):
        return self.residual(x) + self.shortcut(x)



class First_Residual_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3, stride = 1,
                spectral_normed = False):
        super(First_Residual_D, self).__init__()
        
        self.avgpool_short = nn.AvgPool2d(2, 2, padding = 1)
        
        self.conv1 = conv2d(in_channels, out_channels, spectral_normed = spectral_normed,
                            kernel_size = kernel, stride = stride, padding = 1)
    
        self.conv2 = conv2d(out_channels, out_channels, spectral_normed = spectral_normed,
                            kernel_size = kernel, stride = stride, padding = 1)
       
        self.avgpool2 = nn.AvgPool2d(2, 2, padding = 1)
        self.relu = nn.ReLU()
        
          
        self.res_conv = conv2d(in_channels, out_channels, spectral_normed = spectral_normed,
                               kernel_size = 1, stride = 1, padding = 0)
        self.conv_short = conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0,
                                spectral_normed = spectral_normed)  



    def shortcut(self, x):
        short = x
        short = self.conv_short(short)
        return self.avgpool_short(short)

        
        
    def residual(self, x):
        
        conv1 = self.relu(self.conv1(x))
        conv2 = self.conv2(conv1)
        resi = self.avgpool2(conv2)
        return resi
        
    def forward(self, x):

        return self.residual(x) + self.shortcut(x)
       
          
       
class SNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, spectral_normed = False, bias=True):
        super(SNLinear, self).__init__()

        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        torch.nn.init.normal_(self.linear.weight.data, std=0.02)
            
        if spectral_normed:
            self.linear = spectral_norm(self.linear)

    def forward(self, input):
        out = self.linear(input)
        return out
    
    
    
class QSNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, spectral_normed = False, bias=True):
        super(QSNLinear, self).__init__()
        
        self.linear = QuaternionLinear(in_channels, out_channels, bias=bias)
        torch.nn.init.normal_(self.linear.r_weight.data, std=0.02)
        torch.nn.init.normal_(self.linear.i_weight.data, std=0.02)
        torch.nn.init.normal_(self.linear.j_weight.data, std=0.02)
        torch.nn.init.normal_(self.linear.k_weight.data, std=0.02)
            
        if spectral_normed:
            self.linear = Qspectral_norm(self.linear)

    def forward(self, input):
        out = self.linear(input)
        return out    
    
