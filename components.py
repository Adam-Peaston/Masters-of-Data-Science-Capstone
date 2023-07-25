import numpy as np
import pandas as pd
import math, time
import torch
from torch import nn
from torch.nn import init
from torchvision import models
from torchvision.ops import StochasticDepth
from tqdm import tqdm
from scipy.stats import beta
from sklearn.metrics import top_k_accuracy_score
from tabulate import tabulate
from IPython.display import clear_output

class ToRGB(object):
    """Convert greyscale images to RGB. Necessary for treating greyscale images with only one input channel."""
    def __call__(self, tensor):
        # incoming tensor is of shape [?, R, R], should expand to [3, R, R]
        return tensor.expand(3,*tensor.shape[1:])

class ConvNormAct(nn.Module):
    def __init__(self, in_chnls, out_chnls, kernel, stride):
        '''
        Input / Output: (N, chnls, H, W)
        '''
        super(ConvNormAct, self).__init__()
        # First point-wise convolution
        self.conv = nn.Conv2d(
            in_channels = in_chnls, 
            out_channels = out_chnls, 
            kernel_size = kernel,
            padding = int((kernel - 1) / 2), 
            stride = stride,
            bias = False
        )
        self.bn = nn.BatchNorm2d(out_chnls)
        self.act = nn.SiLU()

        self.layer = nn.Sequential(
            self.conv,
            self.bn,
            self.act
        )

        self.conv.reset_parameters()

    def forward(self, x):
        z = self.layer(x)
        return z

class SEBlock(nn.Module):
    def __init__(self, chnls, bneck_r, bmin=2):
        '''
        Input / Output: (N, chnls, H, W)
        bneck_r: bottleneck ratio of fully connected layers
        bmin: min hidden layer size
        '''
        super(SEBlock, self).__init__()
        self.chnls = chnls
        bottleneck = max(int(chnls // bneck_r), bmin)
        self.sqz = nn.AdaptiveAvgPool2d(1)
        self.w0 = nn.Conv2d(in_channels=chnls, out_channels=bottleneck, kernel_size=1, stride=1)
        self.w1 = nn.Conv2d(in_channels=bottleneck, out_channels=chnls, kernel_size=1, stride=1)
        self.act = nn.SiLU()
        self.scale_act = nn.Sigmoid()
        self.layer = nn.Sequential(self.sqz, self.w0, self.act, self.w1, self.scale_act)

        # Initialize weights
        self.w0.reset_parameters()
        self.w1.reset_parameters()
    
    def forward(self, x):
        z = self.layer(x)
        return x * z.expand_as(x)


class MBConv(nn.Module):
    def __init__(self, in_chnls, out_chnls, expansion, kernel, stride, bneck_r, sd_prob, bmin=2):
        '''
        Input / Output: (N, C, H, W)
        expansion: channel inner expansion ratio
        kernel: convolutional kernel size, assume always odd integer
        bneck_r: bottleneck ratio of fully connected layers
        '''
        super(MBConv, self).__init__()
        # Global resources
        # self.act = nn.ReLU()
        self.act = nn.SiLU()
        exp_chnls = int(in_chnls * expansion)
        
        # First point-wise convolution
        self.conv0 = nn.Conv2d(
            in_channels = in_chnls, 
            out_channels = exp_chnls, 
            kernel_size = 1, 
            bias = False
        )
        self.bn0 = nn.BatchNorm2d(exp_chnls)
        
        # Depth-wise convolution
        self.conv1 = nn.Conv2d(
            in_channels = exp_chnls, 
            out_channels = exp_chnls, 
            kernel_size = kernel,
            stride = stride,
            groups = exp_chnls,
            padding = int((kernel - 1) / 2), 
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(exp_chnls)
        
        # Squeese-Excitation
        self.se = SEBlock(exp_chnls, bneck_r, bmin)
        
        # Second point-wise convolution
        self.conv2 = nn.Conv2d(
            in_channels = exp_chnls, 
            out_channels = out_chnls, 
            kernel_size = 1, 
            bias = False
        )
        self.bn2 = nn.BatchNorm2d(out_chnls) # Maybe
        
        self.layer = nn.Sequential(
            self.conv0,
            self.bn0,
            self.act,
            self.conv1,
            self.bn1,
            self.act,
            self.se,
            self.conv2,
            self.bn2
            # Note no final ReLU per MobileNetV2 paper.
        )
        
        # Implementing stochastic depth
        self.stochastic_depth = StochasticDepth(sd_prob, "row")
        
        # Init parameters
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
    def forward(self, x):
        z = self.layer(x)
        if x.shape == z.shape: # Apply skip connection by default for matching input/output tensor shapes.
            z = self.stochastic_depth(z) # Apply stochasic depth
            z += x
        return z

class FusedMBConv(nn.Module):
    def __init__(self, in_chnls, out_chnls, expansion, kernel, stride, bneck_r, sd_prob, bmin=2):
        '''
        Input / Output: (N, C, H, W)
        expansion: channel inner expansion ratio
        kernel: convolutional kernel size
        bneck_r: bottleneck ratio of fully connected layers
        '''
        super(FusedMBConv, self).__init__()
        # Global resources
        # self.act = nn.ReLU()
        self.act = nn.SiLU()
        exp_chnls = int(in_chnls * expansion)
        
        # First ordinary convolution
        self.conv0 = nn.Conv2d(
            in_channels = in_chnls, 
            out_channels = exp_chnls, 
            kernel_size = kernel, 
            stride = stride,
            padding = int((kernel - 1) / 2), 
            bias = False
        )
        self.bn0 = nn.BatchNorm2d(exp_chnls)
        
        # # Squeese-Excitation
        # self.se = SEBlock(exp_chnls, bneck_r, bmin)
        
        # Second point-wise convolution
        self.conv1 = nn.Conv2d(
            in_channels = exp_chnls, 
            out_channels = out_chnls, 
            kernel_size = 1, 
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(out_chnls) # Maybe
        
        self.layer = nn.Sequential(
            self.conv0,
            self.bn0,
            self.act,
            # self.se,
            self.conv1,
            self.bn1
            # Note no final ReLU per MobileNetV2 paper.
        )
        
        # Implementing stochastic depth
        self.stochastic_depth = StochasticDepth(sd_prob, "row")
        
        # Init parameters
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        
    def forward(self, x):
        z = self.layer(x)
        if x.shape == z.shape: # Apply skip connection by default for matching input/output tensor shapes.
            z = self.stochastic_depth(z)
            z += x
        return z



class BetaNet2(nn.Module):
    def __init__(self, R, k, K, Da, Db, Wa, Wb, G0, G1, B, E0, E1, S0, S1, SD0, SD1, dp, output_size):
        super(BetaNet2, self).__init__()
        self.dropout = dp
        
        # Determine L based on R - recall L describes the number of spatial contraction steps (halvings of spatial resoltion).
        # Stride 2 convolutions will be used to halve the spatial resolution until the spatial resolution is less than 2^k.
        L = self.half_ceilings(R, 2**k)

        # Determine model depth based on L and K
        nlayers = int(K * L)
        layer_locs = np.linspace(0, 1, 2*nlayers+1)[1::2] # describes the proportion of the network depth each layer sits at.

        # Description of each layer in the network.
        # type, in_chnls, out_chnls, expansion, kernel, stride, bneck_r
        self.layer_specs = {
            l:{
                'type':'FMBC', 
                'in_chnls':0, 
                'out_chnls':0, 
                'expansion':0, 
                'kernel':3, 
                'stride':1, 
                'bneck_r':0,
                'sd_prob':0,
            } for l in range(nlayers)
        }
        
        # Trying different parameterisation as beta distribution.
        # Da and Db floats constrained to interval from [-1,1] as log distribution of beta parameters looks better.
        D_dist = beta(10**Da, 10**Db) # Defining the beta distribution
        spec = D_dist.ppf(np.linspace(0, 1, L+1))[1:-1] # Computing the CDF at points corresponding to L.

        # Generate all valid reduction layer proposals for L-1 layers
        proposals = self.generate_proposals(nlayers-1, L-1)
        # Describe layer locs for the first layer onward.
        layer_locs_ = np.linspace(0, 1, 2*(nlayers-1)+1)[1::2]
        # Find the proposal best matching the specification
        min_delta = np.array([self.delta(prop, layer_locs_, spec) for prop in proposals]).argmin()
        # Define the reduction layers based on best matching proposal, putting the zero'th spec back.
        reduction_steps = [prop+1 for prop in proposals[min_delta]] + [0]
        
        # Use G to work out the block channels
        # input_size = 3 * R**2 # Assume RGB input
        self.output_channels = int(G0 * R) # As opposed to G * input_size (?)
        self.G1 = G1
        
        # Beta-distribution parameterization of width scaling
        W_dist = beta(10**Wa, 10**Wb) # Defining the beta distribution
        resolutions = np.array([math.ceil(R*0.5**i) for i in range(0,L+1)]) # Resolutions determined by halvings of max resolution
        res = np.log2(resolutions) # Transform to base log2
        max_res = math.log(R, 2) # Interpolation boundary
        min_res = math.log(resolutions[-1], 2)
        x = ((res - max_res)/(min_res - max_res)) * (1 - 0) + 0
        y = W_dist.cdf(x)
        channel_sizes = ((y - 0)/(1 - 0)) * (self.output_channels - 3) + 3
        channel_sizes = channel_sizes.astype(int)
        channel_steps = list(zip(channel_sizes[:-1],channel_sizes[1:]))
        
        # First layer is of type ConvNormAct
        self.layer_specs[0]['type'] = 'CNA'
        for layer in range(nlayers):
            if layer in reduction_steps: # The first layer is always a reduction layer
                step = channel_steps.pop(0)
                self.layer_specs[layer]['stride'] = 2
                self.layer_specs[layer]['in_chnls'] = step[0]
                self.layer_specs[layer]['out_chnls'] = step[1]
            else:
                self.layer_specs[layer]['in_chnls'] = step[1]
                self.layer_specs[layer]['out_chnls'] = step[1]
                
            # Apply transition to MBConv
            if layer_locs[layer] >= B:
                self.layer_specs[layer]['type'] = 'MBC'
                
        # Apply Stochastic-depth probability gradient
        for layer in range(nlayers):
            if layer not in reduction_steps: # Only apply if simple residual
                prob = (layer / (nlayers - 1))*(SD1 - SD0) + SD0
                self.layer_specs[layer]['sd_prob'] = prob

            # Compute interpolated E and S
            E = (layer / (nlayers - 1))*(E1 - E0) + E0
            self.layer_specs[layer]['expansion'] = E
            S = (layer / (nlayers - 1))*(S1 - S0) + S0
            self.layer_specs[layer]['bneck_r'] = S

        # Pass layer specs to build model function.
        # This way we can also pass model specs to this function directly after instantiation if we like.
        self.build_model(self.layer_specs, output_size)
                
    def build_model(self, layer_specs, output_size):
        # Construct network
        self.model = nn.Sequential()

        for l in range(len(layer_specs)):
            spec = layer_specs[l]
            if spec['type'] == 'CNA':
                layer = ConvNormAct(
                    in_chnls = spec['in_chnls'], 
                    out_chnls = spec['out_chnls'], 
                    kernel = spec['kernel'], 
                    stride = spec['stride'], 
                )
                self.model.append(layer)
            if spec['type'] == 'FMBC':
                layer = FusedMBConv(
                    in_chnls = spec['in_chnls'], 
                    out_chnls = spec['out_chnls'], 
                    expansion = spec['expansion'], 
                    kernel = spec['kernel'], 
                    stride = spec['stride'], 
                    bneck_r = spec['bneck_r'],
                    sd_prob = spec['sd_prob']
                )
                self.model.append(layer)
            elif spec['type'] == 'MBC':
                layer = MBConv(
                    in_chnls = spec['in_chnls'], 
                    out_chnls = spec['out_chnls'], 
                    expansion = spec['expansion'], 
                    kernel = spec['kernel'], 
                    stride = spec['stride'], 
                    bneck_r = spec['bneck_r'],
                    sd_prob = spec['sd_prob']
                )
                self.model.append(layer)

        # Final Conv1x1
        latent_size = int(self.G1 * self.output_channels)
        self.model.append(
            ConvNormAct(
                in_chnls = self.output_channels, 
                out_chnls = latent_size, # Parameterise?
                kernel = 1, 
                stride = 1
                )
            )

        # Average pool final feature map over residial spatial ranges.
        self.model.append(nn.AdaptiveAvgPool2d(1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout, inplace=False),
            nn.Linear(latent_size, output_size, bias=True)
        )
        
    ## - HELPER FUNCTIONS - ##

    def half_ceilings(self, R, k):
        L = 0
        while R > k:
            R = math.ceil(R * 0.5)
            L += 1
        return L

    def nextperm(self, L, M):
        # permutes list of integers L given max val M
        # e.g. [3,2,1,0] --> [4,2,1,0]
        for i, v in enumerate(L):
            if v < M - i - 1:    
                L[i] += 1    
                for j in range(i):     
                     L[j] = L[i] + i - j     
                return L 
        return None

    def generate_proposals(self, M, S):
        '''
        Authors note: It was only after I wrote this algorithm that I realised that I could probably have just made use of the itertools.permutations e.g.
        R = itertools.permutations(range(M), r=S), but it was fun coming up with this algorithm anyway.
        '''
        # generates list of index tuples.
        # based on total model layers M
        # and number of step layers S
        L = list(range(S-1, -1, -1))
        R = []
        while L:
            R.append(tuple(L))
            L = self.nextperm(L, M)  
        return R

    def delta(self, prop, locs, spec):
        prop = sorted(prop)
        prop_locs = [locs[p] for p in prop]
        return sum([abs(pl-s) for pl,s in zip(prop_locs, spec)])
        
    def forward(self, x):
        z = self.model(x).flatten(start_dim=1)
        return self.classifier(z)

class BetaNet3(nn.Module):
    def __init__(self, R, k, K, Da, Db, WLa, WLb, WCa, WCb, G0, G1, B, E0, E1, S0, S1, SD0, SD1, dp, output_size):
        super(BetaNet3, self).__init__()
        self.dropout = dp
        
        # Determine L based on R - recall L describes the number of spatial contraction steps (halvings of spatial resoltion).
        # Stride 2 convolutions will be used to halve the spatial resolution until the spatial resolution is less than 2^k.
        L = self.half_ceilings(R, 2**k)

        # Determine model depth based on L and K
        nlayers = int(K * L)
        layer_locs = np.linspace(0, 1, 2*nlayers+1)[1::2] # describes the proportion of the network depth each layer sits at.

        # Description of each layer in the network.
        # type, in_chnls, out_chnls, expansion, kernel, stride, bneck_r
        self.layer_specs = {
            l:{
                'type':'FMBC', 
                'in_chnls':0, 
                'out_chnls':0, 
                'expansion':0, 
                'kernel':3, 
                'stride':1, 
                'bneck_r':0,
                'sd_prob':0,
            } for l in range(nlayers)
        }
        
        # Generate layer proposals once and reuse for resolution and channel adjustment layer specifications
        layer_proposals = self.generate_proposals(nlayers-1, L-1)
        layer_locs_ = np.linspace(0, 1, 2*(nlayers-1)+1)[1::2]

        # Da and Db floats constrained to interval from [-1,1] as log distribution of beta parameters looks better.
        D_dist = beta(10**Da, 10**Db) # Defining the beta distribution
        spec = D_dist.ppf(np.linspace(0, 1, L+1))[1:-1] # Computing the CDF at number of points corresponding to L.
        min_delta = np.array([self.delta(prop, layer_locs_, spec) for prop in layer_proposals]).argmin()
        contraction_steps = [prop+1 for prop in layer_proposals[min_delta]] + [0]
        
        # Beta-distribution parameterization of width change layers
        WL_dist = beta(10**WLa, 10**WLb) # Defining the beta distribution
        spec = WL_dist.ppf(np.linspace(0, 1, L+1))[1:-1] # Computing the CDF at number of points corresponding to L.
        min_delta = np.array([self.delta(prop, layer_locs_, spec) for prop in layer_proposals]).argmin()
        widening_steps = [prop+1 for prop in layer_proposals[min_delta]] + [0]
        widening_steps = list(set(contraction_steps) | set(widening_steps)) # Also adjust width at spatial contractions 

        # Use G to work out the block channels
        # input_size = 3 * R**2 # Assume RGB input
        self.output_channels = int(G0 * R) # As opposed to G * input_size (?)
        self.G1 = G1 # Latent space multiple of output_channels of final spatially extended feature map

        # Beta-distribution parameterization of width as function of depth
        WC_dist = beta(10**WCa, 10**WCb) # Defining the beta distribution
        widening_step_locs = layer_locs[widening_steps]
        y = WC_dist.cdf(widening_step_locs)
        self.channel_sizes = ((y - 0)/(1 - 0)) * (self.output_channels - 3) + 3
        self.channel_sizes = self.channel_sizes.astype(int)
        channel_steps = [(3,self.channel_sizes[0])] + list(zip(self.channel_sizes[:-1],self.channel_sizes[1:]))
        
        # First layer is of type ConvNormAct
        self.layer_specs[0]['type'] = 'CNA'
        for layer in range(nlayers):
            if layer in widening_steps: # The first layer is always a reduction layer
                step = channel_steps.pop(0)
                self.layer_specs[layer]['in_chnls'] = step[0]
                self.layer_specs[layer]['out_chnls'] = step[1]
                if layer in contraction_steps:
                    self.layer_specs[layer]['stride'] = 2
            else:
                self.layer_specs[layer]['in_chnls'] = step[1]
                self.layer_specs[layer]['out_chnls'] = step[1]
                
            # Apply transition to MBConv
            if layer_locs[layer] >= B:
                self.layer_specs[layer]['type'] = 'MBC'
                
        # Apply Stochastic-depth probability gradient
        for layer in range(nlayers):
            if layer not in widening_steps: # Only apply if simple residual
                prob = (layer / (nlayers - 1))*(SD1 - SD0) + SD0
                self.layer_specs[layer]['sd_prob'] = prob

            # Compute interpolated E and S
            E = (layer / (nlayers - 1))*(E1 - E0) + E0
            self.layer_specs[layer]['expansion'] = E
            S = (layer / (nlayers - 1))*(S1 - S0) + S0
            self.layer_specs[layer]['bneck_r'] = S

        # Pass layer specs to build model function.
        # This way we can also pass model specs to this function directly after instantiation if we like.
        self.build_model(self.layer_specs, output_size)
                
    def build_model(self, layer_specs, output_size):
        # Construct network
        self.model = nn.Sequential()
        for l in range(len(layer_specs)):
            spec = layer_specs[l]
            if spec['type'] == 'CNA':
                layer = ConvNormAct(
                    in_chnls = spec['in_chnls'], 
                    out_chnls = spec['out_chnls'], 
                    kernel = spec['kernel'], 
                    stride = spec['stride'], 
                )
                self.model.append(layer)
            if spec['type'] == 'FMBC':
                layer = FusedMBConv(
                    in_chnls = spec['in_chnls'], 
                    out_chnls = spec['out_chnls'], 
                    expansion = spec['expansion'], 
                    kernel = spec['kernel'], 
                    stride = spec['stride'], 
                    bneck_r = spec['bneck_r'],
                    sd_prob = spec['sd_prob']
                )
                self.model.append(layer)
            elif spec['type'] == 'MBC':
                layer = MBConv(
                    in_chnls = spec['in_chnls'], 
                    out_chnls = spec['out_chnls'], 
                    expansion = spec['expansion'], 
                    kernel = spec['kernel'], 
                    stride = spec['stride'], 
                    bneck_r = spec['bneck_r'],
                    sd_prob = spec['sd_prob']
                )
                self.model.append(layer)

        # Final Conv1x1
        latent_size = int(self.G1 * self.output_channels)
        self.model.append(
            ConvNormAct(
                in_chnls = self.channel_sizes[-1], 
                out_chnls = latent_size, # Parameterise?
                kernel = 1, 
                stride = 1
                )
            )

        # Average pool final feature map over residial spatial ranges.
        self.model.append(nn.AdaptiveAvgPool2d(1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout, inplace=False),
            nn.Linear(latent_size, output_size, bias=True)
        )
        
    ## - HELPER FUNCTIONS - ##

    def half_ceilings(self, R, k):
        L = 0
        while R > k:
            R = math.ceil(R * 0.5)
            L += 1
        return L

    def nextperm(self, L, M):
        # permutes list of integers L given max val M
        # e.g. [3,2,1,0] --> [4,2,1,0]
        for i, v in enumerate(L):
            if v < M - i - 1:    
                L[i] += 1    
                for j in range(i):     
                     L[j] = L[i] + i - j     
                return L 
        return None

    def generate_proposals(self, M, S):
        '''
        Authors note: It was only after I wrote this algorithm that I realised that I could probably have just made use of the itertools.permutations e.g.
        R = itertools.permutations(range(M), r=S), but it was fun coming up with this algorithm anyway.
        '''
        # generates list of index tuples.
        # based on total model layers M
        # and number of step layers S
        L = list(range(S-1, -1, -1))
        R = []
        while L:
            R.append(tuple(L))
            L = self.nextperm(L, M)  
        return R

    def delta(self, prop, locs, spec):
        prop = sorted(prop)
        prop_locs = [locs[p] for p in prop]
        return sum([abs(pl-s) for pl,s in zip(prop_locs, spec)])
        
    def forward(self, x):
        z = self.model(x).flatten(start_dim=1)
        return self.classifier(z)

def topk_accuracy(preds, targs, topk=1, normalize=True):
    topk_preds = preds.argsort(axis=1, descending=True)[:,:topk]
    topk_accurate = np.array([[t in p] for t,p in zip(targs,topk_preds)])
    if normalize:
        return topk_accurate.sum() / len(targs)
    else:
        return topk_accurate.sum()


class EffNetV1(nn.Module):
    def __init__(self, output_size):
        super(EffNetV1, self).__init__()
        self.model = models.efficientnet_b0() # EffNetV1 Model
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_ftrs, output_size))
        # Reset and re-initialize weights same as ViolinModel
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                layer.reset_parameters()
            elif isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        # Self assessment
        # params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, x):
        return self.model(x)

class EffNetV2(nn.Module):
    def __init__(self, output_size):
        super(EffNetV2, self).__init__()
        self.model = models.efficientnet_v2_s() # EffNetV2 Model
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_ftrs, output_size))
        # Reset and re-initialize weights same as ViolinModel
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                layer.reset_parameters()
            elif isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        # Self assessment
        # params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, x):
        return self.model(x)

class MobileNetV2(nn.Module):
    def __init__(self, output_size):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2()
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_ftrs, output_size))
        # Reset and re-initialize weights same as ViolinModel
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                layer.reset_parameters()
            elif isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        # Self assessment
        # params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module):
    def __init__(self, output_size):
        super(ResNet50, self).__init__()
        self.model = models.resnet50()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_ftrs, output_size))
        # Reset and re-initialize weights same as ViolinModel
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                layer.reset_parameters()
            elif isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        # Self assessment
        # params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, x):
        return self.model(x)