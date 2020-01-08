import math
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from .py_utils.loss_utils import _regr_loss, _neg_loss
from torch.autograd import Variable
from .resnet_features import resnet50_features, resnet18_features, resnet101_features, resnext101_32x8d, wide_resnet101_2
from .py_utils.utils import conv1x1, conv3x3

def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x
    
class MatrixNet(nn.Module):
    def __init__(self, resnet, layers):
        super(MatrixNet, self).__init__()
         
        self.resnet = resnet
        # We first create a graph of how the layers in the 5x5 matrix are connected. 12 means row 1 column 2 and so on
        self.incidence= {33:[34,43,44,22], 34:[35], 43:[53], 44:[45,55,54], 22:[11,32,23], 11:[12,21], 32:[42], 23:[24], 12:[13], 21:[31] ,42:[52],24:[25], 13:[14], 31:[41], 14:[15], 41:[51] }
        # Visited tracks the set of nodes that need to visited 
        self.visited = set()
    
        self.layers=layers
        # keeps tracks the nodes that need to be returned (-1 in the layer_ranges implies that node is not visited)
        self.keeps=set()

        for i,l in enumerate(self.layers):
            for j,e in enumerate(l):
                if e !=-1:
                    self.keeps.add((j+1)*10+(i+1))
        # we do a BFS to find the nodes that need to be visited using the incidence list of the graph above
        def _bfs(graph, start,end):
            queue=[]
            queue.append([start])
            while queue:
                path= queue.pop(0)
                node=path[-1]
                if node == end:
                    return path
                for n in graph.get(node, []):
                    new_path = list(path)
                    new_path.append(n)
                    queue.append(new_path)

        _keeps=self.keeps.copy()
        
        while _keeps:
            node=_keeps.pop()
            vs = set(_bfs(self.incidence, 33, node)) #for us start is at 33  as that's the first node
            self.visited = vs | self.visited
            _keeps = _keeps - self.visited


        # applied in a pyramid
        self.pyramid_transformation_3 = conv1x1(512, 256)
        self.pyramid_transformation_4 = conv1x1(1024, 256)
        self.pyramid_transformation_5 = conv1x1(2048, 256)

        # both based around resnet_feature_5
        self.pyramid_transformation_6 = conv3x3(2048, 256, padding=1, stride=2)
        self.pyramid_transformation_7 = conv3x3(256, 256, padding=1, stride=2)

        # applied after upsampling
        self.upsample_transform_1 = conv3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3(256, 256, padding=1)

        self.downsample_transformation_12 = conv3x3(256, 256, padding=1, stride=(1,2))
        self.downsample_transformation_21 = conv3x3(256, 256, padding=1, stride=(2,1))


    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        height, width = scaled_feature.size()[2:]
        return F.interpolate(original_feature, scale_factor=scale_factor)[:, :, :height, :width]

    def forward(self, x):
        # don't need resnet_feature_2 as it is too large
        _, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)
        
        _dict={}
        
        # we only update the layers in self.visited

        if 44 in self.visited:
            _dict[44] = self.pyramid_transformation_6(resnet_feature_5)
            
        if 55 in self.visited:
            _dict[55]=self.pyramid_transformation_7(F.relu(_dict[44]))

        if 33 in self.visited:
            _dict[33] = self.pyramid_transformation_5(resnet_feature_5)

        if 22 in self.visited:
            _dict[22] = self.pyramid_transformation_4(resnet_feature_4)
        
        if 33 in self.visited and 22 in self.visited:
            upsampled_feature_5 = self._upsample(_dict[33], _dict[22])
        
        if 22 in self.visited:
            _dict[22] = self.upsample_transform_1(torch.add(upsampled_feature_5, _dict[22]))
        
        if 11 in self.visited:
            _dict[11] = self.pyramid_transformation_3(resnet_feature_3)
        
        if 11 in self.visited and 22 in self.visited:
            upsampled_feature_4 = self._upsample(_dict[22], _dict[11])
        
        if 11 in self.visited:
            _dict[11] = self.upsample_transform_2(torch.add(upsampled_feature_4, _dict[11]))
        
        if 12 in self.visited:
            _dict[12] = self.downsample_transformation_12(_dict[11])
        if 13 in self.visited:
            _dict[13] = self.downsample_transformation_12(_dict[12])
        if 14 in self.visited:
            _dict[14] = self.downsample_transformation_12(_dict[13])
        if 15 in self.visited:
            _dict[15] = self.downsamole_transformation_12(_dict[14])

        if 21 in self.visited:
            _dict[21] = self.downsample_transformation_21(_dict[11])
        if 31 in self.visited:
            _dict[31] = self.downsample_transformation_21(_dict[21])
        if 41 in self.visited:
            _dict[41] = self.downsample_transformation_21(_dict[31])
        if 51 in self.visited:
            _dict[51] = self.downsample_transformation_21(_dict[41])
        
        if 23 in self.visited:
            _dict[23] = self.downsample_transformation_12(_dict[22])
        if 24 in self.visited:
            _dict[24] = self.downsample_transformation_12(_dict[23])
        if 25 in self.visited:
            _dict[25] = self.downsample_transformation_12(_dict[24])

        if 32 in self.visited:
            _dict[32] = self.downsample_transformation_21(_dict[22])
        if 42 in self.visited:
            _dict[42] = self.downsample_transformation_21(_dict[32])
        if 52 in self.visited:
            _dict[52] = self.downsample_transformation_21(_dict[42])

        if 34 in self.visited:
            _dict[34] = self.downsample_transformation_12(_dict[33])
        if 35 in self.visited:
            _dict[35] = self.downsample_transformation_12(_dict[34])
        
        if 43 in self.visited:
            _dict[43] = self.downsample_transformation_21(_dict[33])
        if 53 in self.visited:
            _dict[53] = self.downsample_transformation_21(_dict[43])
        
        if 45 in self.visited:
            _dict[45] = self.downsample_transformation_12(_dict[44])
        if 54 in self.visited:
            _dict[54] = self.downsample_transformation_21(_dict[44])
        
        #the layer ranges is defined column first so we invert the indexes for sorting

        order_keeps = {(i%10)*10+(i//10):i for i in self.keeps}
        
        #we return only the layers in self.keeps

        return [ _dict[order_keeps[i]] for i in sorted(order_keeps)]
          
            
class SubNet(nn.Module):

    def __init__(self, mode, classes=80, depth=4,
                 base_activation=F.relu,
                 output_activation=F.sigmoid):
        super(SubNet, self).__init__()
        self.classes = classes
        self.depth = depth
        self.base_activation = base_activation
        self.output_activation = output_activation

        self.subnet_base = nn.ModuleList([conv3x3(256, 256, padding=1)
                                          for _ in range(depth)])

        if mode == 'corners':
            self.subnet_output = conv3x3(256, 2, padding=1)

        if mode == 'centers':
            self.subnet_output = conv3x3(256, 2 , padding=1)
        elif mode == 'classes':
            # add an extra dim for confidence
            self.subnet_output = conv3x3(256, self.classes, padding=1)

    def forward(self, x):
        for layer in self.subnet_base:
            x = self.base_activation(layer(x))

        x = self.subnet_output(x)
        return x

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs



  
