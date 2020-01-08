import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .py_utils.loss_utils import _regr_loss, _neg_loss
from torch.autograd import Variable
from .resnet_features import resnet152_features, resnet50_features, resnet18_features, resnet101_features, resnext101_32x8d, wide_resnet101_2
from .py_utils.utils import conv1x1, conv3x3
from .matrixnet import _sigmoid, MatrixNet, _gather_feat, _tranpose_and_gather_feat, _topk, _nms


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
            self.subnet_output = conv3x3(256, 4, padding=1)
            
        if mode == 'tl_corners':
            self.subnet_output = conv3x3(256, 2, padding=1)
        if mode == 'br_corners':
            self.subnet_output = conv3x3(256, 2, padding=1)
            
        elif mode == 'classes':
            # add an extra dim for confidence
            self.subnet_output = conv3x3(256, self.classes, padding=1)

    def forward(self, x):
        for layer in self.subnet_base:
            x = self.base_activation(layer(x))
        
        
        x = self.subnet_output(x)
        return x   

    
class MatrixNetAnchors(nn.Module):
    def __init__(self, classes, resnet, layers):
        super(MatrixNetAnchors, self).__init__()
        self.classes = classes
        self.resnet = resnet

        if self.resnet == "resnext101_32x8d":
            _resnet = resnext101_32x8d(pretrained=True)
        elif self.resnet == "resnet101":
            _resnet = resnet101_features(pretrained =True)
        elif self.resnet == "resnet50":
            _resnet = resnet50_features(pretrained =True)
        elif self.resnet == "resnet152":
            _resnet = resnet152_features(pretrained =True)

        try: 
            self.matrix_net = MatrixNet(_resnet, layers)
        except : 
            print("ERROR: ivalid resnet")
            sys.exit()

        self.subnet_tl_corners_regr = SubNet(mode='tl_corners')
        self.subnet_br_corners_regr = SubNet(mode='br_corners')
        self.subnet_anchors_heats = SubNet(mode='classes')

    def forward(self, x):
        features = self.matrix_net(x)
        anchors_tl_corners_regr = [self.subnet_tl_corners_regr(feature) for feature in features]
        anchors_br_corners_regr = [self.subnet_br_corners_regr(feature) for feature in features]
        anchors_heatmaps = [_sigmoid(self.subnet_anchors_heats(feature)) for feature in features]
        return anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr
    
    
class model(nn.Module):
    def __init__(self, db):
        super(model, self).__init__()
        classes = db.configs["categories"]
        resnet  = db.configs["backbone"]
        layers  = db.configs["layers_range"]
        self.net = MatrixNetAnchors(classes, resnet, layers)
        self._decode = _decode
        
    def _train(self, *xs):
        image = xs[0][0]
        anchors_inds = xs[1]
        outs = self.net.forward(image)


        for ind in range(len(anchors_inds)):
            outs[1][ind] = _tranpose_and_gather_feat(outs[1][ind], anchors_inds[ind])
            outs[2][ind] = _tranpose_and_gather_feat(outs[2][ind], anchors_inds[ind])
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0][0]
        
        outs = self.net.forward(image)
        return self._decode(*outs, **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)
    
class MatrixNetAnchorsLoss(nn.Module):
    def __init__(self, corner_regr_weight=1, center_regr_weight=0.1, focal_loss=_neg_loss):
        super(MatrixNetAnchorsLoss, self).__init__()
        self.corner_regr_weight = corner_regr_weight
        self.center_regr_weight = center_regr_weight
        self.focal_loss  = focal_loss
        self.regr_loss   = _regr_loss
        
    def forward(self, outs, targets):
        # focal loss
        focal_loss = 0
        corner_regr_loss = 0

        anchors_heats = outs[0]
        anchors_tl_corners_regrs = outs[1]
        anchors_br_corners_regrs = outs[2]


        gt_anchors_heat = targets[0]
        gt_tl_corners_regr = targets[1]
        gt_br_corners_regr = targets[2]
        gt_mask = targets[3]
        
        numf = 0
        numr = 0
        for i in range(len(anchors_heats)):
            floss, num = self.focal_loss([anchors_heats[i]], gt_anchors_heat[i])
            focal_loss += floss
            numf += num
            rloss, num = self.regr_loss(anchors_br_corners_regrs[i], gt_br_corners_regr[i], gt_mask[i])
            numr += num
            corner_regr_loss += rloss
            
            rloss, num = self.regr_loss(anchors_tl_corners_regrs[i], gt_tl_corners_regr[i], gt_mask[i])
            numr += num
            corner_regr_loss += rloss            

        if numr > 0:
            corner_regr_loss = corner_regr_loss / numr
           
        if numf > 0:
            focal_loss = focal_loss / numf
            
#       
        loss = (focal_loss + corner_regr_loss)
        return loss.unsqueeze(0)
    
loss = MatrixNetAnchorsLoss()

def _decode(
    anchors_heats, corners_tl_regrs, corners_br_regrs,
    K=100, kernel=1, dist_threshold=0.2, num_dets=1000,layers_range = None,
    output_kernel_size = None, output_sizes = None, input_size=None, base_layer_range=None
):
    top_k = K
    batch, cat, height_0, width_0 = anchors_heats[0].size()
    
    for i in range(len(anchors_heats)):
        
        
        anchors_heat = anchors_heats[i]
        corners_tl_regr = corners_tl_regrs[i]
        corners_br_regr = corners_br_regrs[i]

        batch, cat, height, width = anchors_heat.size()
        height_scale = height_0 / height
        width_scale = width_0 / width
    
        anchors_scores, anchors_inds, anchors_clses, anchors_ys, anchors_xs = _topk(anchors_heat, K=K)

        anchors_ys = anchors_ys.view(batch, K, 1)
        anchors_xs = anchors_xs.view(batch, K, 1)


            
        if corners_br_regr is not None:
            corners_tl_regr = _tranpose_and_gather_feat(corners_tl_regr, anchors_inds)
            corners_tl_regr = corners_tl_regr.view(batch, K, 1, 2)
            corners_br_regr = _tranpose_and_gather_feat(corners_br_regr, anchors_inds)
            corners_br_regr = corners_br_regr.view(batch, K, 1, 2)
            
            min_y, max_y, min_x, max_x = map(lambda x:x/8/2,base_layer_range) #This is the range of object sizes within the layers 
            # We devide by 2 since we want to compute the distances from center to corners.
            
            tl_xs = anchors_xs -  (((max_x - min_x) * corners_tl_regr[..., 0]) + (max_x + min_x)/2) 
            tl_ys = anchors_ys -  (((max_y - min_y) * corners_tl_regr[..., 1]) + (max_y + min_y)/2)
            br_xs = anchors_xs +  (((max_x - min_x) * corners_br_regr[..., 0]) + (max_x + min_x)/2)
            br_ys = anchors_ys +  (((max_y - min_y) * corners_br_regr[..., 1]) + (max_y + min_y)/2)

        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
        scores    = anchors_scores.view(batch, K, 1)
        

        
        width_inds  = (br_xs < tl_xs)
        height_inds = (br_ys < tl_ys)

        scores[width_inds]  = -1
        scores[height_inds] = -1

        scores = scores.view(batch, -1)

        scores, inds = torch.topk(scores, min(num_dets, scores.shape[1]))
        scores = scores.unsqueeze(2)

        bboxes = bboxes.view(batch, -1, 4)
        bboxes = _gather_feat(bboxes, inds)

        clses  = anchors_clses.contiguous().view(batch, -1, 1)
        clses  = _gather_feat(clses, inds).float()
         
#         
        bboxes[:, :, 0] *= width_scale
        bboxes[:, :, 1] *= height_scale
        bboxes[:, :, 2] *= width_scale
        bboxes[:, :, 3] *= height_scale
        
        
        if i == 0:
            detections = torch.cat([bboxes, scores,scores,scores, clses], dim=2)
        else:
            detections = torch.cat([detections, torch.cat([bboxes, scores,scores,scores, clses], dim=2)], dim = 1)
   
    
    top_scores, top_inds = torch.topk(detections[:, :, 4], 300)
    detections = _gather_feat(detections, top_inds)
    return detections



