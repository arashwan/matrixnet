import math
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from .py_utils.loss_utils import _regr_loss, _neg_loss
from torch.autograd import Variable
from .resnet_features import resnet50_features, resnet152_features, resnet18_features, resnet101_features, resnext101_32x8d, wide_resnet101_2
from .py_utils.utils import conv1x1, conv3x3
from .matrixnet import _sigmoid, MatrixNet, SubNet, _gather_feat, _tranpose_and_gather_feat, _topk, _nms

          
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

class MatrixNetCorners(nn.Module):
    def __init__(self, classes, resnet, layers):
        super(MatrixNetCorners, self).__init__()
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
            self.feature_pyramid = MatrixNet(_resnet, layers)
        except : 
            print("ERROR: ivalid resnet")
            sys.exit()

        self.subnet_tl_corners_regr = SubNet(mode='corners')
        self.subnet_tl_centers_regr = SubNet(mode='centers')

        self.subnet_br_corners_regr = SubNet(mode='corners')
        self.subnet_br_centers_regr = SubNet(mode='centers')

        self.subnet_tl_heats = SubNet(mode='classes')
        self.subnet_br_heats = SubNet(mode='classes')

    def forward(self, x):

        features = self.feature_pyramid(x)

        tl_corners_regr = [self.subnet_tl_corners_regr(feature) for feature in features]
        tl_centers_regr = [F.relu(self.subnet_tl_centers_regr(feature)) for feature in features]

        br_corners_regr = [self.subnet_br_corners_regr(feature) for feature in features]
        br_centers_regr = [F.relu(self.subnet_br_centers_regr(feature)) for feature in features]

        tl_heatmaps = [_sigmoid(self.subnet_tl_heats(feature)) for feature in features]

        br_heatmaps = [_sigmoid(self.subnet_br_heats(feature)) for feature in features]

        return tl_heatmaps, br_heatmaps, tl_corners_regr, br_corners_regr, tl_centers_regr, br_centers_regr


class model(nn.Module):
    def __init__(self, db):
        super(model, self).__init__()
        classes = db.configs["categories"]
        resnet  = db.configs["backbone"]
        layers  = db.configs["layers_range"]
        self.net = MatrixNetCorners(classes, resnet, layers)
        self._decode = _decode

    def _train(self, *xs):
        image = xs[0][0]
        tl_inds = xs[1]
        br_inds = xs[2]

        outs = self.net.forward(image)


        for ind in range(len(tl_inds)):
            outs[2][ind] = _tranpose_and_gather_feat(outs[2][ind], tl_inds[ind])
            outs[3][ind] = _tranpose_and_gather_feat(outs[3][ind], br_inds[ind])

            outs[4][ind] = _tranpose_and_gather_feat(outs[4][ind], tl_inds[ind])
            outs[5][ind] = _tranpose_and_gather_feat(outs[5][ind], br_inds[ind])

        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0][0]
        outs = self.net.forward(image)

        return self._decode(*outs, **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class MatrixNetCornerLoss(nn.Module):
    def __init__(self, corner_regr_weight=1, center_regr_weight=0.1, focal_loss=_neg_loss):
        super(MatrixNetCornerLoss, self).__init__()
        self.corner_regr_weight = corner_regr_weight
        self.center_regr_weight = center_regr_weight
        self.focal_loss  = focal_loss
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):
        # focal loss
        focal_loss = 0
        corner_regr_loss = 0
        center_regr_loss = 0

        tl_heats = outs[0] 
        br_heats = outs[1] 
        tl_regrs = outs[2] 
        br_regrs = outs[3]
        center_tl_regrs = outs[4]
        center_br_regrs = outs[5]


        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]
        gt_center_tl_regr = targets[5]
        gt_center_br_regr = targets[5]

        numf = 0
        numr = 0
        for i in range(len(tl_heats)):
            floss, num = self.focal_loss([tl_heats[i]], gt_tl_heat[i])
            focal_loss += floss
            numf += num
            floss, num = self.focal_loss([br_heats[i]], gt_br_heat[i])
            focal_loss += floss
            
            rloss, num = self.regr_loss(tl_regrs[i], gt_tl_regr[i], gt_mask[i])
            numr += num
            corner_regr_loss += rloss
            rloss, num = self.regr_loss(br_regrs[i], gt_br_regr[i], gt_mask[i])
            corner_regr_loss += rloss

            rloss, num = self.regr_loss(center_tl_regrs[i], gt_center_tl_regr[i], gt_mask[i])
            center_regr_loss += rloss
            rloss, num = self.regr_loss(center_br_regrs[i], gt_center_br_regr[i], gt_mask[i])
            center_regr_loss += rloss

        
        corner_regr_loss = self.corner_regr_weight * corner_regr_loss
        center_regr_loss = self.center_regr_weight * center_regr_loss
        if numr > 0:
            corner_regr_loss = corner_regr_loss / numr
            center_regr_loss = center_regr_loss / numr
           
        if numf > 0:
            focal_loss = focal_loss / numf
            

        loss = (focal_loss + corner_regr_loss + center_regr_loss)
        return loss.unsqueeze(0)

loss = MatrixNetCornerLoss()

def _decode(
    tl_heats, br_heats, tl_regrs, br_regrs, tl_centers_regrs, br_centers_regrs,
    K=100, kernel=1, dist_threshold=0.2, num_dets=1000,layers_range = None,
    output_kernel_size = None, output_sizes = None, input_size=None, base_layer_range = None
):
    top_k = K
    batch, cat, height_0, width_0 = tl_heats[0].size()

    for i in range(len(tl_heats)):
        tl_heat = tl_heats[i]
        br_heat = br_heats[i]
        tl_regr = tl_regrs[i]
        br_regr = br_regrs[i]
        tl_centers_regr = tl_centers_regrs[i]
        br_centers_regr = br_centers_regrs[i]


        batch, cat, height, width = tl_heat.size()
        height_scale = height_0 / height
        width_scale = width_0 / width
       
        tl_heat = _nms(tl_heat, kernel=output_kernel_size)
        br_heat = _nms(br_heat, kernel=output_kernel_size)

        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
        br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

        tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
        tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
        br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
        br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

            
        if tl_regr is not None and br_regr is not None and tl_centers_regr is not None and br_centers_regr is not None:
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            tl_regr = tl_regr.view(batch, K, 1, 2)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
            br_regr = br_regr.view(batch, 1, K, 2)

            tl_centers_regr = _tranpose_and_gather_feat(tl_centers_regr, tl_inds)
            tl_centers_regr = tl_centers_regr.view(batch, K, 1, 2)
            br_centers_regr = _tranpose_and_gather_feat(br_centers_regr, br_inds)
            br_centers_regr = br_centers_regr.view(batch, 1, K, 2)

            tl_xs = tl_xs + tl_regr[..., 0]
            tl_ys = tl_ys + tl_regr[..., 1]
            br_xs = br_xs + br_regr[..., 0]
            br_ys = br_ys + br_regr[..., 1]

        # all possible boxes based on top k corners (ignoring class)
        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

        # computing distances

        boxes_widths = (br_xs - tl_xs)
        boxes_heights = (br_ys - tl_ys)
        distsx = torch.abs(1 - (output_sizes[-1][1] * (tl_centers_regr[..., 0] + br_centers_regr[..., 0]))/ (boxes_widths))
        distsy = torch.abs(1 - (output_sizes[-1][0] * (tl_centers_regr[..., 1] + br_centers_regr[..., 1]))/ (boxes_heights))

        dists = torch.abs(br_centers_regr - tl_centers_regr)
        
        dists = (dists[...,1] + dists[...,0])

        tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
        br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
        scores    = (tl_scores + br_scores) / 2 #- 1 * dists

        # reject boxes based on classes
        tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
        br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
        cls_inds = (tl_clses != br_clses)

        # reject boxes based on widths heights
            
        if layers_range != None:
            layer_range = layers_range[i]
            diff_x = (br_xs - tl_xs)
            diff_y = (br_ys - tl_ys)
            wrange_ind = (diff_x < 0.8 * layer_range[2]) | (diff_x > 1.3 * layer_range[3])
            hrange_ind = (diff_y < 0.8 * layer_range[0]) | (diff_y > 1.3 * layer_range[1])
            scores[wrange_ind]    = -1    
            scores[hrange_ind]    = -1
        
        # reject boxes based on distances

        dist_inds = (distsx > dist_threshold) | (distsy > dist_threshold) | (dists > 0.25)

        width_inds  = (br_xs < tl_xs)
        height_inds = (br_ys < tl_ys)

        scores[cls_inds]    = -1
        scores[dist_inds]   = -1
        scores[width_inds]  = -1
        scores[height_inds] = -1



        scores = scores.view(batch, -1)

        scores, inds = torch.topk(scores, min(num_dets, scores.shape[1]))
        scores = scores.unsqueeze(2)

        bboxes = bboxes.view(batch, -1, 4)
        bboxes = _gather_feat(bboxes, inds)

        clses  = tl_clses.contiguous().view(batch, -1, 1)
        clses  = _gather_feat(clses, inds).float()

        tl_scores = tl_scores.contiguous().view(batch, -1, 1)
        tl_scores = _gather_feat(tl_scores, inds).float()
        br_scores = br_scores.contiguous().view(batch, -1, 1)
        br_scores = _gather_feat(br_scores, inds).float()
         
#         print(width_scale, height_scale)
        bboxes[:, :, 0] *= width_scale
        bboxes[:, :, 1] *= height_scale
        bboxes[:, :, 2] *= width_scale
        bboxes[:, :, 3] *= height_scale

        if i == 0:
            detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
        else:
            detections = torch.cat([detections, torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)], dim = 1)

    top_scores, top_inds = torch.topk(detections[:, :, 4], 5 * num_dets)
    detections = _gather_feat(detections, top_inds)
    return detections


