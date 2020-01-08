import os
import cv2
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
from external.nms import soft_nms, soft_nms_merge

def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

def save_image(data, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(fn, dpi = height)
    plt.close()

def kp_decode(nnet, images, K, matching_threshold=0.5, kernel=3, layers_range = None, output_kernel_size = None,output_sizes=None, input_size = None, base_layer_range = None):
    detections = nnet.test([[images]], dist_threshold=matching_threshold, K=K, kernel=kernel,
                           layers_range = layers_range, output_kernel_size = output_kernel_size,output_sizes=output_sizes,  input_size = input_size,
                           base_layer_range = base_layer_range)
    detections = detections.data.cpu().numpy()
    return detections

def test_MatrixNetCorners(db, nnet, result_dir, debug=False, decode_func=kp_decode):
    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if db.split != "trainval":
        db_inds = db.db_inds[:200] if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds[:100]
    num_images = db_inds.size

    K             = db.configs["top_k"]
    matching_threshold  = db.configs["matching_threshold"]
    nms_kernel    = db.configs["nms_kernel"]
    flag_flip_images=db.configs["test_flip_images"]
    max_dim = db.configs["test_image_max_dim"]
    
    scales        = db.configs["test_scales"]
    weight_exp    = db.configs["weight_exp"]
    merge_bbox    = db.configs["merge_bbox"]
    categories    = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    layers_range = db.configs["layers_range"]
    input_size = db.configs["input_size"]
    output_kernel_size = db.configs["output_kernel_size"]
    
    _dict={}
    output_sizes=[]
    for i,l in enumerate(layers_range):
        for j,e in enumerate(l):
            if e !=-1:
                output_sizes.append([input_size[0]//(8*2**(j)), input_size[1]//(8*2**(i))])
                _dict[(i+1)*10+(j+1)]=e
    layers_range=[_dict[i] for i in sorted(_dict)]

    
    layers_range = [[lr[0] * os[0]/input_size[0], lr[1] * os[0]/input_size[0],
                    lr[2] * os[1]/input_size[1], lr[3] * os[1]/input_size[1]] for (lr, os) in zip (layers_range, output_sizes)]
    
    
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]
    

    top_bboxes = {}
    for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        db_ind = db_inds[ind]

        image_id   = db.image_ids(db_ind)
        image_file = db.image_file(db_ind)
        image      = cv2.imread(image_file)

        height, width = image.shape[0:2]
        detections = []
        for scale in scales:
            org_scale = scale

            scale = scale * min((max_dim)/float(height), (max_dim)/float(width))
            new_height = int(height * scale)
            new_width  = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])

            inp_height = ((new_height // 128) + 1) * 128
            inp_width  = ((new_width  // 128) + 1) * 128
        
            images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            ratios  = np.zeros((1, 2), dtype=np.float32)
            borders = np.zeros((1, 4), dtype=np.float32)
            sizes   = np.zeros((1, 2), dtype=np.float32)

            out_height, out_width = ((inp_height) // 8, (inp_width) // 8)
            height_ratio = out_height / inp_height
            width_ratio  = out_width  / inp_width

            resized_image = cv2.resize(image, (new_width, new_height))
            resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

            resized_image = resized_image / 255.
            

            images[0]  = resized_image.transpose((2, 0, 1))
            borders[0] = border
            sizes[0]   = [int(height * scale), int(width * scale)]
            ratios[0]  = [height_ratio, width_ratio]
            if flag_flip_images:
                
                images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            images = torch.from_numpy(images)
            dets   = decode_func(nnet, images, K, matching_threshold=matching_threshold, kernel=nms_kernel,
                                 layers_range=layers_range, output_kernel_size = output_kernel_size, output_sizes=output_sizes,input_size=input_size)
        
            if flag_flip_images:
                dets   = dets.reshape(2, -1, 8)
                dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
                dets   = dets.reshape(1, -1, 8)

            _rescale_dets(dets, ratios, borders, sizes)
            dets[:, :, 0:4] /= scale
            detections.append(dets)

        detections = np.concatenate(detections, axis=1)

        classes    = detections[..., -1]
        classes    = classes[0]
        detections = detections[0]

        # reject detections with negative scores
        keep_inds  = (detections[:, 4] > 0)
        detections = detections[keep_inds]
        classes    = classes[keep_inds]

        top_bboxes[image_id] = {}
        for j in range(categories):
            keep_inds = (classes == j)
            top_bboxes[image_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
            if merge_bbox:
                soft_nms_merge(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
            else:
                soft_nms(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm)
            top_bboxes[image_id][j + 1] = top_bboxes[image_id][j + 1][:, 0:5]

        scores = np.hstack([
            top_bboxes[image_id][j][:, -1] 
            for j in range(1, categories + 1)
        ])
        if len(scores) > max_per_image:
            kth    = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, -1] >= thresh)
                top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]
        if debug:
            image_file = db.image_file(db_ind)
            image      = cv2.imread(image_file)
            bboxes = {}
            for j in range(categories, 0, -1):
                keep_inds = (top_bboxes[image_id][j][:, -1] > 0.2)
                cat_name  = db.class_name(j)
                cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                color     = np.random.random((3, )) * 0.6 + 0.4
                color     = color * 255
                color     = color.astype(np.int32).tolist()
                for bbox in top_bboxes[image_id][j][keep_inds]:
                    bbox  = bbox[0:4].astype(np.int32)
                    if bbox[1] - cat_size[1] - 2 < 0:
                        cv2.rectangle(image,
                            (bbox[0], bbox[1] + 2),
                            (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                            color, -1
                        )
                        cv2.putText(image, cat_name, 
                            (bbox[0], bbox[1] + cat_size[1] + 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                        )
                    else:
                        cv2.rectangle(image,
                            (bbox[0], bbox[1] - cat_size[1] - 2),
                            (bbox[0] + cat_size[0], bbox[1] - 2),
                            color, -1
                        )
                        cv2.putText(image, cat_name, 
                            (bbox[0], bbox[1] - 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                        )
                    cv2.rectangle(image,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        color, 2
                    )
            debug_file = os.path.join(debug_dir, "{}.jpg".format(db_ind))
            print(debug_file)
            cv2.imwrite(debug_file,image)

    result_json = os.path.join(result_dir, "results.json")

    detections  = db.convert_to_coco(top_bboxes)
    with open(result_json, "w") as f:
        json.dump(detections, f)
    
    cls_ids   = list(range(1, categories + 1))
    image_ids = [db.image_ids(ind) for ind in db_inds]
    db.evaluate(result_json, cls_ids, image_ids)
    return 0



def test_MatrixNetAnchors(db, nnet, result_dir, debug=False, decode_func=kp_decode):
    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if db.split != "trainval":
        db_inds = db.db_inds[:200] if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds[:100]
    num_images = db_inds.size

    K             = db.configs["top_k"]
    matching_threshold  = db.configs["matching_threshold"]
    nms_kernel    = db.configs["nms_kernel"]
    flag_flip_images=db.configs["test_flip_images"]
    max_dim = db.configs["test_image_max_dim"]

    scales        = db.configs["test_scales"]
    weight_exp    = db.configs["weight_exp"]
    merge_bbox    = db.configs["merge_bbox"]
    categories    = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    layers_range = db.configs["layers_range"]
    input_size = db.configs["input_size"]
    output_kernel_size = db.configs["output_kernel_size"]
    base_layer_range = db.configs["base_layer_range"]
    
    _dict={}
    output_sizes=[]
    for i,l in enumerate(layers_range):
        for j,e in enumerate(l):
            if e !=-1:
                output_sizes.append([input_size[0]//(8*2**(j)), input_size[1]//(8*2**(i))])
                _dict[(i+1)*10+(j+1)]=e
    layers_range=[_dict[i] for i in sorted(_dict)]
    
    layers_range = [[lr[0] * os[0]/input_size[0], lr[1] * os[0]/input_size[0],
                    lr[2] * os[1]/input_size[1], lr[3] * os[1]/input_size[1]] for (lr, os) in zip (layers_range, output_sizes)]

    
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]
    
    top_bboxes = {}
    for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        db_ind = db_inds[ind]

        image_id   = db.image_ids(db_ind)
        image_file = db.image_file(db_ind)
        image      = cv2.imread(image_file)

        height, width = image.shape[0:2]
        detections = [] 
        for scale in scales:
            org_scale = scale
            scale = scale * min((max_dim)/float(height), (max_dim)/float(width))
            new_height = int(height * scale)
            new_width  = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])

            if len(scales) == 0:
                inp_height = input_size[0]
                inp_width = input_size[1]
            else:
                if (new_height % 128) == 0:
                    inp_height = new_height
                else:
                    inp_height = ((new_height // 128) + 1) * 128
                if (new_width % 128) == 0:
                    inp_width  = new_width
                else:
                    inp_width  = ((new_width  // 128) + 1) * 128                  
            images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            ratios  = np.zeros((1, 2), dtype=np.float32)
            borders = np.zeros((1, 4), dtype=np.float32)
            sizes   = np.zeros((1, 2), dtype=np.float32)

            out_height, out_width = ((inp_height) // 8, (inp_width) // 8)
            height_ratio = out_height / inp_height
            width_ratio  = out_width  / inp_width

            resized_image = cv2.resize(image, (new_width, new_height))
            resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

            resized_image = resized_image / 255.

            images[0]  = resized_image.transpose((2, 0, 1))
            borders[0] = border
            sizes[0]   = [int(height * scale), int(width * scale)]
            ratios[0]  = [height_ratio, width_ratio]

            if flag_flip_images:                          
                images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            
            images = torch.from_numpy(images)
            dets   = decode_func(nnet, images, K, matching_threshold=matching_threshold, kernel=nms_kernel,layers_range=layers_range, output_kernel_size = output_kernel_size, output_sizes=output_sizes,input_size=input_size, base_layer_range = base_layer_range)
            if flag_flip_images:
                dets   = dets.reshape(2, -1, 8)
                dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
                dets   = dets.reshape(1, -1, 8)
            
            _rescale_dets(dets, ratios, borders, sizes)
            dets[:, :, 0:4] /= scale
            detections.append(dets)

        detections = np.concatenate(detections, axis=1)

        classes    = detections[..., -1]
        classes    = classes[0]
        detections = detections[0]

        # reject detections with negative scores
        keep_inds  = (detections[:, 4] > 0)
        detections = detections[keep_inds]
        classes    = classes[keep_inds]

        top_bboxes[image_id] = {}
        for j in range(categories):
            keep_inds = (classes == j)
            top_bboxes[image_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)

            if  merge_bbox:
                soft_nms_merge(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
            else:
                soft_nms(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm)
            top_bboxes[image_id][j + 1] = top_bboxes[image_id][j + 1][:, 0:5]

        scores = np.hstack([
            top_bboxes[image_id][j][:, -1] 
            for j in range(1, categories + 1)
        ])
        if len(scores) > max_per_image:
            kth    = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, -1] >= thresh)
                top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]
        if debug:
            image_file = db.image_file(db_ind)
            image      = cv2.imread(image_file)
            bboxes = {}
            for j in range(categories, 0, -1):
                keep_inds = (top_bboxes[image_id][j][:, -1] > 0.3)
                cat_name  = db.class_name(j)
                cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                color     = np.random.random((3, )) * 0.6 + 0.4
                color     = color * 255
                color     = color.astype(np.int32).tolist()
                for bbox in top_bboxes[image_id][j][keep_inds]:
#                     print(bbox)
                    bbox  = bbox[0:4].astype(np.int32)
                    if bbox[1] - cat_size[1] - 2 < 0:
                        cv2.rectangle(image,
                            (bbox[0], bbox[1] + 2),
                            (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                            color, -1
                        )
                        cv2.putText(image, cat_name, 
                            (bbox[0], bbox[1] + cat_size[1] + 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                        )
                    else:
                        cv2.rectangle(image,
                            (bbox[0], bbox[1] - cat_size[1] - 2),
                            (bbox[0] + cat_size[0], bbox[1] - 2),
                            color, -1
                        )
                        cv2.putText(image, cat_name, 
                            (bbox[0], bbox[1] - 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                        )
                    cv2.rectangle(image,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        color, 2
                    )
            debug_file = os.path.join(debug_dir, "{}.jpg".format(db_ind))
            print(debug_file)
            cv2.imwrite(debug_file,image)

    result_json = os.path.join(result_dir, "results.json")

    detections  = db.convert_to_coco(top_bboxes)
    with open(result_json, "w") as f:
        json.dump(detections, f)
    
    cls_ids   = list(range(1, categories + 1))
    image_ids = [db.image_ids(ind) for ind in db_inds]
    db.evaluate(result_json, cls_ids, image_ids)
    return 0




def testing(db, nnet, result_dir, debug=False):
    return globals()["test_"+system_configs.model_name](db, nnet, result_dir, debug=debug)
