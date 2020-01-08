import numpy as np
from db.base import BASE

class DETECTION(BASE):
    def __init__(self, db_config):
        super(DETECTION, self).__init__()

        self._configs["categories"]      = 80
        self._configs["rand_scales"]     = [1]
        self._configs["rand_scale_min"]  = 0.8
        self._configs["rand_scale_max"]  = 1.4
        self._configs["rand_scale_step"] = 0.2
        self._configs["backbone"]        = "resnext101_32x8d"
        self._configs["input_size"]      = [511]


        self._configs["width_thresholds"] = None
        self._configs["height_thresholds"] = None
        self._configs["layers_range"] = None
        self._configs["output_kernel_size"] = None
        self._configs["base_layer_range"] = None
        self._configs["nms_threshold"]   = 0.5
        self._configs["max_per_image"]   = 100
        self._configs["top_k"]           = 100
        self._configs["matching_threshold"]    = 0.5
        self._configs["nms_kernel"]      = 3

        self._configs["nms_algorithm"]   = "exp_soft_nms"
        self._configs["weight_exp"]      = 8
        self._configs["merge_bbox"]      = False
        
        self._configs["data_aug"]        = True
        self._configs["lighting"]        = True
        self._configs["train_image_max_dim"] = 800
        self._configs["border"]          = 128
        self._configs["gaussian_bump"]   = True
        self._configs["gaussian_iou"]    = 0.7
        self._configs["gaussian_radius"] = -1
        self._configs["rand_crop"]       = False
        self._configs["rand_color"]      = False
        self._configs["rand_pushes"]     = False
        self._configs["rand_samples"]    = False
        self._configs["special_crop"]    = False
        self._configs["cutout"]          = False 
        self._configs["test_scales"]     = [1]
        self._configs["test_flip_images"]=True
        self._configs["test_image_max_dim"]   =800
        self.update_config(db_config)

        if self._configs["rand_scales"] is None:
            self._configs["rand_scales"] = np.arange(
                self._configs["rand_scale_min"], 
                self._configs["rand_scale_max"],
                self._configs["rand_scale_step"]
            )
