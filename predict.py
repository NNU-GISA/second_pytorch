"""
Predict with SECOND with single image and
point cloud frame

"""
import os
import os
import pathlib
import pickle
import shutil
import time
from functools import partial

import fire
import numpy as np
import torch
from google.protobuf import text_format
from tensorboardX import SummaryWriter

import torchplus
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                      lr_scheduler_builder, optimizer_builder,
                                      second_builder)
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import ProgressBar

from second.protos import input_reader_pb2
from second.data.dataset import KittiDataset
from second.data.preprocess import prep_pointcloud, prepare_v9_for_predict
import numpy as np
from second.builder import dbsampler_builder
from functools import partial
from second.core import box_np_ops


class_names = ['Car']


class Second3DDetector(object):

    def __init__(self, model_dir, config_f):
        self.model_dir = model_dir
        self.config_f = config_f
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self._init_net()

    def _init_net(self):
        self.config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(self.config_f, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, self.config)

        self.input_cfg = self.config.eval_input_reader
        self.model_cfg = self.config.model.second
        self.train_cfg = self.config.train_config
        self.class_names = list(self.input_cfg.class_names)
        self.center_limit_range = self.model_cfg.post_center_limit_range

        # BUILD VOXEL GENERATOR
        voxel_generator = voxel_builder.build(self.model_cfg.voxel_generator)
        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        box_coder = box_coder_builder.build(self.model_cfg.box_coder)
        target_assigner_cfg = self.model_cfg.target_assigner
        self.target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                             bv_range, box_coder)

        self.net = second_builder.build(self.model_cfg, voxel_generator, self.target_assigner)
        self.net.cuda()
        if self.train_cfg.enable_mixed_precision:
            self.net.half()
            self.net.metrics_to_float()
            self.net.convert_norm_to_float(self.net)
        torchplus.train.try_restore_latest_checkpoints(self.model_dir, [self.net])
        print('Success load latest checkpoint in {}'.format(self.model_dir))

    def example_convert_to_torch(self, example, dtype=torch.float32):
        example_torch = {}
        float_names = [
            "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
            "Trv2c", "P2"]
        for k, v in example.items():
            if k in float_names:
                example_torch[k] = torch.Tensor(v, dtype=dtype).to(self.device)
            elif k in ["coordinates", "labels", "num_points"]:
                example_torch[k] = torch.Tensor(v, dtype=torch.int32).to(self.device)
            elif k in ["anchors_mask"]:
                example_torch[k] = torch.Tensor(v, dtype=torch.uint8).to(self.device)
            else:
                example_torch[k] = v
        return example_torch

    def construct_example_for_predict(self, img, pc, rect, tr, p2):
        """
        using params and input to construct an example
        :param img:
        :param pc:
        :param rect:
        :param tr:
        :param p2:
        :return:
        """
        generate_bev = self.model_cfg.use_bev
        without_reflectivity = self.model_cfg.without_reflectivity
        num_point_features = self.model_cfg.num_point_features
        out_size_factor = self.model_cfg.rpn.layer_strides[0] // self.model_cfg.rpn.upsample_strides[0]

        cfg = self.input_cfg
        db_sampler_cfg = self.input_cfg.database_sampler
        db_sampler = None
        if len(db_sampler_cfg.sample_groups) > 0:  # enable sample
            db_sampler = dbsampler_builder.build(db_sampler_cfg)
        u_db_sampler_cfg = self.input_cfg.unlabeled_database_sampler
        u_db_sampler = None
        if len(u_db_sampler_cfg.sample_groups) > 0:  # enable sample
            u_db_sampler = dbsampler_builder.build(u_db_sampler_cfg)

        voxel_generator = voxel_builder.build(self.model_cfg.voxel_generator)
        grid_size = voxel_generator.grid_size
        # [352, 400]
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]

        ret = self.target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
        }

        # preparing point cloud
        prep_func = partial(
            prep_pointcloud,
            root_path=cfg.kitti_root_path,
            class_names=list(cfg.class_names),
            voxel_generator=voxel_generator,
            target_assigner=self.target_assigner,
            training=False,
            max_voxels=cfg.max_number_of_voxels,
            remove_outside_points=False,
            remove_unknown=cfg.remove_unknown_examples,
            create_targets=False,
            shuffle_points=cfg.shuffle_points,
            gt_rotation_noise=list(cfg.groundtruth_rotation_uniform_noise),
            gt_loc_noise_std=list(cfg.groundtruth_localization_noise_std),
            global_rotation_noise=list(cfg.global_rotation_uniform_noise),
            global_scaling_noise=list(cfg.global_scaling_uniform_noise),
            global_random_rot_range=list(
                cfg.global_random_rotation_range_per_object),
            db_sampler=db_sampler,
            unlabeled_db_sampler=u_db_sampler,
            generate_bev=generate_bev,
            without_reflectivity=without_reflectivity,
            num_point_features=num_point_features,
            anchor_area_threshold=cfg.anchor_area_threshold,
            gt_points_drop=cfg.groundtruth_points_drop_percentage,
            gt_drop_max_keep=cfg.groundtruth_drop_max_keep_points,
            remove_points_after_sample=cfg.remove_points_after_sample,
            remove_environment=cfg.remove_environment,
            use_group_id=cfg.use_group_id,
            out_size_factor=out_size_factor,
            anchor_cache=anchor_cache,
        )
        example = prepare_v9_for_predict(img, pc, num_point_features,
                                         r0_rect=rect, tr_velo_2_cam=tr, p2=p2, prep_func=prep_func)
        return example

    def predict(self, example):
        # should construct an example first for prediction
        example = self.example_convert_to_torch(example)
        batch_image_shape = example['image_shape']
        batch_imgidx = example['image_idx']
        print('input example: ', example)
        predictions_dicts = self.net(example)
        print(predictions_dicts)

        result_lines = []
        for i, preds_dict in enumerate(predictions_dicts):
            try:
                image_shape = batch_image_shape[i]
                img_idx = preds_dict["image_idx"]
                if preds_dict["bbox"] is not None:
                    box_2d_preds = preds_dict["bbox"].data.cpu().numpy()
                    box_preds = preds_dict["box3d_camera"].data.cpu().numpy()
                    scores = preds_dict["scores"].data.cpu().numpy()
                    box_preds_lidar = preds_dict["box3d_lidar"].data.cpu().numpy()
                    # write pred to file
                    box_preds = box_preds[:, [0, 1, 2, 4, 5, 3, 6]]  # lhw->hwl(label file format)
                    label_preds = preds_dict["label_preds"].data.cpu().numpy()

                    # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
                    result_lines = []
                    for box, box_lidar, bbox, score, label in zip(box_preds, box_preds_lidar,
                                                                  box_2d_preds, scores, label_preds):
                        bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                        bbox[:2] = np.maximum(bbox[:2], [0, 0])
                        result_dict = {
                            'name': class_names[int(label)],
                            'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                            'bbox': bbox,
                            'location': box[:3],
                            'dimensions': box[3:6],
                            'rotation_y': box[6],
                            'score': score,
                        }
                        result_line = kitti.kitti_result_line(result_dict)
                        result_lines.append(result_line)
                else:
                    result_lines = []
            except Exception as e:
                print(e)
                continue
        print('Result in kitti label format:\n{}'.format(result_lines))
        return result_lines


def main():

    second_3d_detector = Second3DDetector(model_dir='./models', config_f='./second/configs/car.config')

    # TODO: finish this
    img = ''
    pc = ''
    example = second_3d_detector.construct_example_for_predict()
    res = second_3d_detector.predict(example)
    print(res)


if __name__ == '__main__':
    pass