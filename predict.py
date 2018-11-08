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


def evaluate(config_path,
             model_dir,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=False):
    model_dir = pathlib.Path(model_dir)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)

    net = second_builder.build(model_cfg, voxel_generator, target_assigner)
    net.cuda()
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)

    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=input_cfg.batch_size,
        shuffle=False,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    dt_annos = []
    global_set = None
    print("Generate output labels...")

    for example in iter(eval_dataloader):
        example = example_convert_to_torch(example, float_dtype)
        if pickle_result:
            dt_annos += predict_kitti_to_anno(
                net, example, class_names, center_limit_range,
                model_cfg.lidar_input, global_set)
        else:
            print('Predicting on one example...')
            _predict_kitti_to_file(net, example, result_path_step, class_names,
                                   center_limit_range, model_cfg.lidar_input)

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')

    print(f"avg forward time per example: {net.avg_forward_time:.3f}")
    print(f"avg postprocess time per example: {net.avg_postprocess_time:.3f}")
    if not predict_test:
        gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
        if not pickle_result:
            dt_annos = kitti.get_label_annos(result_path_step)
        result = get_official_eval_result(gt_annos, dt_annos, class_names)
        print(result)
        result = get_coco_eval_result(gt_annos, dt_annos, class_names)
        print(result)
        if pickle_result:
            with open(result_path_step / "result.pkl", 'wb') as f:
                pickle.dump(dt_annos, f)


def _predict_kitti_to_file(net,
                           example,
                           result_save_path,
                           class_names,
                           center_limit_range=None,
                           lidar_input=False):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    print('input example: ', example)
    predictions_dicts = net(example)
    # t = time.time()
    print(predictions_dicts)
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
                box_preds = box_preds[:, [0, 1, 2, 4, 5, 3,
                                          6]]  # lhw->hwl(label file format)
                label_preds = preds_dict["label_preds"].data.cpu().numpy()
                # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
                result_lines = []
                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    if not lidar_input:
                        if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                            continue
                        if bbox[2] < 0 or bbox[3] < 0:
                            continue
                    # print(img_shape)
                    if center_limit_range is not None:
                        limit_range = np.array(center_limit_range)
                        if (np.any(box_lidar[:3] < limit_range[:3])
                                or np.any(box_lidar[:3] > limit_range[3:])):
                            continue
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
            result_file = f"{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt"
            result_str = '\n'.join(result_lines)
            print('saved into: ', result_file)
            with open(result_file, 'w') as f:
                f.write(result_str)
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    pass