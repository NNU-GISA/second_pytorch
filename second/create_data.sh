#!/usr/bin/env bash

# change this line to your machine kitti root path
KITTI_ROOT=/media/jintain/sg/permanent/datasets/KITTI/object

echo '================= Creating kitti infos  ====================='
python3 create_data.py create_kitti_info_file --data_path=${KITTI_ROOT}

echo '================= Creating reduces point cloud ====================='
python3 create_data.py create_reduced_point_cloud --data_path=${KITTI_ROOT}

echo '================= Creating ground truth database  ====================='
python3 create_data.py create_groundtruth_database --data_path=${KITTI_ROOT}
