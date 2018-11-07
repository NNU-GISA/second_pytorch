# SECOND 3D 
this is 3D object detection method with lidar point cloud data. Some video or gif will be added later on.

![](https://s1.ax1x.com/2018/11/07/i79lIs.gif)



## Install



1. Install requirements:

   ```shell
   # clone this repo (of course)
   sudo pip3 install shapely fire pybind11 pyqtgraph tensorboardX protobuf numba
   # You must also install pytorch 1.0
   # install from pytorch official site or using conda (I don't like conda)
   ```


2. Install SparseConvNet from facebook research repo.

   Note that SparseConvNet is using for 3D convolution but not include inside pytorch yet. Must be installed manually.



3. Add some enviroment variables into your `~/.bashrc` or `~/.zshrc` (**optional**):

   ```shell
   export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
   export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
   export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
   ```

4. Add repo `second_pytorch` path into your `~/.bashrc` or `~/.zshrc` PYTHONPATH:

   ```shell
   export PYTHONPATH=$PYTHONPATH:/path/to/second_pytorchPrepare dataset
   ```



## Prepare data



Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```



1. run create_data:

   ```shell
   cd second
   ./create_data.sh
   ```

2. Edit config file:

   ```
   train_input_reader: {
     ...
     database_sampler {
       database_info_path: "/path/to/kitti_dbinfos_train.pkl"
       ...
     }
     kitti_info_path: "/path/to/kitti_infos_train.pkl"
     kitti_root_path: "KITTI_DATASET_ROOT"
   }
   ...
   eval_input_reader: {
     ...
     kitti_info_path: "/path/to/kitti_infos_val.pkl"
     kitti_root_path: "KITTI_DATASET_ROOT"
   }
   ```




## Train



```bash
python ./pytorch/train.py train --config_path=./configs/car.config --model_dir=/path/to/model_dir
```

* Make sure "/path/to/model_dir" doesn't exist if you want to train new model. A new directory will be created if the model_dir doesn't exist, otherwise will read checkpoints in it.

* training process use batchsize=3 as default for 1080Ti, you need to reduce batchsize if your GPU has less memory.

* Currently only support single GPU training, but train a model only needs 20 hours (165 epoch) in a single 1080Ti and only needs 40 epoch to reach 74 AP in car moderate 3D in Kitti validation dateset.





## Concepts

Shall update on my blog.