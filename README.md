# distributed_train
分布式训练模型

## 一、tensoflow

代码在`tensorflow\tf_v1\src`目录下运行

1. 先在终端输入`python multi_input_data.py `命令生成tfrecords数据

2. 开三个终端分别运行

   > python train.py --job_name=ps --task_index=0
   >
   > python train.py --job_name=worker --task_index=0
   >
   > python train.py --job_name=worker --task_index=1

