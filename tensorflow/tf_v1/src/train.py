import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
from model import Three_branch_net
import tempfile
import time
from utils import feed_dict, loss, optimizer, accuracy, read_tfrecord, feed_dict_tfrecord, get_data, get_lstm

# https://blog.csdn.net/sydpz1987/article/details/51340277

# BATCH_SIZE = 100
# LEARNING_RATE_BASE =  0.005 
# LEARNING_RATE_DECAY = 0.99 
# REGULARIZER = 0.0001 
# STEPS = 50000 
# MOVING_AVERAGE_DECAY = 0.99 
MODEL_SAVE_PATH="../ckpt/" 
MODEL_NAME="Three_branch_net" 
learning_rate = 0.01

flags = tf.app.flags
# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', 'localhost:2221', 'Comma-separated list of hostname:port pairs')
# 两个worker节点
flags.DEFINE_string('worker_hosts', 'localhost:2222,localhost:2223',
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

FLAGS = flags.FLAGS

import os
if FLAGS.job_name == 'ps':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    start_time = time.time()
    
    # 第1步：命令行参数解析，获取集群的信息ps_hosts和worker_hosts，以及当前节点的角色信息job_name和task_index
    print("\n\n\n", start_time, "\n\n")
    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print('task_index : %d' % FLAGS.task_index)
    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 第2步：创建当前task结点的Server
    # num_worker = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    # 第3步：如果当前节点是ps，则调用server.join()无休止等待；如果是worker，则执行第4步。
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    # worker_device = '/job:worker/task%d/cpu:0' % FLAGS.task_index

    # 读入数据
    train_next_element = get_data("../data/train.tfrecords")
    test_next_element = get_data("../data/test.tfrecords")

    # 定义网络输入
    with tf.name_scope('net_input'):
        p1 = tf.placeholder(tf.float32, [None, 20, 40, 3])
        p2 = tf.placeholder(tf.float32, [None, 8, 10])
        p3 = tf.placeholder(tf.float32, [None, 10])
        y_ = tf.placeholder(tf.float32,[None,1])
    x = [p1, p2, p3]


    # Assigns ops to the local worker by default.
    # 将op 挂载到各个本地的worker上
    # tf.train.replica_device_setter()会根据job名，将with内的Variable op放到ps tasks，
    # 将其他计算op放到worker tasks。默认分配策略是轮询。
    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
        # 全局优化次数，主要用于分布式
        with tf.name_scope('global_step'):
            global_step = tf.Variable(0, trainable=False) 
        # 第4步：则构建要训练的模型
        model = Three_branch_net()
        y_out = model.forward(x)
        model_loss = tf.losses.mean_squared_error(y_, y_out)
        train_op = optimizer(model_loss, learning_rate, global_step)

        saver = tf.train.Saver() 
        # 用于tensorboard
        #直接获取所有的数据汇总
        summary_op = tf.summary.merge_all() 
        # 生成本地的参数初始化操作init_op
        # init_op = tf.global_variables_initializer()
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

        train_dir = tempfile.mkdtemp()

        # 第5步：创建tf.train.Supervisor来管理模型的训练过程
        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, init_op=init_op, summary_op=summary_op,
                                 recovery_wait_secs=1,
                                 global_step=global_step)

        if is_chief:
            print('Worker %d: Initailizing session...' % FLAGS.task_index)
        else:
            print('Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index)
        
        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        gpu_options = tf.GPUOptions(allow_growth=True)
        with sv.prepare_or_wait_for_session(server.target, config=tf.ConfigProto(gpu_options=gpu_options)) as sess: 
            # 用于tensorboard
            train_writer = tf.summary.FileWriter(MODEL_SAVE_PATH + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(MODEL_SAVE_PATH + '/test')

            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) 
            if ckpt and ckpt.model_checkpoint_path:
                print("\n restore model\n")
                saver.restore(sess, ckpt.model_checkpoint_path) 
            else:
                print("\n new model train \n")
            
            # init_op = tf.global_variables_initializer() 
            sess.run(init_op)

            i = 0
            while True:
                if i % 10 == 0:
                    try:
                        p1_run, p2_run, p3_run, label_run = sess.run([test_next_element[0],
                                                        test_next_element[1],
                                                        test_next_element[2],
                                                        test_next_element[3]])
                        losses, summary, step = sess.run([model_loss,summary_op, global_step], feed_dict={
                                p1: p1_run, p2: p2_run, p3: p3_run, y_: label_run})                     
                        test_writer.add_summary(summary, step)
                        print("test loss at step %s:(global step %s) :%s " %(i, step, losses))
                    except tf.errors.OutOfRangeError:
                        break

                else:
                    try:
                        p1_run, p2_run, p3_run, label_run = sess.run([train_next_element[0],
                                                        train_next_element[1],
                                                        train_next_element[2],
                                                        train_next_element[3]])
                        # print(p1_run.shape, p2_run.shape,  p3_run.shape, label_run.shape)
                        if i % 100 == 1:
                            pass
                            #定义tensorflow运行选项。
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            #定义运行的元信息。可以记录下来运算的时间、内存占用这些信息。
                            run_metadata = tf.RunMetadata()
                            losses, _, summary, step = sess.run([model_loss, train_op,summary_op, global_step], feed_dict={
                                p1: p1_run, p2: p2_run, p3: p3_run, y_: label_run}) 
                            train_writer.add_run_metadata(run_metadata, 'step%03d'%step)
                            train_writer.add_summary(summary, step)
                            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)
                            print("adding run metadata for ", step) 
                        else:
                            losses, _, summary, step = sess.run([model_loss, train_op,summary_op, global_step], feed_dict={
                                p1: p1_run, p2: p2_run, p3: p3_run, y_: label_run})                                                                                 
                            train_writer.add_summary(summary, step)
                            # print("accuracy at step %s:(global step %s) :%s " %(i, step, losses))
                    except tf.errors.OutOfRangeError:
                        break
                i += 1
            train_writer.close()
            test_writer.close()
        print("All time {}s".format(time.time()-start_time))

if __name__ == "__main__":
    train()
