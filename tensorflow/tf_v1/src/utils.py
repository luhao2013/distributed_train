#coding:utf-8
import tensorflow as tf
# IMAGE_SIZE = 28
# NUM_CHANNELS = 1
# CONV1_SIZE = 5
# CONV1_KERNEL_NUM = 32
# CONV2_SIZE = 5
# CONV2_KERNEL_NUM = 64
# FC_SIZE = 512
# OUTPUT_NODE = 10

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape): 
    b = tf.Variable(tf.zeros(shape))  
    return b

def conv2d(x,w):  
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):  
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME') 

#定义对所用变量的数据汇总
def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		#记录变量的直方图数据
		tf.summary.histogram('histogram', var)

#全链接层 Relu(Wx + b)
def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights)
        
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.1, shape=[output_dim]))
            variable_summaries(biases)
        
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activate', preactivate)
        
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        
        return activations

def conv_layer(x, kernel_size, num_channels, num_conv_kernels, regularizer, pool = max_pool_2x2, act = tf.nn.relu):
    conv_w = get_weight([kernel_size, kernel_size, num_channels, num_conv_kernels], regularizer)
    conv_b = get_bias([num_conv_kernels]) 
    conv = conv2d(x, conv_w)
    conv = tf.nn.bias_add(conv, conv_b)
    conv = tf.nn.relu(conv) 
    conv = pool(conv) 

    return conv

# lstm层
def get_lstm(n_hidden, keep_prob,name):
    lstm = tf.nn.rnn_cell.LSTMCell(n_hidden, name = name)
    dropped = tf.nn.rnn_cell.DropoutWrapper(lstm, keep_prob)
    return dropped

def dropout_layer(input_tensor, keep_prob, layer_name):
    with tf.name_scope(layer_name):
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        hidden = tf.nn.dropout(input_tensor, keep_prob)
    return hidden

#定义feed_dict
def feed_dict(x, y_, keep_prob, mnist, train, dropout=1.0):
    if train:
        xs ,ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs , ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x:xs , y_:ys , keep_prob: k}

#定义feed_dict
def feed_dict_tfrecord(sess, xs, ys, x, y_, keep_prob, train, dropout=1.0):
    if train:
        xs_, ys_ = sess.run([xs, ys])
        k = dropout
    else:
        #TODO 怎么把所有的测试数据都读完
        xs_, ys_ = sess.run([xs, ys])
        k = 1.0
    # print('====y', ys_)
    return {x:xs_ , y_: ys_ , keep_prob: k}

#定义feed_dict
def feed_dict_v1(model, mnist, train, dropout=1.0):
    if train:
        xs ,ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs , ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {model.x:xs , model.y_:ys , model.keep_prob: k}

def loss(y_out, y_):
    #计算网络输出y_out和标签数据的交叉熵（cross entropy），并保存到tensorboard中
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=y_)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy

def optimizer(cross_entropy, learning_rate, global_step):
    #将得到的cross entropy作为损失函数，利用Adam优化算法最小化损失函数
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step) 
    
    return train_step

def accuracy(y_out, y_, x):
    #计算准确率，并将识别正确的图片和识别错误的图片采样保存到tensorboard中
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
            wrong_prediction = tf.not_equal(tf.argmax(y_out,1), tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope('input_reshape'):
            image_shape_input = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('correct_image'):
        correct_pred = tf.nn.embedding_lookup(image_shape_input,tf.reshape(tf.where(correct_prediction),[-1]))
        tf.summary.image('correct_pred', correct_pred, 10)
    with tf.name_scope('wrong_image'):
        wrong_pred = tf.nn.embedding_lookup(image_shape_input,tf.reshape(tf.where(wrong_prediction),[-1]))
        tf.summary.image('wrong_pred', wrong_pred, 10)

    tf.summary.scalar('accuracy', accuracy)

    return accuracy

def read_tfrecord(datapath, batch_size=50):
    reader=tf.TFRecordReader()
    #tf.train.string_input_producer()和下面的tf.train.start_queue_runners()相对应，前者创建输入队列，后者启动队列
    filename_queue=tf.train.string_input_producer([datapath])   
    _,serialized_example=reader.read(filename_queue)  #从文件中读取一个样例
    features=tf.parse_single_example(serialized_example,features={'image_raw':tf.FixedLenFeature([],tf.string),'label':tf.FixedLenFeature([],tf.int64)})
    #tf.FixedLenFeature()函数解析得到的结果是一个Tensor
    images=tf.decode_raw(features['image_raw'],tf.uint8)      #tf.decode_raw用于将字符串转换成unit8的张量
    images = tf.reshape(images, [784,])  # 不固定形状会报错
    images = tf.cast(images, tf.float32) * (1. / 255)  # 没有归一化，无法优化，但疑惑的mnist自带迭代器应该也没有归一化，怎么能优化
    labels=tf.cast(features['label'],tf.int32)   #将目标变量转换成tf.int32格式
    labels = tf.one_hot(labels, depth=10)
    #tf.decode_raw可以将字符串解析成图像对应的像素数组
    shuffle_batch = True
    if shuffle_batch:
        images,labels = tf.train.shuffle_batch([images, labels],
                                                batch_size=batch_size,
                                                capacity=1000,
                                                num_threads=8,
                                                min_after_dequeue=700)
    else:
        images,labels = tf.train.batch([images, labels],
                                        batch_size=batch_size,
                                        capacity=1000,
                                        num_threads=8)
    
    return images, labels
    
    # init_op = tf.group(tf.global_variables_initializer(),
    #                tf.local_variables_initializer())


def parse_function(example_proto):
    # 定义解析的字典
    dics = {

        'p1': tf.io.VarLenFeature(dtype = tf.float32),

        'p1_shape': tf.io.FixedLenFeature(shape=(3,), dtype = tf.int64),

        'p2': tf.io.VarLenFeature( dtype = tf.float32),
        'p2_shape': tf.io.FixedLenFeature(shape=(2,), dtype =tf.int64),

        'p3': tf.io.FixedLenFeature(shape= (10), dtype = tf.float32),

        'p3_shape': tf.io.FixedLenFeature([], tf.int64),

        'label': tf.io.FixedLenFeature(shape=(1), dtype=tf.float32)
    }
    parsed_example = tf.io.parse_single_example(serialized=example_proto, features=dics)
    """
    一定注意！！！
    
    对于二维和三维的tensor，解析的时候要用不定长的tf.io.VarLenFeature
    并且要把稀疏的转化为稠密的，
    然后再做reshape
    另外，解析的时候输入的type要和输出的type保持一致
    """

    p1 = tf.sparse_tensor_to_dense(parsed_example['p1'])
    p1 = tf.reshape(p1,(20, 40,3))
    p2 = tf.sparse_tensor_to_dense(parsed_example['p2'])
    p2 = tf.reshape(p2,(8, 10))
    p3 = parsed_example["p3"]

    label = parsed_example["label"]

    return p1,p2,p3,label


def get_data(filename):
    dataset = tf.data.TFRecordDataset(filenames=[filename])
    #print("读取tfrecoder 成功...")
    dataset = dataset.map(parse_function)
    #dataset = dataset.shuffle(buffer_size=C.TRAIN_DATA_SIZE)
    dataset = dataset.batch(36)
    # 用迭代器进行batch的读取
    dataset = dataset.repeat(10)  # epoch
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # 返回next_element后，反复的run它，就可以得到不同的batch
    # 这就是输入数据和训练模型时的连接点
    return next_element
