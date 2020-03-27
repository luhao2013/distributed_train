import tensorflow as tf
from utils import get_weight, get_bias, conv2d, max_pool_2x2, fc_layer, dropout_layer, conv_layer, get_lstm
import numpy as np

class My_model(object):
    def __init__(self):
        pass
    
    def forward(self):
        pass


class MLP_net(My_model):
    def __init__(self, input_dim=784, output_dim=10):
        super(MLP_net,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = 50
        self.output_dim = output_dim 

        self.learning_rate = 0.001
    
    def forward(self, x, keep_prob=1.0):
        #DNN
        #第一层全连接层
        # hidden1 = fc_layer(x, self.input_dim, self.hidden_dim,'net1_layer1')
        y = fc_layer(x, self.input_dim, self.output_dim,'layer1')
        #dropout
        # hidden1 = dropout_layer(hidden1, keep_prob, 'net1_layer1_dropout')

        #第二层全连接层
        # y = fc_layer(hidden1, self.hidden_dim, self.output_dim ,'net1_layer2', act=tf.identity)

        return y

class Conv_net(My_model):
    def __init__(self):
        super(Conv_net,self).__init__()
        self.regularizer = 0.0001 

    def forward(self, x):
        hidden = conv_layer(x ,5, 3, 6, self.regularizer)
        hidden = conv_layer(hidden, 3, 6, 12, self.regularizer)
        hidden = tf.reshape(hidden, [-1, 20*40*12])
        y = fc_layer(hidden, 20*40*12, 10, 'layer_after_conv')

        return  y

class LSTM_net(My_model):
    def __init__(self):
        super(LSTM_net,self).__init__()
        self.regularizer = 0.0001 

    def forward(self, x):
        # rate2 = tf.placeholder(tf.float32)
        with tf.variable_scope("lstm1"):
            lstm_fw_cell = get_lstm(20, 1-0.3,name= "lstm_fw")
        # Backward direction cell
        with tf.variable_scope("lstm2"):
            lstm_bw_cell = get_lstm(20, 1-0.3, name= "lstm_bw")

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                          cell_bw=lstm_bw_cell,
                                                          dtype=tf.float32,
                                                          inputs=x)

        output_fw, output_bw = outputs
        # states_fw, states_bw = states
        #print(output_fw,output_bw)
        #print(states_fw,states_bw)
        lstm_output = tf.concat([output_fw, output_bw], 2)
        #print(lstm_output.shape)
        #lstm_output = tf.reshape(lstm_output, [None, 8*40])
        lstm_output = tf.layers.flatten(lstm_output)
        #print(lstm_output.shape)
        # 输出层
        y = fc_layer(lstm_output, 8*40, 10,'layer_after_lstm', act=tf.nn.sigmoid)

        return  y

class Three_branch_net(My_model):
    def __init__(self):
        super(Three_branch_net,self).__init__()
        self.net1 = MLP_net(input_dim=10, output_dim=10)
        self.net2 = Conv_net()
        self.net3 = LSTM_net()
        self.concat_net = MLP_net(input_dim=10, output_dim=1)
    
    def forward(self, x):
        embedding1 = self.net1.forward(x[-1])
        embedding2 = self.net2.forward(x[0])
        embedding3 = self.net3.forward(x[1])
        with tf.name_scope("concat"):
            logit = tf.add(embedding1, embedding2)
            logit = tf.add(logit, embedding3)
            logit = self.concat_net.forward(logit)
        return logit
        

def MyNet(p1,p2,p3):



    with tf.variable_scope("p1_3d_tensor_process"):
        # 可以用placeholder来看shape的变化
        #p1 = tf.placeholder(tf.float32, [None, 20, 40, 3])
        #p1 = tf.reshape(p1,[None,20, 40, 3])
        # 第一层卷积：5×5×1卷积核6个 [5，5，3，6]
        W_conv1 = weight_variable([5, 5, 3, 6])
        b_conv1 = bias_variable([6])
        h_conv1 = tf.nn.relu(conv2d1(p1, W_conv1) + b_conv1)
        # 第一个pooling 层
        h_pool1 = max_pool_2x2(h_conv1)
        # 第二层卷积：3×3×6卷积核12个 [3，3，6，12]
        W_conv2 = weight_variable([3, 3, 6, 12])
        b_conv2 = bias_variable([12])
        h_conv2 = tf.nn.relu(conv2d2(h_pool1, W_conv2) + b_conv2)

        # 第二个pooling 层,输出(?, 20, 40, 12)
        h_pool2 = max_pool_2x2(h_conv2)

        #print(p1,h_conv1,h_pool1,h_conv2,h_pool2)

        # flatten
        h_pool2_flat = tf.reshape(h_pool2, [-1, 20*40*12])

        # fc1
        W_fc1 = weight_variable([20*40*12, 10])
        b_fc1 = bias_variable([10])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
        ##rate1 = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, rate=C.DROPOUT_RATE)
        # 输出层
        W_fc2 = weight_variable([10, 10])
        b_fc2 = bias_variable([10])
        p1_output = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        #print(p1_output)

    with tf.variable_scope("p2_2d_tensor_process",reuse=tf.AUTO_REUSE):
        #p2 = tf.placeholder(tf.float32,[None,8, 10])
        #p2 = tf.reshape(p2, [None,8, 10])
        # Define lstm cells with tensorflo
        # Forward direction cell
        rate2 = tf.placeholder(tf.float32)
        with tf.variable_scope("lstm1"):
            lstm_fw_cell = get_lstm(C.HIDDEN_SIZE,1-C.DROPOUT_RATE,name= "lstm_fw")
        # Backward direction cell
        with tf.variable_scope("lstm2"):
            lstm_bw_cell = get_lstm(C.HIDDEN_SIZE,1-C.DROPOUT_RATE,name= "lstm_bw")

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                          cell_bw=lstm_bw_cell,
                                                          dtype=tf.float32,
                                                          inputs=p2)

        output_fw, output_bw = outputs
        states_fw, states_bw = states
        #print(output_fw,output_bw)
        #print(states_fw,states_bw)
        lstm_output = tf.concat([output_fw, output_bw], 2)
        #print(lstm_output.shape)
        #lstm_output = tf.reshape(lstm_output, [None, 8*40])
        lstm_output = tf.layers.flatten(lstm_output)
        #print(lstm_output.shape)
        # 输出层
        W_fc_lstm = weight_variable([8*40, 10])
        b_fc_lstm = bias_variable([10],relu=False)

        p2_output = tf.nn.sigmoid(tf.matmul(lstm_output, W_fc_lstm) + b_fc_lstm)
        #print(p2_output)

    with tf.variable_scope("p3_1d_tensor_process"):
        #p3 = tf.placeholder(tf.float32,[None,10])
        #p3 = tf.reshape(p3, [None,10])
        W_fc_p3 = weight_variable([10, 10])
        b_fc_p3 = bias_variable([10], relu=False)
        output = tf.nn.sigmoid(tf.matmul(p3, W_fc_p3) + b_fc_p3)

        W_fc_p3_ = weight_variable([10, 10])
        b_fc_p3_ = bias_variable([10], relu=False)
        p3_output = tf.nn.sigmoid(tf.matmul(output, W_fc_p3_) + b_fc_p3_)
    #print(p1_output,p2_output,p3_output)

    all_concat = tf.concat([p1_output,p2_output,p3_output],1)
    W_fc_all = weight_variable([3*10, 1])
    b_fc_all = bias_variable([1], relu=False)

    y_pred = tf.nn.sigmoid(tf.matmul(all_concat, W_fc_all) + b_fc_all)
    #print(y_pred)
    return y_pred

# class Lenet5(My_model):
#     def __init__(self):
#         super(Lenet5, self).__init__()
#         self.IMAGE_SIZE = 28
#         self.NUM_CHANNELS = 1

#         self.BATCH_SIZE = 100
#         self.LEARNING_RATE_BASE =  0.005 
#         self.LEARNING_RATE_DECAY = 0.99 
#         self.REGULARIZER = 0.0001 
#         self.STEPS = 50000 
#         self.MOVING_AVERAGE_DECAY = 0.99 
#         self.OUTPUT_NODE = 10
#         self.MODEL_SAVE_PATH="./model/" 
#         self.MODEL_NAME="lenet5_model" 
#         self._build()

#     def _build(self):
#         self._loss()
#         self._optimizer()
    
#     def predict(self, x, train=False):
#         y = self.forward(x, train, self.REGULARIZER)
        
#         return y
        
    
#     def forward(self, x, train, regularizer):
#         IMAGE_SIZE = 28
#         NUM_CHANNELS = 1
#         CONV1_SIZE = 5
#         CONV1_KERNEL_NUM = 32
#         CONV2_SIZE = 5
#         CONV2_KERNEL_NUM = 64
#         FC_SIZE = 512
#         OUTPUT_NODE = 10

#         conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
#         conv1_b = get_bias([CONV1_KERNEL_NUM]) 
#         conv1 = conv2d(x, conv1_w) 
#         relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b)) 
#         pool1 = max_pool_2x2(relu1) 

#         conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM],regularizer) 
#         conv2_b = get_bias([CONV2_KERNEL_NUM])
#         conv2 = conv2d(pool1, conv2_w) 
#         relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
#         pool2 = max_pool_2x2(relu2)

#         pool_shape = pool2.get_shape().as_list() 
#         nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] 
#         reshaped = tf.reshape(pool2, [pool_shape[0], nodes]) 

#         fc1_w = get_weight([nodes, FC_SIZE], regularizer) 
#         fc1_b = get_bias([FC_SIZE]) 
#         fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b) 
#         if train: fc1 = tf.nn.dropout(fc1, 0.5)

#         fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
#         fc2_b = get_bias([OUTPUT_NODE])
#         y = tf.matmul(fc1, fc2_w) + fc2_b
#         return y 
    
#     def get_input(self):
#         self.x = tf.placeholder(tf.float32,[
#             self.BATCH_SIZE,
#             self.IMAGE_SIZE,
#             self.IMAGE_SIZE,
#             self.NUM_CHANNELS]) 
#         self.y_ = tf.placeholder(tf.float32, [None, self.OUTPUT_NODE])
#         return self.x, self.y_

#     def _loss(self):
#         x, y_ = self.get_input()
#         y = self.predict(x, True)
#         ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
#         cem = tf.reduce_mean(ce) 
#         self.loss = cem + tf.add_n(tf.get_collection('losses')) 

#     def _optimizer(self):
#         self.global_step = tf.Variable(0, trainable=False) 

#         learning_rate = tf.train.exponential_decay( 
#             self.LEARNING_RATE_BASE,
#             self.global_step,
#             # mnist.train.num_examples / self.BATCH_SIZE, 
#             55000 / self.BATCH_SIZE,
#             self.LEARNING_RATE_DECAY,
#             staircase=True)     
#         train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)

#         ema = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, self.global_step)
#         ema_op = ema.apply(tf.trainable_variables())
#         with tf.control_dependencies([train_step, ema_op]): 
#             self.train_op = tf.no_op(name='train')

#     def train_one_step(self, sess, xs, ys):
#         reshaped_xs = np.reshape(xs,(  
#             self.BATCH_SIZE,
#             self.IMAGE_SIZE,
#             self.IMAGE_SIZE,
#             self.NUM_CHANNELS))
#         _, loss_value, step = sess.run([self.train_op, self.loss, self.global_step], feed_dict={self.x: reshaped_xs, self.y_: ys}) 
#         return loss_value, step

class MLP_mnist(My_model):
    def __init__(self):
        with tf.name_scope('input'):
            self.input_dim = 28 * 28
            self.hidden_dim = 50
            self.ouput_dim = 10 

            self.learning_rate = 0.001
        self.build()

    def build(self):
        #定义程序输入数据: 
        #				x : mnist数据集中图片数据
        #				y_ : mnist数据集中图片对应的标签数据
        #				keep_prob : 定义dropout
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            self.y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
            self.keep_prob = tf.placeholder(tf.float32)
        # 全局优化次数，主要用于分布式
        self.global_step = tf.Variable(0, trainable=False) 

        #调用前向传播网络得到输出：y_out
        y_out = self.forward(self.x, self.keep_prob)
        cross_entropy = self.loss(y_out, self.y_)
        self.train_step = self.optimizer(cross_entropy)
        self.acc = self.accuracy(y_out, self.y_, self.x)

    
    def forward(self, x, keep_prob):
        #DNN
        #第一层全连接层
        hidden1 = fc_layer(x, self.input_dim, self.hidden_dim,'layer1')
        #dropout
        hidden1 = dropout_layer(hidden1, keep_prob, 'layer1_dropout')

        #第二层全连接层
        y = fc_layer(hidden1, self.hidden_dim, self.ouput_dim ,'layer2', act=tf.identity)

        return y
    
    def predict(self, x):
        logit = self.forward(x, self.keep_prob)
        y = tf.nn.softmax(logit, name='softmax')

        return y

    def loss(self, y_out, y_):
        #计算网络输出y_out和标签数据的交叉熵（cross entropy），并保存到tensorboard中
        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=y_)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)
        return cross_entropy
    
    def optimizer(self, cross_entropy):
        #将得到的cross entropy作为损失函数，利用Adam优化算法最小化损失函数
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy, global_step=self.global_step) 
        
        return train_step

    def accuracy(self, y_out, y_, x):
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

    # def train_one_step(self, mnist, merged, is_train):
    #     if is_train:
    #         summary, _ = sess.run([merged, self.train_step], feed_dict = feed_dict(mnist, is_train))
    #         return summary
    #     else:
    #         summary, acc = sess.run([merged, self.acc], feed_dict = feed_dict(mnist, is_train))
    #         return summary, acc

class Multi_net(My_model):
    def __init__(self):
        super(Multi_net,self).__init__()
        self.net1 = Only_net()
        self.net2 = Only_net()
    
    def forward(self, x, keep_prob):
        embedding1 = self.net1.forward(x, keep_prob)
        embedding2 = self.net2.forward(x, keep_prob)
        with tf.name_scope("concat"):
            logit = tf.add(embedding1, embedding2)
        return logit