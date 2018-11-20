import tensorflow as tf
import cPickle


class DIS():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []
 
    #----------------------------------------------------------该部分在处理D的输入
        with tf.variable_scope('discriminator'): #with语句应该是能处理异常    嵌入层的随机定义?
            if self.param == None #无参时
                self.user_embeddings = tf.Variable( #该函数用于定义图变量
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta, #从均匀分布中输出随机值
                                      dtype=tf.float32)) #[]中是输出张量的形状,生成的值在min-max范围内遵循均匀分布,含min不含max
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.itemNum])) #一维数组里放itemNum个0值
            else: #有参时,参数应为三列
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(self.param[2])

        self.d_params = [self.user_embeddings, self.item_embeddings, self.item_bias] #嵌入层随机安排好之后变成D的参数,所以D的参数转变为嵌入层的集合

        # placeholder definition---------------------------------------定义参数,运行时要传入参数,用session进行运行的过程--在下方save-model定义中
        #---------------------------------------------------------------该部分在处理数据,从自己挑的数据中得到预测logits然后计算偏置,一直学习
        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u) #在user_embeddings张量里面寻找第u(即传过来的参数)个元素
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)#在item....
        self.i_bias = tf.gather(self.item_bias, self.i)#在item_bias中返回位置i的元素

        #-----------------------------------------------------------------------------------------------------------
        
        self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias#用户和项目对应元素相乘,得到的新矩阵按行求和之后加上项目的偏置--可能是想得到理论的结果
        self.pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
                                                                logits=self.pre_logits) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias)
        ) #sigmoid逻辑损失,衡量分类任务中的概率误差,加上lamda乘以l2范数(防止过拟合)的各个嵌入层的损失

        #----------------------------------------------------------------设置优化的过程
        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate) #设定优化率
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)  #理论的结果和原始参数,pre_loss是待减小的值,优化更新训练的模型参数

        #------------------------------------------------------------------------------
        
        self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding),
                                           1) + self.i_bias #可能时优化更新之后的预测值,跟之前的式子没有分别
      
        self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5) #按照目标预测进行分类了
        
        #-----------------------------------------------------------------------------------------

        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias #得到线性方程

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias #还是之前的式子
        self.NLL = -tf.reduce_mean(tf.log(
            tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]), self.i))#reshape成一行若干列的矩阵,softmax解决n分类的概率问题
        ) #得到位置? 计算log后,通过张量的维数计算平均值,取负 #这一步不知道在干啥子
        # for dns sample
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias

    def save_model(self, sess, filename):
        param = sess.run(self.d_params)
        cPickle.dump(param, open(filename, 'w'))
