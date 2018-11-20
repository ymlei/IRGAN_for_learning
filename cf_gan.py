import tensorflow as tf
from dis_model import DIS
from gen_model import GEN
import cPickle
import numpy as np
import utils as ut
import multiprocessing

cores = multiprocessing.cpu_count()

#########################################################################################
# Hyper-parameters
#########################################################################################
EMB_DIM = 5 #???这个是人为设定的还是算出来的?
USER_NUM = 943 #训练数据中user的数量/代号
ITEM_NUM = 1683
BATCH_SIZE = 16
INIT_DELTA = 0.05

all_items = set(range(ITEM_NUM))
workdir = 'ml-100k/'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'

#########################################################################################
# Load data
#########################################################################################
user_pos_train = {} #上传训练数据
with open(workdir + 'movielens-100k-train.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]

user_pos_test = {}#上传测试数据
with open(workdir + 'movielens-100k-test.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]

all_users = user_pos_train.keys()
all_users.sort()


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def simple_test_one_user(x):
    rating = x[0]
    u = x[1]

    test_items = list(all_items - set(user_pos_train[u]))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])

#################################################################搞清楚这个函数在干什么
def simple_test(sess, model):
    result = np.array([0.] * 6)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    test_users = user_pos_test.keys()
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret


def generate_for_d(sess, model, filename): #generate_for_d(sess, generator, DIS_TRAIN_FILE)  DIS_TRAIN_FILE = workdir + 'dis-train.txt'
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]

        rating = sess.run(model.all_rating, {model.u: [u]})#和下面得到的all_logits不同,得到的是all_rating,本来有很多的rating,但只取第一列
        rating = np.array(rating[0]) / 0.2  # Temperature
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating) #计算标签和数据的分布概率,单个rating/sum rating

        neg = np.random.choice(np.arange(ITEM_NUM), size=len(pos), p=prob) #以prob这个概率,从ITEM_NUM里随机选取pos个项目
        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))  #最后得到采样文件的信息,但直接用的是prob没有重点采样


def main():
    print "load model..."
    param = cPickle.load(open(workdir + "model_dns_ori.pkl"))  #载入model_dns_ori.pkl,导入用户项目等参数?可能是.txt文件相同信息的不同格式
    generator = GEN(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.0 / BATCH_SIZE, param=param, initdelta=INIT_DELTA,
                    learning_rate=0.001)
    discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.1 / BATCH_SIZE, param=None, initdelta=INIT_DELTA,
                        learning_rate=0.001)

    #------------------------------------------------------动态申请显存
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) #初始化
    #-----------------------------------------------------------

    print "gen ", simple_test(sess, generator)#这里不知道在干什么
    print "dis ", simple_test(sess, discriminator)

    dis_log = open(workdir + 'dis_log.txt', 'w') #pkl文件和txt文件的关系不明确
    gen_log = open(workdir + 'gen_log.txt', 'w')
##########################################################################################输入和初始化
    # minimax training------------------------------------------------------------------------------D部分生成负样本和正样本结合
    best = 0.  #???
    for epoch in range(15):
        if epoch >= 0:
            for d_epoch in range(100):
                if d_epoch % 5 == 0:
                    generate_for_d(sess, generator, DIS_TRAIN_FILE) #根据分布率得到采样文件,根据用户得到item
                    train_size = ut.file_len(DIS_TRAIN_FILE) #从dis-train.txt得到文件数
        ############################################################################################搭建模型前让雷一鸣搞清楚这一段的原因            
                index = 1
                while True:
                    if index > train_size:
                        break
                    if index + BATCH_SIZE <= train_size + 1: #这个判别条件的原理不知道
                        input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index, BATCH_SIZE)
                    else:
                        input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index,
                                                                                train_size - index + 1)
                    index += BATCH_SIZE
        ###########################################################################################################
                    _ = sess.run(discriminator.d_updates,  #猜测是不断修改模型的过程
                                 feed_dict={discriminator.u: input_user, discriminator.i: input_item,
                                            discriminator.label: input_label}) #给空出来的的placeholder传输值,定义好整个图之后才会用sess.run??把从G中得到的参数给了D

####################### Train G################################policy gradient生成item更新参数#########################
            for g_epoch in range(50):  # 50
                for u in user_pos_train: #u训练数据集
                    sample_lambda = 0.2
                    pos = user_pos_train[u]
                #-----------------------------------------------------------------------------很重要,需要明白一下
                    rating = sess.run(generator.all_logits, {generator.u: u})#{}中是字典型数据,猜测是在所有的猜测数据中检索字典型数据或者仅仅是把这个数据输入到模型中去,放到图中开始运行,为了取回fetch内容,在参数中加入需要输入的数据
                    exp_rating = np.exp(rating)#计算e指数
                    prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta 分布率

                    pn = (1 - sample_lambda) * prob
                    pn[pos] += sample_lambda * 1.0 / len(pos)
                    # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta
                #不明白---------------------------------但很重要####################p和pn####################3
                    sample = np.random.choice(np.arange(ITEM_NUM), 2 * len(pos), p=pn)  #这里应该是选择文档的过程给出item索引,在索引中随机以p这个概率选取2 * len(pos)个
                    #得到概率并且抽样了 根据用户的输入得到item
                    ###########################################################################
                    # Get reward and adapt it with importance sampling在D中才有reward
                    ###########################################################################
                    reward = sess.run(discriminator.reward, {discriminator.u: u, discriminator.i: sample}) #用户和抽样的项目放入D中计算反馈,sess实体运行图取回括号中的参数  sess.run(fetches,feed_dict),给placeholder创建出来的变量赋值
                    reward = reward * prob[sample] / pn[sample]
                    ###########################################################################,挑出来的是训练数据中的采样和重点sample
                    # Update G
                    ###########################################################################
                    _ = sess.run(generator.gan_updates,
                                 {generator.u: u, generator.i: sample, generator.reward: reward})

                result = simple_test(sess, generator)
                print "epoch ", epoch, "gen: ", result
                buf = '\t'.join([str(x) for x in result]) #把结果搞到一起了
                gen_log.write(str(epoch) + '\t' + buf + '\n') #输出准确率
                gen_log.flush()

                p_5 = result[1]
                if p_5 > best:
                    print 'best: ', result
                    best = p_5
                    generator.save_model(sess, "ml-100k/gan_generator.pkl")

    gen_log.close()
    dis_log.close()


if __name__ == '__main__':
    main()
