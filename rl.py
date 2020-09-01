
import numpy as np
import tensorflow as tf
#####################  hyper parameters  ####################

LR_A = 0.0013    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 20000
BATCH_SIZE = 32

# s = (e(t), \delta e(t), \delta\delta e(t), kp, ki, kd, 1/0)
# a = (#(\dleta kp, \delta ki, \delta kp)
# r = as in the paper. 1/(1 + exp...)
class DDPG(object):

    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]  # ???
        self.S = tf.placeholder(tf.float32, [None, s_dim], name='s')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], name='s_') # S_ represent the next state
        self.R = tf.placeholder(tf.float32, [None, 1], name='r') # reward

        with tf.variable_scope('Actor'):
            # self.a is the action for current state S(to calculate Qt(s,a))  (Also, known as behavior action by the behavior policy)
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            # a_ is the action for next state S_ (to calculate Qt+1(s_,a_))
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True) # critic the current Q_t(s,a)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False) # critic next Q_t+1(s_,a_)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')

        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement(soft uodate)
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_

        # in the feed_dic for the td_error, the self.a should change to actions in memory
        # to minimize the TD error
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        # to maximize the actor performance (SGA) q is the judgement of the actor network
        a_loss = -tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def _build_a(self, s, scope, trainable): # 构造actor网络
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 300, activation=tf.nn.tanh, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable): # 构造critic网络
        with tf.variable_scope(scope):
            n_l1 = 300
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1) # why + tf.matmul(a, w1_a)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    # based on current state, choose an action to environment.
    def choose_action(self, s):
        action = self.sess.run(self.a, {self.S: s[None, :]})[0]  # why [0]?
        return action

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))  # splice as a row
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition    # store the experience
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:      # indicator for learning, only pointer > MEMORY_CAPACITY, begin to train.
            self.memory_full = True

    def save(self):
        saver = tf.train.Saver()
        # why write_meta_graph is False ???
        saver.save(self.sess, './Data/RL_params_1', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './Data/RL_params_1')

