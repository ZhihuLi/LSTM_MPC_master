import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']

class LSTM():
    def __init__(self):

        self.graph = tf.Graph()
        self.sess1 = tf.Session(graph=self.graph)
        with self.sess1.as_default():
            with self.graph.as_default():
                self.HIDDEN_SIZE = 64
                self.NUM_LAYERS = 2
                self.TRAINING_STEPS = 10000
                self.BATCH_SIZE = 32
                # self.sess = tf.Session(graph=self.graph)
                self.TIMESTEPS_forward = 100
                self.TIMESTEPS_afterward = 100
                # self.TIMESTEPS = self.TIMESTEPS_forward + self.TIMESTEPS_afterward
                self.TIMESTEPS = 200
                self.LEARNING_RATE = 0.01
                self.OUTPUT_NUM = 1
                self.keep_prob = 1.0
                self.mu_train_X = [8, 15]
                self.mu_train_y = [2]
                self.sigma_train_X = [8,15]
                self.sigma_train_y = [2]

                # input the data
                # self.train_X, self.train_y, self.test_X, self.test_y = self.input_model_data()
                # define the input and output of LSTM
                self.X = tf.placeholder(tf.float32, [None, self.TIMESTEPS, 2], name='input')
                self.y = tf.placeholder(tf.float32, [None, self.OUTPUT_NUM], name='output')
                # obtain the prediction of LSTM model

                with tf.variable_scope('model'):
                    self.pred = self._build_lstm(self.X)
                    # define the loss and the training operator
                self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.pred)
                self.train_op = tf.contrib.layers.optimize_loss(self.loss, tf.train.get_global_step(),
                                                                optimizer="Adagrad", learning_rate=self.LEARNING_RATE)
                # initialize all the parameters
                self.sess1.run(tf.global_variables_initializer())

    def _build_lstm(self, X):
                cell1 = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.BasicLSTMCell(self.HIDDEN_SIZE) for _ in range(self.NUM_LAYERS)])
                cell = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=self.keep_prob)
                outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
                output = outputs[:, -1, :]
                predictions = tf.contrib.layers.fully_connected(output, self.OUTPUT_NUM, activation_fn=None)
                return predictions

    def train(self):
                for i in range(self.TRAINING_STEPS):
                    indices = np.random.choice(len(self.train_X), size=self.BATCH_SIZE)
                    X = self.train_X[indices]
                    y = self.train_y[indices]
                    _, l = self.sess1.run([self.train_op, self.loss], feed_dict={self.X: X, self.y: y})
                    # decrease the learning rate if the loss is small enough
                    if l < 0.2:
                        self.LEARNING_RATE = 0.0001
                    if i % 100 == 0:
                        print('train step: ' + str(i) + ', loss: ' + str(l))

    def save(self):
        with self.sess1.as_default():
            with self.graph.as_default():
                saver = tf.train.Saver()
                saver.save(self.sess1, './Data/lstm_params', write_meta_graph=True)
        # self.height_graph = tf.Graph()
        # with self.height_graph.as_default():
        #     saver = tf.train.Saver()
        # self.sess = tf.Session(graph=self.height_graph)
        # with self.sess.as_default():
        #     with self.height_graph.as_default():
        #         saver.save(self.sess, '../Data/lstm_params', write_meta_graph=True)

    def restore(self):
        # saver = tf.train.Saver()
        # saver.restore(self.sess1, '../Data/lstm_params')

        # self.height_graph = tf.Graph()
        # with self.height_graph.as_default():
        #     saver = tf.train.Saver()
        # self.sess = tf.Session(graph=self.height_graph)
        # with self.sess.as_default():
        #     with self.height_graph.as_default():
        #         saver.restore(self.sess, '../Data/lstm_params')

        with self.sess1.as_default():
            with self.graph.as_default():
                # self.sess1.run(tf.global_variables_initializer())
                # tf.global_variables_initializer().run()
                # saver = tf.train.import_meta_graph('../Data/lstm_params.meta')
                saver = tf.train.Saver()
                saver.restore(self.sess1, './Data/lstm_params')

    def standardization(self,data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma, mu, sigma

    def eval(self):
                predictions = []
                labels = []
                for i in range(self.TEST_LENGTH ):
                    X = self.test_X[[i]]
                    y = self.test_y[[i]]
                    p, l = self.sess1.run([self.pred, self.loss], feed_dict={'input:0': X, 'output:0': y})
                    # array date type
                    predictions.append(p[0][-1])
                    labels.append(y[0][-1])

                # rmse error
                predictions = np.array(predictions)
                labels = np.array(labels)
                # mu: [9.65353383 14.83684211  1.98342659  6.33300273]
                # sigma: [2.60548127 5.03382131 0.2475538  0.86990533]
                predictions = predictions * 0.2475538 +1.98342659
                labels = labels * 0.2475538 + 1.98342659
                # np.mean(axis=0) is to get the column mean value
                rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
                print('Root mean square error is:%3f' % rmse)
                # save the evaluation result
                eval_result = np.hstack((predictions[:, np.newaxis], labels[:, np.newaxis]))
                np.savetxt('../Data/eval_result.txt', eval_result)
                plt.figure()
                plt.plot(predictions, label='prediction')
                plt.plot(labels, label='real_value')
                plt.xlabel('Number of samples')
                plt.ylabel('Height(mm)')
                plt.legend()
                # plt.show()

    def input_model_data(self,number):
        # # 画出焊道数据的图像
        # for num in range (1,46):
        #     welding_data = np.loadtxt('../Data/数据_6月校正后（200+100）/Height_Width_Bead'+str(num)+'.txt')
        #     print(np.shape(welding_data))
        #     plot1 = plt.plot(welding_data[:, 0], welding_data[:, 1], 's', label='original values')
        #     plt.xlabel('x')
        #     plt.ylabel('y')
        #     plt.legend(loc=4)  # 指定legend的位置右下角
        #     plt.title('拟合结果')
        #     plt.show()
        #     plot2 = plt.plot(welding_data[:, 0], welding_data[:, 2], 's', label='original values')
        #     plt.xlabel('x')
        #     plt.ylabel('y')
        #     plt.legend(loc=4)  # 指定legend的位置右下角
        #     plt.title('拟合结果')
        #     plt.show()
        all_data = np.zeros((950, 4))
        one_data = np.zeros((950, 4))
        for num in range(12, 40):
            # print(num)
            raw_data_input = np.loadtxt('./Data/输入数据_6月（处理后）/input' + str(num + 1) + '.txt')
            raw_data_output = np.loadtxt('./Data/数据_6月校正后（200+100）/Height_Width_Bead' + str(num + 1) + '.txt')
            data_input = raw_data_input[650:1600, :]
            data_output = raw_data_output[650:1600, :]
            if num == 12:
                all_data[:, 0:2] = data_input[:, 1:3]
                all_data[:, 2:4] = data_output[:, 1:3]
            else:
                one_data[:, 0:2] = data_input[:, 1:3]
                one_data[:, 2:4] = data_output[:, 1:3]
                all_data = np.append(all_data, one_data, axis=0)
        # print("all_data:", np.shape(all_data))
        # print(all_data)
        _ ,mu, sigma =  self.standardization(all_data)
        print("mu:",mu)
        print("sigma:",sigma)

        self.train_X = []
        self.train_y = []
        for num in range (12,40):
            print(num)
            raw_data_input = np.loadtxt('./Data/输入数据_6月（处理后）/input'+ str(num+1) + '.txt')
            raw_data_output = np.loadtxt('./Data/数据_6月校正后（200+100）/Height_Width_Bead'+ str(num+1) + '.txt')
            # 起弧阶段去除30毫米，灭弧阶段去除15毫米
            training_examples_input = raw_data_input[(650-self.TIMESTEPS_forward+1):(1600+self.TIMESTEPS_afterward),1:3]
            training_examples_input[:,0] = (training_examples_input[:,0] - mu[0]) / sigma[0]
            training_examples_input[:, 1] = (training_examples_input[:, 1] - mu[1]) / sigma[1]
            training_examples_output = raw_data_output[650:1600,1]
            training_examples_output = (training_examples_output-mu[2]) / sigma[2]
            self.train_X, self.train_y = self.model_data_pack(training_examples_input, training_examples_output,self.train_X,self.train_y)
        self.test_X = []
        self.test_y = []
        # for num in range (number,number+1):
        for num in range(0, 12):
        #     print("number:",num)
            print(num+1)
            raw_data_input = np.loadtxt('./Data/输入数据_6月（处理后）/input'+ str(num+1) + '.txt')
            raw_data_output = np.loadtxt('./Data/数据_6月校正后（200+100）/Height_Width_Bead'+ str(num+1) + '.txt')
            # 起弧阶段去除25毫米，灭弧阶段去除10毫米
            training_examples_input = raw_data_input[(650-self.TIMESTEPS_forward+1):(1600+self.TIMESTEPS_afterward),1:3]
            training_examples_input[:, 0] = (training_examples_input[:, 0] - mu[0]) / sigma[0]
            training_examples_input[:, 1] = (training_examples_input[:, 1] - mu[1]) / sigma[1]
            training_examples_output = raw_data_output[650:1600,1]
            training_examples_output = (training_examples_output - mu[2]) / sigma[2]
            self.test_X, self.test_y = self.model_data_pack(training_examples_input, training_examples_output,self.test_X,self.test_y)

        # raw_data = np.loadtxt('../Data/Height_100.txt')
        # training_examples = raw_data[0:int(0.7 * len(raw_data)), :]
        # print(np.shape(training_examples))

        # testing_examples = raw_data[0:int(0.70833 * len(raw_data)), :]
        # testing_examples = raw_data[int(0.70833 * len(raw_data)):int(1* len(raw_data)), :]
        self.TEST_LENGTH = len(self.test_X)
        # print("length:",self.TEST_LENGTH)
        #
        # self.train_X, self.train_y = self.model_data_pack(training_examples[:, 0 : 2], training_examples[:, -1])
        # self.test_X, self.test_y = self.model_data_pack(testing_examples[:, 0 : 2], testing_examples[:, -1])
        # self.train_X ,self.mu_train_X, self.sigma_train_X =  self.standardization(self.train_X)
        # self.train_y, self.mu_train_y, self.sigma_train_y= self.standardization(self.train_y)
        # print("mu_train_X:",self.mu_train_X)
        # print("sigma_train_X:",self.sigma_train_X)
        # print("mu_train_y:", self.mu_train_y)
        # print("sigma_train_y:", self.sigma_train_y)
        # self.test_X = (self.test_X - self.mu_train_X) / self.sigma_train_X
        # self.test_y = (self.test_y - self.mu_train_y) / self.sigma_train_y
        # self.test_X = (self.test_X - self.mu_train_X) / self.sigma_train_X
        # self.test_y = (self.test_y - self.mu_train_y) / self.sigma_train_y
        # print("train_X:",np.shape(self.train_X))
        # print("train_y:",np.shape(self.train_y))

        return self.train_X, self.train_y, self.test_X, self.test_y

    def model_data_pack(self, X_seq, y_sep, X, y):
        X_list = np.array(X)
        y_list = np.array(y)
        X_list = X_list.tolist()
        y_list = y_list.tolist()
        for i in range(len(X_seq) - self.TIMESTEPS ):
            # print(i)
            X_list.append(X_seq[i : i + self.TIMESTEPS])
            y_list.append(y_sep[i : i + self.OUTPUT_NUM])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    # problem ?
    def welding_pred(self, Rs_list, Ws_list):
        Rs_list1 = np.array(Rs_list)
        Ws_list1 = np.array(Ws_list)
        # mu: [9.65353383 14.83684211  1.98342659  6.33300273]
        # sigma: [2.60548127 5.03382131 0.2475538  0.86990533]
        Rs_list1 = (Rs_list1 - 9.65353383) / 2.60548127
        Ws_list1 = (Ws_list1 - 14.83684211) / 5.03382131

        # print('Rs_list:',Rs_list1)
        # print('Ws_list:', Ws_list1)
        Param = np.transpose(np.vstack((Rs_list1, Ws_list1)))
        tem1 = []
        tem1.append(Param)
        X = np.array(tem1, dtype=np.float32)
        predition = self.sess1.run([self.pred], feed_dict={self.X : X})
        # return the array datatype
        return (predition[0][-1]* 0.2475538 + 1.98342659)

if __name__ == '__main__':

    mylstm = LSTM()
    #mylstm.simulate_data()
    mylstm.input_model_data(0)

    #print('train_X :' + str(mylstm.train_X), 'train_y:' + str(mylstm.train_y))
    # mylstm.train()
    # mylstm.save()

    mylstm.restore()
    # Rs_list = []
    # Ws_list = []
    # for i in range(mylstm.TIMESTEPS - 1):
    #     Rs_list.append(0)
    #     Ws_list.append(0)
    #
    # for i in range(5):
    #     Rs_list.append(5)
    #     Ws_list.append(5)
    #
    #
    # print(mylstm.welding_pred(Rs_list[-65 : ], Ws_list[-65 : ]))
# 1,35,10,33,26,11,31
    mylstm.eval()
#     for i in range (0,40):
#         mylstm.input_model_data(i)
#         mylstm.eval()
