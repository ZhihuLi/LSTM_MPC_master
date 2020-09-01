from Model_5_18 import LSTM
from Model_width_5_18 import LSTM_Width
import numpy as np
import copy

height_lstm = LSTM()
height_lstm.restore()

width_lstm = LSTM_Width()
width_lstm.restore()

class Predictor():
    def __init__(self, sample_nums, timesteps):
        self.K = sample_nums # number of sample trajectories
        self.T = timesteps # timesteps to predict
        self.trajectory_length = 1700
        self.input_sequence_len = height_lstm.TIMESTEPS
        # state
        self.h_state = np.zeros([self.K, self.trajectory_length + self.T, 1])
        self.w_state = np.zeros([self.K, self.trajectory_length + self.T, 1])
        self.Welding_feed_list = np.zeros([self.K, self.trajectory_length + self.T, 1])
        self.Robot_speed_list = np.zeros([self.K, self.trajectory_length + self.T, 1])

        self.count = 0 # reset to zero every catch_up, count how many states have been changed
        self.h_target_list = np.zeros([1700])
        self.w_target_list = np.zeros([1700])

    def get_real_shape(self, Welding_feed_list, Robot_speed_list):
        h_real = height_lstm.welding_pred(Welding_feed_list, Robot_speed_list)
        w_real = width_lstm.welding_pred(Welding_feed_list, Robot_speed_list)
        return h_real, w_real

    def catch_up(self,Welding_feed_list, Robot_speed_list,  h_state, w_state, h_target_list, w_target_list, step):
        """
        update the current state and trajectory history of this episode for this sample agent
        :param h_state: np.array(step + 200, 1)
        :param w_state: np.array(step + 200, 1)
        :param h_target_list: np.array(1700, 1)
        :param w_target_list: np.array(1700, 1)
        :param step: (int) time_step
        :return:
        """
        assert (np.asarray(Welding_feed_list)).shape == (step + 200, )
        assert (np.asarray(Robot_speed_list)).shape == (step + 200, )
        assert (np.asarray(h_state)).shape == (step + 200, )
        assert (np.asarray(w_state)).shape == (step + 200, )
        assert (np.asarray(h_target_list)).shape == (1700, )
        assert (np.asarray(w_target_list)).shape == (1700, )

        # state (for input of the model and for cost)
        self.Welding_feed_list[:, :(step + 200),0] = Welding_feed_list[:]
        self.Robot_speed_list[:, :(step + 200),0] = Robot_speed_list[:]
        self.h_state[:, :(step + 200),0] = h_state[:]
        self.w_state[:, :(step + 200),0] = w_state[:]

        self.h_target_list = h_target_list
        self.w_target_list = w_target_list

        # how many states it has predicted
        self.count = 0 # reset count

    def cost_fun(self, h_predict_K, w_predict_K, h_target_K, w_target_K):
        assert h_predict_K.shape == (self.K, )
        assert w_predict_K.shape == (self.K, )

        cost = np.square(h_predict_K - h_target_K) + np.square(w_predict_K - h_target_K)
        return cost

    def predict(self, action, step):
        self.step = step
        assert action.shape == (self.K, 2)
        action_ = copy.copy(action)
        input_welding_feed_list = np.zeros([self.K, self.input_sequence_len, 1])
        input_robot_speed_list = np.zeros([self.K, self.input_sequence_len, 1])

        self.Welding_feed_list[:, step + self.count + 200, 0] \
            = self.Welding_feed_list[:, step + self.count + 199, 0] + action[:, 0]
        self.Robot_speed_list[:, step + self.count + 200, 0] \
            = self.Robot_speed_list[:, step + self.count + 199, 0] + action[:, 1]

        # update the action data for current state
        # get input sequence for model, the model need self.input_sequence_len steps sequence as input
        for i in range (self.input_sequence_len):
            input_welding_feed_list[:, i] = self.Welding_feed_list[:, i + step + self.count + 1]
            input_robot_speed_list[:, i] = self.Robot_speed_list[:, i + step + self.count + 1]

        h_predict_K, w_predict_K = self.Model_predict(input_welding_feed_list, input_robot_speed_list)

        self.h_state[:, step + self.count + 200, 0] = copy.copy(h_predict_K)
        self.w_state[:, step + self.count + 200, 0] = copy.copy(w_predict_K)

        h_target_K = self.h_target_list[step + self.count + 200]
        w_target_K = self.w_target_list[step + self.count + 200]
        # compute the cost
        cost = self.cost_fun(h_predict_K, w_predict_K, h_target_K, w_target_K)
        assert cost.shape == (self.K, )

        # update count
        self.count += 1
        return cost

    def Model_predict(self, input_welding_feed_list, input_robot_speed_list):
        input_welding_feed_list_s = copy.copy(input_welding_feed_list)
        input_robot_speed_list_s = copy.copy(input_robot_speed_list)
        h_predict_K = np.zeros([self.K])
        w_predict_K = np.zeros([self.K])
        for i in range (self.K):
            h_predict_K[i] = height_lstm.welding_pred(input_welding_feed_list_s[i, :, 0], input_robot_speed_list_s[i, :, 0])
            w_predict_K[i] = width_lstm.welding_pred(input_welding_feed_list_s[i, :, 0], input_robot_speed_list_s[i, :, 0])

        return h_predict_K, w_predict_K



