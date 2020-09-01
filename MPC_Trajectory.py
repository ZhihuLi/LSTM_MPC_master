import numpy as np
import copy

class Trajectory():
    """
    collect trajectory history and preprocess the data making it more suitable for the input of predictor
    """
    def __init__(self):
        self.Welding_feed_list = [8] * 200
        self.Robot_speed_list = [12] * 200
        self.h_target_list = np.zeros([1700])
        self.w_target_list = np.zeros([1700])
        self.h_state = [1.9819496]*200
        self.w_state = [6.2371025]*200
        self.count = 0

    def set_goal(self, h_target_list, w_target_list):
        self.h_target_list = h_target_list
        self.w_target_list = w_target_list

    def update_state(self, h_real, w_real):
        h_real = copy.copy(h_real)
        w_real = copy.copy(w_real)

        if self.count > 0:
            self.h_state.append(h_real)
            self.w_state.append(w_real)
            self.count += 1
        # update state

    def get_h_state(self):
        h_state = copy.copy(self.h_state)
        return np.asarray(h_state)

    def get_w_state(self):
        w_state = copy.copy(self.w_state)
        return np.asarray(w_state)

    def get_h_target(self):
        return self.h_target_list

    def get_w_target(self):
        return self.w_target_list

    def update_welding_parameter(self, target_action_Wf, target_action_Rs):
        target_action_Wf_ = copy.copy(target_action_Wf)
        target_action_Rs_ = copy.copy(target_action_Rs)

        self.Welding_feed_list.append(self.Welding_feed_list[-1] + target_action_Wf_)
        self.Robot_speed_list.append(self.Robot_speed_list[-1] + target_action_Rs_)

    def update_shape(self, h_real, w_real):
        h_real_ = copy.copy(h_real)
        w_real_ = copy.copy(w_real)

        self.h_state.append(h_real_)
        self.w_state.append(w_real_)



