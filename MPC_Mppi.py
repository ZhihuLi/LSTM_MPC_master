from MPC_Predictor import Predictor
from MPC_Trajectory import Trajectory
import numpy as np
import copy

class MPPI():
    """
    MPPI algorithm
    """
    def __init__(self, K, T, A):
        self.K = K # sample trajectories
        self.T = T # timesteps
        self.A = A # action scale
        self.lambd = 1

        self.predictor = Predictor(self.K, self.T)
        self.trajectory = Trajectory()
        self.dim_delta = 2
        self.delta_init = [0.0, 0.0]

        # action init
        self.Delta_reset()
        self.Delta_init = np.array([0.0, 0.0])
        self.cost = np.zeros([self.K])
        self.noise = np.zeros([self.K, self.T, self.dim_delta])

    def get_real_shape(self):
        h_real, w_real = self.predictor.get_real_shape(self.trajectory.Welding_feed_list[-200:], self.trajectory.Robot_speed_list[-200:])
        return h_real, w_real

    def trajectory_set_goal(self, h_target_list, w_target_list):
        self.trajectory.set_goal(h_target_list, w_target_list)

    def Delta_reset(self):
        self.Delta = np.zeros([self.T, self.dim_delta])

    def Delta_update(self):
        self.Delta = np.roll(self.Delta, -1, axis=0)
        self.Delta[-1] = self.delta_init

    def trajectory_update_state(self, h_real, w_real):
        self.trajectory.update_state(h_real, w_real)

    def compute_cost(self, step):
        action_list = np.zeros([self.K, self.T, 2])
        # 噪声使用正太分布进行采样，然后每一步的噪声中有80%是上一步的噪声
        self.noise = np.clip(np.random.normal(loc=0, scale=1, size=(self.K, self.T, self.dim_delta)), -2, 2)
        self.predictor.catch_up(self.trajectory.Welding_feed_list, self.trajectory.Robot_speed_list,
                                self.trajectory.get_h_state(), self.trajectory.get_w_state(),
                                self.trajectory.get_h_target(), self.trajectory.get_w_target(), step)
        eps = copy.copy(self.noise)

        # compute for T timesteps
        for t in range (self.T):
            # print("timestep: ", t)
            if t > 0:
                eps[:, t] = 0.8*eps[:, t-1] + 0.2*eps[:, t]
            self.noise[:, t] = copy.copy(eps[:, t])
            action = self.Delta[t] + eps[:,t]
            assert action.shape == (self.K, 2)
            cost = self.predictor.predict(action, step)
            assert cost.shape == (self.K, )
            self.cost += cost

    def compute_noise_action(self):
         beta = np.min(self.cost)
         eta = np.sum(np.exp((-1/self.lambd) * (self.cost - beta))) + 1e-6
         w = (1/eta) * np.exp((-1/self.lambd) * (self.cost - beta))

         self.Delta += [np.dot(w, self.noise[:, t]) for t in range (self.T)]

         action_list = self.Delta
         action_Wf_list = self.Delta[:, 0]
         action_Rs_list = self.Delta[:, 1]
         action_Wf = copy.copy(action_Wf_list[0])
         action_Rs = copy.copy(action_Rs_list[0])

         return action_Wf_list, action_Rs_list, action_Wf, action_Rs

    def trajectory_update_shape(self, target_action_Wf, target_action_Rs):
        self.trajectory.update_welding_parameter(target_action_Wf, target_action_Rs)
        h_real, w_real = self.get_real_shape()
        self.trajectory.update_shape(h_real, w_real)

    def cost_clear(self):
        self.cost = np.zeros([self.K])

    def get_real_shape_list(self):
        return self.trajectory.h_state, self.trajectory.w_state

    def get_real_parameter_list(self):
        return self.trajectory.Welding_feed_list, self.trajectory.Robot_speed_list
