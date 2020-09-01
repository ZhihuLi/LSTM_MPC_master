# 输入各个库
from Model_5_18 import LSTM
from Model_width_5_18 import LSTM_Width
import numpy as np
import pyglet
import PID
import time
import matplotlib
matplotlib.use("TkAgg")

# 设置超参数
KP = 0.1
KI = 0.1
KD = 0.1
KP2 = 0.1
KI2 = 0.1
KD2 = 0.1
learning_rate = 1.0

height_lstm = LSTM()
height_lstm.restore()

width_lstm = LSTM_Width()
width_lstm.restore()

filename1 = "./Data/prediction_net_Height1.txt"
dataset1 = np.loadtxt(filename1)
filename2 = './Data/prediction_net_Width1.txt'
dataset2 = np.loadtxt(filename2)
dataset = np.zeros((8166,4))
dataset[:,0] = dataset1[:,0]
dataset[:,1] = dataset1[:,1]
dataset[:,2] = dataset1[:,2]
dataset[:,3] = dataset2[:,2]
dataset = np.array(dataset)

class Welding_Env():# 该函数的作用是搭建焊接环境，供强化学习训练和验证时交互使用
    viewer = None

    def __init__(self):# 初始化
        ## define the original values of the variables
        self.Kp = KP
        self.Ki = KI
        self.Kd = KD
        self.Kp2 = KP2
        self.Ki2 = KI2
        self.Kd2 = KD2
        self.H_error = 0
        self.W_error = 0
        self.H_del_error = 0
        self.W_del_error = 0
        self.H_error_last = 0
        self.W_error_last = 0
        self.H_target = 0
        self.W_target = 0
        self.H_prediction = 0
        self.W_prediction = 0
        self.H_actual = 0
        self.W_actual = 0

        self.on_goal = 0
        self.goal = 0
        self.action_bound = [-1, 1]
        self.action_dim = 6
        self.state_dim = 8
        self.counter = 0
        # !!!!! wrong coding
        self.Rs_list = [8] *200
        self.Ws_list = [12] *200
        self.Rs = 8
        self.Ws = 12

        self.delta_A = 0
        self.delta_B = 0
        self.delta_Rs = 0
        self.delta_Ws = 0
        self.flag = 0

    def step(self, action):# 焊接过程走一步的函数，即根据强化学习的动作值得出下一步焊道的形状
        done = False
        self.counter += 1
        # obtain error and delta error
        self.H_error = self.H_target - self.H_prediction# 误差值
        self.H_del_error = self.H_error - self.H_error_last# 误差值的差分
        self.H_error_last = self.H_error

        self.W_error = self.W_target - self.W_prediction# 误差值
        self.W_del_error = self.W_error - self.W_error_last# 误差值的差分
        self.W_error_last = self.W_error

        # pid control the robot speed
        # 底层为PID控制器，强化学习的作用为调节PID控制器的参数1
        # pid1 = PID.PID(self.Kp + learning_rate * action[0]*0.1, self.Ki + learning_rate * action[1]*0.1, self.Kd + learning_rate * action[2]*0.1 )
        # pid2 = PID.PID(self.Kp2 + learning_rate * action[3]*0.1, self.Ki2 + learning_rate * action[4]*0.1, self.Kd2 + learning_rate * action[5]*0.1)
        pid1 = PID.PID(self.Kp, self.Ki, self.Kd)
        pid2 = PID.PID(self.Kp2, self.Ki2 , self.Kd2)
        action_np = np.zeros((6,1),dtype="float32")
        action_np[0,0] = action[0]
        action_np[1,0] = action[1]
        action_np[2,0] = action[2]
        action_np[3,0] = action[3]
        action_np[4, 0] = action[4]
        action_np[5, 0] = action[5]

        self.delta_A = pid1.update(self.H_error,self.H_del_error)
        self.delta_B = pid2.update(self.W_error,self.W_del_error)

        # 28组数据反解
        self.delta_Rs = -12.1674 * self.delta_A + 3.0336 * self.delta_B
        self.delta_Ws = -26.0291 * self.delta_A + 2.1745 * self.delta_B
        self.Rs += self.delta_Rs
        self.Ws += self.delta_Ws

        self.Rs = np.clip(self.Rs, 3, 13)
        self.Ws = np.clip(self.Ws, self.Rs, 2*self.Rs)
        print(self.Rs,self.Ws)
        self.H_actual = height_lstm.welding_pred(self.Rs_list[-height_lstm.TIMESTEPS:],
                                                     self.Ws_list[-height_lstm.TIMESTEPS:])
        self.W_actual = width_lstm.welding_pred(self.Rs_list[-width_lstm.TIMESTEPS:],
                                                    self.Ws_list[-width_lstm.TIMESTEPS:])

        for num in range(0, 60):
            self.Rs_list.append(self.Rs[0]) ## wrong ???
            self.Ws_list.append(self.Ws[0])
        # print("original:",np.shape(self.Rs_list))
        # 将新的焊接参数序列输入焊接过程模型，得到新的焊道形状预测值
        self.H_prediction = height_lstm.welding_pred(self.Rs_list[-height_lstm.TIMESTEPS:], self.Ws_list[-height_lstm.TIMESTEPS:])
        self.W_prediction = width_lstm.welding_pred(self.Rs_list[-width_lstm.TIMESTEPS:], self.Ws_list[-width_lstm.TIMESTEPS:])
        # print(self.Rs_list[-1 : ], self.Ws_list[-1 : ], self.H_prediction, self.target)
        for num in range(0, 59):
            self.Rs_list.pop() ## wrong ???
            self.Ws_list.pop()
        # print("after:", np.shape(self.Rs_list))

        # reward
        # if abs(self.H_target - self.H_prediction) < 0.002 and abs(self.W_target - self.W_prediction) < 0.005:
        #     self.on_goal += 1
        #     r = 1
        #     if self.on_goal > 80:
        #         done = True
        # else:
        #     r = 1 / (1 + np.exp((abs(self.H_target - self.H_prediction) + abs(self.W_target - self.W_prediction)))) - 0.5
        #     # r = - (abs(self.H_target - self.H_prediction) + abs(self.W_target - self.W_prediction))
        #     # self.on_goal = 0
        if abs(self.H_target - self.H_prediction) < 0.002 and abs(self.W_target - self.W_prediction) < 0.005:
            self.on_goal += 1
            r = 1
            if self.on_goal > 80:
                done = True
        else:
            r = 1 / (1 + np.exp(abs(self.H_target - self.H_prediction) + abs(self.W_target - self.W_prediction))) - 0.5
            self.on_goal = 0

        # if self.flag>60:
        #     done = True
        #     r = -100
        # state
        s = np.hstack(((self.H_prediction - 2), self.H_error, self.H_del_error, ((self.W_prediction - 5) / 3),self.W_error, self.W_del_error, (self.Rs - 8) / 5, ((self.Ws - 14.5) / 11.5)))
        # s = np.hstack(((self.H_prediction - 2), self.H_target, ((self.W_prediction - 5) / 3),
        #                self.W_target))
        #print(s,r)
        return s, r, done

    # set the initilize values
    def reset(self):
        self.flag = 0
        self.H_error = 0
        self.H_del_error = 0
        self.H_error_last = 0

        self.W_error = 0
        self.W_del_error = 0
        self.W_error_last = 0

        self.Rs = 8
        self.Ws = 12
        #self.Wf = 10
        # self.H_prediction = np.random.uniform(low=1.65, high=2.4, size=1)
        # self.W_prediction = np.random.uniform(low=4.7, high=8, size=1)
        # self.H_target = np.random.uniform(low=1.65, high=2.4, size=1)
        # self.W_target = np.random.uniform(low=4.7, high=8, size=1)
        ran_num1 = np.random.randint(0,8166)
        self.H_prediction = [dataset[ran_num1,2]]
        self.W_prediction = [dataset[ran_num1,3]]
        ran_num2 = np.random.randint(0, 8166)
        self.H_target =[dataset[ran_num2,2]]
        self.W_target = [dataset[ran_num2,3]]
        # self.H_prediction = [dataset[ran_num,2]]
        # print("self.H:",self.H_prediction)
        self.H_prediction = np.array(self.H_prediction)
        self.W_prediction = np.array(self.W_prediction)
        self.H_target = np.array(self.H_target)
        self.W_target = np.array(self.W_target)

        s = np.hstack(((self.H_prediction - 2), self.H_error, self.H_del_error, ((self.W_prediction - 5) / 3),self.W_error, self.W_del_error, (self.Rs - 8) / 5, ((self.Ws - 14.5) / 11.5)))
        # s = np.hstack(((self.H_prediction - 2), self.H_target, ((self.W_prediction - 5) / 3),
        #                self.W_target))
        return s


    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.H_prediction, self.H_target)

        self.viewer.render(self.H_prediction, self.H_target)

    def sample_action(self):
        return np.random.rand(2) - 0.5



class Viewer(pyglet.window.Window):# 显示函数

    def __init__(self, Y_t, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.Y_t = Y_t
        # print("init")
        self.goal_info = goal

        self.batch = pyglet.graphics.Batch()  # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [100, 100 + self.goal_info,  # location
                     100, 105 + self.goal_info,
                     300, 105 + self.goal_info,
                     300, 100 + self.goal_info]),
            ('c3B', (86, 109, 249) * 4))  # color
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,  # location
                     250, 255,
                     255, 255,
                     255, 250]),
            ('c3B', (249, 86, 86) * 4,))  # color

    def render(self, H_prediction, target):
        self.Y_t = H_prediction
        self._update_arm(H_prediction, target)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
        # print(self.goal_info['h'])

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self, H_prdiction, target):
        # update goal

        self.goal.vertices = (
            100, 100 + target * 10,
            100, 105 + target * 10,
            300, 105 + target * 10,
            300, 100 + target * 10)

        # update arm
        #height = self.Y_t
        #print(H_prdiction)
        self.arm1.vertices = (
            195, 100 + H_prdiction * 10,
            195, 105 + H_prdiction * 10,
            205, 105 + H_prdiction * 10,
            205, 100 + H_prdiction * 10)

if __name__ == '__main__':
    env = Welding_Env()
    while True:
        print("new epoch")
        s = env.reset()
        for i in range(100):
            env.render()
            env.step(env.sample_action())
            time.sleep(0.01)





