"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
# 输入各个库
from rl import DDPG
from Env_PID import Welding_Env

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import copy

# 超参数定义
MAX_EPISODES = 400 #最大训练场景数
MAX_EP_STEPS = 200 #每个场景最大训练步数
ON_TRAIN = False
#设置焊接环境，定义状态和动作的维数和动作的取值范围
env = Welding_Env()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# 设置DDPG方法
# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)
cumulative_reward = [] # 累计奖励值
steps_until_done = [] # 每个场景结束前的训练次数
prediction_error = [] # 预测的误差
prediction_error_width = [] # 预测的误差
#action_list = []

# 训练函数
def train():
    # for every episode
    for i in range(MAX_EPISODES):
        s = env.reset()# 每个场景的初始状态设定
        ep_r = 0.# 初始化累计奖励和为0
        # for every step
        for j in range(MAX_EP_STEPS):
            env.render()# 调用显示函数，显示当前的预测值和目标值
            # based on DDPG, choose an action
            # if i<100:
            #     a1 = np.random.uniform(low=-1, high=1, size=1)
            #     a2 = np.random.uniform(low=-1, high=1, size=1)
            #     a3 = np.random.uniform(low=-1, high=1, size=1)
            #     a4 = np.random.uniform(low=-1, high=1, size=1)
            #     # a5 = np.random.uniform(low=-0.1, high=0.1, size=1)
            #     # a6 = np.random.uniform(low=-0.1, high=0.1, size=1)
            #     # a7 = np.random.uniform(low=-0.1, high=0.1, size=1)
            #     # a8 = np.random.uniform(low=-0.1, high=0.1, size=1)
            # a = np.array([a1[0],a2[0],a3[0],a4[0]])
            #     print(a)
            # else:
            a = rl.choose_action(s)  # 根据当前状态s选择当前要采取的动作a
            s_, r, done = env.step(a)  # 选取动作以后，让当前的动作作用于环境，得到下一步的状态和采取这个动作的奖励值，以及判定当前场景是否结束
            rl.store_transition(s, a, r, s_)  # 将得到的当前状态，动作，奖励值，下一步的状态作为一个训练元组储存到经验回放池，待储存满后在进行训练，消除顺序的影响
            ep_r += r  # 累计奖励和为每一步的奖励累加
            if rl.memory_full:  # 如果经验回放池满，则开始进行学习
                rl.learn()  # 学习过程：利用经验回放池中的元组进行神经网络的学习
            # uodate the state to next state
            s = s_  # 更新状态，准备进行下一步
            print(s, a)  # 打印当前状态和动作值
            if done or j == MAX_EP_STEPS - 1:  # 如果场景触发结束条件或者达到最大步数
                print('Ep: %i | %s | ep_r: %.1f | step: %i | Error: %env.error | action: %a' %
                      (i, '----' if not done else 'done', ep_r, j, env.H_error, a))
                cumulative_reward.append(ep_r)  # 存储累计奖励和
                steps_until_done.append(j)  # 存储总步数
                break
    rl.save()  # 训练结束后将所得到的模型存储，以供验证时调用

def eval(): # 验证强化学习效果函数
    rl.restore() # 将强化学习得到的模型进行恢复
    env.render() # 调用显示函数，显示预测值和真实值
    env.viewer.set_vsync(True)
    s = env.reset() # 初始化状态


    feedback_list = []
    setpoint_list = []
    feedback_list_width = []
    setpoint_list_width = []
    # target_all = np.zeros((75,1), float) # 每条焊道长度为150mm, 1mm有10个采样点，共有1500个采样点，每2mm改变一次焊道高度，共有75个不同的焊道高度值
    #time_list.append(0)
    for i in range(0, 1500):  # 设置目标高度值,每条焊道长度为150mm, 1mm有10个采样点，共有1500个采样点
    #     # 两个正弦
    #     # H_target = 2.0 + 0.2 * np.sin((i + 10) / 50)  # 将目标高度值设置为正弦形状，还可设置为其他不同的形状
    #     # W_target = 5.7 + 0.5 * np.sin(i / 100)
    #         # 一个正弦，一个阶跃
    #     H_target = 2.0 + 0.2 * np.sin((i + 10) / 50)  # + 0.2 * np.sin((i + 10) / 2.5)  # 将目标高度值设置为正弦形状，还可设置为其他不同的形状
    #     if 0 <= i <= 300:
    #         W_target = 5.2
    #     elif 300 < i <= 600:
    #         W_target = 6.2
    #     elif 600 < i <= 900:
    #         W_target = 5.2
    #     elif 900 < i <= 1200:
    #         W_target = 6.2
    #     elif 1200 < i <= 1500:
    #         W_target = 5.2 # 将目标高度值设置为阶跃形状，还可设置为其他不同的形状
        #     # W_target = 6.5 + 0.1 * np.sin(i / 5)
    # for i in range(0, 75):  # 设置目标高度值,每条焊道长度为150mm, 1mm有10个采样点，共有1500个采样点
    #
    #     # 一个直线，一个正弦
        H_target = 2.0 + 0.1 * np.sin((i + 100) / 100)  # 将目标高度值设置为正弦形状，还可设置为其他不同的形状
        W_target = 6.0 + 0.5 * np.sin(i / 100)
        # if 0 <= i <= 300:
        #     W_target = 5.2 +i/300
        # elif 300 < i <= 600:
        #     W_target = 6.2 -(i-300)/ 300
        # elif 600 < i <= 900:
        #     W_target = 5.2+(i-600)/ 300
        # elif 900 < i <= 1200:
        #     W_target = 6.2-(i-900)/300
        # elif 1200 < i <= 1500:
        #     W_target = 5.2+(i-1200)/300 # 将目标高度值设置为阶跃形状，还可设置为其他不同的形状
        # if 0 <= i <= 300:
        #     W_target = 5.2
        # elif 300 < i <= 600:
        #     W_target = 6.2
        # elif 600 < i <= 900:
        #     W_target = 5.2
        # elif 900 < i <= 1200:
        #     W_target = 6.2
        # elif 1200 < i <= 1500:
        #     W_target = 5.2 # 将目标高度值设置为阶跃形状，还可设置为其他不同的形状
        # 一个直线，一个斜坡
        # H_target = 2.2  # + 0.2 * np.sin((i + 10) / 2.5)  # 将目标高度值设置为正弦形状，还可设置为其他不同的形状
        # if 0 <= i <= 300:
        #     W_target = 5.5 + i / 150
        # elif 300 < i <= 600:
        #     W_target = 7.5 - (i - 300) / 150
        # elif 600 < i <= 900:
        #     W_target = 5.5 + (i - 600) / 150
        # elif 900 < i <= 1200:
        #     W_target = 7.5 - (i - 900) / 150
        # elif 1200 < i <= 1500:
        #     W_target = 5.5 + (i - 1200) / 150 # 将目标高度值设置为阶跃形状，还可设置为其他不同的形状
        # W_target = 5.7 + 0.5 * np.sin((i + 10) / 100)  # 将目标高度值设置为正弦形状，还可设置为其他不同的形状
        # if 0 <= i <= 300:
        #     H_target = 1.8
        # elif 300 < i <= 600:
        #     H_target = 2.2
        # elif 600 < i <= 900:
        #     H_target = 1.8
        # elif 900 < i <= 1200:
        #     H_target = 2.2
        # elif 1200 < i <= 1500:
        #     H_target = 1.8  # 将目标高度值设置为阶跃形状，还可设置为其他不同的形状
        # H_target = 2
        # # W_target = 6.5 + 0.1 * np.sin(i / 5)
        # 一个直线，一个阶跃
        # H_target = 2.2  # + 0.2 * np.sin((i + 10) / 2.5)  # 将目标高度值设置为正弦形状，还可设置为其他不同的形状
        # if 0 <= i <= 60:
        #     W_target = 5.2
        # elif 60 < i <= 120:
        #     W_target = 6.2
        # elif 120 < i <= 180:
        #     W_target = 5.2
        # elif 180 < i <= 240:
        #     W_target = 6.2
        # elif 240 < i <= 300:
        #     W_target = 5.2 # 将目标高度值设置为阶跃形状，还可设置为其他不同的形状
        #     # W_target = 6.5 + 0.1 * np.sin(i / 5)
        # 两个正弦
        # H_target = 2.0 + 0.2 * np.sin((i + 10) / 2.5)  # 将目标高度值设置为正弦形状，还可设置为其他不同的形状
        # W_target = 5.7 + 0.5 * np.sin(i / 5)
        # # 一个正弦，一个斜坡
        # H_target = 2.0 + 0.2 * np.sin((i + 10) / 2.5)  # + 0.2 * np.sin((i + 10) / 2.5)  # 将目标高度值设置为正弦形状，还可设置为其他不同的形状
        # if 0 <= i <= 15:
        #     W_target = 5.2 +i/15
        # elif 15 < i <= 30:
        #     W_target = 6.2 -(i-15)/ 15
        # elif 30 < i <= 45:
        #     W_target = 5.2+(i-30)/ 15
        # elif 45 < i <= 60:
        #     W_target = 6.2-(i-45)/15
        # elif 60 < i <= 75:
        #     W_target = 5.2+(i-60)/15 # 将目标高度值设置为阶跃形状，还可设置为其他不同的形状
        #     # 一个正弦，一个阶跃
        # H_target = 2.0 + 0.2 * np.sin((i + 10) / 2.5)  # + 0.2 * np.sin((i + 10) / 2.5)  # 将目标高度值设置为正弦形状，还可设置为其他不同的形状
        # if 0 <= i <= 15:
        #     W_target = 5.2
        # elif 15 < i <= 30:
        #     W_target = 6.2
        # elif 30 < i <= 45:
        #     W_target = 5.2
        # elif 45 < i <= 60:
        #     W_target = 6.2
        # elif 60 < i <= 75:
        #     W_target = 5.2 # 将目标高度值设置为阶跃形状，还可设置为其他不同的形状
        #     # W_target = 6.5 + 0.1 * np.sin(i / 5)

        env.H_target = H_target
        env.W_target = W_target
    #     if 0 <= i <= 15:
    #         env.H_target = 1.9
    #     elif 15 < i <=30:
    #         env.H_target = 2.2
    #     elif 30 < i <= 45:
    #         env.H_target = 1.9
    #     elif 45 < i <= 60:
    #         env.H_target = 2.2
    #     elif 60 < i <= 75:
    #         env.H_target = 1.9  # 将目标高度值设置为阶跃形状，还可设置为其他不同的形状
    #
    #     # if 0 <= i <= 700:
    #     #     env.H_target = 1.95
    #     # elif 700 < i <= 1000:
    #     #     env.H_target = 2
    #     # elif 1000 < i <= 1500:
    #     #     env.H_target = 1.95# 将目标高度值设置为阶跃形状，还可设置为其他不同的形状
    #
    #     # env.H_target = 2 + 0.1 * np.sin((i) / 50)
    #     # env.H_target = 2.0
    #     # for i in range(0,1500): # 设置目标高度值,每条焊道长度为150mm, 1mm有10个采样点，共有1500个采样点
    #     #     if 0 <= i <= 400:
    #     #         env.W_target = 4
    #     #     elif 400 < i <= 800:
    #     #         env.W_target = 5
    #     #     elif 800 < i <= 1200:
    #     #         env.W_target = 4
    #     #     elif 1200 < i <= 1500:
    #     #         env.W_target = 4.5
    #     # elif 1300 < i <= 1500:
    #     #     env.W_target = 4
    #     # env.H_target = 2 # 将目标高度值设置为正弦形状，还可设置为其他不同的形状
    #     env.H_target = 2.05 + 0.2 * np.sin((i + 10) / 2.5)  # 将目标高度值设置为正弦形状，还可设置为其他不同的形状
    #     env.W_target = 6.1 + 0.4 * np.sin((i) / 5)
    # # for i in range(0, 1500):# 设置目标高度值,每条焊道长度为150mm, 1mm有10个采样点，共有1500个采样点
    #
    #     if 0 <= i <= 400:
    #         env.H_target = 1.9
    #     elif 400 < i <= 700:
    #         env.H_target = 2.2
    #     elif 700 < i <= 1000:
    #         env.H_target = 1.9
    #     elif 1000 < i <= 1300:
    #         env.H_target = 2.2
    #     elif 1300 < i <= 1500:
    #         env.H_target = 1.9# 将目标高度值设置为阶跃形状，还可设置为其他不同的形状
    #
    #     # if 0 <= i <= 700:
    #     #     env.H_target = 1.95
    #     # elif 700 < i <= 1000:
    #     #     env.H_target = 2
    #     # elif 1000 < i <= 1500:
    #     #     env.H_target = 1.95# 将目标高度值设置为阶跃形状，还可设置为其他不同的形状
    #
    #     # env.H_target = 2 + 0.1 * np.sin((i) / 50)
    #     # env.H_target = 2.0
    # # for i in range(0,1500): # 设置目标高度值,每条焊道长度为150mm, 1mm有10个采样点，共有1500个采样点
    # #     if 0 <= i <= 400:
    # #         env.W_target = 4
    # #     elif 400 < i <= 800:
    # #         env.W_target = 5
    # #     elif 800 < i <= 1200:
    # #         env.W_target = 4
    # #     elif 1200 < i <= 1500:
    # #         env.W_target = 4.5
    #     # elif 1300 < i <= 1500:
    #     #     env.W_target = 4
    #     # env.H_target = 2 # 将目标高度值设置为正弦形状，还可设置为其他不同的形状
    #     env.W_target = 6.1+ 0.4* np.sin((i) / 100)

        a = rl.choose_action(s)# 根据当前的状态选取动作
        #action_list.append(a)
        s, r, done = env.step(a)# 根据动作更新状态并得到奖励值以及判定是否结束
        s = np.array(s)# 将状态进行格式转化，以便进行矩阵操作
        speed = s[np.newaxis,-2:]# 根据状态的定义，后两维便是速度
        speed[0,0] = speed[0,0] * 5 +8
        speed[0,1] = speed[0,1] * 11.5 + 14.5
        if i ==0:
            speed_list = speed
            # target_all[int(i / 20)] = env.target
        # elif i%20 == 10:
        else:
            # target_all[int(i/20)] = env.target
            speed_list=np.vstack((speed_list,speed))
        # speed_list = np.hstack((speed_list,target_all))
        # env.render()
        # feedback_list.append(env.W_prediction)
        # setpoint_list.append(env.W_target)
        # prediction_error.append(env.W_target - env.W_prediction)
        env.render()
        feedback_list.append(env.H_actual)
        setpoint_list.append(env.H_target)
        prediction_error.append(env.H_target - env.H_prediction)
        feedback_list_width.append(env.W_actual)
        setpoint_list_width.append(env.W_target)
        prediction_error_width.append(env.W_target - env.W_prediction)
    # speed_list = np.hstack((speed_list,target_all))

    plt.figure(0)
    plt.plot(feedback_list[60:])
    plt.plot(setpoint_list)
    plt.xlim((0,1500))
    plt.ylim(0,8)
    plt.xlabel('time (s)')
    # plt.ylabel('Height(mm)')
    plt.ylabel('Height(mm)')
    plt.title('Control result')

    plt.figure(2)
    plt.plot(feedback_list_width[60:])
    plt.plot(setpoint_list_width)
    plt.xlim((0, 1500))
    plt.ylim(0,8)
    plt.xlabel('time (s)')
    # plt.ylabel('Height(mm)')
    plt.ylabel('Width(mm)')
    plt.title('Control result')
    plt.show()

    plt.figure(1)
    plt.plot(prediction_error)
    plt.xlabel('time (s)')
    # plt.ylabel('Height error')
    plt.ylabel('Width error')
    plt.title('Control error result')
    plt.show()


if ON_TRAIN:
    train()
    plt.scatter(list(range(MAX_EPISODES)), cumulative_reward)
    plt.show()
    plt.scatter(list(range(MAX_EPISODES)), steps_until_done)
    plt.show()

else:
    eval()