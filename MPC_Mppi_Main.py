import matplotlib.pyplot as plt
from MPC_Mppi import MPPI
mppi = MPPI(256, 200, 0.5)
"""
目标形状设立1500个时间步
"""
STEP_LIMIT = 200

def get_real_shape():
    h_real, w_real = mppi.get_real_shape()
    return h_real, w_real

def mppi_main(h_target_list, w_target_list):
    h_target_list_ = h_target_list
    w_target_list_ = w_target_list

    # get shape information from environment
    h_real, w_real = get_real_shape() # 获取目前的形状，焊接开始前将参数设为中位值 [8,12]，以此预测形状
    print("real height and width: ", h_real, w_real)

    mppi.trajectory_set_goal(h_target_list_, w_target_list_)
    mppi.Delta_reset()
    # mppi.trajectory_update_state(h_real, w_real)

    # rollout with mppi algo
    for step in range (STEP_LIMIT):
        print("step: ", step)
        mppi.compute_cost(step)
        target_action_Wf_list, target_action_Rs_list, target_action_Wf, target_action_Rs = mppi.compute_noise_action()
        mppi.trajectory_update_shape(target_action_Wf, target_action_Rs)
        mppi.Delta_update()
        if step <= 200:
            mppi.Delta_reset()

        mppi.cost_clear()

    h_real_list, w_real_list = mppi.get_real_shape_list()
    Wf_real_list, Rs_real_list = mppi.get_real_parameter_list()

    plt.figure(0)
    plt.plot(h_real_list)
    plt.plot(h_target_list_)
    plt.xlim((0, 2000))
    plt.ylim(0, 5)
    plt.xlabel('time (s)')
    # plt.ylabel('Height(mm)')
    plt.ylabel('Height(mm)')
    plt.title('Control result')

    plt.figure(1)
    plt.plot(w_real_list)
    plt.plot(w_target_list_)
    plt.xlim((0, 2000))
    plt.ylim(0, 10)
    plt.xlabel('time (s)')
    # plt.ylabel('Height(mm)')
    plt.ylabel('Width(mm)')
    plt.title('Control result')
    plt.show()

    plt.figure(2)
    plt.plot(Wf_real_list)
    plt.plot(Rs_real_list)
    plt.xlim((0, 2000))
    plt.ylim(0, 30)
    plt.xlabel('time (s)')
    # plt.ylabel('Height(mm)')
    # plt.ylabel('Width(mm)')
    plt.title('Control result')
    plt.show()

if __name__ == '__main__':
    h_target_list = [2]*1700
    w_target_list = [6]*1700
    mppi_main(h_target_list, w_target_list)
