from environment import Environment
from dqn import DQN
import xlsxwriter
from performance import Performance
import matplotlib.pyplot as plt
import pandas as pd

num_of_points_measured = 6300
env = Environment(num_of_points_measured=num_of_points_measured)
model = DQN()


def train():
    print("training")
    workbook = xlsxwriter.Workbook('Ep_reward_train.xlsx')
    worksheet = workbook.add_worksheet('total_r')
    S = [0]
    X = [0]

    for i_episode in range(100):
        state = env.first_state()
        time_in_episode = 0
        ep_reward = 0
        model.learn_time += 0.05

        while True:
            a = model.select_action(state)
            next_state, reward, done = env.step(a, time_in_episode)
            model.store_data(state, a, reward, next_state)
            state = next_state
            ep_reward += reward
            if model.memory_counter > model.MEMORY_CAPACITY:
                model.learn()
            if done:
                print('Ep: ', i_episode, ' |', 'Ep_reward: ', round(ep_reward, 2))
                worksheet.write(0, 0, "number")
                worksheet.write(i_episode + 1, 0, i_episode + 1)
                worksheet.write(0, 1, "total_r")
                worksheet.write(i_episode + 1, 1, round(ep_reward, 2))
                S.append(ep_reward)
                X.append(i_episode)
                break
            time_in_episode += 1

    model.save()

    plt.plot(X, S, '-')
    plt.legend(['PPO'])
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('demo')
    plt.pause(5)

    workbook.close()


def test_once():
    print("testing once")
    state = env.first_state()
    time_in_episode = 0
    ep_reward = 0
    model.learn_time = 100

    while True:
        a = model.select_action(state)
        state, reward, done = env.step(a, time_in_episode)
        ep_reward += reward
        if done:
            break
        time_in_episode += 1

    performance = Performance(Net_state=env.SERVEBASE, SINR=env.SINR, Distance=env.distance)
    performance.all_criteria()
    performance.save_excel('test_once_performance.xlsx')
    print("finished")


def test_average():
    print("testing average")
    workbook = xlsxwriter.Workbook('Ep_reward_test_average.xlsx')
    worksheet = workbook.add_worksheet('total_r')

    model.learn_time = 100

    times = 0
    Sum_Num_of_handover = 0
    Sum_Num_of_failure = 0
    Sum_average_T = 0
    Sum_num_of_HOPP = 0
    Sum_latency = 0
    Sum_ep_r = 0

    for i_episode in range(10):
        state = env.first_state()
        time_in_episode = 0
        ep_reward = 0

        while True:
            a = model.select_action(state)
            state, reward, done = env.step(a, time_in_episode)
            ep_reward += reward
            if done:
                print('Ep: ', i_episode, " | finished! well done!")
                worksheet.write(0, 0, "number")
                worksheet.write(i_episode + 1, 0, i_episode + 1)
                worksheet.write(0, 1, "total_r")
                worksheet.write(i_episode + 1, 1, round(ep_reward, 2))
                break
            time_in_episode += 1

        performance = Performance(Net_state=env.SERVEBASE, SINR=env.SINR, Distance=env.distance)
        performance.all_criteria()

        average_T = performance.average_T

        Sum_Num_of_handover += performance.Num_of_handover
        Sum_Num_of_failure += performance.Num_of_failure
        Sum_average_T += average_T
        Sum_num_of_HOPP += performance.num_of_HOPP[num_of_points_measured - 1]
        Sum_latency += performance.latency
        Sum_ep_r += ep_reward

        times += 1

    Average_Num_of_handover = Sum_Num_of_handover / times
    Average_Num_of_failure = Sum_Num_of_failure / times
    Average_average_T = Sum_average_T / times
    Average_num_of_HOPP = Sum_num_of_HOPP / times
    Average_latency = Sum_latency / times
    Average_ep_r = Sum_ep_r / times

    print('\nAverage here:')
    print('This is Num_of_handover: ', Average_Num_of_handover)
    print('This is Num_of_failure: ', Average_Num_of_failure)
    print('This is average_T: ', Average_average_T)
    print('This is num_of_HOPP: ', Average_num_of_HOPP)
    print('This is latency: ', Average_latency)
    print('This is ep_r: ', Average_ep_r)

    workbook.close()


def get_data_set():
    # [RSS, SINR, distance, x, y, net_state]
    print("getting data set")

    model.learn_time = 100

    dataSet = []
    for i_episode in range(4):
        state = env.first_state()
        time_in_episode = 0

        while True:
            a = model.select_action(state)
            state, reward, done = env.step(a, time_in_episode)
            if done:
                break
            time_in_episode += 1

        for i in range(num_of_points_measured - 1):
            dataSet.append([env.noisy_RSS[env.SERVEBASE[i + 1]][i + 1], env.SINR[env.SERVEBASE[i + 1]][i + 1],
                            env.distance[env.SERVEBASE[i + 1]][i + 1], env.MS_coordinate[i][0], env.MS_coordinate[i][1],
                            env.SERVEBASE[i]])

    name = ['RSS', 'SINR', 'distance', 'x', 'y', 'net_state']
    dataSet = pd.DataFrame(columns=name, data=dataSet)
    dataSet.to_csv('./dataSet.csv')


if __name__ == '__main__':
    # train()
    test_once()
    # test_average()
    # get_data_set()
