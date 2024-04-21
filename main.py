from environment2 import Environment2
from environment16 import Environment16
import xlsxwriter
from performance import Performance
import matplotlib.pyplot as plt
import pandas as pd
import DRL
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', required=True, type=str, help='"train" or "test_once" or "test_average" or "get_data_set"')
parser.add_argument('-t', '--type', required=True, type=str, help='"DQN" or "PPO" or "PG"')
parser.add_argument('-e', '--env', required=True, type=str, help='"2" or "16"')
args = parser.parse_args()

EPOCH = 300
if args.env == "2":
    num_of_points_measured = 6300
    env = Environment2(num_of_points_measured)
    N_STATES = 3
    N_ACTIONS = 2
if args.env == "16":
    num_of_points_measured = 4000
    env = Environment16(num_of_points_measured)
    N_STATES = 5
    N_ACTIONS = 16

def train(type):
    print("training")
    workbook = xlsxwriter.Workbook('Ep_reward_train.xlsx')
    worksheet = workbook.add_worksheet('total_r')
    S = [0]
    X = [0]
    if type == "DQN":
        agent = DRL.DQN(N_STATES, N_ACTIONS, hidden_dim=512, explore_intensity=25, replay_capacity=20000, replay_batch_size=128, target_update_freq=100)
    if type == "PPO":
        agent = DRL.PPO(N_STATES, N_ACTIONS, hidden_dim=512)
    if type == "PG":
        agent = DRL.PG(N_STATES, N_ACTIONS, hidden_dim=512)
    for i in range(EPOCH):
        s = env.first_state()
        time_in_episode = 0
        ep_r = 0
        while True:
            a = agent.select_action(s)
            s_, r, d = env.step(a, time_in_episode)
            agent.store_data(s, a, r, s_, d)
            s = s_
            ep_r += r
            if type == "DQN":
                agent.learn()
            if d:
                print('Ep: ', i, ' |', 'Ep_reward: ', round(ep_r, 2))
                worksheet.write(0, 0, "number")
                worksheet.write(i + 1, 0, i + 1)
                worksheet.write(0, 1, "total_r")
                worksheet.write(i + 1, 1, round(ep_r, 2))
                S.append(ep_r)
                X.append(i)
                break
            time_in_episode += 1
        if type == "PPO" or type == "PG":
            agent.learn()
    agent.save()
    plt.plot(X, S, '-')
    plt.legend(['Agent'])
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training')
    plt.pause(5)
    workbook.close()

def test_once(type):
    print("testing once")
    s = env.first_state()
    time_in_episode = 0
    ep_r = 0
    if type == "DQN":
        agent = DRL.DQN(N_STATES, N_ACTIONS, is_train=False, is_load=True)
    if type == "PPO":
        agent = DRL.PPO(N_STATES, N_ACTIONS, is_train=False, is_load=True)
    if type == "PG":
        agent = DRL.PG(N_STATES, N_ACTIONS, is_train=False, is_load=True)
    while True:
        a = agent.select_action(s)
        s_, r, d = env.step(a, time_in_episode)
        s = s_
        ep_r += r
        if d:
            break
        time_in_episode += 1
    performance = Performance(Net_state=env.SERVEBASE, SINR=env.SINR, Distance=env.distance)
    performance.all_criteria()
    performance.save_excel('test_once_performance.xlsx')

def test_average(type):
    print("testing average")
    workbook = xlsxwriter.Workbook('Ep_reward_test_average.xlsx')
    worksheet = workbook.add_worksheet('total_r')
    times = 0
    Sum_Num_of_handover = 0
    Sum_Num_of_failure = 0
    Sum_average_T = 0
    Sum_num_of_HOPP = 0
    Sum_latency = 0
    Sum_ep_r = 0
    if type == "DQN":
        agent = DRL.DQN(N_STATES, N_ACTIONS, is_train=False, is_load=True)
    if type == "PPO":
        agent = DRL.PPO(N_STATES, N_ACTIONS, is_train=False, is_load=True)
    if type == "PG":
        agent = DRL.PG(N_STATES, N_ACTIONS, is_train=False, is_load=True)
    
    for i in range(10):
        s = env.first_state()
        time_in_episode = 0
        ep_r = 0
        while True:
            a = agent.select_action(s)
            s_, r, d = env.step(a, time_in_episode)
            s = s_
            ep_r += r
            if d:
                print('Ep: ', i, " | finished! well done!")
                worksheet.write(0, 0, "number")
                worksheet.write(i + 1, 0, i + 1)
                worksheet.write(0, 1, "total_r")
                worksheet.write(i + 1, 1, round(ep_r, 2))
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
        Sum_ep_r += ep_r
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


def get_data_set(type):
    # [RSS, SINR, distance, x, y, net_state]
    print("getting data set")
    dataSet = []
    if type == "DQN":
        agent = DRL.DQN(N_STATES, N_ACTIONS, is_train=False, is_load=True)
    if type == "PPO":
        agent = DRL.PPO(N_STATES, N_ACTIONS, is_train=False, is_load=True)
    if type == "PG":
        agent = DRL.PG(N_STATES, N_ACTIONS, is_train=False, is_load=True)
    for _ in range(4):
        s = env.first_state()
        time_in_episode = 0
        while True:
            a = agent.select_action(s)
            s_, r, d = env.step(a, time_in_episode)
            s = s_
            if d:
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
    if args.mode == "train":
        train(args.type)
    elif args.mode == "test_once":
        test_once(args.type)
    elif args.mode == "test_average":
        test_average(args.type)
    elif args.mode == "get_data_set":
        get_data_set(args.type)
