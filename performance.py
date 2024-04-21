import math
import numpy as np
import pandas as pd


class Performance:

    threshold_failure_SINR = 1

    def __init__(self, Net_state, SINR, Distance):
        self.Net_state = Net_state.copy()
        self.SINR = SINR.copy()
        self.num_of_eNBs, self.num_of_points_measured = SINR.shape
        self.Num_of_handover = 0
        self.Num_of_failure = 0
        self.T = np.zeros([self.num_of_points_measured])
        self.num_of_HOPP = np.zeros([self.num_of_points_measured])
        self.distance = Distance.copy()
        self.latency = -1
        self.average_T = 0

    def handover_times(self):
        for i in range(self.num_of_points_measured-1):
            if self.Net_state[i] != self.Net_state[i+1]:
                self.Num_of_handover = self.Num_of_handover+1

    def handover_failure(self):
        for i in range(self.num_of_points_measured-1):
            if self.Net_state[i+1] != self.Net_state[i] and self.SINR[int(self.Net_state[i])][i] <= self.threshold_failure_SINR:
                self.Num_of_failure = self.Num_of_failure+1

    def cal_T(self):
        for i in range(self.num_of_points_measured-1):
            B = (12 * 28 * 275) / 0.001
            S = math.pow(10, self.SINR[int(self.Net_state[i])][i] / 10)
            self.T[i] = (0.93332 * 0.97619 * B * math.log2(1 + S)) / 1000000
        self.T[self.num_of_points_measured - 1] = self.T[self.num_of_points_measured - 2]
        self.average_T = np.mean(self.T)

    def ping_pong(self):
        s = 1
        counter = 0
        wait_interval = 5
        a = -1
        b = -1
        for n in range(1, self.num_of_points_measured, 1):
            HOPP_enter_flag = 0
            if s == 1:
                counter = 0
                if self.Net_state[n] == self.Net_state[n - 1]:
                    s = 1
                elif self.Net_state[n] != self.Net_state[n - 1]:
                    a = self.Net_state[n]
                    b = self.Net_state[n - 1]
                    s = 2

            elif s == 2:
                counter = counter + 1
                if self.Net_state[n] != a and self.Net_state[n] != b:
                    a = self.Net_state[n]
                    b = self.Net_state[n - 1]
                    s = 2
                    counter = 0
                elif counter > wait_interval:
                    s = 1
                elif self.Net_state[n] == a:
                    s = 2
                elif self.Net_state[n] == b:
                    s = 1
                    HOPP_enter_flag = 1

            if HOPP_enter_flag == 1:
                self.num_of_HOPP[n] = self.num_of_HOPP[n - 1] + 1
            else:
                self.num_of_HOPP[n] = self.num_of_HOPP[n - 1]

    def cal_latency(self):
        sum_distance = 0
        for i in range(self.num_of_points_measured):
            sum_distance = sum_distance+self.distance[int(self.Net_state[i])][i]
        self.latency = (sum_distance / 1000 * 10 + sum(100 / self.T) + self.Num_of_handover * 20) / self.num_of_points_measured

    def all_criteria(self):
        self.handover_times()
        self.handover_failure()
        self.cal_T()
        self.ping_pong()
        self.cal_latency()

    def save_excel(self, name):
        writer = pd.ExcelWriter(name)
        data = pd.DataFrame(self.Net_state)
        data.to_excel(writer, 'Net_state', float_format='%.5f')
        data = pd.DataFrame([self.Num_of_handover])
        data.to_excel(writer, 'Num_of_handover', float_format='%.5f')
        data = pd.DataFrame([self.Num_of_failure])
        data.to_excel(writer, 'Num_of_failure', float_format='%.5f')
        data = pd.DataFrame(self.T)
        data.to_excel(writer, 'T', float_format='%.5f')
        data = pd.DataFrame(self.num_of_HOPP)
        data.to_excel(writer, 'num_of_HOPP', float_format='%.5f')
        data = pd.DataFrame([self.latency])
        data.to_excel(writer, 'latency', float_format='%.5f')
        writer._save()
