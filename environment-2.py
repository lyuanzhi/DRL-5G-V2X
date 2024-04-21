import numpy as np
import math


class Environment:
    ht = 30
    hr = 1
    num_of_eNBs = 16
    _calculate_all_flag = 0

    def __init__(self, num_of_points_measured):

        self.num_of_points_measured = num_of_points_measured
        self.fc = np.array([6000 * 10e6 for _ in range(self.num_of_eNBs)])
        self.tx_RSS = np.array([20 for _ in range(self.num_of_eNBs)])
        self.MS_coordinate = np.zeros([self.num_of_points_measured, 2])
        self.distance = np.zeros([self.num_of_eNBs, self.num_of_points_measured])
        self.ideal_RSS = np.zeros([self.num_of_eNBs, self.num_of_points_measured])
        self.noisy_RSS = np.zeros([self.num_of_eNBs, self.num_of_points_measured])
        self.SINR = np.zeros([self.num_of_eNBs, self.num_of_points_measured])

        self.SERVEBASE = [-1 for _ in range(0, self.num_of_points_measured)]
        self.SERVEBASE[0] = 6

        self.rss_max = -10
        self.rss_min = -100
        self.rss_range = self.rss_max - self.rss_min
        self.sinr_max = 40
        self.sinr_min = -20
        self.sinr_range = self.sinr_max - self.sinr_min
        self.d_max = 1800
        self.d_min = 0
        self.d_range = self.d_max - self.d_min

        self.eNB_coordinate = np.zeros([self.num_of_eNBs, 2])

        self.eNB_coordinate[0, :] = [-600, 600]
        self.eNB_coordinate[1, :] = [-200, 600]
        self.eNB_coordinate[2, :] = [200, 600]
        self.eNB_coordinate[3, :] = [600, 600]
        self.eNB_coordinate[4, :] = [-600, 200]
        self.eNB_coordinate[5, :] = [-200, 200]
        self.eNB_coordinate[6, :] = [200, 200]
        self.eNB_coordinate[7, :] = [600, 200]
        self.eNB_coordinate[8, :] = [-600, -200]
        self.eNB_coordinate[9, :] = [-200, -200]
        self.eNB_coordinate[10, :] = [200, -200]
        self.eNB_coordinate[11, :] = [600, -200]
        self.eNB_coordinate[12, :] = [-600, -600]
        self.eNB_coordinate[13, :] = [-200, -600]
        self.eNB_coordinate[14, :] = [200, -600]
        self.eNB_coordinate[15, :] = [600, -600]

    def step(self, A, CURRENT_point):

        done = False
        if A == 0:
            self.SERVEBASE[CURRENT_point + 1] = self.SERVEBASE[CURRENT_point]
        elif A == 1:
            self.SERVEBASE[CURRENT_point + 1] = np.argmax(self.SINR[:, CURRENT_point])

        rss = self.noisy_RSS[self.SERVEBASE[CURRENT_point + 1]][CURRENT_point + 1]
        sinr = self.SINR[self.SERVEBASE[CURRENT_point + 1]][CURRENT_point + 1]
        distance = self.distance[self.SERVEBASE[CURRENT_point + 1]][CURRENT_point + 1]
        s_ = [rss, sinr, distance]
        reward = (rss - self.rss_min) / self.rss_range + (sinr - self.sinr_min) / self.sinr_range + (
                    self.d_max - distance) / self.d_range

        if A == 1:
            reward -= 1.3

        CURRENT_point += 1
        if CURRENT_point == self.num_of_points_measured - 1:
            done = True

        return s_, reward, done

    def first_state(self):
        self.calculate_all_rand_walk()
        rss = self.noisy_RSS[0][0]
        sinr = self.SINR[0][0]
        distance = self.distance[0][0]
        s = [rss, sinr, distance]
        return s

    def base_station_random_setting(self):
        for i in range(self.num_of_eNBs):
            self.fc[i] = np.random.rand()*1000*10e6+6000*10e6
            self.tx_RSS[i] = np.random.rand()*5+20

    def calculate_distance(self):
        for x in range(self.num_of_eNBs):
            for i in range(self.num_of_points_measured):
                self.distance[x][i] = math.sqrt(pow(self.MS_coordinate[i][0] - self.eNB_coordinate[x][0], 2) + pow(self.MS_coordinate[i][1] - self.eNB_coordinate[x][1], 2))

    def get_RSS(self):
        for i in range(self.num_of_points_measured):
            for j in range(self.num_of_eNBs):
                path_loss = 32.4 + 20 * math.log10(self.fc[j] / 1e9) + 30 * math.log10(
                    self.distance[j][i])
                self.ideal_RSS[j][i] = self.tx_RSS[j] - path_loss
                m = 1
                v = 8
                mu = math.log(pow(m, 2) / math.sqrt(v + pow(m, 2)))
                sigma = math.sqrt(math.log(v / pow(m, 2) + 1))
                shadow_fading = -np.random.lognormal(mu, sigma)
                multi_path_fading = 0
                self.noisy_RSS[j][i] = self.ideal_RSS[j][i] + shadow_fading + multi_path_fading

    def calculate_SINR(self):
        target = -1
        for i in range(self.num_of_eNBs):
            for t in range(self.num_of_points_measured):
                sorted_distance = np.sort(self.distance[:, t])
                if sorted_distance[0] == self.distance[i][t]:
                    for j in range(self.num_of_eNBs):
                        if self.distance[j][t] == sorted_distance[1]:
                            target = j
                            break
                else:
                    for j in range(self.num_of_eNBs):
                        if self.distance[j][t] == sorted_distance[0]:
                            target = j
                            break

                if target == -1:
                    print("No target")
                else:
                    self.SINR[i][t] = (self.ideal_RSS[i][t] - (self.ideal_RSS[target][t] + 10 * math.log10(
                        np.random.rayleigh(self.distance[i][t] / 1e5)) + np.random.normal(0, 2))) / 3

    def square_walk(self):
        ini_coordinate = [0, 0]
        unit = 1
        for t in range(700):
            self.MS_coordinate[t] = np.array(ini_coordinate) + np.array([unit * t, 0])
        for t in range(700, 1400):
            self.MS_coordinate[t] = self.MS_coordinate[t - 1] + np.array([0, unit])
        for t in range(1400, 2800):
            self.MS_coordinate[t] = self.MS_coordinate[t - 1] + np.array([-unit, 0])
        for t in range(2800, 4200):
            self.MS_coordinate[t] = self.MS_coordinate[t - 1] + np.array([0, -unit])
        for t in range(4200, 5600):
            self.MS_coordinate[t] = self.MS_coordinate[t - 1] + np.array([unit, 0])
        for t in range(5600, 6300):
            self.MS_coordinate[t] = self.MS_coordinate[t - 1] + np.array([0, unit])

    def calculate_all_rand_walk(self):
        self.base_station_random_setting()
        self.square_walk()
        self.calculate_distance()
        self.get_RSS()
        self.calculate_SINR()
        self._calculate_all_flag = 1
