import random
from locust import HttpUser, task, tag, between
import base64
import os
from pathlib import Path
import logging
import numpy as np
import time
import json

import locust.stats

locust.stats.CSV_STATS_INTERVAL_SEC = 0.5  # second

random.seed(time.time())

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# logging.basicConfig(level=logging.INFO,
#                     # filename='/mnt/locust_log/locust_openwhisk_log.txt',
#                     # filemode='w+',
#                     format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

logging.basicConfig(level=logging.INFO)


def get_user():
    user_id = random.randint(0, 500)
    user_name = 'Cornell_' + str(user_id)
    password = ""
    for i in range(0, 10):
        password = password + str(user_id)
    return user_name, password


mean_iat = 1

from scipy.stats import truncnorm

NOISE_CONFIG = {
    "base_iat": 1,  # 基础请求间隔（秒）
    "noise_type": "composite",  # 可选：gaussian, poisson, composite
    "gaussian": {
        "mu": 0.0,  # 均值
        "sigma": 0.3,  # 标准差
        "clip": (-0.4, 0.4)  # 截断范围
    },
    "impulse": {
        "prob": 0.02,  # 脉冲事件概率
        "multiplier": 5  # 脉冲强度
    },
    "random_walk": {
        "step_size": 0.1,  # 随机游走步长
        "persistence": 0.8  # 趋势保持系数
    }
}


class SocialMediaUser(HttpUser):
    _last_iat = NOISE_CONFIG["base_iat"]  # 随机游走状态保持

    def _gaussian_noise(self):
        """截断高斯噪声生成器"""
        cfg = NOISE_CONFIG["gaussian"]
        a, b = (cfg["clip"][0] - cfg["mu"]) / cfg["sigma"], (cfg["clip"][1] - cfg["mu"]) / cfg["sigma"]
        return truncnorm(a, b, loc=cfg["mu"], scale=cfg["sigma"]).rvs()

    def _impulse_noise(self):
        """脉冲事件生成器"""
        if random.random() < NOISE_CONFIG["impulse"]["prob"]:
            return np.random.exponential(NOISE_CONFIG["impulse"]["multiplier"] * NOISE_CONFIG["base_iat"])
        return 0

    def _random_walk(self):
        """随机游走过程"""
        cfg = NOISE_CONFIG["random_walk"]
        delta = np.random.uniform(-cfg["step_size"], cfg["step_size"])
        self._last_iat = cfg["persistence"] * self._last_iat + (1 - cfg["persistence"]) * delta
        return self._last_iat

    def _composite_noise(self):
        """组合噪声生成"""
        components = [self._gaussian_noise(), self._impulse_noise(), self._random_walk()]
        return sum(components) / len(components)

    def wait_time(self):
        # 生成基础间隔时间
        base_interval = NOISE_CONFIG["base_iat"]

        # 选择噪声类型
        if NOISE_CONFIG["noise_type"] == "poisson":
            # 纯泊松过程
            return max(0.05, np.random.exponential(base_interval))

        noise_modulation = {
            "gaussian": lambda: 1 + self._gaussian_noise(),
            "composite": lambda: 1 + self._composite_noise()
        }[NOISE_CONFIG["noise_type"]]()

        # 应用噪声调制
        modulated_iat = base_interval * noise_modulation

        # 最终间隔时间（不低于50ms）
        return max(0.05, modulated_iat)

    @task(600)
    @tag('search_hotel')
    def search_hotel(self):
        in_date = random.randint(9, 23)
        out_date = random.randint(in_date + 1, 24)
        if in_date <= 9:
            in_date = "2015-04-0" + str(in_date)
        else:
            in_date = "2015-04-" + str(in_date)

        if out_date <= 9:
            out_date = "2015-04-0" + str(out_date)
        else:
            out_date = "2015-04-" + str(out_date)

        lat = 38.0235 + (random.randint(0, 481) - 240.5) / 1000.0
        lon = -122.095 + (random.randint(0, 325) - 157.0) / 1000.0

        url = '/hotels?inDate=' + in_date + '&outDate=' + out_date + \
            '&lat=' + str(lat) + "&lon=" + str(lon)

        r = self.client.get(url, name='search_hotel', timeout=10)
        if r.status_code > 202:
            logging.warning('search_hotel resp.status = %d, text=%s' % (r.status_code, r.text))

    @task(390)
    @tag('recommend')
    def recommend(self):
        coin = random.random()
        if coin < 0.33:
            req = 'dis'
        elif coin < 0.66:
            req = 'rate'
        else:
            req = 'price'

        lat = 38.0235 + (random.randint(0, 481) - 240.5) / 1000.0
        lon = -122.095 + (random.randint(0, 325) - 157.0) / 1000.0

        url = '/recommendations?require=' + req + \
            "&lat=" + str(lat) + "&lon=" + str(lon)

        r = self.client.get(url, name='recommend', timeout=10)
        if r.status_code > 202:
            logging.warning('recommend resp.status = %d, text=%s' % (r.status_code, r.text))

    @task(5)
    @tag('reserve')
    def reserve(self):
        in_date = random.randint(9, 23)
        out_date = random.randint(in_date + 1, 24)

        if in_date <= 9:
            in_date = "2015-04-0" + str(in_date)
        else:
            in_date = "2015-04-" + str(in_date)

        if out_date <= 9:
            out_date = "2015-04-0" + str(out_date)
        else:
            out_date = "2015-04-" + str(out_date)

        lat = 38.0235 + (random.randint(0, 481) - 240.5) / 1000.0
        lon = -122.095 + (random.randint(0, 325) - 157.0) / 1000.0

        hotel_id = str(random.randint(1, 80))
        user_name, password = get_user()

        num_room = 1

        url = '/reservation?inDate=' + in_date + "&outDate=" + out_date + \
            "&lat=" + str(lat) + "&lon=" + str(lon) + "&hotelId=" + hotel_id + \
            "&customerName=" + user_name + "&username=" + user_name + \
            "&password=" + password + "&number=" + str(num_room)

        r = self.client.post(url, name='reserve', timeout=10)

    @task(5)
    @tag('user_login')
    def read_user_timeline(self):
        user_name, password = get_user()
        url = '/user?username=' + user_name + "&password=" + password

        r = self.client.get(url, name='user_login', timeout=10)

        if r.status_code > 202:
            logging.warning('read_user_timeline resp.status = %d, text=%s' % (r.status_code, r.text))
