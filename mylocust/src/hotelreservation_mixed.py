import random
import threading
from locust import HttpUser, task, tag, between
import base64
import os
from pathlib import Path
import logging
import numpy as np
import time
import json
from scipy.stats import truncnorm
import locust.stats
from locust.env import Environment
from locust.runners import MasterRunner, WorkerRunner
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


# ======================
# 全局配置和状态管理
# ======================
class LoadConfig:
    # 阶段持续时间（秒）
    PHASE_DURATION = 250  # 修改为250秒
    CYCLE_DURATION = PHASE_DURATION * 4  # 修改为1000秒

    # 阶段顺序配置
    PHASE_ORDER = ["constant", "daynight", "burst", "noise"]

    # 各阶段配置参数
    CONFIGURATIONS = {
        "constant": {
            "mean_iat": 1
        },
        "daynight": {
            "period": 250,
            "base_iat": 1,
            "peak_iat": 1.5,
            "day_start": 0.2,
            "day_duration": 0.6
        },
        "burst": {
            "base_iat": 1.3,
            "burst_iat": 0.9,
            "cycle_duration": 250,
            "burst_duration": 50,
            "min_interval": 150
        },
        "noise": {
            "base_iat": 1,
            "noise_type": "composite",
            "gaussian": {
                "mu": 0.0,
                "sigma": 0.3,
                "clip": (-0.4, 0.4)
            },
            "impulse": {
                "prob": 0.02,
                "multiplier": 5
            },
            "random_walk": {
                "step_size": 0.1,
                "persistence": 0.8
            }
        }
    }


class GlobalState:

    def __init__(self):
        self.test_start = time.time()
        self.current_phase = 0
        self.cycle_count = 0
        self.user_count = 0
        self.active_users = 0
        self._lock = threading.Lock()

    def get_phase(self):
        elapsed = time.time() - self.test_start
        phase_index = int((elapsed % LoadConfig.CYCLE_DURATION) // LoadConfig.PHASE_DURATION)
        return LoadConfig.PHASE_ORDER[phase_index]

    def update_users(self, env: Environment):
        with self._lock:
            # 每完整循环增加用户数
            full_cycles = int((time.time() - self.test_start) // LoadConfig.CYCLE_DURATION)
            if full_cycles > self.cycle_count:
                self.cycle_count = full_cycles
                new_count = self.user_count + LoadConfig.USER_INCREMENT

                # 安全停止并重启
                if env.runner and not isinstance(env.runner, WorkerRunner):
                    # env.runner.stop()
                    env.runner.start(user_count=new_count, spawn_rate=new_count // 10)
                    self.user_count = new_count
                    print(f"\n[Cycle {self.cycle_count}] Users increased to {new_count}")


# 初始化全局状态
global_state = GlobalState()


class SocialMediaUser(HttpUser):
    # 初始化每个用户的阶段相关状态
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phase_specific_init()
        # 初始化所有负载模式可能需要的属性
        self._cycle_start = time.time()  # 昼夜负载
        self._test_start = time.time()  # 突发负载
        self._last_iat = 1.0  # 噪声负载
        self._burst_schedule = {}  # 突发负载

    def _phase_specific_init(self):
        current_phase = global_state.get_phase()

        if current_phase == "burst":
            self._test_start = time.time()
            self._burst_schedule = {}

        elif current_phase == "daynight":
            self._cycle_start = time.time()

        elif current_phase == "noise":
            self._last_iat = LoadConfig.CONFIGURATIONS["noise"]["base_iat"]

    def wait_time(self):
        current_phase = global_state.get_phase()

        if current_phase == "constant":
            return self._constant_wait()
        elif current_phase == "daynight":
            return self._daynight_wait()
        elif current_phase == "burst":
            return self._burst_wait()
        elif current_phase == "noise":
            return self._noise_wait()
        return 1.0

    # 恒定负载
    def _constant_wait(self):
        cfg = LoadConfig.CONFIGURATIONS["constant"]
        return max(0.05, np.random.exponential(cfg["mean_iat"]))

    # 昼夜负载
    def _daynight_wait(self):
        cfg = LoadConfig.CONFIGURATIONS["daynight"]
        elapsed = time.time() - self._cycle_start
        cycle_pos = elapsed % cfg["period"]

        day_start = cfg["period"] * cfg["day_start"]
        day_end = day_start + cfg["period"] * cfg["day_duration"]
        transition = 30

        core_day_start = day_start + transition
        core_day_end = day_end - transition

        if cycle_pos < day_start - transition:
            factor = 0.0
        elif cycle_pos < day_start + transition:
            factor = (cycle_pos - (day_start - transition)) / (2 * transition)
        elif cycle_pos < core_day_end:
            factor = 1.0
        elif cycle_pos < day_end + transition:
            factor = 1 - (cycle_pos - core_day_end) / (2 * transition)
        else:
            factor = 0.0

        factor = max(0.0, min(1.0, factor))
        current_iat = cfg["base_iat"] + (cfg["peak_iat"] - cfg["base_iat"]) * (1 - factor)
        return max(0.05, np.random.exponential(current_iat))

    # 突发负载
    def _burst_wait(self):
        cfg = LoadConfig.CONFIGURATIONS["burst"]
        cycle_num, cycle_pos = self._get_current_cycle()
        burst = self._get_burst_window(cycle_num)

        if burst['start'] <= cycle_pos < burst['end']:
            return max(0.05, np.random.exponential(cfg["burst_iat"]))
        return max(0.05, np.random.exponential(cfg["base_iat"]))

    def _get_current_cycle(self):
        cfg = LoadConfig.CONFIGURATIONS["burst"]
        elapsed = time.time() - self._test_start
        cycle_num = int(elapsed // cfg["cycle_duration"])
        cycle_pos = elapsed % cfg["cycle_duration"]
        return cycle_num, cycle_pos

    def _get_burst_window(self, cycle_num):
        cfg = LoadConfig.CONFIGURATIONS["burst"]
        if cycle_num not in self._burst_schedule:
            prev_end = self._burst_schedule.get(cycle_num - 1, {}).get('end', 0)
            min_start = max(0, prev_end + cfg["min_interval"] - cfg["cycle_duration"])
            start = random.uniform(min_start, cfg["cycle_duration"] - cfg["burst_duration"])
            self._burst_schedule[cycle_num] = {'start': start, 'end': start + cfg["burst_duration"]}
        return self._burst_schedule[cycle_num]

    # 随机负载
    def _noise_wait(self):
        cfg = LoadConfig.CONFIGURATIONS["noise"]

        if cfg["noise_type"] == "poisson":
            return max(0.05, np.random.exponential(cfg["base_iat"]))

        # 组合噪声生成
        noise = self._gaussian_noise() + self._impulse_noise() + self._random_walk()
        modulated_iat = cfg["base_iat"] * (1 + noise / 3)
        return max(0.05, modulated_iat)

    def _gaussian_noise(self):
        # 正确从全局配置获取噪声参数
        noise_cfg = LoadConfig.CONFIGURATIONS["noise"]
        cfg = noise_cfg["gaussian"]  # 从全局配置获取gaussian子配置
        a, b = (cfg["clip"][0] - cfg["mu"]) / cfg["sigma"], (cfg["clip"][1] - cfg["mu"]) / cfg["sigma"]
        return truncnorm(a, b, loc=cfg["mu"], scale=cfg["sigma"]).rvs()

    def _impulse_noise(self):
        # 获取全局noise配置
        noise_cfg = LoadConfig.CONFIGURATIONS["noise"]
        # 从父级配置获取base_iat
        base_iat = noise_cfg["base_iat"]
        # 获取impulse子配置
        impulse_cfg = noise_cfg["impulse"]

        if random.random() < impulse_cfg["prob"]:
            return np.random.exponential(impulse_cfg["multiplier"] * base_iat)
        return 0

    def _random_walk(self):
        cfg = LoadConfig.CONFIGURATIONS["noise"]
        cfg = cfg["random_walk"]
        delta = np.random.uniform(-cfg["step_size"], cfg["step_size"])
        self._last_iat = cfg["persistence"] * self._last_iat + (1 - cfg["persistence"]) * delta
        return self._last_iat

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
