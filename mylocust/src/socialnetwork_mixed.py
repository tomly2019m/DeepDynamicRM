import random
from locust import HttpUser, task, tag, events
import os
import logging
import numpy as np
import time
import json
import threading
from locust.env import Environment
from locust.runners import MasterRunner, WorkerRunner
import locust.stats
from scipy.stats import truncnorm

locust.stats.CSV_STATS_INTERVAL_SEC = 0.5  # second

random.seed(time.time())

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

project_root = "~/DeepDynamicRM"

image_dir = f"{project_root}/mylocust/base64_images"
image_dir = os.path.expanduser(image_dir)

image_data = {}
image_names = []

logging.basicConfig(level=logging.INFO)

# data
for img in os.listdir(str(image_dir)):
    full_path = os.path.join(image_dir, img)
    image_names.append(img)
    with open(str(full_path), 'r') as f:
        image_data[img] = f.read()

charset = [
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v',
    'b', 'n', 'm', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z',
    'X', 'C', 'V', 'B', 'N', 'M', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
]

decset = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

user_id_by_follower_num = {}
user_id_by_follower_num[10] = [
    3, 12, 14, 21, 24, 27, 29, 33, 37, 41, 42, 43, 51, 58, 62, 69, 80, 84, 86, 92, 97, 117, 122, 123, 124, 133, 135,
    159, 167, 170, 181, 185, 187, 193, 195, 213, 215, 218, 219, 224, 253, 254, 263, 267, 271, 272, 281, 289, 294, 297,
    298, 300, 303, 306, 314, 316, 318, 342, 344, 346, 347, 349, 350, 355, 358, 368, 372, 375, 378, 380, 392, 397, 401,
    404, 414, 421, 425, 427, 431, 442, 447, 449, 453, 459, 462, 477, 478, 479, 485, 486, 492, 503, 506, 507, 509, 511,
    513, 514, 520, 533, 546, 547, 548, 551, 553, 560, 564, 565, 573, 577, 588, 598, 619, 621, 622, 623, 628, 641, 643,
    653, 654, 657, 658, 661, 663, 665, 672, 675, 677, 680, 684, 695, 696, 702, 707, 724, 727, 730, 732, 736, 745, 759,
    764, 766, 768, 776, 778, 779, 782, 783, 784, 787, 798, 800, 801, 803, 810, 814, 816, 817, 820, 827, 831, 834, 835,
    839, 840, 844, 850, 859, 862, 881, 885, 890, 891, 893, 897, 899, 902, 907, 910, 911, 921, 925, 931, 932, 939, 942,
    943, 961
]
user_id_by_follower_num[30] = [
    4, 6, 8, 13, 15, 20, 22, 23, 26, 28, 31, 32, 36, 38, 50, 53, 54, 55, 65, 66, 68, 70, 74, 76, 78, 79, 81, 83, 87, 89,
    91, 93, 94, 96, 100, 102, 106, 107, 108, 109, 111, 119, 120, 125, 129, 130, 137, 140, 142, 144, 148, 151, 152, 153,
    158, 165, 168, 169, 171, 173, 174, 175, 178, 180, 186, 189, 190, 197, 200, 202, 205, 207, 209, 211, 212, 217, 221,
    223, 225, 227, 228, 233, 234, 235, 239, 240, 241, 242, 245, 247, 258, 262, 266, 269, 273, 274, 277, 278, 279, 280,
    282, 283, 286, 290, 292, 304, 305, 310, 319, 322, 323, 330, 331, 333, 334, 335, 338, 341, 345, 352, 356, 357, 359,
    360, 361, 363, 365, 367, 370, 376, 383, 385, 386, 387, 389, 391, 394, 395, 402, 409, 412, 415, 417, 422, 428, 433,
    435, 437, 440, 445, 446, 448, 450, 454, 457, 463, 465, 469, 473, 481, 482, 483, 484, 487, 490, 491, 497, 498, 499,
    500, 504, 510, 512, 515, 519, 523, 524, 527, 528, 535, 536, 537, 540, 541, 543, 544, 545, 549, 557, 559, 566, 567,
    568, 569, 570, 571, 581, 583, 584, 587, 590, 591, 593, 594, 596, 597, 599, 600, 605, 608, 609, 611, 613, 614, 625,
    626, 629, 630, 634, 635, 636, 638, 639, 642, 648, 652, 660, 662, 664, 666, 668, 673, 674, 678, 683, 692, 693, 694,
    698, 700, 705, 708, 711, 714, 717, 721, 725, 726, 735, 741, 743, 749, 750, 755, 756, 762, 763, 767, 771, 772, 780,
    788, 789, 790, 799, 802, 804, 807, 809, 815, 818, 819, 822, 823, 825, 826, 828, 829, 832, 836, 843, 848, 849, 855,
    857, 858, 861, 865, 868, 871, 872, 878, 882, 883, 887, 894, 896, 901, 903, 914, 917, 920, 922, 923, 929, 944, 945,
    946, 947, 949, 952, 956
]
user_id_by_follower_num[50] = [
    2, 5, 7, 11, 16, 17, 18, 34, 35, 39, 40, 45, 46, 47, 49, 56, 60, 63, 67, 71, 73, 82, 85, 90, 95, 104, 105, 110, 112,
    113, 114, 115, 116, 121, 126, 131, 138, 139, 141, 143, 156, 157, 164, 166, 176, 177, 179, 182, 183, 191, 196, 199,
    208, 220, 226, 237, 243, 244, 252, 257, 276, 285, 307, 320, 324, 327, 332, 336, 340, 351, 354, 364, 371, 374, 377,
    379, 381, 388, 390, 393, 398, 406, 408, 411, 418, 423, 424, 426, 432, 434, 438, 439, 444, 451, 452, 455, 464, 466,
    467, 470, 471, 472, 475, 508, 516, 517, 518, 522, 530, 532, 534, 539, 552, 554, 556, 578, 582, 589, 601, 603, 604,
    607, 612, 615, 616, 620, 631, 632, 647, 655, 659, 670, 676, 686, 687, 699, 703, 713, 715, 719, 720, 728, 731, 734,
    737, 739, 740, 751, 752, 753, 757, 761, 765, 769, 773, 777, 785, 786, 791, 796, 821, 824, 841, 842, 845, 846, 851,
    853, 854, 863, 867, 869, 874, 875, 876, 880, 884, 898, 900, 904, 905, 915, 916, 919, 930, 934, 935, 948, 951, 955,
    960, 962
]
user_id_by_follower_num[80] = [
    1, 9, 10, 25, 30, 44, 57, 59, 61, 72, 75, 77, 88, 98, 99, 118, 127, 136, 150, 162, 172, 188, 192, 194, 198, 201,
    210, 216, 231, 248, 249, 251, 259, 284, 287, 288, 295, 296, 301, 308, 317, 325, 326, 337, 339, 343, 348, 353, 362,
    366, 369, 382, 384, 396, 399, 400, 407, 420, 429, 430, 441, 443, 458, 460, 461, 468, 474, 476, 480, 488, 489, 493,
    494, 502, 521, 525, 531, 538, 542, 550, 561, 563, 572, 576, 585, 586, 595, 602, 606, 610, 618, 624, 637, 640, 651,
    667, 681, 682, 685, 688, 689, 690, 691, 697, 701, 704, 706, 712, 722, 723, 729, 733, 738, 742, 744, 758, 770, 774,
    775, 794, 797, 811, 812, 813, 833, 838, 847, 852, 864, 877, 886, 906, 909, 912, 913, 918, 924, 926, 927, 928, 933,
    936, 937, 941, 950, 953, 954, 957, 959
]
user_id_by_follower_num[100] = [
    19, 48, 64, 101, 128, 132, 134, 145, 146, 149, 154, 161, 163, 184, 203, 232, 238, 250, 255, 256, 261, 264, 268, 291,
    299, 302, 312, 313, 329, 403, 405, 410, 416, 419, 436, 495, 496, 501, 505, 574, 575, 579, 580, 627, 644, 645, 649,
    650, 656, 671, 716, 754, 781, 792, 793, 806, 860, 870, 879, 888, 895, 938, 940
]
user_id_by_follower_num[300] = [
    52, 103, 147, 155, 160, 204, 206, 214, 222, 229, 230, 236, 246, 260, 265, 270, 275, 293, 309, 311, 315, 321, 328,
    373, 413, 456, 526, 529, 555, 558, 562, 592, 617, 633, 646, 669, 679, 709, 710, 718, 746, 747, 748, 760, 795, 805,
    808, 830, 837, 856, 866, 873, 889, 892, 908, 958
]


def random_string(length):
    global charset
    if length > 0:
        s = ""
        for i in range(0, length):
            s += random.choice(charset)
        return s
    else:
        return ""


def random_decimal(length):
    global decset
    if length > 0:
        s = ""
        for i in range(0, length):
            s += random.choice(decset)
        return s
    else:
        return ""


def compose_random_text():
    coin = random.random() * 100
    if coin <= 30.0:
        length = random.randint(10, 50)
    elif coin <= 58.2:
        length = random.randint(51, 100)
    elif coin <= 76.5:
        length = random.randint(101, 150)
    elif coin <= 85.3:
        length = random.randint(151, 200)
    elif coin <= 92.6:
        length = random.randint(201, 250)
    else:
        length = random.randint(251, 280)
    return random_string(length)


def compose_random_user():
    user = 0
    coin = random.random() * 100
    if coin <= 0.4:
        user = random.choice(user_id_by_follower_num[10])
    elif coin <= 6.1:
        user = random.choice(user_id_by_follower_num[30])
    elif coin <= 16.6:
        user = random.choice(user_id_by_follower_num[50])
    elif coin <= 43.8:
        user = random.choice(user_id_by_follower_num[80])
    elif coin <= 66.8:
        user = random.choice(user_id_by_follower_num[100])
    else:
        user = random.choice(user_id_by_follower_num[300])
    return str(user)


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
        cfg = LoadConfig.CONFIGURATIONS["noise"]
        cfg = cfg["impulse"]
        if random.random() < cfg["prob"]:
            return np.random.exponential(cfg["multiplier"] * cfg["base_iat"])
        return 0

    def _random_walk(self):
        cfg = LoadConfig.CONFIGURATIONS["noise"]
        cfg = cfg["random_walk"]
        delta = np.random.uniform(-cfg["step_size"], cfg["step_size"])
        self._last_iat = cfg["persistence"] * self._last_iat + (1 - cfg["persistence"]) * delta
        return self._last_iat

    @task(5)
    @tag('compose_post')
    def compose_post(self):
        global image_names
        global image_data
        #----------------- contents -------------------#
        user_id = compose_random_user()
        username = 'username_' + user_id
        text = compose_random_text()
        # #---- user mentions ----#
        # for i in range(0, 5):
        #     user_mention_id = random.randint(1, 2)
        #     while True:
        #         user_mention_id = random.randint(1, 962)
        #         if user_id != user_mention_id:
        #             break
        #     text = text + " @username_" + str(user_mention_id)

        #---- urls ----#
        for i in range(0, 5):
            if random.random() <= 0.2:
                num_urls = random.randint(1, 5)
                for i in range(0, num_urls):
                    text = text + " https://www.bilibili.com/av" + random_decimal(8)

        #---- media ----#
        num_media = 0
        medium = []
        media_types = []
        if random.random() < 0.25:
            num_media = random.randint(1, 4)
            # num_media = 1
        for i in range(0, num_media):
            img_name = random.choice(image_names)
            if 'jpg' in img_name:
                media_types.append('jpg')
            elif 'png' in img_name:
                media_types.append('png')
            else:
                continue
            medium.append(image_data[img_name])

        params = {}
        # params['blocking'] = 'true'
        # params['result'] = 'true'

        url = '/wrk2-api/post/compose'
        img = random.choice(image_names)
        body = {}
        if num_media > 0:
            body['username'] = username
            body['user_id'] = user_id
            body['text'] = text
            body['medium'] = json.dumps(medium)
            body['media_types'] = json.dumps(media_types)
            body['post_type'] = '0'
        else:
            body['username'] = username
            body['user_id'] = user_id
            body['text'] = text
            body['medium'] = ''
            body['media_types'] = ''
            body['post_type'] = '0'

        # r = self.client.post(url, params=params,
        #     data=body, name="/compose_post")

        # TODO 修改为10
        r = self.client.post(url, params=params, data=body, name='compose_post', timeout=1)

        if r.status_code > 202:
            logging.warning('compose_post resp.status = %d, text=%s' % (r.status_code, r.text))

    @task(80)
    @tag('read_home_timeline')
    def read_home_timeline(self):
        start = random.randint(0, 100)
        stop = start + 10

        url = '/wrk2-api/home-timeline/read'
        args = {}
        args['user_id'] = str(random.randint(1, 962))
        args['start'] = str(start)
        args['stop'] = str(stop)

        # r = self.client.get(url, params=args,
        #     verify=False, name='/read_home_timeline')

        r = self.client.get(url, params=args, name='read_home_line', timeout=1)

        if r.status_code > 202:
            logging.warning('read_home_timeline resp.status = %d, text=%s' % (r.status_code, r.text))

    @task(15)
    @tag('read_user_timeline')
    def read_user_timeline(self):
        start = random.randint(0, 100)
        stop = start + 10

        url = '/wrk2-api/user-timeline/read'
        args = {}
        args['user_id'] = str(random.randint(1, 962))
        args['start'] = str(start)
        args['stop'] = str(stop)

        # r = self.client.get(url, params=args,
        #     verify=False, name='/read_home_timeline')

        r = self.client.get(url, params=args, name='read_user_timeline', timeout=1)

        if r.status_code > 202:
            logging.warning('read_user_timeline resp.status = %d, text=%s' % (r.status_code, r.text))
