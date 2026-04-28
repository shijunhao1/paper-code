import datetime
import random
import time

import numpy as np
import pytz
import torch


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cur_time(timezone: str = "Asia/Shanghai", t_format: str = "%m-%d %H:%M:%S") -> str:
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def pred_community_analysis(pred_comms):
    if len(pred_comms) == 0:
        print("Predicted communities #0")
        return
    lengths = [len(com) for com in pred_comms]
    avg_length = np.mean(np.asarray(lengths))
    print(f"Predicted communities #{len(pred_comms)}, avg size {avg_length:.4f}")
