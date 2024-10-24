import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange, tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import json
from pnp_utils import check_safety

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from run_features_extraction import load_model_from_config
import matplotlib.pyplot as plt

def diff():
    receive_start= torch.load("secret/receive_512_attn_start.pt")
    receive_sender=torch.load("secret/receive_512_attn_sender.pt")

    steg_64_start= torch.load("secret/steg_64_attn_start.pt")
    steg_64_sender=torch.load("secret/steg_64_attn_sender.pt")

    secret_start= torch.load("secret/secret_attn_start.pt")
    secret_sender=torch.load("secret/secret_attn_sender.pt")

    diff_receive=torch.sum(torch.abs(receive_sender-receive_start))
    diff_steg_64=torch.sum(torch.abs(steg_64_sender-steg_64_start))
    diff_secret=torch.sum(torch.abs(secret_sender-secret_start))

    print("receive=",diff_receive,"steg_64=",diff_steg_64,"secret=",diff_secret)


def plot_twin(t,_y1, _y2, _ylabel1, _ylabel2):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('lamda_L',fontsize=14)
    ax1.set_ylabel(_ylabel1, color=color,fontsize=14)
    ax1.plot(t,_y1, color=color,label="SSIM",lw='3')
    ax1.tick_params(axis='y', labelcolor=color,labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    #ax1.set_ylim(0,0.97)
    ax1.legend(loc=6)

    ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

    color = 'tab:red'
    ax2.set_ylabel(_ylabel2, color=color,fontsize=14)
    ax2.plot(t,_y2, color=color,label="Bit acc.",lw='3')
    ax2.tick_params(axis='y', labelcolor=color,labelsize=11)
    #ax2.set_ylim(0.926,1)

    fig.tight_layout()
    #plt.xlabel("lamda_L")
    plt.grid()
    plt.legend(loc=7)
    plt.savefig("lamda.png")


if __name__ == '__main__':
    import numpy as np
    # 创建模拟数据
    t = np.array([0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8])
    data1 = np.array([0.961,0.895,0.828,0.767,0.710,0.656,0.605,0.556])
    data2 = np.array([0.927,0.935,0.953,0.957,0.959,0.964,0.966,0.967])
    plot_twin(t,data1, data2, 'SSIM', 'Bit acc.')

