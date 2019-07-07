import sys
import os
import urllib.parse
import subprocess
import argparse
import audiolabel
import numpy as np
import ultratils.pysonix.bprreader
import pandas as pd
import matplotlib.pyplot as plt
import parselmouth
import seaborn as sns
from scipy import stats
import struct
from struct import error
from struct import unpack
import audosync as audo
import derivatives as der



parser = argparse.ArgumentParser()
#parser.add_argument("datadir", help = "Experiment directory containing bprs, tgs, sync")
#parser.add_argument("subject", help = "subject number")
parser.add_argument("savedir", help = "where saved figs should go")
parser.add_argument("csv", help = "csv containing matchstreak info")

args = parser.parse_args()
data = args.csv
#testsubj = args.subject
savedir = args.savedir

subs = [121, 122, 123, 124, 125, 126, 127, 128]
# for testing
#subs = [121]

df = pd.read_csv(data)
#df = df['us_diff'].interpolate(method='pad')
df = df.replace(0, np.NaN)
df.dropna(axis = 0, how = 'any', inplace=True)
df = df[df.phone != 'sp']

df05 = df[df.p < 0.05]
df10 = df[df.p < 0.10]

fig, ax = plt.subplots(1,1)
fig.set_size_inches(11,6)
p05 = sns.lineplot(x="pos", y="r", hue = "windowlength",estimator = np.median, data = df05, alpha = 0.7, ax = ax)
p05.legend(loc='upper right')
pic05 = 'windows_p05.png'
savepic = os.path.join(savedir,pic05)
fig.savefig(savepic)

fig2, ax2 = plt.subplots(1,1)
p10 = sns.lineplot(x="pos", y="r", hue = "windowlength",estimator = np.median, data = df10, alpha = 0.7, ax = ax2)
p10.legend(loc='upper right')
pic10 = 'windows_p10.png'
savepic = os.path.join(savedir,pic10)
fig2.savefig(savepic)

fig3, ax3 = plt.subplots(1,1)
pall = sns.lineplot(x="pos", y = "r", hue = "windowlength", estimator = np.median, data = df, alpha =0.7, ax = ax3)
picall = 'windows_allvals.png'
savepic = os.path.join(savedir,picall)
fig3.savefig(savepic)
