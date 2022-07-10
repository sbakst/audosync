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
parser.add_argument('--csvs', nargs = '*', dest = 'csvs', help = 'all cvs')

#parser.add_argument("poscsv", help = "csv containing matchstreak info: pos offset")
#parser.add_argument("negcsv", help = "csv with neg offset")

args = parser.parse_args()
#poswin = args.poscsv
#negwin = args.negcsv


#poswindf = pd.read_csv(poswin)
#print(poswindf.head())
#negwindf = pd.read_csv(negwin)
#print(negwindf.head())
#if len(csvs) > 1:
#    for n in np.arange(1,len(csvs)):
#        if n == 1:
#            df = pd.concat(
csvs = args.csvs
# df = []
for csv in csvs:
    c = pd.read_csv(csv)
    print(c.head())
    try:
        df
    except NameError:
        df = c
    else:
        df = pd.concat([df, c])



df.drop_duplicates()

#df = pd.concat([csvs])
#df = pd.concat([poswindf,negwindf])

#testsubj = args.subject
savedir = args.savedir

subs = [121, 122, 123, 124, 125, 126, 127, 128]
# for testing
#subs = [121]




# df = pd.read_csv(data)
#df = df['us_diff'].interpolate(method='pad')
df = df.replace(0, np.NaN)
df.dropna(axis = 0, how = 'any', inplace=True)
df = df[df.phone != 'sp']

df05 = df[df.p < 0.05]
df10 = df[df.p < 0.10]

vtp = df[(df.phone == 'AY1') |( df.phone == 'OW1')|( df.phone == 'AA1' )|( df.phone == 'IY1' )|( df.phone == 'AH0' )|( df.phone == 'AH1')| (df.phone == 'EH1')| ( df.phone == 'AE1')|(df.phone=='IH1')|(df.phone=='EY1')|(df.phone=='AO1')]

v05 = vtp[vtp.p < 0.05]
v10 = vtp[vtp.p < 0.10]


fig, ax = plt.subplots(1,1)
fig.set_size_inches(11,6)
p05 = sns.lineplot(x="offset", y="r", hue = "windowlength",estimator = np.median, data = df05, alpha = 0.7, ax = ax)
p05.legend(loc='upper right')
pic05 = 'tensalloffsets_windows_p05.png'
savepic = os.path.join(savedir,pic05)
fig.savefig(savepic)

fig2, ax2 = plt.subplots(1,1)
fig2.set_size_inches(11,6)
p10 = sns.lineplot(x="offset", y="r", hue = "windowlength",estimator = np.median, data = df10, alpha = 0.7, ax = ax2)
p10.legend(loc='upper right')
pic10 = 'tensalloffsets_windows_p10.png'
savepic = os.path.join(savedir,pic10)
fig2.savefig(savepic)

fig3, ax3 = plt.subplots(1,1)
fig3.set_size_inches(11,6)
pall = sns.lineplot(x="offset", y = "r", hue = "windowlength", estimator = np.median, data = df, alpha =0.7, ax = ax3)
picall = 'tensalloffsets_windows_allvals.png'
savepic = os.path.join(savedir,picall)
fig3.savefig(savepic)

fig, ax = plt.subplots(1,1)
fig.set_size_inches(11,6)
p05 = sns.lineplot(x="offset", y="r", hue = "windowlength",estimator = np.median, data = v05, alpha = 0.7, ax = ax)
p05.legend(loc='upper right')
pic05 = 'tensalloff_vowels_windows_p05.png'
savepic = os.path.join(savedir,pic05)
fig.savefig(savepic)

fig2, ax2 = plt.subplots(1,1)
fig2.set_size_inches(11,6)
p10 = sns.lineplot(x="offset", y="r", hue = "windowlength",estimator = np.median, data = v10, alpha = 0.7, ax = ax2)
p10.legend(loc='upper right')
pic10 = 'tensalloff_vowels_windows_p10.png'
savepic = os.path.join(savedir,pic10)
fig2.savefig(savepic)

fig3, ax3 = plt.subplots(1,1)
fig3.set_size_inches(11,6)
pall = sns.lineplot(x="offset", y = "r", hue = "windowlength", estimator = np.median, data = vtp, alpha =0.7, ax = ax3)
picall = 'tensalloff_vowels_windows_allvals.png'
savepic = os.path.join(savedir,picall)
fig3.savefig(savepic)
savepic = os.path.join(savedir,picall)
fig3.savefig(savepic)
