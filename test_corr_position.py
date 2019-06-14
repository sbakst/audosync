import sys
import os
import argparse
import audiolabel
import numpy as np
import ultratils.pysonix.bprreader
import pandas as pd
import matplotlib.pyplot as plt
import parselmouth
import seaborn as sns
from scipy import stats
import audosync as audo
import derivatives as der


parser = argparse.ArgumentParser()
parser.add_argument("datadir", help = "Experiment directory containing bprs, tgs, sync")
parser.add_argument("subject", help = "subject number")

args = parser.parse_args()
datadir = args.datadir
testsubj = args.subject

tss = next(os.walk(os.path.join(datadir,testsubj)))[1]
testcases = tss[3:15]
#testcase = '2015-10-30T104019-0700'

frame_times = None
for testcase in testcases:

# read in the BPR
    bprloc = os.path.join(datadir, testsubj, testcase, testcase+'.bpr')
    bpr = ultratils.pysonix.bprreader.BprReader(bprloc)

# get tg
    tg = os.path.join(datadir, testsubj, testcase, testcase +'.TextGrid')
    if not os.path.isfile(tg):
        continue
# read in the audio
    auloc = os.path.join(datadir, testsubj, testcase, testcase+'.bpr.wav')
    au = parselmouth.Sound(auloc)
    au = au.extract_channel(1)

# get sync file
    syncloc = os.path.join(datadir, testsubj, testcase, testcase+'.bpr.sync.txt')
    utt_frametimes = audo.get_datadf_2der(bpr, au, syncloc, maxhz = 200)
    windowlen = 0.075
    utt_frametimes = audo.syncmatch(utt_frametimes, windowlen=windowlen, offset=0)
    utt_frametimes = audo.get_corr_pos(utt_frametimes,tg)
    # get r and p for acceleration
    rs = []
    ps = []
    acc_df = None
    acc_df = utt_frametimes[np.isfinite(utt_frametimes['us_acc'])]
    print(acc_df.head())
    starttimes = [t for t in acc_df['time'] if t+windowlen < np.max(acc_df['time'])]
    for starttime in starttimes:
        subdf = acc_df[(acc_df['time'] >= starttime-windowlen/2) & (acc_df['time'] <= starttime+windowlen/2)]
        sts = stats.linregress(x=subdf['us_acc'], y=subdf['au_acc'])
    #    print(sts.rvalue)
    #    print(sts.pvalue)
        rs = rs + [sts.rvalue]
        ps = ps + [sts.pvalue]

    acc_df = pd.concat([acc_df,pd.DataFrame({'r_acc':rs})], ignore_index=False, axis=1)
    acc_df = pd.concat([acc_df,pd.DataFrame({'p_acc':ps})], ignore_index=False, axis=1)
    
    
    
    if frame_times is None:
        frame_times = acc_df
    else:
        dfs = (frame_times,acc_df)
        frame_times = pd.concat(dfs)
print(frame_times.head())
ftp = frame_times[frame_times.p_acc < 0.2]
ftp = ftp[ftp.phone != 'sp']
ftp.plot(x = 'pos', y = 'r_acc',kind = 'scatter')
# ftpp = sns.regplot(data=ftp, x="pos", y="r", fit_reg=False, marker="o")
# for line in range(0,ftp.shape[0]):
#     ftpp.text(ftp.pos[line]+0.2, ftp.r[line], ftp.phone[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
plt.show()
#frame_times.plot(x='time',y='au_diff',secondary_y=True,ax=fig)
