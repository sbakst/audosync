import sys
import os
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
import audosync as audo
import derivatives as der


parser = argparse.ArgumentParser()
parser.add_argument("datadir", help = "Experiment directory containing bprs, tgs, sync")
parser.add_argument("subject", help = "subject number")

# optional
parser.add_argument("subset", choices = ['all','vowels','sonorants','nasals','sibilants','stops'],help = "subset of words")#, action = "store_true")
#parser.add_argument("-r", help = "sonorants only", action = "store_true")
#parser.add_argument("-m", help = "nasals only", action = "store_true")
#parser.add_argument("-s", help = "sibilants only", action = "store_true")
#parser.add_argument("-k", help = "stops only", action = "store_true")

args = parser.parse_args()
datadir = args.datadir
testsubj = args.subject

if args.subset != 'all':
    if args.subset == 'sonorants':
        words = ['rah','Rome','rome','ream','bar','bore','beer', 'Lee', 'lee', 'lob','lobe','ball','bowl','meal','wad','wand','Watt','watt','want','wan']
    elif args.subset == 'nasals':
        words = ['meal','ream','wand','want','wan','canned','cant','can','ben','bend','bent','Ben']
    elif args.subset == 'sibilants':
        words = ['sob','sew','sea','boss','dose','piece','shaw','show','she','posh','gauche','quiche']
    elif args.subset == 'stops':
        words = ['bar','bore','beer','lob','ball','lobe','bowl','wad','wand','Watt','want','bed','bet','ben','Ben','bend','bent','gauche','posh','piece','quiche','dose','cat','cad','canned','can','cant']

tss = next(os.walk(os.path.join(datadir,testsubj)))[1]
testcases = tss[1:50]
#testcase = '2015-10-30T104019-0700'
frame_times = None
for testcase in testcases:

#    if args.subset:
# get stim
    stimfi = os.path.join(datadir, testsubj, testcase, 'stim.txt')
    stimo = open(stimfi)
    stim = stimo.read()
    print(stim)
    if args.subset != 'all':
        if stim not in words:
            continue
# read in the BPR
    bprloc = os.path.join(datadir, testsubj, testcase, testcase+'.bpr')
    bpr = ultratils.pysonix.bprreader.BprReader(bprloc)

# get tg
    tg = os.path.join(datadir, testsubj, testcase, testcase +'.TextGrid')
# read in the audio
    
    auloc = os.path.join(datadir, testsubj, testcase, testcase+'.bpr.wav')
    if not auloc:
        continue
    au = parselmouth.Sound(auloc)
    au = au.extract_channel(1)
#    stimfile = os.path.join(datadir, testsubj, testcase,'stim.txt')
    if not os.path.isfile(tg):
        continue
#        wav = os.path.join(datadir, testsubj, testcase, testcase+'.bpr.ch1.wav')
#        proc = subprocess.check_call(['pyalign',wav,stimfile,tg])

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
#    print(acc_df.head())`
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
ftp = frame_times[frame_times.p < 0.05]
ftp = ftp[ftp.phone != 'sp']
# ftp.plot(x = 'pos', y = 'r',kind = 'scatter')
vtp = ftp[(ftp.phone == 'AY1') |( ftp.phone == 'OW1')|( ftp.phone == 'AA1' )|( ftp.phone == 'IY1' )|( ftp.phone == 'AH0' )|( ftp.phone == 'AH1')| (ftp.phone == 'EH1')| ( ftp.phone == 'AE1')|(ftp.phone=='IH1')]
stp = ftp[(ftp.phone == 'S' )|( ftp.phone == 'SH')]
rtp = ftp[(ftp.phone == 'L' )|( ftp.phone == 'R') | (ftp.phone == 'W')]
ktp = ftp[(ftp.phone == 'B' )|( ftp.phone == 'P') | (ftp.phone == 'K') | (ftp.phone == 'G' )|( ftp.phone == 'D') |( ftp.phone == 'T')]
ntp = ftp[(ftp.phone == 'N' )|(  ftp.phone == 'M')]

fig, ([ax1, ax2, ax3, ax4, ax5], [ax6, ax7, ax8, ax9, ax10]) = plt.subplots(2, 5)

vtpp = sns.scatterplot(x="pos", y="r", hue = 'phone', data = vtp, ax = ax1)
stpp = sns.scatterplot(x="pos", y="r", hue = 'phone', data = stp, ax = ax2)
rtpp = sns.scatterplot(x="pos", y="r", hue = 'phone', data = rtp, ax = ax3)
ktpp = sns.scatterplot(x="pos", y="r", hue = 'phone', data = ktp, ax = ax4)
ntpp = sns.scatterplot(x="pos", y="r", hue = 'phone', data = ntp, ax = ax5)
vtpp = sns.scatterplot(x="pos", y="r_acc", hue = 'phone', data = vtp, ax = ax6)
stpp = sns.scatterplot(x="pos", y="r_acc", hue = 'phone', data = stp, ax = ax7)
rtpp = sns.scatterplot(x="pos", y="r_acc", hue = 'phone', data = rtp, ax = ax8)
ktpp = sns.scatterplot(x="pos", y="r_acc", hue = 'phone', data = ktp, ax = ax9)
ntpp = sns.scatterplot(x="pos", y="r_acc", hue = 'phone', data = ntp, ax = ax10)


# for line in range(0,ftp.shape[0]):
#     ftpp.text(ftp.pos[line]+0.2, ftp.r[line], ftp.phone[line], horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.show()
#frame_times.plot(x='time',y='au_diff',secondary_y=True,ax=fig)
