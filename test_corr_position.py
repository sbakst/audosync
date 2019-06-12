import sys
import os
import argparse
import audiolabel
import numpy as np
import ultratils.pysonix.bprreader
import pandas as pd
import matplotlib.pyplot as plt
import parselmouth
from scipy import stats
import audosync as audo
import derivatives as der


parser = argparse.ArgumentParser()
parser.add_argument("datadir", help = "Experiment directory containing bprs, tgs, sync")
parser.add_argument("subject", help = "subject number")

args = parser.parse_args()
datadir = args.datadir
testsubj = args.subject

def get_datadf_simple(rawUSinput, au, syncloc, der = 1):

    frame_times = audiolabel.LabelManager(from_file = syncloc, from_type='table', t1_col='seconds').as_df()[1]
    frame_times = frame_times.rename(columns={'text':'frameN','t1':'time'})

    if isinstance(rawUSinput,np.ndarray):
    #os.path.splitext(rawUSinput)[1] == '.bpr' : 
        frames = rawUSinput
    else :
        frames = [bpr.get_frame(i) for i in range(0, bpr.nframes)]

    frames_diff = [np.mean(np.abs(frames[i]-frames[i-1])) for i in range(1, len(frames))]

    frame_times['us_diff'] = frame_times['frameN'].apply(lambda x: frames_diff[int(x)-1]
                                                         if (x!='NA' and int(x)>0) else np.nan)
    for i in range(1, len(frame_times)):
        if frame_times['frameN'][i-1]=='NA':
            frame_times.loc[i,'us_diff']=np.nan

    pmfcc = au.to_mfcc()
    mfcc = np.transpose(pmfcc.to_array())  # transpose this to get time (frames) on the first dimension

    au_diff = [np.mean(np.abs(mfcc[i]-mfcc[i-1])) for i in range(1, len(mfcc))]
    frame_times['au_diff']=frame_times.time.apply(lambda x: au_diff[int(pmfcc.get_frame_number_from_time(x)+1)])

    pmint = au.to_intensity()
    frame_times['au_int'] = frame_times.time.apply(lambda x: pmint.get_value(x))

    if der == 2: # would a third derivative mean anything? Could put in a loop here I suppose.
#        for i in range(2, len(frame_times)):
        us_acc = [frame_times['us_diff'][i]-frame_times['us_diff'][i-1] for i in range(2, len(frame_times))]
        frame_times['us_acc'] = frame_times['frameN'].apply(lambda x: us_acc[int(x)-1]
                                                         if (x!='NA' and int(x)>1) else np.nan)
        au_acc = [frame_times['au_diff'][i] - frame_times['au_diff'][i-1] for i in range (1, len(frame_times))]
        frame_times['au_acc'] = np.nan
        frame_times['au_acc'][1:len(frame_times)] = au_acc
        #frame_times['frameN'].apply(lambda x: us_acc[int(x)-1]
        #                                                 if (x!='NA' and int(x)>1) else np.nan)
       #
        return frame_times#, au_acc

#frame_times = get_datadf_simple(bpr,au,syncloc,der = 2)

# print(au_acc[0:5])


#testsubj = '121'
tss = next(os.walk(os.path.join(datadir,testsubj)))[1]
testcases = tss[3:50]
#testcase = '2015-10-30T104019-0700'

frame_times = None
for testcase in testcases:

# read in the BPR
    bprloc = os.path.join(datadir, testsubj, testcase, testcase+'.bpr')
    bpr = ultratils.pysonix.bprreader.BprReader(bprloc)

# get tg
    tg = os.path.join(datadir, testsubj, testcase, testcase +'.TextGrid')
# read in the audio
    auloc = os.path.join(datadir, testsubj, testcase, testcase+'.bpr.wav')
    au = parselmouth.Sound(auloc)
    au = au.extract_channel(1)

# get sync file
    syncloc = os.path.join(datadir, testsubj, testcase, testcase+'.bpr.sync.txt')
    utt_frametimes = get_datadf_simple(bpr, au, syncloc, der = 2)
    utt_frametimes = audo.syncmatch(utt_frametimes, windowlen=0.08, offset=0)
    utt_frametimes = audo.get_corr_pos(utt_frametimes,tg)
    if frame_times is None:
        frame_times = utt_frametimes
    else:
        dfs = (frame_times,utt_frametimes)
        frame_times = pd.concat(dfs)
print(frame_times.head())
ftp = frame_times[frame_times.p < 0.05]
ftp = ftp[ftp.phon != 'sp']
ftp.plot(x = 'pos', y = 'r',kind = 'scatter')
plt.show()
#frame_times.plot(x='time',y='au_diff',secondary_y=True,ax=fig)
