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
parser.add_argument("datadir", help = "Experiment directory containing bprs, tgs, sync")
parser.add_argument("subject", help = "subject number")
parser.add_argument("savedir", help = "where saved figs should go")

# optional
parser.add_argument("subset", choices = ['all','vowels','sonorants','nasals','sibilants','stops'],help = "subset of words")#, action = "store_true")
#parser.add_argument("-r", help = "sonorants only", action = "store_true")
#parser.add_argument("-m", help = "nasals only", action = "store_true")
#parser.add_argument("-s", help = "sibilants only", action = "store_true")
#parser.add_argument("-k", help = "stops only", action = "store_true")

args = parser.parse_args()
datadir = args.datadir
testsubj = args.subject
savedir = args.savedir

if args.subset != 'all':
    if args.subset == 'sonorants':
        words = ['rah','Rome','rome','ream','bar','bore','beer', 'Lee', 'lee', 'lob','lobe','ball','bowl','meal','wad','wand','Watt','watt','want','wan']
    elif args.subset == 'nasals':
        words = ['meal','ream','wand','want','wan','canned',"can't",'can','ben','bend','bent','Ben','don','pain','paint','pained']
    elif args.subset == 'sibilants':
        words = ['sob','sew','sea','boss','dose','piece','shah','show','she','posh','gauche','quiche']
    elif args.subset == 'stops':
        words = ['doe','don','bar','bore','beer','lob','ball','lobe','bowl','wad','wand','Watt','watt','want','bed','bet','ben','Ben','bend','bent','gauche','posh','piece','quiche','dose','cat','cad','canned','can',"can't",'paid','pate','pained','pain','paint']

tss = next(os.walk(os.path.join(datadir,testsubj)))[1]
testcases = tss
#testcase = '2015-10-30T104019-0700'
frame_times = None
for testcase in testcases:

#    if args.subset:
# get stim
    stimfi = os.path.join(datadir, testsubj, testcase, 'stim.txt')
    stimo = open(stimfi)
    stim = stimo.read()
    if stim == 'bolus':
        continue
    print(stim + ' ' + str(testcase))
    if args.subset != 'all':
        if stim not in words:
            continue
# read in the BPR
    bprloc = os.path.join(datadir, testsubj, testcase, testcase+'.bpr')
    if os.path.getsize(bprloc) ==  0:
        continue
    bpr = ultratils.pysonix.bprreader.BprReader(bprloc)

# get tg
    tg = os.path.join(datadir, testsubj, testcase, testcase +'.TextGrid')
# read in the audio
    
    auloc = os.path.join(datadir, testsubj, testcase, testcase+'.bpr.wav')
    if not os.path.isfile(auloc):
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
    if not os.path.isfile(syncloc):
        continue
    try:
        utt_frametimes = audo.get_datadf_2der(bpr, au, syncloc, maxhz = 270)
    except IndexError:
        continue
    except struct.error as err:
        print(err)
        continue
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

savcsv = 'csv_all_'+testsubj+'.csv'
saveme = os.path.join(savedir,savcsv)
frame_times.to_csv(saveme)


fftp = frame_times#[frame_times.p < 0.05]

si = ['pos','neg']
for sign in si:
    if sign == 'pos':
        ftp = fftp[fftp.r > 0]
    else:
        ftp = fftp[fftp.r < 0]

        
    ftp = ftp[ftp.phone != 'sp']
    # ftp.plot(x = 'pos', y = 'r',kind = 'scatter')
    vtp = ftp[(ftp.phone == 'AY1')|( ftp.phone == 'AA1' )|( ftp.phone == 'IY1' )|( ftp.phone == 'AH0')]
#    vtp = ftp[(ftp.phone == 'AY1') |( ftp.phone == 'OW1')|( ftp.phone == 'AA1' )|( ftp.phone == 'IY1' )|( ftp.phone == 'AH0' )|( ftp.phone == 'AH1')| (ftp.phone == 'EH1')| ( ftp.phone == 'AE1')|(ftp.phone=='IH1')]
    stp = ftp[(ftp.phone == 'S' )|( ftp.phone == 'SH')]
    rtp = ftp[(ftp.phone == 'L' )|( ftp.phone == 'R') | (ftp.phone == 'W')]
    ktp = ftp[(ftp.phone == 'B' )|( ftp.phone == 'P') | (ftp.phone == 'K') | (ftp.phone == 'G' )|( ftp.phone == 'D') |( ftp.phone == 'T')]
    ntp = ftp[(ftp.phone == 'N' )|(  ftp.phone == 'M')]
    
    fig, ([ax1, ax2, ax3, ax4, ax5], [ax6, ax7, ax8, ax9, ax10]) = plt.subplots(2, 5)
    fig.set_size_inches(35, 8)    

    vtpp = sns.scatterplot(x="pos", y="r", hue = 'phone', data = vtp, ax = ax1)
    stpp = sns.scatterplot(x="pos", y="r", hue = 'phone', data = stp, ax = ax2)
    rtpp = sns.scatterplot(x="pos", y="r", hue = 'phone', data = rtp, ax = ax3)
    ktpp = sns.scatterplot(x="pos", y="r", hue = 'phone', data = ktp, ax = ax4)
    ntpp = sns.scatterplot(x="pos", y="r", hue = 'phone', data = ntp, ax = ax5)
    #vtpp = sns.scatterplot(x="pos", y="r_acc", hue = 'phone', data = vtp, ax = ax6)
    #stpp = sns.scatterplot(x="pos", y="r_acc", hue = 'phone', data = stp, ax = ax7)
    #rtpp = sns.scatterplot(x="pos", y="r_acc", hue = 'phone', data = rtp, ax = ax8)
    #ktpp = sns.scatterplot(x="pos", y="r_acc", hue = 'phone', data = ktp, ax = ax9)
    #ntpp = sns.scatterplot(x="pos", y="r_acc", hue = 'phone', data = ntp, ax = ax10)
    
    #vtpp = sns.scatterplot(x="us_diff", y="au_diff", hue = 'phone', data = vtp, ax = ax6)
    #stpp = sns.scatterplot(x="us_diff", y="au_diff", hue = 'phone', data = stp, ax = ax7)
    #rtpp = sns.scatterplot(x="us_diff", y="au_diff", hue = 'phone', data = rtp, ax = ax8)
    #ktpp = sns.scatterplot(x="us_diff", y="au_diff", hue = 'phone', data = ktp, ax = ax9)
    #ntpp = sns.scatterplot(x="us_diff", y="au_diff", hue = 'phone', data = ntp, ax = ax10)
    
    vtpp = sns.scatterplot(x="pos", y="au_diff", hue = 'phone', data = vtp, alpha = 0.5, ax = ax6)
    vtpp2 = vtpp.twinx()
    sns.scatterplot(x = "pos", y = "us_diff", hue = 'phone', marker = '+',data = vtp, ax = vtpp2)
#    h1, l1 = ax6.get_legend_handles_labels()
#    h2, l2 = vtpp2.get_legend_handles_labels()
#    ax6.legend(h1+h2, l1+l2, loc=3)
    ax6.get_legend().set_visible(False)

    stpp = sns.scatterplot(x="pos", y="au_diff", hue = 'phone', data = stp,  alpha = 0.5,ax = ax7)
    stpp2 = stpp.twinx()
    sns.scatterplot(x = "pos", y = "us_diff", hue = 'phone', marker = '+',data = stp,ax = stpp2)
    
    rtpp = sns.scatterplot(x="pos", y="au_diff", hue = 'phone', data = rtp,  alpha = 0.5,ax = ax8)
    rtpp2 = rtpp.twinx()
    sns.scatterplot(x = "pos", y = "us_diff", hue = 'phone', marker = '+',data = rtp,ax = rtpp2)
    
    ktpp = sns.scatterplot(x="pos", y="au_diff", hue = 'phone', data = ktp,  alpha = 0.5,ax = ax9)
    ktpp2 = ktpp.twinx()
    sns.scatterplot(x = "pos", y = "us_diff", hue = 'phone', marker = '+', data = ktp,ax = ktpp2)
    
    ntpp = sns.scatterplot(x="pos", y="au_diff", hue = 'phone', data = ntp,  alpha = 0.5,ax = ax10)
    ntpp2 = ntpp.twinx()
    sns.scatterplot(x = "pos", y = "us_diff", hue = 'phone', marker = '+', data = ntp,ax = ntpp2)
    
    
    #fig = validdf.plot(x='time',y='r')
    #validdf.plot(x='time',y='p',secondary_y=True,ax=fig)
    # for line in range(0,ftp.shape[0]):
    #     ftpp.text(ftp.pos[line]+0.2, ftp.r[line], ftp.phone[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
    
    figname = sign +'_testallwords_allp_' + testsubj + '_byposition.png'
    savename = os.path.join(savedir,figname)
    fig.savefig(savename)
    #plt.show()
