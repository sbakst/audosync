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
#parser.add_argument("subject", help = "subject number")
parser.add_argument("savedir", help = "where saved figs should go")

args = parser.parse_args()
datadir = args.datadir
#testsubj = args.subject
savedir = args.savedir

subs = [121, 122, 123, 124, 125, 126, 127, 128]
# for testing
#subs = [121]


frame_times = None
for testsubj in subs:
    testsubj = str(testsubj)
    tss = next(os.walk(os.path.join(datadir,testsubj)))[1]
    testcases = tss
    #testcase = '2015-10-30T104019-0700'
#    frame_times = None
    for testcase in testcases:

    # get stim
        stimfi = os.path.join(datadir, testsubj, testcase, 'stim.txt')
        stimo = open(stimfi)
        stim = stimo.read()
        if stim == 'bolus':
            continue
        print(stim + ' ' + str(testcase))

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
        if not os.path.isfile(tg):
            continue

    # get sync file
        syncloc = os.path.join(datadir, testsubj, testcase, testcase+'.bpr.sync.txt')
        if not os.path.isfile(syncloc):
            continue
        try:
            utt_frametimes_base = audo.get_datadf_simple(bpr, au, syncloc, maxhz = 270)
            #print(utt_frametimes_base.head())
        except IndexError:
            continue
        except struct.error as err:
            print(err)
            continue
        windowlens = [.150, .175, .200, .225]
        for win in windowlens:
            print(win)
            offsets = [0, 0.05, 0.1, 0.15]
            for ofs in offsets:
                utt_frametimes = audo.syncmatch(utt_frametimes_base, windowlen=win, offset=ofs)
                utt_frametimes = audo.get_corr_pos(utt_frametimes,tg)
                nutts = len(utt_frametimes.index)
                windowlength = [win]*nutts
                subject = [testsubj]*nutts
                offset = [ofs]*nutts
                utt_frametimes = pd.concat([utt_frametimes,pd.DataFrame({'windowlength':windowlength})], ignore_index = False, axis = 1)
                utt_frametimes = pd.concat([utt_frametimes,pd.DataFrame({'subject':subject})],ignore_index = False, axis = 1)
                utt_frametimes = pd.concat([utt_frametimes,pd.DataFrame({'offset':offset})],ignore_index = False, axis = 1)
                #print(utt_frametimes.head())
                nmatch = audo.matchstreak(utt_frametimes, p_crit = 0.05, r_crit = 0.4)
                matches = [nmatch]*nutts
                utt_frametimes = pd.concat([utt_frametimes,pd.DataFrame({'matchstreak':matches})],ignore_index = False, axis = 1)

                if frame_times is None:
                    #print('yay')
                    frame_times = utt_frametimes
                else:
                    #print('boo')
                    dfs = (frame_times,utt_frametimes)
                    frame_times = pd.concat(dfs)


#print(frame_times.head())

savcsv = 'csv_all_subs_matchstreak.csv'
saveme = os.path.join(savedir,savcsv)
frame_times.to_csv(saveme)

