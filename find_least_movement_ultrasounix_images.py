# PCA with bmps
# based off Susan's PCA (reading in bmps) and Matt's but also my frame-subtracting technique for finding R steady state
# assumes bmps have already been made

import sys
import os
import re
import argparse
import subprocess
from subprocess import call
import audiolabel
import numpy as np
import shutil
import datetime
from PIL import Image
from scipy import misc
from scipy import ndimage
from scipy import sparse
import glob
import matplotlib.pyplot as plt
from ultratils.exp import Exp
from itertools import *

from sklearn import decomposition
from sklearn.decomposition import PCA

# this could be made a separate file I read in, but this was already here from an ancient script soooo

WORDS = ['rah', 'rome', 'Rome', 'ream', 'bar', 'bore', 'beer', 'RAH', 'ROME', 'REAM','BAR','BORE','BEER']


parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Experiment directory containing all subjects' bmps and jpgs")
parser.add_argument("otherdir", help="Experiment directory containing all subjects' tgs and sync files and wavs etc, because Sarah's images are a little disorganized.")
parser.add_argument("subject", help="subjnumber")


args = parser.parse_args()


subbmpdir = os.path.join(args.directory,args.subject)
utildir = os.path.join(args.otherdir,args.subject)

rpca = None
mrpca = None


for dirs, times, files in os.walk(subbmpdir):
# When you see this a year from now, you can remember sitting in the Illini union. ;)
    for tindex, timestamp in enumerate(times):
# append some kind of placeholder for when you skip a file
        ts.append(timestamp)
        subjrframelist = []
        utt = os.path.join(utildir,str(timestamp))
        syncfile = os.path.join(utt,(timestamp+'.bpr.sync.txt'))
       # print(syncfile)
        if not os.path.isfile(syncfile):
            print("can't find syncfile")
#        print(syncfile)
#        if not syncfile:
            continue
        stimfile = os.path.join(utt, 'stim.txt')
        stim = open(stimfile)
        stimtext = stim.read()
        print (stimtext)
        stimulus.append(str(stimtext))
        if any(substring in stimtext for substring in WORDS): 
            tg = os.path.join(utt,(timestamp+'.TextGrid')) # may need to change to just TextGrid depending on when data is from
            wav = os.path.join(utt,(timestamp+'.wav'))
            if not os.path.isfile(tg):
                tg = os.path.join(utt,(timestamp+'.bpr.ch1.TextGrid'))
            if not os.path.isfile(wav):
                wav = os.path.join(utt,(timestamp+'.bpr.ch1.wav')) # may need to change to bpr.ch1.wav depending on when data is from
            print(wav)


            # This next bit gets the frame corresponding to the acoustic midpoint (for comparison purposes.)
            pm = audiolabel.LabelManager(from_file = tg, from_type = 'praat')
            for lab in pm.tier('phone') :
                if (re.match('R',lab.text)) :
                    label = lab.text
                    rt1 = lab.t1
                    rt2 = lab.t2
                    rt_frmtime = ((rt1 + rt2)/2)
            subjrframelist = re.compile('.*\.jpg')
            regex = re.compile('(pc[0-9])|(mean).jpg')
            stimex = re.compile(stimtext) 
            bmpdir = os.path.join(subbmpdir, timestamp)
            imlist = [i for i in os.listdir(bmpdir) if (subjrframelist.search(i) and not regex.search(i) and not stimex.search(i))]
            print(imlist)
            try:
                im = np.array(Image.open(os.path.join(bmpdir,(imlist[0])))) #open one image to get the size
            except IndexError as e:
                myrframe = 'NA'
                continue
            if len(imlist) > 3:
#                framenos.append('NA')
#                frmtimes.append('NA')
#                continue 
                q,s = im.shape[0:2] #get the size of the images
                #print(q)
                #print(s)
                imnbr = len(imlist) #get the number of images
                print(imnbr)
                if rpca is None:
                    rpca = np.empty([len(os.listdir(subbmpdir))]+list(im.shape[0:2])) * np.nan
                if mrpca is None:
                    mrpca = np.empty([len(os.listdir(subbmpdir))]+list(im.shape[0:2])) * np.nan
                difflist = []
                for i in range(imnbr):
                    #print(i)
                    print(imlist[i])
                    print(os.path.join(bmpdir,imlist[i]))
                    bmpnorm = np.linalg.norm(np.array(Image.open(os.path.join(bmpdir, imlist[i]))))
                    if bmpnorm > 25000:
                        continue
                     if i > 0:
                        diffmatrix = np.array(Image.open(os.path.join(bmpdir,imlist[i])))-np.array(Image.open(os.path.join(bmpdir,imlist[i-1])))
                        if (np.linalg.norm(diffmatrix)) > 0:
                            difflist.append(np.linalg.norm(diffmatrix))
                        else:
                            difflist.append('NA')
         	        # identify frame where the difference in tongue shape is lowest.
                print(difflist)
                min_val, min_idx = min((val,idx) for (idx, val) in enumerate(difflist))
                print(min_val)
                print(min_idx)
	               # add frame to array
                frame_n = imlist[min_idx]
                print(frame_n)
     
                frame_number = os.path.splitext(os.path.splitext(frame_n)[0])[1][1:] # actual frame number, not INDEX
                print(frame_number)
                framenos.append(frame_number)
                openframe = np.array(Image.open(os.path.join(bmpdir,frame_n)))
                myrframe = ndimage.median_filter(openframe,5)
                print(myrframe)
                rpca[tindex,:,:] = myrframe

