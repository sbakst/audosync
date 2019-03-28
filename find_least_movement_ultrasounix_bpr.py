from __future__ import absolute_import, division, print_function

import os
import re
import numpy as np
import glob
from ultratils.exp import Exp
import audiolabel
import argparse
from scipy.spatial import distance
from scipy import ndimage
from sklearn.metrics.pairwise import euclidean_distances
import itertools
import subprocess


# for plotting
from PIL import Image
import matplotlib.pyplot as plt

# for PCA business
from sklearn import decomposition
from sklearn.decomposition import PCA

vre =  re.compile(
         "^(?P<vowel>AA|AE|AH|AO|EH|ER|EY|IH|IY|OW|UH|UW)(?P<stress>\d)?$"
      )

parser = argparse.ArgumentParser()
parser.add_argument("directory", help = "Experiment directory with all subjects")
parser.add_argument("subject", help = "subject number")
parser.add_argument("-v", "--visualize", help = "Produce pngs of frames", action = "store_true")
parser.add_argument("-c", "--convert", help = "flip frames", action = "store_true")
parser.add_argument("outfile", help = "outfile containing all of the measurements")
args = parser.parse_args()

outfile = os.path.join(args.directory, args.outfile)
out = open(outfile, 'a')
header = '\t'.join(['subject','timestamp', 'mindif', 't1', 'f1','f2', 'f3','f4', 'which_frame', 'num_frames'])
out.write(header + '\n')
# Read in and parse the arguments, getting directory info and whether or not data should flop

subjdir = os.path.join(args.directory, str(i))
try:
    expdir = (os.path.join(subjdir, 'block3'))
    print(expdir)
except IndexError:
    print("\tDirectory provided doesn't exist!")
    sys.exit(2)

e = Exp(expdir)
e.gather()

frames = None
threshhold = 0.020 # threshhold value in s for moving away from acoustic midpoint measure

trial = []
phone = []
tstamp = []

#    if args.convert:
#        conv_frame = e.acquisitions[0].image_reader.get_frame(0)
#        conv_img = test.image_converter.as_bmp(conv_frame)

for idx,a in enumerate(e.acquisitions):

    if frames is None:
        if args.convert:
            frames = np.empty([len(e.acquisitions)] + list(conv_img.shape))
        else:
            frames = np.empty([len(e.acquisitions)] + list(a.image_reader.get_frame(0).shape)) * np.nan
    
    a.gather()

    tg = str(a.abs_image_file + ".ch1.TextGrid")
    fbfile = str(a.abs_image_file + ".ch1.fb")
    wav = str(a.abs_image_file + ".ch1.wav")
    pm = audiolabel.LabelManager(from_file=tg, from_type="praat")
    print(pm)
    v,m = pm.tier('phone').search(vre, return_match=True)[-1] # return last V = target V
    print(v)
#    print('here comes m')
#    print(m)
    #print( pm.tier('phone'))
    word = pm.tier('word').label_at(v.center).text
    print(word)
#    print(phone)    
#    phase.append(a.runvars.phase)
    trial.append(idx)
    tstamp.append(a.timestamp)
    phone.append(v.text)
    

    #        mid, mid_lab, mid_repl = a.frame_at(v.center,missing_val="prev")

    Rframes = a.sync_lm.tier('raw_data_idx').tslice(t1=v.t1, t2=v.t2)
    print(v.t1)
    rfrlist = []
    for rlab in Rframes:
        rf, rf_lab = (a.frame_at(rlab.t1))
        rf = np.array(rf)
        try:
            rf[0:150] = 0 # added this minimask -- could be smaller
        except (IndexError):
            pass
        rfrlist.append(rf)
    tstamp.append(a.timestamp)
    phone.append(v.text)
    mindiffs = []
    mindex = int(len(rfrlist)/2) # this is for trying to force the minimum in the last half of the frames, which typically works better for American /r/.
    for k in range (mindex,(len(rfrlist))): # force last half of the frames
        try:
            Y = (rfrlist[k])-(rfrlist[k-1])
            mindiffs.append(np.linalg.norm(Y))
        except (TypeError, KeyError) as e:
            print ('bad frame')
            pass
            #            mindiffs.append(np.linalg.norm(Y))
    print(mindiffs)
    try:
       val,indx = min((val,idx) for (idx, val) in enumerate(mindiffs))
    except (ValueError) as g:
       break
    if indx == 0:
        indx == indx +1
        # The numbers above are going to help us pick the frame for acoustics analysis and for pca.
    frames[idx,:,:] = rfrlist[indx] # this is the frame for pca

# Everything above this line is articulation. Everything below is acoustics at the timepoint corresponding to the determined frame.

###############################################################################################################################################################################

    # We will need these numbers for acoustics.
    # meas_t1 = (Rframes[indx]).t1
    # print(meas_t1)
    
    

    # We will not do pca or other things in this script besides acoustics. We are looping over all subjects, and masking is done on a per-block basis.
#     t1_start = 0.0
#     t1_step = 0.01
#     try:
#         formant_proc = subprocess.check_call(["rformant", wav])#, stdin=rdc_proc.stdout) # also remove 20
#     except subprocess.CalledProcessError as e:
#         print(e)
#         print(e.stdout)
#     ppl_proc = subprocess.Popen(
#         ["pplain", fbfile],
#         stdout=subprocess.PIPE)
#     print(fbfile)
# 
#     lm = audiolabel.LabelManager(
#         from_file=ppl_proc.stdout,   # read directly from pplain output
#         from_type='table',
#         sep=" ",
#         fields_in_head=False,
#         fields="f1,f2,f3,f4",
#         t1_col=None,                 # esps output doesn't have a t1 column
#         t1_start=0.0,          # autocreate t1 starting with this value and
#         t1_step=0.01)             # increase by this step
#     f4 = lm.tier('f4')
#     f3 = lm.tier('f3')
#     f2 = lm.tier('f2')
#     f1 = lm.tier('f1')
#     framef3 = (f3.label_at(meas_t1)).text
#     if float(framef3) > 2300:
#         framef3 = "NA"
#     framef2 = (f2.label_at(meas_t1)).text
#     framef1 = (f1.label_at(meas_t1)).text
#     framef4 = (f4.label_at(meas_t1)).text
#     row_out = '\t'.join([str(i),str(a.timestamp), str(val), str(meas_t1), framef1, framef2, framef3, framef4, str(indx), str(len(mindiffs))])
#     out.write(row_out+'\n')
