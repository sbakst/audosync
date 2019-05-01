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
parser.add_argument("-p", "--pca", help = "Run PCA", action = "store_true")
parser.add_argument("-v", "--visualize", help = "Produce pngs of pca frames", action = "store_true")
parser.add_argument("-c", "--convert", help = "flip frames", action = "store_true")
parser.add_argument("outfile", help = "outfile containing all of the measurements")
args = parser.parse_args()

outfile = os.path.join(args.directory, args.outfile)
out = open(outfile, 'a')
header = '\t'.join(['subject','timestamp', 'mindif', 't1', 'which_frame', 'num_frames'])
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
        winrframe = rfrlist[indx]
        winrframe = ndimage.median_filter(winrframe,5) # filter
    frames[idx,:,:] =  winrframe# this is the frame for pca

# Everything above this line is articulation. Everything below is acoustics at the timepoint corresponding to the determined frame.

###############################################################################################################################################################################
# PCA
if args.pca:
    frames = np.squeeze(frames)
    print(np.shape(frames))
    trial = np.squeeze(np.array(trial))
    phone = np.squeeze(np.array(phone))
    tstamp = np.squeeze(np.array(tstamp))
    keep_indices = np.where(~np.isnan(frames).any(axis=(1,2)))[0]
    kept_phone = np.array(phone,str)[keep_indices]
    kept_trial = np.array(trial,str)[keep_indices]
    kept_frames = frames[keep_indices]
    kept_tstamp = tstamp[keep_indices]
    
    n_components = 6
    pca = PCA(n_components=n_components)
    
    frames_reshaped = kept_frames.reshape([kept_frames.shape[0], kept_frames.shape[1]*kept_frames.shape[2]])
    
    pca.fit(frames_reshaped)
    analysis = pca.transform(frames_reshaped)
    
    meta_headers = ["trial","timestamp","phone"]
    pc_headers = ["pc"+str(i+1) for i in range(0,n_components)] # determine number of PC columns; changes w.r.t. n_components
    headers = meta_headers + pc_headers

    out_file = args.outfile

    d = np.row_stack((headers,np.column_stack((kept_trial,kept_tstamp,kept_phone,analysis))))
    np.savetxt(out_file, d, fmt="%s", delimiter =',')

    print("Data saved. Explained variance ratio of PCs: %s", str(pca.explained_variance_ratio_))

    expl = './expl.txt'
    fifi = open(expl, 'w')
    fifi.write(str(pca.explained_variance_ratio_))
    fifi.close()
    
    if args.visualize:
        image_shape = (416,69)
        print(e.acquisitions) 
        for n in range(0,n_components):
            d = pca.components_[n].reshape(image_shape)# We will need these numbers for acoustics.
            mag = np.max(d) - np.min(d)
            d = (d-np.min(d))/mag*255
            pcn = np.flipud(e.acquisitions[0].image_converter.as_bmp(d)) # converter from any frame will work; here we use the first
            
            
#     t1_start = 0.0
#            if args.flop:  t1_step = 0.01
#                pcn = np.fliplr(pcn)  try:
#         formant_proc = subprocess.check_call(["rformant", wav])#, stdin=rdc_proc.stdout) # also remove 20
#           plt.title("PC{:} min/max loadings".format(n+1))  except subprocess.CalledProcessError as e:
#           plt.imshow(pcn, cmap="Greys_r")      print(e)
#           savepath = "pc{:}.pdf".format(n+1)      print(e.stdout)
#           plt.savefig(savepath)  ppl_proc = subprocess.Popen(
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


