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
parser.add_argument("-v", "--visualize", help = "Produce pngs of frames", action = "store_true")
parser.add_argument("-c", "--convert", help = "flip frames", action = "store_true")
parser.add_argument("outfile", help = "outfile containing all of the measurements")
args = parser.parse_args()

outfile = os.path.join(args.directory, args.outfile)
out = open(outfile, 'a')
header = '\t'.join(['subject','timestamp', 'mindif', 't1', 'f2', 'f3', 'which_frame', 'num_frames'])
out.write(header + '\n')

# bad_subs = [321, 326]

for i in range (0,1):
#    if i in bad_subs:
#        continue
# Read in and parse the arguments, getting directory info and whether or not data should flop
#    subjdir = os.path.join(args.directory, str(i))
    try:
        expdir = (args.directory)
        print(expdir)
    except IndexError:
        print("\tDirectory provided doesn't exist!")
        sys.exit(2)
    
    # TODO right now this can only be run on one subject's baseline phase at a time. this should loop over subjects.
    # TODO alternately, accept STDIN argument for a single subject or series of subjects
    # for s in subjects:
    #    ... initialize PCA, do PCA, output
    
    e = Exp(expdir)
    e.gather()
    
    frames = None
    threshhold = 0.020 # threshhold value in s for moving away from acoustic midpoint measure
    
    # subject = [] TODO add this in once subject loop is added 
    #phase = []
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
#        print(e)
    #    print(phone)    
    #    phase.append(a.runvars.phase)
        trial.append(idx)
        tstamp.append(a.timestamp)
        phone.append(v.text)
        

#        mid, mid_lab, mid_repl = a.frame_at(v.center,missing_val="prev")
    
        Rframes = a.sync_lm.tier('raw_data_idx').tslice(t1=v.t1, t2=v.t2)
        print(v.t1)
#        print(e)
#        print('did that work?')
        rfrlist = []
        for rlab in Rframes:
            rf, rf_lab = (a.frame_at(rlab.t1))
            rf = np.array(rf)
            try:
                rf[0:150] = 0 # added this minimask
            except (IndexError):
                pass
            rfrlist.append(rf)
        tstamp.append(a.timestamp)
        phone.append(v.text)
        mindiffs = []
        mindex = int(len(rfrlist)/2)
        for k in range (mindex,(len(rfrlist))): # force last half of the frames
            try:
                Y = (rfrlist[k])-(rfrlist[k-1])
                mindiffs.append(np.linalg.norm(Y))
            except (TypeError, KeyError) as erk:
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
        myrframe = rfrlist[indx]
        myrframe = ndimage.median_filter(myrframe,5)
        frames[idx,:,:] = myrframe # this is the frame for pca

    frames[:,0:170,:]=0 # is this how we mask?

#    print(e)

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

    n_components = 15
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
#    np.savetxt('expl.txt', str(pca.explained_variance_ratio_), delimiter = ',', fmt="%s")


    if args.visualize:
        image_shape = (416,69)
#        print(e.acquisitions) 
        for n in range(0,n_components):
            d = pca.components_[n].reshape(image_shape)
#            print ('look tis d')
#            print (d)
            mag = np.max(d) - np.min(d)
            d = (d-np.min(d))/mag*255

            pcn = np.flipud(e.acquisitions[0].image_converter.as_bmp(d)) # converter from any frame will work; here we use the first

#            if args.flop:
#                pcn = np.fliplr(pcn)

            plt.title("PC{:} min/max loadings".format(n+1))
            plt.imshow(pcn, cmap="Greys_r")
            savepath = "pc{:}.pdf".format(n+1)
            plt.savefig(savepath)



#     frames = np.empty([len(e.acquisitions)] + list(a.image_reader.get_frame(0).shape)) * np.nan

