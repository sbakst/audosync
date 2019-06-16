import sys
import audiolabel
import numpy as np
import ultratils.pysonix.bprreader
import pandas as pd
import parselmouth
# try:
from parselmouth.praat import call as pcall  
# except ModuleNotFoundError as e:
from scipy import stats



# could pass in frame array instead of bpr
# if maxhz != 0, filter out frequencies under maxhz before mfcc transformation
def get_datadf_simple(rawUSinput, au, syncloc, maxhz=300):
    
    frame_times = audiolabel.LabelManager(from_file = syncloc, from_type='table', t1_col='seconds').as_df()[1]
    frame_times = frame_times.rename(columns={'text':'frameN','t1':'time'})

    if isinstance(rawUSinput,np.ndarray):
        frames = rawUSinput
    else :
        frames = [rawUSinput.get_frame(i) for i in range(0, rawUSinput.nframes)]
    
    frames_diff = [np.mean(np.abs(frames[i]-frames[i-1])) for i in range(1, len(frames))]
    
    frame_times['us_diff'] = frame_times['frameN'].apply(lambda x: frames_diff[int(x)-1] 
                                                         if (x!='NA' and int(x)>0) else np.nan)
    for i in range(1, len(frame_times)):
        if frame_times['frameN'][i-1]=='NA':
            frame_times.loc[i,'us_diff']=np.nan

    if maxhz > 0:
        au = pcall(au, 'Filter (stop Hann band)...', 0, maxhz, 100)  # filter voicing

    pmfcc = au.to_mfcc()
    mfcc = np.transpose(pmfcc.to_array())  # transpose this to get time (frames) on the first dimension

    au_diff = [np.mean(np.abs(mfcc[i]-mfcc[i-1])) for i in range(1, len(mfcc))]
    frame_times['au_diff']=frame_times.time.apply(lambda x: au_diff[int(pmfcc.get_frame_number_from_time(x)+1)])

    pmint = au.to_intensity()
    frame_times['au_int'] = frame_times.time.apply(lambda x: pmint.get_value(x))
    
    return frame_times

def get_datadf_2der(rawUSinput, au, syncloc, maxhz=300):
    
    frame_times = audiolabel.LabelManager(from_file = syncloc, from_type='table', t1_col='seconds').as_df()[1]
    frame_times = frame_times.rename(columns={'text':'frameN','t1':'time'})

    if isinstance(rawUSinput,np.ndarray):
        frames = rawUSinput
    else :
        frames = [rawUSinput.get_frame(i) for i in range(0, rawUSinput.nframes)]
    
    frames_diff = [np.mean(np.abs(frames[i]-frames[i-1])) for i in range(1, len(frames))]
    
    frame_times['us_diff'] = frame_times['frameN'].apply(lambda x: frames_diff[int(x)-1] 
                                                         if (x!='NA' and int(x)>0) else np.nan)
    for i in range(1, len(frame_times)):
        if frame_times['frameN'][i-1]=='NA':
            frame_times.loc[i,'us_diff']=np.nan

    if maxhz > 0:
        au = pcall(au, 'Filter (stop Hann band)...', 0, maxhz, 100)  # filter voicing

    pmfcc = au.to_mfcc()
    mfcc = np.transpose(pmfcc.to_array())  # transpose this to get time (frames) on the first dimension

    au_diff = [np.mean(np.abs(mfcc[i]-mfcc[i-1])) for i in range(1, len(mfcc))]
    frame_times['au_diff']=frame_times.time.apply(lambda x: au_diff[int(pmfcc.get_frame_number_from_time(x)+1)])

    pmint = au.to_intensity()
    frame_times['au_int'] = frame_times.time.apply(lambda x: pmint.get_value(x))
    
    us_acc = [frame_times['us_diff'][i]-frame_times['us_diff'][i-1] for i in range(2, len(frame_times))]
    frame_times['us_acc'] = frame_times['frameN'].apply(lambda x: us_acc[int(x)-1]
                                                             if (x!='NA' and int(x)>1) else np.nan)
    au_acc = [frame_times['au_diff'][i] - frame_times['au_diff'][i-1] for i in range (1, len(frame_times))]
    frame_times['au_acc'] = np.nan
    frame_times['au_acc'][1:len(frame_times)] = au_acc    
    
    return frame_times

    


# bpr: BPR file
# au: Parselmouth Sound object, mono
# sync: LabelManager object
# windowlen: window length for analysis, in seconds
def syncmatch(frame_times, windowlen=0.2, offset=0, limitamp=False):

    if offset!=0:
        auonly = frame_times.drop(columns=['frameN','us_diff'])
        usonly = frame_times.drop(columns=['au_diff','au_int'])
        auonly.time = auonly.time.apply(lambda x: x+offset)
        frame_times = pd.merge_asof(usonly, auonly, on='time')

    if limitamp:
        min_intensity = frame_times['au_int'].max()-25
        validdf = frame_times[(frame_times['us_diff'] > 0) & (frame_times['au_diff'] > 0) 
                              & (frame_times['au_int'] > min_intensity)].reset_index()
    else:
        validdf = frame_times[(frame_times['us_diff'] > 0) & (frame_times['au_diff'] > 0)].reset_index()

    rs = []
    ps = []
    starttimes = [t for t in validdf['time'] if t+windowlen < np.max(validdf['time'])]
    for starttime in starttimes:
        subdf = validdf[(validdf['time'] >= starttime-windowlen/2) & (validdf['time'] <= starttime+windowlen/2)]
        sts = stats.linregress(x=subdf['us_diff'], y=subdf['au_diff'])
        rs = rs + [sts.rvalue]
        ps = ps + [sts.pvalue]

    validdf = pd.concat([validdf,pd.DataFrame({'r':rs})], ignore_index=False, axis=1)
    validdf = pd.concat([validdf,pd.DataFrame({'p':ps})], ignore_index=False, axis=1)

    return validdf


def matchstreak(syncdf, p_crit=0.05, r_crit=0.5):

    inrange=False
    streak=0
    maxstreak=0
    streakonset=[]
    streakoffset=[]
    for i in range(0, len(syncdf)-1):
        if syncdf.p[i]<p_crit and syncdf.r[i]>r_crit:
            if inrange:
                streak+=1
            else:
                inrange=True
                streak=1
                streakonset = streakonset+[syncdf.time[i]]
        else:
            if inrange:
                inrange=False
                streakoffset = streakoffset+[syncdf.time[i]]
        maxstreak=max(maxstreak,streak)

    return (maxstreak, streakonset, streakoffset)
    
def get_corr_pos(validdf, tg):
    word = []
    phone = []
    pos = []
    pm = audiolabel.LabelManager(from_file = tg, from_type = 'praat')    
    for i in range(0, len(validdf)-1):
        synctime = validdf.time[i]
        labword = pm.tier('word').label_at(synctime).text
        labmatch = pm.tier('phone').label_at(synctime).text
        labt1 = float(pm.tier('phone').label_at(synctime).t1)
        labt2 = float(pm.tier('phone').label_at(synctime).t2)
        labperc = float((synctime-labt1)/(labt2-labt1))            
        phone.append(labmatch)
        pos.append(labperc)
        word.append(labword)
    posdf = pd.concat([validdf,pd.DataFrame({'phone':phone})], ignore_index = False, axis = 1)
    posdf = pd.concat([posdf,pd.DataFrame({'pos':pos})], ignore_index = False, axis = 1)i
    posdf = pd.concat([posdf,pd.DataFrame({'word':word})], ignore_index = False, axis = 1)
    return posdf
                
