import sys
import audiolabel
import numpy as np
import ultratils.pysonix.bprreader
import pandas as pd
import parselmouth
from scipy import stats



# could pass in frame array instead of bpr
def get_datadf_simple(rawUSinput, au, syncloc):
    
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

    pmfcc = au.to_mfcc()
    mfcc = np.transpose(pmfcc.to_array())  # transpose this to get time (frames) on the first dimension

    au_diff = [np.mean(np.abs(mfcc[i]-mfcc[i-1])) for i in range(1, len(mfcc))]
    frame_times['au_diff']=frame_times.time.apply(lambda x: au_diff[int(pmfcc.get_frame_number_from_time(x)+1)])

    pmint = au.to_intensity()
    frame_times['au_int'] = frame_times.time.apply(lambda x: pmint.get_value(x))
    
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
    phone = []
    pos = []
    pm = audiolabel.LabelManager(from_file = tg, from_type = 'praat')    
    for i in range(0, len(validdf)-1):
        synctime = validdf.time[i]
        labmatch = pm.tier('phone').label_at(synctime).text
        labt1 = int(pm.tier('phone').label_at(synctime).t1)
        labt2 = int(pm.tier('phone').label_at(synctime).t2)
        labperc = float((synctime-labt1)/(labt2-labt1))            
        phone.append(labmatch)
        pos.append(labperc)
    posdf = pd.concat([validdf,pd.DataFrame({'phone':phone})], ignore_index = False, axis = 1)
    posdf = pd.concat([posdf,pd.DataFrame({'pos':pos})], ignore_index = False, axis = 1)
    return posdf
                