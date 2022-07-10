import sys
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import audosync as audo
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("csv", help = "csv file of data")
parser.add_argument("savedir", help = "where plot should be saved")


args = parser.parse_args()
data = args.csv
savedir = args.savedir

df = pd.Series.from_csv(data, header = 0, parse_dates = False,index_col = 'pos')
#df = df['us_diff'].interpolate(method='pad')
df = df.replace(0, np.NaN)
df.dropna(axis = 0, how = 'any', inplace=True)
df.drop_duplicates()

#df = pd.TimeSeries(df,
df.interpolate(inplace=True)
df = df[(df.offset == 0)]

df = df[(df.phone != 'AY1')]
# df = df[(df.p<0.05)]


signs = ['pos','neg']
for sign in signs:
    if sign == 'pos':
        ftp = df[df.r>0]
    else:
        ftp = df[df.r<0]
    
    ftp = ftp[ftp.phone != 'sp']

#    vtp = ftp[(ftp.phone == 'AY1')|( ftp.phone == 'AA1' )|( ftp.phone == 'IY1' )|( ftp.phone == 'AH0')]
    vtp = ftp[(ftp.phone == 'AY1') |( ftp.phone == 'OW1')|( ftp.phone == 'AA1' )|( ftp.phone == 'IY1' )|( ftp.phone == 'AH0' )|( ftp.phone == 'AH1')| (ftp.phone == 'EH1')| ( ftp.phone == 'AE1')|(ftp.phone=='IH1')|(ftp.phone=='EY1')|(ftp.phone=='AO1')]
    #stp = ftp[(ftp.phone == 'S' )|( ftp.phone == 'SH')]
    #rtp = ftp[(ftp.phone == 'L' )|( ftp.phone == 'R') | (ftp.phone == 'W')]
    #ktp = ftp[(ftp.phone == 'B' )|( ftp.phone == 'P') | (ftp.phone == 'K') | (ftp.phone == 'G' )|( ftp.phone == 'D') |( ftp.phone == 'T')]
    #ntp = ftp[(ftp.phone == 'N' )|(  ftp.phone == 'M')]

    fig, ax1 = plt.subplots(1,1)
    fig.set_size_inches(11,6)

    vtpp = sns.lineplot(x="pos", y="au_diff", hue = 'phone',estimator = np.median, data = vtp, alpha = 0.5, ax = ax1)
    vtpp2 = vtpp.twinx()
    sns.lineplot(x = "pos", y = "us_diff", hue = 'phone', estimator = np.median,data = vtp, ax = vtpp2)
    ax1.get_legend().set_visible(False)
    numines = vtpp2.get_lines()
    print(len(numines))
    nums = int(((len(numines))-1)/2)
    for p in range(0,nums):
#    for q in range(0,nums):
#        p = q+nums
        vtpp2.lines[p].set_linestyle("--")    
    vtpp2.legend(loc='upper right')
    pic = sign+'_vplot_allsubs_smooth1.png'
    savepic = os.path.join(savedir,pic)
    fig.savefig(savepic)