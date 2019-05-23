'''
Speech derivatives
get change in tongue position
get change in change in tongue position
'''


def diffs([frame_array],der_num=1):
    diff_array = None
    diff_array = np.empty([len(frame_array)-1]+list(frame_array.shape[1:])) * np.nan
    curder = 0    
    while curder < der_num:
        for i in np.arange(1,len(frame_array)):
            diff = frame_array[i] - frame_array[i-1]
            diff_array[i-1,:,:] = diff
        frame_array = diff_array
        curder = curder + 1        
    return frame_array        