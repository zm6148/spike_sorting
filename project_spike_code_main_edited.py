import h5py
import tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import scipy as sp
from operator import itemgetter
from itertools import groupby
from sklearn.feature_selection import VarianceThreshold
from sklearn import cluster
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import heapq
import cPickle
import os
import fun
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA
import Tkinter, tkFileDialog

##################################################
## load recording files from selected directory ##
##################################################
file_path = os.path.dirname(os.path.realpath(__file__))
root = Tkinter.Tk()
root.withdraw()
dirname = tkFileDialog.askdirectory(parent=root,initialdir=file_path,title='Please select a directory')
filename_list=[]
for filename in os.listdir(dirname):
    print(filename)
    filename_list.append(filename)
    filename_path=dirname+'/'+filename
    fh=tables.openFile(filename_path,'r')
    nodes = fh.root._f_listNodes()
    node_name=str(nodes[0]).split()[0]
    ## trial_log data trial info
    ## tmr 0, ts_start 61, ttype 62
    trial_info=[]
    trial_log=nodes[0].data.trial_log
    for ele in trial_log:
        trial_info.append([ele[0],ele[61],ele[62]])
    trial_info=np.asarray(trial_info)
    ## load row data          
    f=h5py.File(filename_path,'r')
    data_raw=f.get(node_name+'/data/physiology/raw')
    
    ## data properties
    Fs=24414.0625
    ts=trial_info[:,1]
    ts=ts.astype(float)
    spike_step=np.arange(0,32/Fs,1/Fs)

    ## mean across all channel
    all_channel_mean=np.mean(data_raw,axis=0)

    ## for each channel zero mean and filter
    channel_id=range(data_raw.shape[0])
    channel_id.remove(7)
    for jj in channel_id:
        data_channel=data_raw[jj,:]-all_channel_mean
        mean=np.mean(data_channel)
        data_channel_zero_mean=data_channel-mean
    
        ## band pass filter data 
        data_channel_zero_mean_filtered=fun.butter_bandpass_filter(data_channel_zero_mean,300,3000,Fs,3)
    
        ## for each channel determine noise threshold
        theta=np.median(np.absolute(data_channel_zero_mean_filtered)/0.6745)
        thr_low=5*theta
        thr_high=20*theta
    
        ## find voltage that is above thr_low and below thr_high
        ## extract 32 data points 10 before peak and 22 after peak
        ## align them at the peak.
        old_extracted_peak_index=fun.extrac_peak_index(data_channel_zero_mean_filtered,thr_low,thr_high)  
        spike_trace_channel=[]
        extracted_peak_index=[]
        for index in  old_extracted_peak_index:
            spike_window=[index-10,index+22]
            spike_trace=data_channel_zero_mean_filtered[spike_window[0]:spike_window[1]]
            if len(spike_trace)==32:
                spike_trace_channel.append(spike_trace)
                extracted_peak_index.append(index)
        spike_trace_channel=np.asarray(spike_trace_channel)
        ## remove bad extraction (max at 10 data point)
        spike_trace_channel,extracted_peak_index=fun.spike_trace_selection(spike_trace_channel,extracted_peak_index)
    
        ### do PCA analysis befroe wavelet anaylis
        ### inital 8 PC
        ### plot top 2 PCA component in 2d graph
        #pca_b = PCA(n_components=8)
        #pca_b.fit(spike_trace_channel)
        #eigen_b=pca_b.explained_variance_ratio_
        #eigen_vector_b=pca_b.components_
        #spike_trace_channel_r_b = pca_b.fit(spike_trace_channel).transform(spike_trace_channel)
    
        ## wavelet transform of each spike trace of this channel
        ## select top 10 least likely to be random as feature later
        ## and build matrix for k-mean clustering
        wave_coef_matrix=[]
        for spike_trace in spike_trace_channel:
            wave_coef=fun.discreteHaarWaveletTransform(spike_trace)
            wave_coef_matrix.append(wave_coef)
        wave_coef_matrix=np.asarray(wave_coef_matrix)
    
        ## do PCA analysis after wavelet anaylis
        ## inital 8 PC
        ## plot top 2 PCA component in 2d graph 
        pca_a = PCA(n_components=8)
        pca_a.fit(wave_coef_matrix)
        eigen_a=pca_a.explained_variance_ratio_
        eigen_vector_a=pca_a.components_
        spike_trace_channel_r_a = pca_a.fit(spike_trace_channel).transform(spike_trace_channel)
    
        ## based on PCA analysis 
        ## initialize 3 cluster and use k-mean to classify
        ## visulized results on 2d top 2 pca plot
        k_means = cluster.KMeans(init='k-means++',n_clusters=3, n_init=10, n_jobs=-2)
        res=k_means.fit(wave_coef_matrix)
        labels=res.labels_
    
        ## visulize results on 2d PCA plots
        ## average of each class
        ## raster plot of each class
        ## classification results labels
        class_1_index=np.where(labels == 0)[0]
        class_2_index=np.where(labels == 1)[0]
        class_3_index=np.where(labels == 2)[0]
        
        ## extracted spike index
        extracted_peak_index_1=extracted_peak_index[class_1_index]
        extracted_peak_index_2=extracted_peak_index[class_2_index]
        extracted_peak_index_3=extracted_peak_index[class_3_index]
        
        ## raster plot of each class
        ## us ts as window time stamp for now
        ## ts and 1s after ts is the window
        ## ts: ts+2*20000
        ## 2ms as the step 2*20000/1000 in raster plot
        window=20000
        step=40
        block=window/step
        go_index,nogo_index=fun.trial_selection(trial_info)
        ts_go=ts[go_index]
        ts_nogo=ts[nogo_index]
        ## go condition with max tmr
        raster_index_1_go=fun.raster(extracted_peak_index_1,ts_go,window,step)
        raster_index_2_go=fun.raster(extracted_peak_index_2,ts_go,window,step)
        raster_index_3_go=fun.raster(extracted_peak_index_3,ts_go,window,step)
        ## no go condition with any tmr
        raster_index_1_nogo=fun.raster(extracted_peak_index_1,ts_nogo,window,step)
        raster_index_2_nogo=fun.raster(extracted_peak_index_2,ts_nogo,window,step)
        raster_index_3_nogo=fun.raster(extracted_peak_index_3,ts_nogo,window,step)

###############################################################################
        fig = plt.figure(filename+' channel '+str(jj),figsize=(25, 12))    

        # Top row
        ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=3)
        ax2 = plt.subplot2grid((3, 6), (0, 3), colspan=3)

        # Middle row
        ax3 = plt.subplot2grid((3, 6), (1, 0), colspan=2)
        ax4 = plt.subplot2grid((3, 6), (1, 2), colspan=2)
        ax5 = plt.subplot2grid((3, 6), (1, 4), colspan=2)

        # Bottom row
        ax6 = plt.subplot2grid((3, 6), (2, 0), colspan=1)
        ax7 = plt.subplot2grid((3, 6), (2, 1), colspan=1)
        ax8 = plt.subplot2grid((3, 6), (2, 2), colspan=1)
        ax9 = plt.subplot2grid((3, 6), (2, 3), colspan=1)
        ax10 = plt.subplot2grid((3, 6), (2, 4), colspan=1)
        ax11 = plt.subplot2grid((3, 6), (2, 5), colspan=1)

        ## eigen value variance precentage
        ax1.plot(eigen_a,marker='o', linestyle='--', color='r', label='eigen value')
        ax1.set_title('variance explained by Eigen values')
        ax1.legend()


        ## projected top 2 eigen vector plot
        ax2.plot(spike_trace_channel_r_a[class_1_index,0],spike_trace_channel_r_a[class_1_index,1],'o',color='r',label='Class 1')
        ax2.plot(spike_trace_channel_r_a[class_2_index,0],spike_trace_channel_r_a[class_2_index,1],'^',color='g',label='Class 2')
        ax2.plot(spike_trace_channel_r_a[class_3_index,0],spike_trace_channel_r_a[class_3_index,1],'D',color='b',label='Class 3')
        ax2.set_title('Top 2 PCA plot')
        ax2.legend(loc=8,ncol=3, mode="expand", borderaxespad=0.)



        ## class 1 trace and average
        for ele in spike_trace_channel[class_1_index,:]:
            ax3.plot(spike_step,ele, alpha=0.2)
        ax3.plot(spike_step,np.mean(spike_trace_channel[class_1_index,:],axis=0), linewidth=3, color='r',label=' class 1 average')
        ax3.set_title('class 1 trace')
        ax3.legend()
        ax3.set_xlabel('Time is seconds')

        ## class 2 trace and average
        for ele in spike_trace_channel[class_2_index,:]:
            ax4.plot(spike_step,ele, alpha=0.2)
        ax4.plot(spike_step,np.mean(spike_trace_channel[class_2_index,:],axis=0),linewidth=3, color='g',label='class 2 average')
        ax4.set_title('class 2 trace')
        ax4.legend()
        ax4.set_xlabel('Time is seconds')

        ## class 3 trace and average
        for ele in spike_trace_channel[class_3_index,:]:
            ax5.plot(spike_step,ele, alpha=0.2)
        ax5.plot(spike_step,np.mean(spike_trace_channel[class_3_index,:],axis=0), linewidth=3, color='b',label='class 3 average')
        ax5.set_title('class 3 trace')
        ax5.legend()
        ax5.set_xlabel('Time is seconds')

        ### class 1 raster stim 6
        ax6.set_title('class 1 stim on')
        for ii in range(len(raster_index_1_go)):
            y=(int(500/len(raster_index_1_go))*(ii+1))/2
            if len(raster_index_1_go[ii])!=0:
                y_value=[y]*len(raster_index_1_go[ii])
                ax6.plot(np.divide(raster_index_1_go[ii],Fs),y_value,'ro')
        ax6.set_xlim([0,block/Fs])
        ax6.set_ylim([0,300])
        ax6.set_xlabel('Time is seconds')
        
    
        ### class 1 raster no stim 7
        ax7.set_title('class 1 stim off')
        for ii in range(len(raster_index_1_nogo)):
            y=(int(500/len(raster_index_1_nogo))*(ii+1))/2
            if len(raster_index_1_nogo[ii])!=0:
                y_value=[y]*len(raster_index_1_nogo[ii])
                ax6.plot(np.divide(raster_index_1_nogo[ii],Fs),y_value,'ro')
        ax7.set_xlim([0,block/Fs])
        ax7.set_ylim([0,300])
        ax7.set_xlabel('Time is seconds')

        ### class 2 raster stim 8
        ax8.set_title('class 2 stim on')
        for ii in range(len(raster_index_2_go)):
            y=(int(500/len(raster_index_2_go))*(ii+1))/2
            if len(raster_index_2_go[ii])!=0:
                y_value=[y]*len(raster_index_2_go[ii])
                ax8.plot(np.divide(raster_index_2_go[ii],Fs),y_value,'ro')
        ax8.set_xlim([0,block/Fs])
        ax8.set_ylim([0,300])
        ax8.set_xlabel('Time is seconds')

        ### class 2 raster no stim 9
        ax9.set_title('class 2 stim off')
        for ii in range(len(raster_index_2_nogo)):
            y=(int(500/len(raster_index_2_nogo))*(ii+1))/2
            if len(raster_index_2_nogo[ii])!=0:
                y_value=[y]*len(raster_index_2_nogo[ii])
                ax9.plot(np.divide(raster_index_2_nogo[ii],Fs),y_value,'ro')
        ax9.set_xlim([0,block/Fs])
        ax9.set_ylim([0,300])
        ax9.set_xlabel('Time is seconds')

        ### class 3 raster stim 10 
        ax10.set_title('class 3 stim on')
        for ii in range(len(raster_index_3_go)):
            y=(int(500/len(raster_index_3_go))*(ii+1))/2
            if len(raster_index_3_go[ii])!=0:
                y_value=[y]*len(raster_index_3_go[ii])
                ax10.plot(np.divide(raster_index_3_go[ii],Fs),y_value,'ro')
        ax10.set_xlim([0,block/Fs])
        ax10.set_ylim([0,300])
        ax10.set_xlabel('Time is seconds')

        ### class 3 raster no stim 11
        ax11.set_title('class 3 stim off')
        for ii in range(len(raster_index_3_nogo)):
            y=(int(500/len(raster_index_3_nogo))*(ii+1))/2
            if len(raster_index_3_nogo[ii])!=0:
                y_value=[y]*len(raster_index_3_nogo[ii])
                ax10.plot(np.divide(raster_index_3_nogo[ii],Fs),y_value,'ro')
        ax11.set_xlim([0,block/Fs])
        ax11.set_ylim([0,300])
        ax11.set_xlabel('Time is seconds')


        plt.tight_layout()
        #plt.show()

        ## save to file ##
        fig_path=os.path.dirname(dirname)+'/results'+'/'+filename+'/'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig_name = fig_path+filename+'_channel_'+str(jj)+'.png'
        fig_filename = os.path.join(fig_path , fig_name)
        fig.savefig(fig_filename)
    
    
    
    
        

    