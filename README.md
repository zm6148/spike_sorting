# spike_sorting
k-means spike sorting on animal recordings

sample data can be downloaded 
https://drive.google.com/drive/folders/0B9quQa4Md4D-UGtHeFVNendmSWs?usp=sharing

project_spike_code_main_edited.py will ask for the directory where electrophysiological data were stored (.hdf5 files). For each file, it will store and output a result summary for each channel as shown below:

![Image description](https://lh3.googleusercontent.com/LEtBmcLb1v78630GlJAHtVZisEdHhtlgiBgvJ3GlGzWpYphP1rKunXE7kboN76EKNeBJrlzO-uH4ToPi1VpUa1b9xNirefaxmPAkfmlXPj2WnhhI808ILlwuBJKffzqAb1e6FIDIBQ=w2400)

It has 
1) Eigen value plot (guidance to how many clusters there may be for k-means)
2) Top 2 PCA component plot for each classes
3) Block average of possible classes
4) Raster plot for neural firing event during one task window.

This particular channel has two neural firing types that behave differently to stimuli. One is active during target onset, the other is active during noise masker.
