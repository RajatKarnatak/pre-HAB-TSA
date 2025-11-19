The script extracts pre-event windows in the supplied time series. Then constructs a feature space using temporal (within the time window) mean/median and 
standard deviation - we will need more appropriate features as the TS gets noisy. Then using these features, we cluster these different pre-event windows into 
different groups (clusters) to explore the mechanisms. To determine an optimal number of clusters (mechanisms), we use a total of 17 different metrices and look at 
their consesus. The script produces several analytical plots which include: (a) cluster validation indices, (b) dendrogram, (c) inter-event interval plots, 
(d) cluster transition probabilities, and (e) a folder adv_plots which contains a variety of pre-event time series plots.
