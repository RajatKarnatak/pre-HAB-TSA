The script extracts pre-event windows in the supplied time series. Then constructs a feature space using temporal (within the time window) mean/median and 
standard deviation - we will need more appripriate features as the TS gets noisy. Then using these features, we cluster these different pre-event windows into 
different groups (clusters) to explore the mechanisms. To determine an optimal number of clusters (mechanisms), we use a total of 17 different metrices and look at 
their consesus.
