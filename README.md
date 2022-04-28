# tSMOTE

tSMOTE is a non-parametric time series sampling and imputation technique described in "Imputing, predicting, and classifying observations with time sliced minority oversampling technique and recurrent neural networks"

The idea is to define a time series based on non-overlapping bins ("slices") with equal elements, generate synthetic data in each of these bins, build a 
Markov model using a sample's existing observations and the synthetic elements, and impute with the most probable path through the model.

It is presented as a suite of functions meant to be used in succession. The main objects needed are a python list the of the data set of shape (nSamples, nObservations, nFeatures). 
