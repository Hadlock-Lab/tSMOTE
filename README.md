# tSMOTE

tSMOTE is a non-parametric time series sampling and imputation technique described in "Imputing, predicting, and classifying observations with time sliced minority oversampling technique and recurrent neural networks": https://arxiv.org/abs/2201.05634 (version 2 will be posted shortly)

The idea is to define a time series based on non-overlapping bins ("slices") with equal elements, generate synthetic data in each of these bins, build a Markov model using a sample's existing observations and the synthetic elements, and impute with the most probable path through the model.

It is presented as a suite of functions meant to be used in succession. The main objects needed are a python list of the data set of shape (nSamples, nObservations, nFeatures). 

The jupyter notebook 2D_Oscillator_tsmote.ipynb has the code used for the 2D oscillator example in the paper. It can be treated as a tutorial for tSMOTE usage and visualizaiton. I apologize in advance since the model as written is rather messy, but the tSMOTE usage steps should be clear. For any questions please open an issue or contact andrew.baumgartner@isbscience.org

## Dependencies
See the preamble in both tSmote.py and 2D_Oscillators_tSmote.ipynb for a full list of dependencies.
