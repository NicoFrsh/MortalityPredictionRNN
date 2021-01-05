# MortalityPredictionRNN

This repository contains R code to model mortality rates in 9 different countries using the Human Mortality Database (HMD) dataset which is stored in the 'data' directive. 
The predictions are made using Recurrent Neural Networks, both LSTM and GRU architectures are possible. 

How to work with this repository:

There are two options: 

If you only want to create and fit a specific model with specific parameters you can use the script rnn_mortality.R and specify the parameters in
the head of the code. 

To perform a complete hyperparameter search to find the best performing models based on the Mean-Squared-Error using different combinations
of parameters do the following:

1. Run the grid_search script using your parameter options in the beginning of the code. Note that, for performance issues, the tuning_run function in the last line of the code
has the 'sample' parameter set to 0.01, so that it takes only a 1% sample of the possible parameter combinations. The script will then perform a grid search saving all models
in a new subdirectory called 'grid_search_LSTM' or 'grid_search_GRU'. 

2. To evaluate the results of the grid search you can use the evaluate_grid_search.R script.
