# Multivariate-pandemic-electricity-load-prediction
The data was gathered from 2017 all the way till the end of 2020 where it was analyzed to see the impact of covid on the electricity usage and how well could a mobility factor improve the predictions

We have inputted weather features for a traditional Artificial Neural Network model (ANN), a Long-Short Term Memory Model (LSTM) and finally an Autoregressive Integrated Moving Average w/ Exogenous Variables statistical model (ARIMAX) for a multivariate time series prediction of electricity peak loads during COVID pandemic. We have implemented a random search for the ANN and LSTM where we have created an experiment generator which runs the two models ANN and LSTM with random parameters and storing them at the end of a run with their results, then generated a script which retrieves the best results anc checks which parameters were used and implements them.


in collaboration with: https://github.com/m13ammed
