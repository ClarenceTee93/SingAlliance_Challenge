# SingAlliance Code Challenge

This repository contains the submission for the SingAlliance Code Challenge. 

The overall workflow is as follows: 
1) Connect to the Huobi API to pull btcusdt, ethusdt & ltcusdt futures data from the exchange. Data is processed from json format into a pandas dataframe for subsequent easy manipulation.
2) Two ways of obtaining weights are presented:
    a) Mean Variance Optimization, where the portfolio volatility is minimized by constraining portfoio returns.
    b) Constructing 10,000 random portfolios by generating 10,000 random weights for the 3 assets (btcusdt, ethusdt, ltcusdt). Based on these weights and the expected return and volatility of 



Results & Discussion:
