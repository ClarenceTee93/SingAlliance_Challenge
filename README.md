# SingAlliance Code Challenge

This repository contains the submission for the SingAlliance Code Challenge. 

The overall workflow is as follows: 
1) Connect to the Huobi API to pull btcusdt, ethusdt & ltcusdt futures data from the exchange. Data is processed from json format into a pandas dataframe for subsequent easy manipulation.
2) Two ways of obtaining weights are presented:
    a) Mean Variance Optimization, where the portfolio volatility is minimized given a constraint on portfoio returns.
    b) Generating random weights and constructing a portfolio based on these weights



Results & Discussion:
