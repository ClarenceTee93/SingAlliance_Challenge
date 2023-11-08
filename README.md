# SingAlliance Code Challenge

This repository contains the submission for the SingAlliance Code Challenge. 

The overall workflow is as follows: 
1) Connect to the Huobi API to pull btcusdt, ethusdt & ltcusdt futures data from the exchange. Data is processed from json format into a pandas dataframe for subsequent easy manipulation.
2) Two ways of obtaining optimal portfolio weights are presented:
    a) Mean Variance Optimization, where the portfolio volatility is minimized by constraining portfoio returns to solve for the optimal weights.
    b) Constructing 10,000 random portfolios by generating random weights of the 3 assets (btcusdt, ethusdt, ltcusdt) for each random portfolio. Based on these weights and the expected return
       and volatility of the data from 2023-09-01 00:00:00 to 2023-09-01 23:00:00, a risk-return profile is obtained where the efficient frontier can be observed. 
4) A plot of the efficient frontier and a log file will be output.

Results & Discussion:

