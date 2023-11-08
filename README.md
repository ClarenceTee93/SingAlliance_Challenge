# SingAlliance Code Challenge
Introduction:
This repository contains the submission for the SingAlliance Code Challenge. 

Steps to run:
1) Clone this Github Repository: git clone https://github.com/ClarenceTee93/SingAlliance_Challenge.git
2) cd into the directory "SingAlliance_Challenge"
3) pipenv install -r requirements.txt
4) pipenv shell
5) python main.py

Description of workflow:    
1) Connect to the Huobi API to pull btcusdt, ethusdt & ltcusdt futures data from the exchange. Data is processed from json format into a pandas dataframe for subsequent easy manipulation.
2) Two ways of obtaining optimal portfolio weights are presented:
    a) Mean Variance Optimization, where the portfolio volatility is minimized by constraining portfoio returns to solve for the optimal weights.
    b) Constructing 10,000 random portfolios by generating random weights of the 3 assets (btcusdt, ethusdt, ltcusdt) for each random portfolio. Based on these weights and the expected return
       and volatility of the data from 2023-09-01 00:00:00 to 2023-09-01 23:00:00, a risk-return profile is obtained where the efficient frontier can be observed. 
4) A plot of the efficient frontier and a log file will be output.

Results:
Given that the average returns for all 3 assets from 2023-09-01 00:00:00 to 2023-09-10 23:00:00 are negative (also observed from plotting the prices of all 3 assets), the portfolio weights obtained is deemed as a sub-optimal solution which is under 
the belly of the efficient frontier, thus making it an "uninvestable" portfolio.
