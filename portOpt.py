import gzip
import json
import time
import pprint
from datetime import datetime
from pandas import json_normalize
import pandas as pd
import websocket
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import sys

def on_open(ws):
    """
    Function for WebSocketApp
    """
    data = {
        "req": "market." + ticker + ".kline.60min",
        "id": "id1",
        "from": int(time.mktime(datetime.strptime("2023-09-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple())),
        "to": int(time.mktime(datetime.strptime("2023-09-01 23:00:00", "%Y-%m-%d %H:%M:%S").timetuple()))}
    send_message(ws, data)

def send_message(ws, message_dict):
    """
    Function for WebSocketApp 
    """
    data = json.dumps(message_dict).encode()
    print("Sending Message:")
    pprint.pprint(message_dict)
    ws.send(data)

def on_message(ws, message):
    """
    Function for WebSocketApp
    """
    unzipped_data = gzip.decompress(message).decode()
    msg_dict = json.loads(unzipped_data)
    print("Recieved Message: ")
    pprint.pprint(msg_dict)
    data_output.append(msg_dict)
    if 'ping' in msg_dict:
        data = {
            "pong": msg_dict['ping']
        }
        send_message(ws, data)
        on_close(ws)
        print("Closing Connection")

def on_error(ws, error):
    """
    Function for WebSocketApp
    """
    print("Error: " + str(error))
    error = gzip.decompress(error).decode()
    print(error)
    
def on_close(ws):
    """
    Function for WebSocketApp
    """
    ws.close()
    print("### Connection closed ###")

def createDf():
    """
    Processes and cleans output extracted from the Huobi Exchange API into pandas dataframes.

    Parameters:
    -----------
    None

    Returns:
    --------
    portfolio1 : pd.DataFrame, prices of the 3 assets (btcusdt, ethusdt, ltcusdt)
    portfolio1_ret : pd.DataFrame, returns (fractional change) of the 3 assets
    """
    cleaned_df = []
    for j in data_output:
        if 'data' in j.keys():
            df = json_normalize(j['data'])
            df['datetime'] = pd.to_datetime(df.id, unit = 's') + pd.Timedelta(hours = 8)
            df['asset'] = list(set(j['rep'].split(".")).intersection(["btcusdt", "ethusdt", "ltcusdt"]))[0]
            cleaned_df.append(df)

    df = pd.concat(cleaned_df)
    portfolio1 = pd.pivot(df[["close", "datetime", "asset"]], index='datetime', columns = 'asset', values='close')
    portfolio1_ret = portfolio1.pct_change().dropna()
    
    return portfolio1, portfolio1_ret

def efficientFrontier(df, ret, cov):
    """
    To compute the efficient frontier, fix a target return level and minimize volatility for each target return.

    Parameters
    ----------
    df : pd.DataFrame, prices of the three assets (btcusdt, ethusdt, ltcusdt)
    ret : pd.Series, average return of each asset over the predefined period
    cov : pd.DataFrame, covariance matrix of the three assets

    Returns
    -------
    output_arr_min_var : list, portfolio volatilty
    ret_out : list, portfolio returns
    weightsAndSharpe : pd.DataFrame, pandas dataframe containing weights of assets and the corresponding risk-adjusted returns and volatility
    """
    output_arr_min_var = []
    ret_out = []
    targetRangeRet = np.linspace(-0.01, 0.01, 500)
    portWeights = []
    
    def portfolio_returns(weights):
        return (np.sum(ret * weights))

    def portfolio_sd(weights):
        return np.sqrt(np.transpose(weights) @ (cov) @ weights)

    def sharpe(weights):
        return (portfolio_returns(weights) / portfolio_sd(weights))

    def minimumVarOpt(constraints = None):
        if constraints is not None:
            constraints = constraints
        else:
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        nAssets = df.shape[1]
        bounds = tuple(
            (0, 1) for j in range(nAssets)
        )
        eq_wts = np.array([1 / nAssets] * nAssets)

        min_var = minimize(
            fun = portfolio_sd, 
            x0 = eq_wts,
            method = 'SLSQP',
            bounds = bounds,
            constraints = constraints,
            options={'maxiter':300}
        )
        return min_var
    
    for target_return in targetRangeRet:
        constraints_min_var = (
            {'type': 'eq', 'fun': lambda x: portfolio_returns(x) - target_return}, 
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        res_out = minimumVarOpt(constraints=constraints_min_var)
        if res_out['success']:
            output_arr_min_var.append(res_out['fun'])
            ret_out.append(target_return)
            weights = res_out['x'].tolist()
            weights.append(sharpe(res_out['x']))
            weights.append(portfolio_sd(res_out['x']))
            portWeights.append(weights)
        else:
            break
    
    wsCol = df.columns.to_list()
    wsCol.append('SharpeRatio')
    wsCol.append('SD')
    weightsAndSharpe = pd.DataFrame(portWeights, columns = wsCol)

    return output_arr_min_var, ret_out, weightsAndSharpe

def generateRandPorts(df_returns):
    """
    Generates 10,000 random portfolios with the expected return and volatility from a sample of btc, eth and ltc 
    during the interval 2023-09-01T00:00:00 to 2023-09-01T23:00:00. The purpose is to obtain a risk-return profile 
    and the efficient frontier.

    Parameters:
    -----------
    df_returns : pd.DataFrame, pandas dataframe of returns on all 3 assets.

    Returns
    -------
    max_sharpe_port_wts : dict, dictionary of asset names and their corresponding weights.
    """

    returns_list = []
    vol_list = []
    wts_list = []
    expRet = df_returns.mean().values * 100
    data_cov = (df_returns * 100).cov()

    fnt_size = 10

    for i in range(10000):
        # Generate 3 random numbers
        wts = np.random.random(3)
        # Normalize the 3 random numbers so they add up to 1.
        norm_wts = wts / np.sum(wts)
        wts_list.append(norm_wts)

        # Compute the return on the random portfolio
        randPortReturn = expRet.dot(norm_wts)
        returns_list.append(randPortReturn)

        # Compute the volatility of the random portfolio
        randPortVar = np.dot(np.dot(norm_wts.T, data_cov), norm_wts)
        randPortStd = np.sqrt(randPortVar)
        vol_list.append(randPortStd)

    ports = pd.DataFrame({'Return':returns_list, 'Vol':vol_list})
    wts_df = pd.DataFrame(np.vstack(tuple(wts_list)), columns=['btcusdt', 'ethusdt', 'ltcusdt'])
    rand_ports = pd.concat([ports, wts_df], axis=1)
    rand_ports['sharpe'] = rand_ports['Return'] / rand_ports['Vol']

    frontier1 = rand_ports.sort_values(by='Vol')[["Return", "Vol"]].rolling(250).max().dropna().rolling(100).mean()
    max_sharpe_port_wts = rand_ports[rand_ports.sharpe == rand_ports.sharpe.max()][["btcusdt", "ethusdt", "ltcusdt"]].to_dict('records')[0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)

    ax.scatter(ports.Vol, ports.Return)
    ax.plot(frontier1.Vol, frontier1.Return, color = 'red', label="Efficient Frontier")
    ax.set_xlabel("Expected Vol (%)", fontsize=fnt_size)
    ax.set_ylabel("Expected Return (%)", fontsize=fnt_size)
    ax.xaxis.set_tick_params(which='both', labelbottom=True, labelsize=fnt_size)
    ax.yaxis.set_tick_params(which='both', labelbottom=True, labelsize=fnt_size)

    fig.suptitle("Random Portfolios")
    fig.savefig("efficient_frontier.png")

    print("Portfolio Weights: ")
    print(max_sharpe_port_wts)
    
    return max_sharpe_port_wts

if __name__ == '__main__':

    data_output = []
    ticker_list = ["btcusdt", "ethusdt", "ltcusdt"]
    for ticker in ticker_list:
        try:
            ws = websocket.WebSocketApp(
                "wss://api.huobi.pro/ws",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever()
        except Exception:
            print("error")
    
    df, df_returns = createDf()
    print(df)
    # Check if this returns are computed correctly throughout.
    returns = (1 + df_returns).prod() - 1
    cov_mat = (df_returns).cov()
    
    vol, ret, w_s = efficientFrontier(df=df, ret=returns, cov=cov_mat)
    generateRandPorts(df_returns)

