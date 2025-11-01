import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter, DayLocator
from sklearn.tree import DecisionTreeClassifier, plot_tree

def plot_kd_chart(data):
    low_min = data['Low'].rolling(window=9).min()
    high_max = data['High'].rolling(window=9).max()
    data['K'] = (data['Close'] - low_min) / (high_max - low_min) * 100
    data['D'] = data['K'].rolling(window=3).mean()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data.index, data['K'], label='K')
    ax.plot(data.index, data['D'], label='D')
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, label='K=50')
    ax.set_title('KD指標圖')
    ax.legend()
    return fig

def plot_ma_chart(data):
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, data['MA20'], label='20-Day MA')
    ax.plot(data.index, data['MA50'], label='50-Day MA')
    ax.set_title('均價指標圖')
    ax.legend()
    return fig

def plot_normal_distribution(data):

    # returns = data['Close'].pct_change().dropna()         # 20251101 原始, NG, 
    # Error: Length of values (31) does not match length of index (1)
    data['returns'] = data['Close'].pct_change()
    returns = data['returns']
    
    mu = returns.mean()
    sigma = returns.std()
    fig, ax = plt.subplots(figsize=(8, 4))
    count, bins, ignored = ax.hist(returns, bins=30, density=True, alpha=0.6, color='g')
    ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
    ax.set_title('常態分佈圖')
    return fig

def plot_boxplot(data):
    returns = data['Close'].pct_change().dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(returns, vert=False)
    ax.set_title('盒鬚圖')
    ax.set_xlabel('日回報率')
    return fig

def plot_rsi(data):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data.index, data['RSI'], label='RSI')
    ax.axhline(70, color='r', linestyle='--')
    ax.axhline(30, color='g', linestyle='--')
    ax.set_title('相對強弱指數（RSI）')
    ax.legend()
    return fig

def plot_heatmap(data):
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    pivot_table = data.pivot_table(values='Close', index='Month', columns='Year', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title('股價熱力圖')
    return fig

def plot_scatter_chart(stock_data, tickers):
    fig, ax = plt.subplots(figsize=(8, 4))

    # 20251101 出現錯誤Error: x and y must be the same size
    # ax.scatter(stock_data[tickers[0]]['Close'], stock_data[tickers[1]]['Close'])      # 原始
    # 1. 提取兩隻股票的收盤價 Series
    close_prices_ticker0 = stock_data[tickers[0]]['Close']
    close_prices_ticker1 = stock_data[tickers[1]]['Close']
    # 2. 將它們合併到一個新的 DataFrame 中，Pandas 會自動按索引（日期）對齊
    combined_closes = pd.concat([close_prices_ticker0, close_prices_ticker1], axis=1)
    combined_closes.columns = [tickers[0], tickers[1]] # 重新命名欄位
    # 3. 刪除任何含有 NaN 值的行 (確保兩欄長度一致)
    combined_closes.dropna(inplace=True)
    # 4. 現在使用對齊後的數據進行繪圖
    ax.scatter(combined_closes[tickers[0]], combined_closes[tickers[1]])

    ax.set_xlabel(f'{tickers[0]} Close Price')
    ax.set_ylabel(f'{tickers[1]} Close Price')
    ax.set_title('散佈圖')
    return fig

def plot_regression_chart(stock_data, tickers):
    from sklearn.linear_model import LinearRegression
    
    # 20251101, 出現錯誤, Error: Found input variables with inconsistent numbers of samples: [421, 422],
    # model = LinearRegression()
    # x = stock_data[tickers[0]]['Close'].values.reshape(-1, 1)
    # y = stock_data[tickers[1]]['Close'].values
    # 1. 提取兩隻股票的收盤價 Series
    # 使用 tickers[0] 獲取列表中的第一個股票代碼字串
    close_prices_0 = stock_data[tickers[0]]['Close'] 
    # 使用 tickers[1] 獲取列表中的第二個股票代碼字串
    close_prices_1 = stock_data[tickers[1]]['Close'] 
    # 2. 將它們合併到一個新的 DataFrame 中，並刪除任何包含 NaN 值的行
    combined_data = pd.concat([close_prices_0, close_prices_1], axis=1).dropna()
    combined_data.columns = ['Stock_A_Close', 'Stock_B_Close'] # 重新命名欄位以便識別
    # 3. 從對齊後的數據中準備 x 和 y
    x = combined_data['Stock_A_Close'].values.reshape(-1, 1)
    y = combined_data['Stock_B_Close'].values
    # 4. 現在 x 和 y 具有相同的長度，可以訓練模型了
    model = LinearRegression()
    model.fit(x, y)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, color='blue')
    ax.plot(x, model.predict(x), color='red')
    ax.set_xlabel(f'{tickers[0]} Close Price')
    ax.set_ylabel(f'{tickers[1]} Close Price')
    ax.set_title('迴歸分析圖')
    return fig

def plot_price_chart(stock_data, tickers):
    fig, ax = plt.subplots(figsize=(8, 4))

    """
    for ticker in tickers:
        ax.plot(stock_data[ticker].index, stock_data[ticker]['Close'], label=ticker)
    ax.set_title('多股票價格圖')
    """
    ax2 = ax.twinx()
    # 定義顏色
    color1 = 'tab:blue'
    color2 = 'tab:red'
    # 使用 tickers 列表的第一個元素作為 ticker1
    ticker1 = tickers[0]
    # 使用 tickers 列表的第二個元素作為 ticker2
    ticker2 = tickers[1]
    # 繪製第一隻股票的數據在左側 Y 軸
    ax.plot(stock_data[ticker1].index, stock_data[ticker1]['Close'], label=ticker1, color=color1)
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{ticker1} 收盤價', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    # 繪製第二隻股票的數據在右側 Y 軸
    ax2.plot(stock_data[ticker2].index, stock_data[ticker2]['Close'], label=ticker2, color=color2)
    ax2.set_ylabel(f'{ticker2} 收盤價', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title(f'多股票價格圖 for {ticker1} and {ticker2} with Dual Y-Axis')

    ax.legend()
    return fig

def plot_decision_tree(stock_data, tickers):
    """
    # 舊的語法, 會出現index錯誤
    data = pd.DataFrame({
        tickers[0]: stock_data[tickers[0]]['Close'],
        tickers[1]: stock_data[tickers[1]]['Close']
    }).dropna()
    
    # 建立字典來存放新的DataFrame數據, 
    # 會出現Error: Data must be 1-dimensional, got ndarray of shape (437, 1) instead
    data_dict = {
        tickers[0]: stock_data[tickers[0]]['Close'],
        tickers[1]: stock_data[tickers[1]]['Close']
    }
    # 建立新的DataFrame，並使用其中一個Series的索引
    # 由於兩個Series的索引相同，所以使用哪一個都可以
    data = pd.DataFrame(data_dict, index=stock_data[tickers[0]]['Close'].index).dropna()
    """
    # 使用 pd.concat 沿著軸 1 (columns) 合併 Series
    data = pd.concat([
        stock_data[tickers[0]]['Close'],
        stock_data[tickers[1]]['Close']
    ], axis=1).dropna()
    # 重新命名欄位
    data.columns = [tickers[0], tickers[1]]

    data['Target'] = (data[tickers[1]] > data[tickers[1]].shift(1)).astype(int)
    x = data[tickers[0]].values.reshape(-1, 1)
    y = data['Target'].values

    clf = DecisionTreeClassifier(max_depth=3).fit(x, y)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(clf, ax=ax, feature_names=[tickers[0]], class_names=['下降', '上升'], filled=True, rounded=True, proportion=True)
    ax.set_title('決策樹圖')
    return fig
