import datetime as dt
import statistics

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dateutil import relativedelta
from pandas_datareader import data
from plotly.subplots import make_subplots

from app import app


# 株価データ取得処理
def stock(brand, start, end):
    # df = pd.read_csv(f"stock_data/{brand}.csv", index_col=0)
    df = pd.read_csv(app.config["DATA_FOLDER"] + f"{brand}.csv", index_col=0)
    df.index = pd.to_datetime(df.index)

    date1 = df.tail(1).index[0] + relativedelta.relativedelta(months=1)
    date2 = dt.datetime.now()
    if brand != "GDP" and date2 >= date1 + relativedelta.relativedelta(months=2):
        if brand == "10USYB":
            df2 = data.DataReader("10USY.B", "stooq", date1.date(), date2.date())
        else:
            df2 = data.DataReader(brand, "stooq", date1.date(), date2.date())
        df2 = df2[["Close"]]
        # 株価データを月単位に間引き
        df2 = df2.iloc[::-1]
        df2 = df2.reset_index()
        df2 = df2.groupby([df2["Date"].dt.year, df2["Date"].dt.month]).head(1)
        df2.set_index("Date", inplace=True)
        df = pd.concat([df, df2])
        # df.to_csv(f"stock_data/{brand}.csv")
        df.to_csv(app.config["DATA_FOLDER"] + f"{brand}.csv")

    df = df[start:end]
    return df


# ポートフォリオ作成
# lst[[銘柄,割合],・・・]:銘柄と資産割合
def portfolio(lst, start, end):
    # 各銘柄のデータを取得
    lst2 = []
    lst_start = []
    for lst in lst:
        brand = lst[0]
        p = lst[1]
        df = stock(brand, start, end)
        # インデックスの日にちを１日に統一
        df.index = df.index.strftime("%Y-%m")
        df.index = pd.to_datetime(df.index)
        lst2.append([df, p])
        # 各銘柄のデータ期間を統一
        lst_start.append(df.index.min())
        start = max(lst_start)

    lst3 = []
    P = 0
    for lst in lst2:
        df = lst[0]
        df = df[start:end]
        p = lst[1]
        P = P + p
        lst3.append([df, p])

    # 毎月資産割合をリバランス
    df = pd.DataFrame()
    # freqを'M'とすると月末をとるためendの月が間引かれる
    df = pd.DataFrame(index=pd.date_range(start=start, end=end, freq="D"))
    df.index.name = "Date"
    # indexを月単位に間引き
    df = df.reset_index()
    df = df.groupby([df["Date"].dt.year, df["Date"].dt.month]).head(1)
    df.set_index("Date", inplace=True)
    df["Close"] = 1

    # インデックスの日にちを１日に統一
    df.index = df.index.strftime("%Y-%m")
    df.index = pd.to_datetime(df.index)
    for lst in lst3:
        df = df + lst[0].pct_change() * lst[1] / P
    df.iloc[0, 0] = 1
    df = df.cumprod()

    return df


# 利率計算
def interest(df, periods):
    # 一括投資の利率
    for period in periods:
        # トータル
        df[f"{period} year(s) total return"] = (
            df["Close"].shift(-period * 12) / df["Close"]
        )
        # 年率
        df[f"{period} year(s) return"] = (
            df[f"{period} year(s) total return"] ** (1 / period) - 1
        )
    # 積立投資の利率
    for period in periods:
        y = 0
        for i in range(0, period * 12):
            y = y + 1 / df["Close"].shift(-i)
        df[f"{period} year(s) total return2"] = (
            y * df["Close"].shift(-period * 12) / 12 / period
        )
        df[f"{period} year(s) return2"] = (df[f"{period} year(s) total return2"]) ** (
            1 / period
        ) - 1
    return df


# FIRE達成までのシミュレーション
# df:ポートフォリオ、saving:積立額、value0:初期証券額、cash0:初期現金、income:年間収支、goal:目標金額
def sim_goal(df, saving, value0, cash0, income, goal):
    dfs = []
    INDEX = []
    N = int(len(df) / 12)
    n = len(income)
    for i in range(0, N - n):
        # シミュレーション期間のデータ抽出
        a = df[12 * i : 12 * (i + n) :] / df.iloc[12 * i]

        # 抽出データで計算
        b = sim_goal_sub(a, saving, value0, cash0, income, goal)
        dfs.append(b)

        INDEX.append([i, len(b), float(b["value"].tail(1) + b["cash"].tail(1))])

    # 最高・最低ケースの抽出
    lengths = []
    for i in range(0, N - n):
        lengths.append(INDEX[i][1])

    INDEX_MAXs = []
    INDEX_MINs = []
    for i in range(0, N - n):
        if INDEX[i][1] == min(lengths):
            INDEX_MAXs.append(i)
        elif INDEX[i][1] == max(lengths):
            INDEX_MINs.append(i)

    value_max = float(
        dfs[INDEX_MAXs[0]]["value"].tail(1) + dfs[INDEX_MAXs[0]]["cash"].tail(1)
    )
    for i in range(len(INDEX_MAXs)):
        i = INDEX_MAXs[i]
        if float(dfs[i]["value"].tail(1) + dfs[i]["cash"].tail(1)) >= value_max:
            value_max = float(dfs[i]["value"].tail(1) + dfs[i]["cash"].tail(1))
            INDEX_MAX = i

    value_min = float(
        dfs[INDEX_MINs[0]]["value"].tail(1) + dfs[INDEX_MINs[0]]["cash"].tail(1)
    )
    for i in range(len(INDEX_MINs)):
        i = INDEX_MINs[i]
        if float(dfs[i]["value"].tail(1) + dfs[i]["cash"].tail(1)) <= value_min:
            value_min = float(dfs[i]["value"].tail(1) + dfs[i]["cash"].tail(1))
            INDEX_MIN = i

    # 中央値ケースの抽出
    lst = []
    for i in range(0, N - n):
        lst.append(abs(lengths[i] - statistics.median(lengths)))
    INDEX_MED = lst.index(min(lst))
    # 資産がgoalに達しない場合
    if float(dfs[INDEX_MED]["value"].tail(1) + dfs[INDEX_MED]["cash"].tail(1)) < goal:
        for i in range(0, N - n):
            count = 1
            for j in range(0, N - n):
                if INDEX[i][1] <= INDEX[j][1]:
                    count = count + 1
            if count == int((N - n) / 2):
                INDEX_MED = i
                break

    return [dfs[INDEX_MAX], dfs[INDEX_MED], dfs[INDEX_MIN]], dfs


def sim_goal_sub(df, saving, value0, cash0, income, goal):
    df["value"] = value0
    df["saving"] = saving / 12
    income_monthly = []
    for i in range(len(income)):
        income_monthly.extend([income[i] / 12] * 12)
    df["income"] = income_monthly
    df["cash"] = cash0

    # 処理速度アップのためnumpy arrayに変換
    dfa = np.asarray(df)

    for i in range(0, len(df.index) - 1):
        # 証券価格(1)の計算
        if dfa[i, 3] >= dfa[i, 2]:
            pass
        else:
            if dfa[i, 3] >= 0:
                dfa[i, 2] = dfa[i, 3]
            elif dfa[i, 3] < 0:
                dfa[i, 2] = 0
        dfa[i + 1, 1] = (dfa[i, 1] + dfa[i, 2]) * dfa[i + 1, 0] / dfa[i, 0]

        # 現金(4)の計算
        # dfa[i+1,4]=dfa[i,4]+dfa[i,2]+dfa[i,3]
        dfa[i + 1, 4] = dfa[i, 4] + dfa[i, 3] - dfa[i, 2]

        # 資産(1)+(4)がgoalになった場合
        if dfa[i + 1, 1] + dfa[i + 1, 4] >= goal:
            dfa = dfa[: i + 2]
            break

    # インデックスを経過年数に変換
    l = len(dfa)
    index = list(range(l))
    index2 = np.arange(1 / 12, (l + 1) / 12, 1 / 12)
    for i in range(0, l):
        index[i] = index2[i]

    df = pd.DataFrame(dfa, index=index, columns=df.columns)
    return df


# FIREシミュレーション
# df:ポートフォリオ、r:取り崩し率、m:取り崩し方法、stock0:初期証券額、cash0:初期現金、outgo:年間収支
def sim_fire(df, r, m, value0, cash0, outgo):
    dfs = []
    INDEX = []
    N = int(len(df) / 12)
    n = len(outgo)
    for i in range(0, N - n):
        # シミュレーション期間のデータ抽出
        a = df[12 * i : 12 * (i + n) :] / df.iloc[12 * i]

        # 抽出データで計算
        b = sim_fire_sub(a, r, m, value0, cash0, outgo)
        dfs.append(b)

        INDEX.append([i, len(b), float(b["value"].tail(1) + b["cash"].tail(1))])

    # 最高・最低ケースの抽出
    values = []
    for i in range(0, N - n):
        values.append(INDEX[i][2])

    INDEX_MAXs = []
    INDEX_MINs = []
    for i in range(0, N - n):
        if INDEX[i][2] == max(values):
            INDEX_MAXs.append(i)
        elif INDEX[i][2] == min(values):
            INDEX_MINs.append(i)

    length_max = len(dfs[INDEX_MAXs[0]])
    for i in range(len(INDEX_MAXs)):
        i = INDEX_MAXs[i]
        if len(dfs[i]) >= length_max:
            length_max = len(dfs[i])
            INDEX_MAX = i

    length_min = len(dfs[INDEX_MINs[0]])
    for i in range(len(INDEX_MINs)):
        i = INDEX_MINs[i]
        if len(dfs[i]) <= length_min:
            length_min = len(dfs[i])
            INDEX_MIN = i

    # 中央値ケースの抽出
    lst = []
    for i in range(0, N - n):
        lst.append(abs(values[i] - statistics.median(values)))
    INDEX_MED = lst.index(min(lst))
    # 資産が０の場合
    if float(dfs[INDEX_MED]["value"].tail(1) + dfs[INDEX_MED]["cash"].tail(1)) == 0:
        for i in range(0, N - n):
            count = 1
            for j in range(0, N - n):
                if INDEX[i][1] >= INDEX[j][1]:
                    count = count + 1
            if count == int((N - n) / 2):
                INDEX_MED = i
                break

    return [dfs[INDEX_MAX], dfs[INDEX_MED], dfs[INDEX_MIN]], dfs


def sim_fire_sub(df, r, m, value0, cash0, outgo):
    r = r / 12
    df["value"] = value0
    df["dissaving"] = r * value0
    outgo_monthly = []
    for i in range(len(outgo)):
        outgo_monthly.extend([outgo[i] / 12] * 12)
    df["outgo"] = outgo_monthly
    df["cash"] = cash0

    # 処理速度アップのためnumpy arrayに変換
    dfa = np.asarray(df)

    for i in range(0, len(df.index) - 1):

        # 定率の場合の取り崩し額
        if m == 0:
            dfa[i, 2] = dfa[i, 1] * r

        # 証券価格(1)の計算
        # 証券価格(1)が取り崩し額(2)以上の場合
        if dfa[i, 1] >= dfa[i, 2]:
            # 取り崩し額(2)が支出(3)以上の場合
            if dfa[i, 2] >= -dfa[i, 3]:
                dfa[i, 2] = -dfa[i, 3]
            dfa[i + 1, 1] = (dfa[i, 1] - dfa[i, 2]) * dfa[i + 1, 0] / dfa[i, 0]
        # 証券価格(1)が取り崩し額(2)より小さい場合
        else:
            dfa[i + 1, 1] = 0
            dfa[i, 2] = dfa[i, 1]

        # 現金(4)の計算
        dfa[i + 1, 4] = dfa[i, 4] + dfa[i, 2] + dfa[i, 3]
        # 現金(4)が０になった場合
        if dfa[i + 1, 4] <= 0:
            dfa[i + 1, 1] = dfa[i + 1, 1] + dfa[i + 1, 4]
            dfa[i + 1, 4] = 0
            # 証券価格(1)も０になった場合
            if dfa[i + 1, 1] <= 0:
                dfa[i + 1, 1] = 0
                dfa = dfa[: i + 2]
                break

    # インデックスを経過年数に変換
    l = len(dfa)
    index = list(range(l))
    index2 = np.arange(1 / 12, (l + 1) / 12, 1 / 12)
    for i in range(0, l):
        index[i] = index2[i]

    df = pd.DataFrame(dfa, index=index, columns=df.columns)
    return df


# 投資シミュレーション
def sim_invest(df, start, end):
    # 積立投資
    df["purchases"] = 0
    df.loc[start:end, "purchases"] = 1
    df["total_purchases"] = df["purchases"].cumsum()
    df.loc[start:end, "purchases"] = df.loc[start:end, "purchases"] / max(
        df["total_purchases"]
    )
    df["total_purchases"] = df["purchases"].cumsum()
    df["shares"] = df["purchases"] / df["Close"]
    df["total_shares"] = df["shares"].cumsum()
    df["value"] = df["Close"] * df["total_shares"]
    return df.loc[start:end]


# 取り崩しシミュレーション
def sim_dissaving(df, start, r, n):
    sim = []
    r = r / 12
    # 定率取り崩し
    if n == 0:
        for i in range(0, int(len(df) / 12) - 1):
            sim.append(
                constant_rate(
                    df.copy(), start + relativedelta.relativedelta(years=i), r
                )
            )
    # 定額取り崩し
    if n == 1:
        for i in range(0, int(len(df) / 12) - 1):
            sim.append(
                constant_amount(
                    df.copy(), start + relativedelta.relativedelta(years=i), r
                )
            )
    # 成功率
    sim_sort = pd.DataFrame()
    for df in sim:
        # 年単位に間引き
        df = df.reset_index()
        df = df.groupby([df["Date"].dt.year]).head(1)
        df.set_index("Date", inplace=True)
        sim_sort = pd.concat([sim_sort, df.reset_index()["value"]], axis=1)
    sim_sort_bool0 = sim_sort == 0
    sim_sort_bool1 = sim_sort > 0
    success = pd.DataFrame(
        sim_sort_bool1.sum(axis=1)
        / (sim_sort_bool0.sum(axis=1) + sim_sort_bool1.sum(axis=1))
    )
    return sim, success


# 取り崩しシミュレーション（シミュレーション用）
def sim_dissaving_dash(df):
    # 成功率
    df_sort = pd.DataFrame()
    for df in df:
        df_sort = pd.concat([df_sort, df["value"] + df["cash"]], axis=1)
    df_sort_bool0 = df_sort == 0
    df_sort_bool1 = df_sort > 0
    success = pd.DataFrame(
        (len(df_sort.columns) - df_sort_bool1.sum(axis=1)) / len(df_sort.columns)
    )
    return success


# 取り崩しシミュレーション（シミュレーション用）
def sim_dissaving_dash2(df):
    # 成功率
    df_sort = pd.DataFrame()
    for df in df:
        df_sort = pd.concat([df_sort, df["value"] + df["cash"]], axis=1)
    df_sort_bool0 = df_sort == 0
    df_sort_bool1 = df_sort > 0
    success = pd.DataFrame(df_sort_bool1.sum(axis=1) / len(df_sort.columns))
    return success


# 定率取り崩し
def constant_rate(df, start, r):
    start = min(df[start:].index)
    n = df.index.get_loc(start)
    df["ratio"] = (1 + df["Close"].pct_change(periods=1)) * (1 - r)
    df["ratio"].iloc[: n + 1] = 1
    # 資産額
    df["value"] = df["ratio"].cumprod()
    # 取り崩し額
    df["dissaving"] = df["value"] * r

    # [n:]を途中に入れると警告が出る
    return df[n:]


# 定額取り崩し
def constant_amount(df, start, r):
    start = min(df[start:].index)
    n = df.index.get_loc(start)

    df["ratio"] = 1 + df["Close"].pct_change(periods=1)
    df["ratio"].iloc[: n + 1] = 1
    df["value"] = np.nan
    df["value"].iloc[n] = 1

    # 処理速度アップのためnumpy arrayに変換
    nr = df.columns.get_loc("ratio")
    nv = df.columns.get_loc("value")
    dfa = np.asarray(df)

    for i in range(n + 1, len(df)):
        dfa[i, nv] = dfa[i, nr] * (dfa[i - 1, nv] - r)
        if dfa[i, nv] <= 0:
            dfa[i, nv] = 0
    df = pd.DataFrame(dfa, index=df.index, columns=df.columns)
    df["dissaving"] = r

    # [n:]を途中に入れると警告が出る
    return df.iloc[n:]


# シミュレーション取り出し
def sim_dissaving_extract(df, start):
    for df in df:
        if df.index[0].timestamp() >= pd.to_datetime(start).timestamp():
            df_extract = df
            break
    return df_extract


# 株価チャート
def fig_chart(df1, *df2):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="日付"),
        yaxis=dict(title="増減", type="log", tickformat="0%"),
    )
    fig.add_trace(
        go.Scatter(x=df1.index, y=df1["Close"], mode="lines", name="Case1"),
    )
    for df2 in df2:
        fig.add_trace(
            go.Scatter(x=df2.index, y=df2["Close"], mode="lines", name="Case2"),
        )
    return fig


# 年利グラフ
def fig_interest(df1, *df2):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="日付"),
        yaxis=dict(title="年利", tickformat="0%"),
    )

    fig.add_trace(
        go.Scatter(x=df1.index, y=df1["1 year(s) return"], mode="lines", name="Case1"),
    )
    for df2 in df2:
        fig.add_trace(
            go.Scatter(
                x=df2.index, y=df2["1 year(s) return"], mode="lines", name="Case2"
            ),
        )
    return fig


# 年利分散グラフ
def fig_interest_dispersion(periods, df1, *df2):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="投資年数"),
        yaxis=dict(title="年利", tickformat="0%", range=[-0.3, 0.3]),
        boxmode="group",
    )
    fig.add_trace(
        go.Box(
            x=tr(df1, periods, "return")[:, 1],
            y=tr(df1, periods, "return")[:, 0],
            name="Case1",
        )
    )
    for df2 in df2:
        fig.add_trace(
            go.Box(
                x=tr(df2, periods, "return")[:, 1],
                y=tr(df2, periods, "return")[:, 0],
                name="Case2",
            )
        )
    return fig


# グラフ用データ変換
def tr(df, periods, text):
    r = np.empty((0, 2), int)
    for period in periods:
        d = pd.DataFrame()
        d = df[[f"{period} year(s) " + text]]
        d = d.assign(periods=period)
        r = np.append(r, d.values, axis=0)
    return r


# トータルの利率分散グラフ
def fig_total_interest_dispersion(periods, df1, *df2):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="投資年数"),
        yaxis=dict(title="増減", tickformat="0%"),
        boxmode="group",
    )
    fig.add_trace(
        go.Box(
            x=tr(df1, periods, "total return")[:, 1],
            y=tr(df1, periods, "total return")[:, 0],
            name="Case1",
        )
    )
    for df2 in df2:
        fig.add_trace(
            go.Box(
                x=tr(df2, periods, "total return")[:, 1],
                y=tr(df2, periods, "total return")[:, 0],
                name="Case2",
            )
        )
    return fig


# FIRE達成までのシミュレーションのグラフ
def fig_sim_goal(df):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="経過年"),
        yaxis=dict(title="資産"),
        barmode="stack",
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["cash"],
            # mode='relative',
            name="現金",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["value"],
            # mode='relative',
            name="有価証券",
        ),
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=12 * df["income"], mode="lines", name="収支"),
    )
    return fig


# FIREシミュレーションのグラフ
def fig_sim_fire(df):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="経過年"),
        yaxis=dict(title="資産"),
        barmode="stack",
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["cash"],
            # mode='relative',
            name="現金",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["value"],
            # mode='relative',
            name="有価証券",
        ),
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=12 * df["outgo"], mode="lines", name="収支"),
    )
    return fig


# 資産推移の全パターン
def fig_sim(*df):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="経過年"),
        yaxis1=dict(title="増減", type="log", tickformat="0%"),
        showlegend=False,
    )
    for df in df:
        df = df[::12].reset_index()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["value"],
                mode="lines",
            ),
        )
    return fig


# 資産推移の全パターン（シミュレーション用）
def fig_sim_dash(*df):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="経過年"),
        yaxis1=dict(title="資産"),
        showlegend=False,
    )
    for df in df:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["value"] + df["cash"],
                mode="lines",
            ),
        )
    return fig


# 資産推移の分散のグラフ
def fig_sim_dispersion(df1, *df2):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="経過年"),
        yaxis=dict(title="増減", type="log", tickformat="0%"),
    )

    # 並び替え
    df_sort1 = pd.DataFrame()
    for df1 in df1:
        # 年単位に間引き
        df1 = df1.reset_index()
        df1 = df1.groupby([df1["Date"].dt.year]).head(1)
        df1.set_index("Date", inplace=True)
        df_sort1 = pd.concat([df_sort1, df1.reset_index()["value"]], axis=0)
    fig.add_trace(
        go.Box(x=df_sort1.index, y=df_sort1.iloc[:, 0], name="portfolio #1"),
    )

    for df2 in df2:
        # 並び替え
        df_sort2 = pd.DataFrame()
        for df2 in df2:
            # 年単位に間引き
            df2 = df2.reset_index()
            df2 = df2.groupby([df2["Date"].dt.year]).head(1)
            df2.set_index("Date", inplace=True)
            df_sort2 = pd.concat([df_sort2, df2.reset_index()["value"]], axis=0)
        fig.add_trace(
            go.Box(x=df_sort2.index, y=df_sort2.iloc[:, 0], name="portfolio #2"),
        )

    return fig


# 資産推移の分散のグラフ（シミュレーション用）
def fig_sim_dispersion_dash(df1, *df2):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="経過年"),
        yaxis=dict(title="資産"),
        boxmode="group",
    )

    # 並び替え
    df_sort1 = pd.DataFrame()
    for df1 in df1:
        # 年単位に間引き
        df1 = df1[::12]
        df_sort1 = pd.concat([df_sort1, df1["value"] + df1["cash"]], axis=0)
    fig.add_trace(
        go.Box(x=df_sort1.index, y=df_sort1.iloc[:, 0], name="portfolio #1"),
    )

    for df2 in df2:
        # 並び替え
        df_sort2 = pd.DataFrame()
        for df2 in df2:
            # 年単位に間引き
            df2 = df2[::12]
            df_sort2 = pd.concat([df_sort2, df2["value"] + df2["cash"]], axis=0)
        fig.add_trace(
            go.Box(x=df_sort2.index, y=df_sort2.iloc[:, 0], name="portfolio #2"),
        )

    return fig


# FIRE成功率のグラフ
def fig_fire_success(df1, *df2):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="経過年"),
        yaxis=dict(title="成功率", tickformat="0%"),
    )
    fig.add_trace(
        go.Scatter(x=df1.index, y=df1.iloc[:, 0], mode="lines", name="portfolio #1"),
    )
    for df2 in df2:
        fig.add_trace(
            go.Scatter(
                x=df2.index, y=df2.iloc[:, 0], mode="lines", name="portfolio #2"
            ),
        )
    return fig


# FIRE成功率のグラフ（シミュレーション用）
def fig_fire_success_dash(df1, *df2):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(title="経過年"),
        yaxis=dict(title="成功率", tickformat="0%", range=[0, 1]),
    )
    fig.add_trace(
        go.Scatter(x=df1.index, y=df1.iloc[:, 0], mode="lines", name="portfolio #1"),
    )
    for df2 in df2:
        fig.add_trace(
            go.Scatter(
                x=df2.index, y=df2.iloc[:, 0], mode="lines", name="portfolio #2"
            ),
        )
    return fig


# FIRE失敗事例
def fig_fire_failure(df, sim_dissaving):
    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        xaxis=dict(title="日付"),
        yaxis1=dict(title="価額", type="log"),
        yaxis2=dict(title="増減", tickformat="0%"),
        showlegend=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
        ),
    )
    for df in sim_dissaving:
        if df["value"][len(df) - 1] == 0:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["value"],
                    mode="lines",
                ),
                secondary_y=True,
            )
    return fig


# 投資シミュレーションのグラフ
def fig_sim_invest(df, sim_invest):
    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        xaxis=dict(title="経過年"),
        yaxis1=dict(title="増減", type="log"),
        yaxis2=dict(title="成功率", tickformat="0%"),
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="株価",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=sim_invest.index,
            y=sim_invest["total_purchases"],
            mode="lines",
            name="元金（ドル・コスト平均法）",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=sim_invest.index, y=sim_invest["value"], mode="lines", name="ドル・コスト平均法"
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=sim_invest.index,
            y=sim_invest["Close"] / sim_invest["Close"],
            mode="lines",
            name="元金（一括投資）",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=sim_invest.index,
            y=sim_invest["Close"] / sim_invest["Close"][0],
            mode="lines",
            name="一括投資",
        ),
        secondary_y=True,
    )
    return fig


# 資産取り崩しシミュレーションのグラフ
def fig_sim_dissaving(df, *df2):
    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        xaxis=dict(title="日付"),
        yaxis1=dict(title="価額", type="log"),
        yaxis2=dict(tickformat="0%"),
        showlegend=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
        ),
    )
    for df2 in df2:
        fig.add_trace(
            go.Scatter(
                x=df2.index,
                y=df2["value"],
                mode="lines",
            ),
            secondary_y=True,
        )
    return fig


# GDPとの比較のグラフ
def fig_gdp(df, df_gdp):

    df = dec(df.copy())
    df_gdp = dec(df_gdp.copy())

    a = max(min(df.index), min(df_gdp.index))
    b = min(max(df.index), max(df_gdp.index))
    df = df.loc[a:b] / df.loc[a:b].iloc[0]
    df_gdp = df_gdp.loc[a:b] / df_gdp.loc[a:b].iloc[0]

    df_ratio = df / df_gdp

    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        xaxis=dict(title="年"),
        yaxis1=dict(title="増減", type="log", tickformat="0%"),
        yaxis2=dict(title="株価/GDP"),
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], mode="lines", name="株価"),
    )

    fig.add_trace(
        go.Scatter(x=df_gdp.index, y=df_gdp["Close"], mode="lines", name="GDP"),
    )

    fig.add_trace(
        go.Bar(
            x=df_ratio.index,
            y=df_ratio["Close"],
            name="株価/GDP",
            width=0.5,
        ),
        secondary_y=True,
    )

    return fig


# 年単位に間引き
def dec(df):
    df = df.reset_index()
    df = df.groupby([df["Date"].dt.year]).tail(1)
    df.set_index("Date", inplace=True)
    df.index = df.index.year
    return df
