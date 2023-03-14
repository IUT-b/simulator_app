import datetime as dt

from flask import Flask, redirect, render_template, request, session, url_for
from flask_bootstrap import Bootstrap

app = Flask(__name__)
app.config.from_object("config.LocalConfig")
bootstrap = Bootstrap(app)

import analysis as an


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        if "outgo" in session:
            outgo = session["outgo"]
        return render_template("index.html")

    if request.method == "POST":
        brand = request.form.getlist("brand")
        p = request.form.getlist("p")
        sim = request.form.get("sim")
        value0_b = request.form.get("value0_b")
        cash0_b = request.form.get("cash0_b")
        saving = request.form.get("saving")
        income = [100] * 30
        if "income" in session:
            income = session["income"]
        goal = request.form.get("goal")

        value0_a = request.form.get("value0_a")
        cash0_a = request.form.get("cash0_a")
        method = request.form.getlist("method")
        r = request.form.getlist("r")
        outgo = [100] * 30
        if "outgo" in session:
            outgo = session["outgo"]

        session["brand"] = brand
        session["p"] = p
        session["sim"] = sim
        session["value0_b"] = value0_b
        session["cash0_b"] = cash0_b
        session["saving"] = saving
        session["goal"] = goal

        session["value0_a"] = value0_a
        session["cash0_a"] = cash0_a
        session["method"] = method
        session["r"] = r

        start = dt.date(year=1920, month=1, day=1)
        end = dt.datetime.now().date()
        periods = [1, 5, 10, 15, 20, 25, 30]
        lst_case1 = [
            [brand[0], float(p[0])],
            [brand[2], float(p[2])],
            [brand[4], float(p[4])],
        ]
        df_case1 = an.portfolio(lst_case1, start, end)
        interest_case1 = an.interest(df_case1.copy(), periods)

        lst_case2 = [
            [brand[1], float(p[1])],
            [brand[3], float(p[3])],
            [brand[5], float(p[5])],
        ]
        df_case2 = an.portfolio(lst_case2, start, end)
        interest_case2 = an.interest(df_case2.copy(), periods)

        fig = an.fig_chart(df_case1.copy(), df_case2.copy())
        fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig1.html")
        fig = an.fig_interest(interest_case1, interest_case2)
        fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig2.html")
        fig = an.fig_interest_dispersion(periods, interest_case1, interest_case2)
        fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig3.html")
        fig = an.fig_total_interest_dispersion(periods, interest_case1, interest_case2)
        fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig4.html")
        page = "portfolio"

        # FIREまでのシミュレーションあり
        if sim == "before":

            for i in range(len(income)):
                income[i] = int(income[i])
            df_goal1, df_goal1_all = an.sim_goal(
                df_case1.copy(),
                int(saving),
                int(value0_b),
                int(cash0_b),
                income,
                int(goal),
            )
            df_goal2, df_goal2_all = an.sim_goal(
                df_case2.copy(),
                int(saving),
                int(value0_b),
                int(cash0_b),
                income,
                int(goal),
            )

            success_goal1 = an.sim_dissaving_dash(df_goal1_all.copy())
            success_goal2 = an.sim_dissaving_dash(df_goal2_all.copy())

            fig = an.fig_fire_success_dash(success_goal1.copy(), success_goal2.copy())
            fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig21.html")
            fig = an.fig_sim_dispersion_dash(df_goal1_all.copy(), df_goal2_all.copy())
            fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig22.html")
            fig = an.fig_sim_dash(*df_goal1_all.copy())
            fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig23.html")
            fig = an.fig_sim_goal(df_goal1[0].copy())
            fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig24.html")
            fig = an.fig_sim_goal(df_goal1[1].copy())
            fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig25.html")
            fig = an.fig_sim_goal(df_goal1[2].copy())
            fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig26.html")

            page = "simulation1"

        # FIRE後のシミュレーションあり
        if sim == "after":
            # sim_invest_case1=an.cal_sim_invest(df_case1.copy(),start_sim_invest,end_sim_invest)
            sim_dissaving_case1, success_dissaving_case1 = an.sim_dissaving(
                df_case1.copy(), start, float(r[0]) / 100, int(method[0])
            )
            # sim_dissaving_extract_case1=an.cal_sim_dissaving_extract(sim_dissaving_case1,start_sim_dissaving)

            # sim_invest_case2=an.cal_sim_invest(df_case2.copy(),start_sim_invest,end_sim_invest)
            sim_dissaving_case2, success_dissaving_case2 = an.sim_dissaving(
                df_case2.copy(), start, float(r[1]) / 100, int(method[1])
            )
            # sim_dissaving_extract_case2=an.cal_sim_dissaving_extract(sim_dissaving_case2,start_sim_dissaving)

            for i in range(len(outgo)):
                outgo[i] = int(outgo[i])
            df_fire1, df_fire1_all = an.sim_fire(
                df_case1.copy(),
                float(r[0]) / 100,
                int(method[0]),
                int(value0_a),
                int(cash0_a),
                outgo,
            )
            df_fire2, df_fire2_all = an.sim_fire(
                df_case2.copy(),
                float(r[1]) / 100,
                int(method[1]),
                int(value0_a),
                int(cash0_a),
                outgo,
            )

            success_fire1 = an.sim_dissaving_dash2(df_fire1_all.copy())
            success_fire2 = an.sim_dissaving_dash2(df_fire2_all.copy())

            fig = an.fig_fire_success_dash(success_fire1.copy(), success_fire2.copy())
            fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig1.html")
            fig = an.fig_sim_dispersion_dash(df_fire1_all.copy(), df_fire2_all.copy())
            fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig2.html")
            fig = an.fig_sim_dash(*df_fire1_all.copy())
            fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig3.html")
            fig = an.fig_sim_fire(df_fire1[0].copy())
            fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig4.html")
            fig = an.fig_sim_fire(df_fire1[1].copy())
            fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig5.html")
            fig = an.fig_sim_fire(df_fire1[2].copy())
            fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig6.html")

            page = "simulation2"

        return redirect(url_for(f"{page}"))


@app.route("/fig/<i>", methods=["GET", "POST"])
def fig(i):
    return render_template(f"fig{i}.html")


@app.route("/portfolio", methods=["GET"])
def portfolio():

    if request.method == "GET":
        brand = session["brand"]
        brand_jp = ["undefined"] * len(brand)
        for i in range(len(brand)):
            if brand[i] == "^DJI":
                brand_jp[i] = "ダウ平均株価"
            if brand[i] == "^SPX":
                brand_jp[i] = "S&P 500"
            if brand[i] == "^NDQ":
                brand_jp[i] = "NASDAQ"
            if brand[i] == "10USYB":
                brand_jp[i] = "米10年債"
        session["brand_jp"] = brand_jp

        return render_template("portfolio.html")


@app.route("/simulation1", methods=["GET", "POST"])
def simulation1():
    if request.method == "GET":
        brand = session["brand"]
        brand_jp = ["undefined"] * len(brand)
        for i in range(len(brand)):
            if brand[i] == "^DJI":
                brand_jp[i] = "ダウ平均株価"
            if brand[i] == "^SPX":
                brand_jp[i] = "S&P 500"
            if brand[i] == "^NDQ":
                brand_jp[i] = "NASDAQ"
            if brand[i] == "10USYB":
                brand_jp[i] = "米10年債"
        session["brand_jp"] = brand_jp

        return render_template("simulation1.html")


@app.route("/simulation2", methods=["GET", "POST"])
def simulation2():
    if request.method == "GET":
        brand = session["brand"]
        brand_jp = ["undefined"] * len(brand)
        for i in range(len(brand)):
            if brand[i] == "^DJI":
                brand_jp[i] = "ダウ平均株価"
            if brand[i] == "^SPX":
                brand_jp[i] = "S&P 500"
            if brand[i] == "^NDQ":
                brand_jp[i] = "NASDAQ"
            if brand[i] == "10USYB":
                brand_jp[i] = "米10年債"
        session["brand_jp"] = brand_jp

        method = session["method"]
        method_jp = ["undefined"] * len(method)
        for i in range(len(method)):
            if method[i] == "0":
                method_jp[i] = "定率"
            if method[i] == "1":
                method_jp[i] = "定額"
        session["method_jp"] = method_jp

        return render_template("simulation2.html")


@app.route("/income", methods=["GET", "POST"])
def income():
    if request.method == "POST":
        income = request.form.getlist("income")
        session["income"] = income
        return redirect(url_for("index"))
    else:
        return render_template("income.html")


@app.route("/outgo", methods=["GET", "POST"])
def outgo():
    if request.method == "POST":
        outgo = request.form.getlist("outgo")
        session["outgo"] = outgo
        return redirect(url_for("index"))
    else:
        return render_template("outgo.html")


@app.route("/reference", methods=["GET"])
def reference():
    import datetime as dt

    import analysis as an

    # 銘柄選択
    brand1 = "^DJI"
    brand2 = "^SPX"
    brand3 = "^NDQ"

    # 株価取得期間
    start = dt.date(year=1920, month=1, day=1)
    end = dt.datetime.now().date()

    df1 = an.stock(brand1, start, end)
    df2 = an.stock(brand2, start, end)
    df3 = an.stock(brand3, start, end)
    df_gdp = an.stock("GDP", start, end)

    fig = an.fig_gdp(df1.copy(), df_gdp.copy())
    fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig101.html")
    fig = an.fig_gdp(df2.copy(), df_gdp.copy())
    fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig102.html")
    fig = an.fig_gdp(df3.copy(), df_gdp.copy())
    fig.write_html(app.config["TEMPLATES_FOLDER"] + "fig103.html")

    return render_template("reference.html")
