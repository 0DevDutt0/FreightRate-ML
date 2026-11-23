"""Exploration outputs used in the report: data-quality counts and the figures behind the key findings."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, OUTLIER_LOG_THRESHOLD, RESULTS_DIR
from .features import LANE_FEATURES, LOAD_FEATURES
from .model import FreightRateModel

log = logging.getLogger(__name__)

INK, MUTED, GRID = "#1F2A30", "#5B6B71", "#D9E2E4"
TEAL, ORANGE, GREY, RED = "#0B6E7F", "#D9822B", "#8A9BA1", "#B23A48"
SLATE, LIGHT_GREY, PALE_TEAL = "#3D5A80", "#C5CED1", "#9CCBD2"
VALIDATION_FILL, VALIDATION_TEXT = "#F6E7D7", "#8A4B12"  # orange is reserved for the validation period

# Names used for the designs in every figure and in the report.
DESIGN_LABELS = {
    "lane_median_baseline": "Lane-median baseline",
    "static_dow": "Member A: lane, load, day of week",
    "market_time": "Member B: A + market index + elapsed time",
    "ensemble (final)": "Final: average of A and B",
}
DESIGN_COLORS = {"lane_median_baseline": LIGHT_GREY, "static_dow": GREY, "market_time": SLATE, "ensemble (final)": TEAL}


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9DAFB3")
    ax.tick_params(colors=MUTED, labelsize=9)


def data_quality_table(raw_dev: pd.DataFrame, raw_val: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    dev_cities = set(raw_dev["pickup"]) | set(raw_dev["delivery"])
    val_new_city = ~raw_val["pickup"].isin(dev_cities) | ~raw_val["delivery"].isin(dev_cities)
    dev_lanes = set(zip(raw_dev["pickup"], raw_dev["delivery"]))
    val_new_lane = [lane not in dev_lanes for lane in zip(raw_val["pickup"], raw_val["delivery"])]
    rows = [
        ("Negative weight (sign error)", (raw_dev["weight"] < 0).sum(), (raw_val["weight"] < 0).sum(),
         "Absolute value: negatives mirror the positive distribution and their rates match normal loads"),
        ("Missing weight", raw_dev["weight"].isna().sum(), raw_val["weight"].isna().sum(),
         "Left missing; LightGBM learns which branch suits a missing value"),
        ("Weight at the 47,500 lb cap", (raw_dev["weight"].abs() == 47_500).sum(), (raw_val["weight"].abs() == 47_500).sum(),
         "Kept: a ceiling in the source data (values clipped to 5,000-47,500), not an error"),
        ("Missing market_index", raw_dev["market_index"].isna().sum(), raw_val["market_index"].isna().sum(),
         "Same-day mean of the other loads (the date explains ~98% of its variance)"),
        ("Corrupted posted_rate (2.2-5.5x or 0.18-0.45x expected)", int(outliers["is_outlier"].sum()), "n/a",
         "Removed from training (screened with an out-of-fold L1 model)"),
        ("quote_signal unreliable (flips sign by month; noise in Nov-Dec)", "all rows", "all rows",
         "Feature excluded"),
        ("Cities never seen in development data", 0, int(val_new_city.sum()),
         "Location enters the model as coordinates, not city IDs"),
        ("Lanes never seen in development data", 0, int(np.sum(val_new_lane)),
         "Same as above; no lane-ID features"),
        ("Duplicate load_id / duplicate rows", int(raw_dev["load_id"].duplicated().sum()),
         int(raw_val["load_id"].duplicated().sum()), "None found; checked on load"),
    ]
    return pd.DataFrame(rows, columns=["issue", "development_rows", "validation_rows", "handling"])


def quote_regimes(dev: pd.DataFrame, X_val: pd.DataFrame, val: pd.DataFrame, model: FreightRateModel) -> pd.DataFrame:
    """Correlation of log(quote_signal) with the actual log rate per mile in each development month,
    and with the model's expected log rate per mile in each validation month (labels are hidden)."""
    rows = []
    rpm = np.log(dev["posted_rate"] / dev["distance"])
    for m, part in dev.groupby(dev["date"].dt.month):
        rows.append({"month": m, "set": "development (vs actual)",
                     "corr": np.corrcoef(np.log(part["quote_signal"]), rpm[part.index])[0, 1]})
    expected = pd.Series(model.predict_log_rpm(X_val), index=val.index)
    for m, part in val.groupby(val["date"].dt.month):
        rows.append({"month": m, "set": "validation (vs model expectation)",
                     "corr": np.corrcoef(np.log(part["quote_signal"]), expected[part.index])[0, 1]})
    return pd.DataFrame(rows)


def plot_quote_regimes(regimes: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=160)
    labels = [pd.Timestamp(2025, int(m), 1).strftime("%b") for m in regimes["month"]]
    corr = regimes["corr"].where(regimes["corr"].abs() >= 0.005, 0.0).to_numpy()
    colors = [TEAL if c > 0.5 else RED if c < -0.5 else GREY for c in corr]
    colors = [c if s.startswith("development") else ORANGE for c, s in zip(colors, regimes["set"])]
    x = np.arange(len(labels))
    ax.bar(x, corr, color=colors, width=0.65)
    near_zero = np.abs(corr) < 0.05  # a bar this short is invisible, so mark the value as well
    ax.scatter(x[near_zero], corr[near_zero], color=np.array(colors)[near_zero], marker="D", s=30, zorder=3)
    ax.set_xticks(x, labels)
    ax.axhline(0, color="#9DAFB3", linewidth=0.8)
    ax.set_ylim(-1.3, 1.25)
    ax.set_ylabel("Correlation", color=MUTED)
    for xi, c in zip(x, corr):
        ax.text(xi, c + (0.07 if c >= 0 else -0.16), f"{c:+.2f}" if c else "0.00", ha="center", fontsize=9, color=INK)
    _style(ax, "Correlation of log(quote_signal) with log(rate per mile), by month")
    ax.text(0, -0.2, "Jan-Oct: against the actual rate. Teal tracks it, red is inverted, grey is noise.\n"
            "Nov-Dec (orange): against the model's expected rate, because validation rates are hidden.",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quote_signal_regimes.png", bbox_inches="tight")
    plt.close(fig)


def daily_rate_level(dev: pd.DataFrame, X_dev: pd.DataFrame, y_dev: pd.Series) -> pd.Series:
    """Mean log residual per day of a model that knows only lane, equipment and weight:
    the part of the price that moves with the date."""
    static = FreightRateModel({"static": LANE_FEATURES + LOAD_FEATURES}).fit(X_dev, y_dev)
    return (y_dev - static.predict_log_rpm(X_dev)).groupby(dev.loc[y_dev.index, "date"]).mean()


def key_facts(dev: pd.DataFrame, val: pd.DataFrame, keep: pd.Series, level: pd.Series, daily_mi: pd.Series) -> dict:
    """Numbers quoted in the report, computed here so the text always matches the data."""
    from pandas.tseries.holiday import USFederalHolidayCalendar

    from .data import haversine_miles

    clean = dev[keep]
    rpm = clean["posted_rate"] / clean["distance"]
    by_equipment = rpm.groupby(clean["equipment"].astype(str)).median()
    within_day = dev["market_index"] - dev.groupby("date")["market_index"].transform("mean")
    straight = haversine_miles(dev["pickup_lat"], dev["pickup_lon"], dev["delivery_lat"], dev["delivery_lon"])
    mi_dow = np.log(daily_mi[daily_mi.index <= dev["date"].max()]).groupby(
        daily_mi[daily_mi.index <= dev["date"].max()].index.day_name()).mean()
    monthly_level = level.groupby(level.index.month).mean()
    # Holiday check: the day's weekday-adjusted level against the three days either side. A short window
    # keeps month-boundary level shifts (e.g. 1 July) from masquerading as holiday effects.
    dow_adjusted = level - level.groupby(level.index.dayofweek).transform("mean")
    holidays = USFederalHolidayCalendar().holidays(dev["date"].min(), dev["date"].max())
    holiday_gaps = {}
    for day in holidays:
        around = dow_adjusted[(abs((dow_adjusted.index - day).days) <= 3) & (dow_adjusted.index != day)]
        holiday_gaps[day.strftime("%Y-%m-%d")] = float((dow_adjusted[day] - around.mean()) * 100)
    lane = dev[(dev["pickup"] == "Lexington") & (dev["delivery"] == "Fort Wayne") & (dev["equipment"] == "Dry Van")
               & (dev["date"].dt.month == 10) & keep]
    august = dev[dev["date"].dt.month == 8]["quote_signal"]
    december = daily_mi[(daily_mi.index >= "2025-12-01") & (daily_mi.index <= "2025-12-31")]
    december_thursdays = december[december.index.dayofweek == 3]
    return {
        "equipment_premium_pct": {k: float((v / by_equipment["Dry Van"] - 1) * 100) for k, v in by_equipment.items()},
        "market_index_share_explained_by_date": float(1 - within_day.var() / dev["market_index"].var()),
        "market_index_lowest_day": mi_dow.idxmin(), "market_index_highest_day": mi_dow.idxmax(),
        "market_index_peak_month": int(daily_mi[daily_mi.index <= dev["date"].max()].groupby(
            daily_mi[daily_mi.index <= dev["date"].max()].index.month).mean().idxmax()),
        "rate_level_change_jan_to_oct_pct": float((np.exp(monthly_level[10] - monthly_level[1]) - 1) * 100),
        "median_circuity": float(np.median(dev["distance"] / np.maximum(straight, 1.0))),
        "holiday_level_gap_pct": holiday_gaps,
        "max_abs_holiday_gap_pct": float(max(abs(v) for v in holiday_gaps.values())),
        "quote_mean_sd_august": [float(august.mean()), float(august.std())],
        "quote_mean_sd_validation": [float(val["quote_signal"].mean()), float(val["quote_signal"].std())],
        "lexington_fort_wayne_dry_van_october": [float(lane["posted_rate"].min()), float(lane["posted_rate"].max())],
        "december_thursday_market_index": {d.strftime("%Y-%m-%d"): float(v) for d, v in december_thursdays.items()},
    }


def plot_market_index_vs_rates(level: pd.Series, daily_mi: pd.Series) -> None:
    """Daily market index (dev + validation) against the daily rate level left after lane,
    equipment and weight are accounted for. Shows the shared weekly cycle and the different
    seasonal shape that makes the raw index misleading for forecasting."""
    dev_mi = daily_mi[daily_mi.index <= level.index.max()]
    peak = pd.Timestamp(2025, int(dev_mi.groupby(dev_mi.index.month).mean().idxmax()), 1).strftime("%B")
    pct = (np.exp(level) - 1) * 100
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), dpi=160, sharex=True, gridspec_kw={"hspace": 0.32})
    top.plot(daily_mi.index, daily_mi.values, color="#B5C0C4", linewidth=0.8, label="Daily")
    top.plot(daily_mi.rolling(7, center=True).mean(), color=INK, linewidth=1.8, label="7-day mean")
    top.set_ylabel("Market index", color=MUTED)
    _style(top, f"Market index: weekly cycle on top of a seasonal hump that peaks in {peak}")
    bottom.plot(pct.index, pct.values, color=PALE_TEAL, linewidth=0.8, label="Daily")
    bottom.plot(pct.rolling(7, center=True).mean(), color=TEAL, linewidth=2, label="7-day mean")
    bottom.axhline(0, color="#9DAFB3", linewidth=0.8)
    bottom.set_ylabel("Rate level vs average (%)", color=MUTED)
    _style(bottom, "Rate level after lane, equipment and weight: a different, rising shape")
    for ax in (top, bottom):
        ax.axvspan(pd.Timestamp("2025-11-01"), daily_mi.index.max(), color=VALIDATION_FILL, lw=0)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncols=2)
    bottom.text(pd.Timestamp("2025-11-04"), bottom.get_ylim()[1] - 0.4, "Validation period:\nrates unknown",
                fontsize=9, color=VALIDATION_TEXT, va="top")
    fig.savefig(FIGURES_DIR / "market_index_vs_rate_level.png", bbox_inches="tight")
    plt.close(fig)


def plot_outlier_residuals(outliers: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    ax.hist(outliers["log_residual"], bins=np.linspace(-2, 2, 161), color=TEAL)
    ax.set_yscale("log")
    for x in (-OUTLIER_LOG_THRESHOLD, OUTLIER_LOG_THRESHOLD):
        ax.axvline(x, color=RED, linestyle="--", linewidth=1, label="Threshold" if x < 0 else None)
    ax.set_xlabel("log(actual / expected rate), out-of-fold", color=MUTED)
    ax.set_ylabel("Loads (log scale)", color=MUTED)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    n = int(outliers["is_outlier"].sum())
    _style(ax, f"Label screening: {n} corrupted rates fall outside ±{OUTLIER_LOG_THRESHOLD}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "label_outlier_residuals.png", bbox_inches="tight")
    plt.close(fig)


def plot_backtest(results: pd.DataFrame) -> None:
    clean = results[results["rows"] == "clean"]
    table = clean.groupby(["origin", "design"])["MAPE"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=160)
    width = 0.2
    x = np.arange(len(table.index))
    for i, name in enumerate(DESIGN_LABELS):
        ax.bar(x + (i - 1.5) * width, table[name], width=width, color=DESIGN_COLORS[name],
               edgecolor="#9DAFB3" if name == "lane_median_baseline" else "none", linewidth=0.6,
               label=DESIGN_LABELS[name])
    month = pd.to_datetime([f"2025-{m:02d}-01" for m in range(1, 13)]).strftime("%b")
    ax.set_xticks(x, [f"train Jan-{month[k - 1]}\ntest {month[k]}-{month[k + 1]}" for k in table.index])
    ax.set_ylabel("MAPE (%)", color=MUTED)
    _style(ax, "Rolling-origin backtest: error on the next two months, per cut-off")
    ax.legend(fontsize=9, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "backtest_mape_by_origin.png", bbox_inches="tight")
    plt.close(fig)


def save_tables(**tables: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(RESULTS_DIR / f"{name}.csv", index=name.endswith("summary"))
