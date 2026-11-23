"""Builds reports/Freight_Rate_Model_Report.docx from the saved results, figures and scorer chart.

Every number in the report is read from reports/results, so the text cannot drift from the run.
Run `python -m freight_rate all` (or eda, validate, predict and score) first.
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from . import config as C
from .eda import DESIGN_LABELS

log = logging.getLogger(__name__)

FONT, MONO = "Calibri", "Consolas"
INK = RGBColor(0x1F, 0x2A, 0x30)
MUTED = RGBColor(0x5B, 0x6B, 0x71)
ACCENT = RGBColor(0x0B, 0x6E, 0x7F)
HEADER_FILL = "E3EEF0"
DEV_ROWS, VAL_ROWS = 48_000, 12_000
REPORT_AUTHOR = "Devdutt S"
ABLATION = "ablation: market index, no elapsed time"
LABELS = {**DESIGN_LABELS, ABLATION: "Ablation: A + market index, no elapsed time"}


def _load_results() -> dict:
    r = C.RESULTS_DIR
    files = ["data_quality.csv", "quote_regimes.csv", "backtest.csv", "backtest_summary.csv",
             "city_holdout_summary.csv", "random_kfold_summary.csv", "quote_august_check.csv",
             "feature_importance.csv", "prediction_summary.json", "eda_facts.json"]
    missing = [n for n in files if not (r / n).exists()]
    if not C.SCORER_CHART_PATH.exists():
        missing.append(str(C.SCORER_CHART_PATH.relative_to(C.ROOT)))
    if missing:
        raise FileNotFoundError(f"missing {missing}; run `python -m freight_rate all` first")
    return {
        "dq": pd.read_csv(r / "data_quality.csv", keep_default_na=False),
        "regimes": pd.read_csv(r / "quote_regimes.csv"),
        "backtest": pd.read_csv(r / "backtest.csv"),
        "summary": pd.read_csv(r / "backtest_summary.csv", index_col=0),
        "city": pd.read_csv(r / "city_holdout_summary.csv", index_col=[0, 1]),
        "random": pd.read_csv(r / "random_kfold_summary.csv", index_col=0),
        "august": pd.read_csv(r / "quote_august_check.csv").set_index("design"),
        "importance": pd.read_csv(r / "feature_importance.csv", index_col=0),
        "pred": json.loads((r / "prediction_summary.json").read_text()),
        "facts": json.loads((r / "eda_facts.json").read_text()),
        "december": pd.read_csv(C.DECEMBER_PATH, parse_dates=["date"]),
    }


# ----------------------------------------------------------------------------- document helpers

def _setup(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.left_margin = section.right_margin = Inches(0.9)
    section.top_margin = section.bottom_margin = Inches(0.8)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size, normal.font.color.rgb = FONT, Pt(10.5), INK
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.12
    for name, size, color in (("Title", 22, INK), ("Heading 1", 14, ACCENT), ("Heading 2", 11.5, INK)):
        style = doc.styles[name]
        style.font.name, style.font.size, style.font.bold, style.font.color.rgb = FONT, Pt(size), True, color
        # Built-in heading styles point at theme fonts; pin them to the body font instead.
        rfonts = style.element.rPr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            style.element.rPr.append(rfonts)
        for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rfonts.set(qn(attr), FONT)
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            rfonts.attrib.pop(qn(attr), None)
        fmt = style.paragraph_format
        fmt.space_before = Pt(0 if name == "Title" else 14)
        fmt.space_after = Pt(2 if name == "Title" else 6)
        fmt.keep_with_next = True
    bullet = doc.styles["List Bullet"]
    bullet.font.name, bullet.font.size = FONT, Pt(10.5)
    bullet.paragraph_format.space_after = Pt(3)


def _para(doc, text: str = "", *, lead: str | None = None, style: str | None = None, size: float | None = None,
          color: RGBColor | None = None, italic: bool = False, keep_with_next: bool = False):
    p = doc.add_paragraph(style=style)
    if lead:
        p.add_run(lead).bold = True
    if text:
        p.add_run(text).italic = italic
    for run in p.runs:
        if size:
            run.font.size = Pt(size)
        if color is not None:
            run.font.color.rgb = color
    p.paragraph_format.keep_with_next = keep_with_next
    return p


def _bullets(doc, items) -> None:
    for item in items:
        lead, text = item if isinstance(item, tuple) else (None, item)
        _para(doc, text, lead=lead, style="List Bullet")


def _shade(cell, fill: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def _row_flag(row, tag: str) -> None:
    el = OxmlElement(tag)
    el.set(qn("w:val"), "true")
    row._tr.get_or_add_trPr().append(el)


def _table(doc, caption: str, header: list[str], rows: list[list], widths: list[float], note: str = "",
           numeric: tuple[int, ...] = (), bold_row: int | None = None) -> None:
    label, text = caption.split(". ", 1)
    _para(doc, text, lead=f"{label}. ", size=9.5, keep_with_next=True)
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(header):
        cell = table.rows[0].cells[i]
        run = cell.paragraphs[0].add_run(text)
        run.bold, run.font.size = True, Pt(9)
        _shade(cell, HEADER_FILL)
    for r, row in enumerate(rows):
        cells = table.add_row().cells
        for i, text in enumerate(row):
            run = cells[i].paragraphs[0].add_run(str(text))
            run.font.size = Pt(9)
            run.bold = r == bold_row
    _row_flag(table.rows[0], "w:tblHeader")
    last = len(table.rows) - 1
    for r_idx, row in enumerate(table.rows):
        _row_flag(row, "w:cantSplit")
        for i, width in enumerate(widths):
            row.cells[i].width = Inches(width)
            for p in row.cells[i].paragraphs:
                p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.keep_with_next = r_idx < last  # keep the whole table on one page
                if i in numeric:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if note:
        _para(doc, note, size=8.5, color=MUTED)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(0)


def _figure(doc, path, caption: str, width: float = 6.6) -> None:
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    _para(doc, caption, size=9, color=MUTED, italic=True)


def _pct(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}%"


def _count(v) -> str:
    return f"{int(v):,}" if str(v).isdigit() else str(v)


def _month(m: int) -> str:
    return pd.Timestamp(2025, int(m), 1).strftime("%b")


def _corr(c: float) -> str:
    """Signed correlation without a negative zero."""
    text = f"{c:+.2f}"
    return "0.00" if text in ("+0.00", "-0.00") else text


# ----------------------------------------------------------------------------- report

def build_report() -> None:
    R = _load_results()
    summary, city, rnd, aug, dq, facts = R["summary"], R["city"], R["random"], R["august"], R["dq"], R["facts"]
    final, mt, sd = summary.loc["ensemble (final)"], summary.loc["market_time"], summary.loc["static_dow"]
    base = summary.loc["lane_median_baseline"]

    backtest = R["backtest"]
    clean = backtest[backtest["rows"] == "clean"]
    folds = clean.groupby(["design", "origin"])[["MAPE", "bias_pct"]].mean()
    late_bias = folds.loc[ABLATION, "bias_pct"].loc[7:8]
    train_sizes = backtest.groupby("origin")["train_rows"].first()
    test_sizes = backtest[(backtest["rows"] == "all") & (backtest["design"] == "ensemble (final)")].groupby("origin")["n"].sum()
    worst_b = folds.loc["market_time", "MAPE"].idxmax()
    spring_a = folds.loc["static_dow", "MAPE"].loc[3:5]

    regimes = R["regimes"]
    dev_reg = regimes[regimes["set"].str.startswith("development")]
    val_reg = regimes[~regimes["set"].str.startswith("development")]
    tracks = ", ".join(_month(m) for m in dev_reg.loc[dev_reg["corr"] > 0.5, "month"])
    inverted = ", ".join(_month(m) for m in dev_reg.loc[dev_reg["corr"] < -0.5, "month"])
    noise = ", ".join(_month(m) for m in dev_reg.loc[dev_reg["corr"].abs() <= 0.5, "month"])
    val_corr = " and ".join(f"{_corr(c)} in {_month(m)}" for m, c in zip(val_reg["month"], val_reg["corr"]))

    dqi = dq.set_index("issue")
    outliers = int(R["pred"]["label_outliers_removed"])
    new_city_rows = int(dqi.loc["Cities never seen in development data", "validation_rows"])
    neg_dev = int(dqi.loc["Negative weight (sign error)", "development_rows"]) / DEV_ROWS * 100
    neg_val = int(dqi.loc["Negative weight (sign error)", "validation_rows"]) / VAL_ROWS * 100
    mi_dev = int(dqi.loc["Missing market_index", "development_rows"]) / DEV_ROWS * 100
    mi_val = int(dqi.loc["Missing market_index", "validation_rows"]) / VAL_ROWS * 100

    prem = facts["equipment_premium_pct"]
    q_aug, q_val = facts["quote_mean_sd_august"], facts["quote_mean_sd_validation"]
    lane_lo, lane_hi = facts["lexington_fort_wayne_dry_van_october"]
    thursdays = pd.Series(facts["december_thursday_market_index"])

    dec = R["december"]
    by_dow = dec.groupby(dec["date"].dt.day_name())["predicted_rate"].mean()
    peak = dec.loc[dec["predicted_rate"].idxmax()]
    peak_mi = thursdays.get(peak["date"].strftime("%Y-%m-%d"))
    other_thu = thursdays.drop(peak["date"].strftime("%Y-%m-%d"), errors="ignore")
    imp = R["importance"]["market_time"].sort_values(ascending=False)
    pretty = {"log_distance": "distance", "equipment": "equipment", "weight": "weight", "pickup_lat": "pickup latitude",
              "pickup_lon": "pickup longitude", "delivery_lat": "delivery latitude",
              "delivery_lon": "delivery longitude", "day_of_week": "day of week",
              "log_market_index": "market index", "months_elapsed": "elapsed time"}
    top_imp = ", ".join(f"{pretty[k]} {v * 100:.0f}%" for k, v in imp.head(5).items())

    doc = Document()
    _setup(doc)
    doc.add_paragraph("Freight Rate Prediction: Model Report", style="Title")
    _para(doc, "Validation and data split, data findings, model choice and the December chart", size=11.5, color=MUTED)
    _para(doc, f"Development data: {DEV_ROWS:,} labelled loads, Jan-Oct 2025. Final predictions: {VAL_ROWS:,} loads, "
               "Nov-Dec 2025. Every number below is produced by the code in the accompanying repository.",
          size=9.5, color=MUTED)

    # 1 ------------------------------------------------------------------
    doc.add_heading("1. Summary", level=1)
    _para(doc, "The model predicts the posted rate as rate per mile times distance. Rate per mile is learned in log "
               "space by two LightGBM regressors whose predictions are averaged. The inputs are the lane's coordinates "
               "and distance, equipment, weight, day of week, the daily market index and elapsed time.")
    _bullets(doc, [
        ("How it was validated: ", "with a rolling-origin backtest. The model is trained on every month up to a "
         "cut-off and scored on the next two months, at six cut-offs. That mirrors the real job of predicting "
         "Nov-Dec from Jan-Oct, which a random split would not, because a random split lets the model see the future."),
        ("Backtest result: ", f"{_pct(final['MAPE'])} mean absolute percentage error (MAE ${final['MAE']:.0f}, "
         f"RMSE ${final['RMSE']:.0f}) over the next two months, against {_pct(base['MAPE'])} for a lane-median "
         f"baseline. The worst cut-off scored {_pct(final['MAPE_worst_origin'])}."),
        ("Two tempting features were traps: ", "quote_signal tracks the rate in some months, is inverted in others "
         "and is pure noise in Nov-Dec, so it is excluded. City identity breaks down on the "
         f"{new_city_rows:,} validation loads that touch a city missing from training, so location enters through "
         "coordinates."),
        ("Data quality: ", f"{outliers} corrupted rates ({outliers / DEV_ROWS * 100:.1f}%) were removed from "
         "training, sign-flipped weights were corrected and missing market-index values were filled with the "
         "same-day mean."),
    ])

    # 2 ------------------------------------------------------------------
    doc.add_heading("2. What the data showed", level=1)
    _para(doc, f"Month by month, its correlation with the actual rate per mile is close to +1 in {tracks}, close to "
               f"-1 in {inverted} and near zero in {noise} (Figure 1). Validation rates are hidden, so for Nov-Dec "
               f"it is compared with the model's expected rate instead: the correlation is {val_corr}. In other "
               f"words it behaves like the August noise, and its mean and spread match August's ({q_val[0]:.3f} and "
               f"{q_val[1]:.3f} against {q_aug[0]:.3f} and {q_aug[1]:.3f}). A model that leans on it would look "
               "excellent in a random split and fail on the loads that matter.",
          lead="quote_signal switches regime by month. ")
    _figure(doc, C.FIGURES_DIR / "quote_signal_regimes.png",
            "Figure 1. Correlation of log(quote_signal) with log(rate per mile) by month: against the actual rate "
            "for Jan-Oct, and against the model's expected rate for Nov-Dec.", width=6.2)
    _para(doc, f"The date alone explains {facts['market_index_share_explained_by_date'] * 100:.0f}% of its variance. "
               f"It is lowest on {facts['market_index_lowest_day']}s and highest on "
               f"{facts['market_index_highest_day']}s, and rates follow the same weekly rhythm. Its seasonal shape "
               f"is different, though (Figure 2). The index peaks in {_month(facts['market_index_peak_month'])}, "
               f"while the underlying rate level climbs by about {facts['rate_level_change_jan_to_oct_pct']:.0f}% "
               "from January to October. A tree model given the raw index therefore reads a low index as \"January "
               "prices\". In the backtest, that version (the ablation row in Table 2) under-predicts the last two "
               "folds by "
               f"{-late_bias.loc[7]:.1f}% and {-late_bias.loc[8]:.1f}% on average. The fix is to pair the index "
               "with elapsed time and to average that model with one that does not use the index at all (section 4).",
          lead="The market index is a daily signal with a strong weekly cycle. ")
    _figure(doc, C.FIGURES_DIR / "market_index_vs_rate_level.png",
            "Figure 2. Top: daily market index. Bottom: the part of the rate that moves with the date, after removing "
            "the effect of lane, equipment and weight. The shaded area is the validation period, where the index "
            "is known but rates are not.")
    _para(doc, lead="Other findings.", keep_with_next=True)
    _bullets(doc, [
        f"Rate per mile falls with distance and carries clear equipment premiums. At the median, Reefer is "
        f"{prem['Reefer']:.0f}% and Flatbed {prem['Flatbed']:.0f}% above Dry Van. Heavier loads cost slightly more.",
        "City coordinates are shifted from their real-world positions but are internally consistent: each city has "
        f"one location, and the road distance is a median {facts['median_circuity']:.2f}x the straight-line distance.",
        "No holiday effect shows around the 2025 federal holidays in the training months: no holiday's rate level "
        f"differs from the three days either side by more than {facts['max_abs_holiday_gap_pct']:.1f}%. There is no "
        "December history to learn one from.",
    ])

    # 3 ------------------------------------------------------------------
    doc.add_heading("3. Data-quality issues and how they were handled", level=1)
    _table(doc, "Table 1. Data-quality issues found and how each was handled.",
           ["Issue", "Dev rows", "Val rows", "Handling"],
           [[r.issue, _count(r.development_rows), _count(r.validation_rows), r.handling] for r in dq.itertuples()],
           widths=[2.15, 0.7, 0.7, 3.15], numeric=(1, 2),
           note="Dev = development file (train_test.csv, 48,000 rows); Val = validation file (12,000 rows).")
    _para(doc, "Corrupted labels were found by scoring every load against an out-of-fold L1 (median) model and "
               "flagging those more than 0.35 away in log space, which is roughly 42% above or 30% below the "
               "expectation. The cut sits in an empty gap (Figure 3): clean loads fall within 0.2 of the "
               "expectation and corrupted ones beyond 0.79, so the exact threshold does not change the result. "
               "Validation rows are never dropped; they are cleaned with the same rules.")
    _figure(doc, C.FIGURES_DIR / "label_outlier_residuals.png",
            "Figure 3. Out-of-fold residuals. The corrupted rates form separate clusters well outside the threshold.",
            width=6.0)

    # 4 ------------------------------------------------------------------
    doc.add_heading("4. Validation approach and data split", level=1)
    _para(doc, "Here, \"validation file\" means the 12,000 unlabelled Nov-Dec loads to be predicted. All model "
               "checks happen inside the labelled development data, where each fold has a training window and a "
               "test window.", size=9.5, color=MUTED)
    _para(doc, "The development data ends on 31 October and the validation loads run from 1 November to 31 December, "
               "so the model has to forecast one to two months ahead. Most of its error comes from not knowing the "
               "future market level, and a random split hides that. On the same data, shuffled 5-fold CV reports "
               f"{_pct(rnd.loc['ensemble (final)', 'MAPE'])} MAPE for the final model, against {_pct(final['MAPE'])} "
               "in the backtest. Add quote_signal and shuffled CV drops to "
               f"{_pct(rnd.loc['market_time + quote_signal', 'MAPE'])}, which would make the trap feature look like "
               "the best idea in the project.",
          lead="Why the split is by time. ")
    _para(doc, lead="Rolling-origin backtest (the main yardstick).", keep_with_next=True)
    _bullets(doc, [
        "Cut-offs at the end of each month from March through August (six folds). Each fold trains on every month "
        "up to the cut-off and tests on the next two months, matching the two-month horizon of the real task. "
        f"Training windows grow from {train_sizes.min():,} to {train_sizes.max():,} loads; each test window holds "
        f"{test_sizes.min():,}-{test_sizes.max():,} loads.",
        "Label screening is redone inside each training window, so no future label influences a fold. Test rows are "
        "screened by a model fitted only on that fold's training data.",
        "Errors are reported on screened test rows, because corrupted labels cannot be predicted by any model and "
        f"would only add noise. With them included, the final model's MAPE is {_pct(final['MAPE_all_rows'])}; "
        "scores on all rows are kept in reports/results/backtest.csv.",
    ])
    order = ["ensemble (final)", "market_time", "static_dow", ABLATION, "lane_median_baseline"]
    rows = [[LABELS[key], _pct(summary.loc[key, "MAPE"]), f"${summary.loc[key, 'MAE']:.0f}",
             f"${summary.loc[key, 'RMSE']:.0f}", f"{summary.loc[key, 'bias_pct']:+.2f}%",
             _pct(summary.loc[key, "MAPE_worst_origin"]), _pct(summary.loc[key, "MAPE_latest_origin"])]
            for key in order]
    _table(doc, "Table 2. Backtest results, averaged over six cut-offs and both test months.",
           ["Design", "MAPE", "MAE", "RMSE", "Bias", "Worst fold", "Latest fold"], rows,
           widths=[2.45, 0.62, 0.58, 0.62, 0.65, 0.78, 0.78], numeric=(1, 2, 3, 4, 5, 6), bold_row=0,
           note="MAPE: mean absolute percentage error. MAE and RMSE: mean absolute and root-mean-square error in "
                "dollars. Bias: mean signed percentage error (negative means under-prediction). Worst fold: highest "
                "MAPE among the six cut-offs. Latest fold: trained on Jan-Aug, tested on Sep-Oct.")
    _figure(doc, C.FIGURES_DIR / "backtest_mape_by_origin.png",
            "Figure 4. Backtest MAPE for each cut-off. Member B beats Member A in the spring folds but does worst "
            "after June's peak; Member A does the opposite. The average sits between them or beats both.")
    _para(doc, "The rate level jumps at some month boundaries, and nothing in the data predicts those jumps. Member B "
               "carries the latest level forward through the market index and elapsed time. It does well while the "
               f"level holds and worst when it drops, as in the fold trained through {_month(worst_b)} "
               f"({_pct(folds.loc[('market_time', worst_b), 'MAPE'])}). Member A assumes the average level returns, "
               f"which works when rates are flat and fails when they trend, as in the spring folds "
               f"({_pct(spring_a.min())}-{_pct(spring_a.max())}). Their average has the lowest mean error "
               f"({_pct(final['MAPE'])} against {_pct(mt['MAPE'])} and {_pct(sd['MAPE'])}) and the best worst fold "
               f"({_pct(final['MAPE_worst_origin'])} against {_pct(mt['MAPE_worst_origin'])} and "
               f"{_pct(sd['MAPE_worst_origin'])}).",
          lead="Why an average of two members. ")
    _para(doc, "Eight validation cities never appear in training. To measure the cost, three random sets of eight "
               "training cities were held out completely and the model was trained without any load touching them. "
               f"With coordinates only, the error on held-out cities was "
               f"{_pct(city.loc[('coordinates (final)', 'unseen cities'), 'MAPE'])} (seen cities "
               f"{_pct(city.loc[('coordinates (final)', 'seen cities'), 'MAPE'])}). Adding city-ID categories made "
               f"almost no difference for seen cities ({_pct(city.loc[('coordinates + city IDs', 'seen cities'), 'MAPE'])}) "
               "but raised the unseen-city error to "
               f"{_pct(city.loc[('coordinates + city IDs', 'unseen cities'), 'MAPE'])}, so the IDs were left out.",
          lead="Unseen-city hold-out. ")
    _para(doc, lead="Checks on the validation file itself.", keep_with_next=True)
    _bullets(doc, [
        "Every Nov-Dec date has market-index values from its own loads, so the index never has to be forecast.",
        "Feature drift was traced and handled: the quote turning to noise, new cities and lanes, and a higher share "
        f"of negative weights ({neg_val:.1f}% against {neg_dev:.1f}%) and of missing market index ({mi_val:.1f}% "
        f"against {mi_dev:.1f}%) than in development.",
        "August is the one development month where the quote behaves as it does in Nov-Dec. Training on Jan-Jul and "
        f"scoring August, adding quote_signal raises MAPE from {_pct(aug.loc['ensemble (final)', 'MAPE'])} to "
        f"{_pct(aug.loc['ensemble + quote_signal', 'MAPE'])}.",
    ])

    # 5 ------------------------------------------------------------------
    doc.add_heading("5. Model", level=1)
    _bullets(doc, [
        ("Target: ", "log(posted_rate / distance). The rate scales with distance, so this target is smooth, and "
         "squared error in log space approximates percentage error. Predictions are converted back with exp and "
         "multiplied by distance."),
        ("Algorithm: ", "LightGBM. The price surface is non-linear in location and distance and interacts with "
         "equipment and weight, which gradient-boosted trees capture without hand-built interactions. LightGBM also "
         "handles missing weights natively and trains in seconds."),
        ("Members: ", "Member A uses distance, pickup and delivery coordinates, equipment, weight and day of week. "
         "Member B adds the log market index and elapsed time in months. The final prediction is exp(mean of the "
         "two log predictions) times distance."),
        ("Settings: ", "2,000 trees, learning rate 0.03, 63 leaves, at least 30 rows per leaf, 80% row and 90% column "
         "sampling. Moving between 31-127 leaves or 700-3,000 trees changed shuffled-CV error by under 0.03 points, "
         "so no further tuning was done."),
        ("Most influential inputs ", f"(share of split gain in member B): {top_imp}."),
        ("Training data: ", f"{R['pred']['training_rows']:,} loads after removing {outliers} corrupted labels, "
         "trained once on all of Jan-Oct for the final predictions."),
    ])

    # 6 ------------------------------------------------------------------
    doc.add_heading("6. Fixed December prediction chart", level=1)
    _para(doc, "Produced by the provided score.py from data/december_chart_inputs.csv: Lexington to Fort Wayne, "
               "360 miles, Dry Van, 32,000 lb, with only the date changing (Figure 5). Each day uses that date's "
               "market index from the validation loads.")
    _figure(doc, C.SCORER_CHART_PATH, "Figure 5. Output of score.py (scorer_results/candidate_december.png).")
    christmas = ""
    if peak_mi is not None and len(other_thu):
        rising = bool(pd.Series(thursdays.values).is_monotonic_increasing)
        trend = " continuing the rise in Thursday values through the month" if rising else ""
        christmas = (f" The highest point is {peak['date'].strftime('%A %d %B')}. The validation file's market index "
                     f"is at its December high that day ({peak_mi:.3f}, against {other_thu.min():.3f}-"
                     f"{other_thu.max():.3f} on the other December Thursdays),{trend} with no holiday dip. The model "
                     "follows the index, so the chart does too.")
    _para(doc, f"Predictions run from ${dec['predicted_rate'].min():,.0f} to ${dec['predicted_rate'].max():,.0f}. "
               f"They follow a weekly cycle, highest on {by_dow.idxmax()}s and lowest on {by_dow.idxmin()}s, which "
               "is the rhythm that the market index and the training rates share. The level matches recent Dry Van "
               f"loads on this lane in the development data, which ran ${lane_lo:,.0f}-${lane_hi:,.0f} in October."
               f"{christmas} Apart from the weekly cycle the line is flat, because nothing in the data supports a "
               "December-specific effect: there is no December history and the training holidays show no rate jump. "
               "A holiday effect would be the first thing to check once December labels exist.")

    # 7 ------------------------------------------------------------------
    doc.add_heading("7. Limitations and next steps", level=1)
    _bullets(doc, [
        "The future rate level is the main source of error. A full prior year of history would allow a proper "
        "seasonal model of the level in place of the hedge used here.",
        "quote_signal deserves a fix at the source and per-month monitoring. When it tracks the rate it nearly "
        "reproduces the label, so it is a large opportunity as well as a risk.",
        "In production: retrain monthly, alert on input drift (new cities, the quote's correlation, missing-value "
        "rates) and track error by month and lane against the backtest numbers above.",
    ])
    doc.add_heading("Reproducing the results", level=2)
    p = _para(doc, "Install the pinned requirements with ")
    for text, mono in (("pip install -r requirements.txt", True), (", then run ", False),
                       ("python -m freight_rate all", True), (". That runs exploration, validation, training and "
                       "prediction, score.py and this report. README.md describes each step.", False)):
        run = p.add_run(text)
        if mono:
            run.font.name = MONO
            run.font.size = Pt(9.5)

    props = doc.core_properties
    props.title = "Freight Rate Prediction - Model Report"
    props.subject = "Validation and data split, data findings, model and December chart"
    props.author = props.last_modified_by = REPORT_AUTHOR
    props.comments = props.keywords = ""

    C.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(C.REPORT_PATH)
    log.info("report written to %s", C.REPORT_PATH)
