#!/usr/bin/env python3

# ============================================================
# plot_csv_etalepton.py
#
# Purpose:
#   Read the lepton-pseudorapidity correction-factor CSV files
#   produced by build_corrections_etalepton.py and create
#   correction-factor + uncertainty plots.
#
# Observable:
#       etalepton = |eta_lepton|
#
# Plot:
#   Top pad:
#       - correction factor
#       - E2 total uncertainty band
#       - E1 correction-factor points
#
#   Bottom pad:
#       - total uncertainty
#       - statistical uncertainty
#       - scale uncertainty
#       - PDF uncertainty
#       - shower uncertainty
#       - model uncertainty
#
# Output:
#       one PDF + one PNG for each CSV
#
# Folder structure:
#
#   lepton_pseudorapidity_work/
#   ├── build_corrections_etalepton_csv/
#   ├── plot_csv_etalepton.py
#   └── plot_csv_etalepton_outputs/
#
# ============================================================

import csv
from pathlib import Path
from array import array

import ROOT


# ============================================================
# ROOT settings
# ============================================================

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetErrorX(0.5)


# ============================================================
# Input / output directories
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_DIR = SCRIPT_DIR / "build_corrections_etalepton_csv"
OUTPUT_DIR = SCRIPT_DIR / "plot_csv_etalepton_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Observable
# ============================================================

OBSERVABLE = "etalepton"


# ============================================================
# Save options
# ============================================================

SAVE_PDF = True
SAVE_PNG = True


# ============================================================
# CSV column names
# ============================================================

CORR_COLUMN = "correction_factor"
TOTAL_COLUMN = "total_unc"
STAT_COLUMN = "stat_unc"
SCALE_COLUMN = "scale_unc"
PDF_COLUMN = "pdf_unc"
SHOWER_COLUMN = "shower_unc"
MODEL_COLUMN = "model_unc"


# ============================================================
# Canvas / pad geometry
# ============================================================

CANVAS_W = 1200
CANVAS_H = 850

LEFT_MARGIN = 0.12
RIGHT_MARGIN = 0.30
TOP_MARGIN = 0.08
BOTTOM_MARGIN = 0.12

TOP_PAD_YMIN = 0.34
TOP_PAD_YMAX = 1.00

BOT_PAD_YMIN = 0.08
BOT_PAD_YMAX = 0.34


# ============================================================
# Font sizes
# ============================================================

TOP_X_LABEL_SIZE = 0.0
TOP_X_TITLE_SIZE = 0.0

TOP_Y_TITLE_SIZE = 0.050
TOP_Y_LABEL_SIZE = 0.042
TOP_Y_TITLE_OFFSET = 1.15

BOT_X_TITLE_SIZE = 0.11
BOT_X_LABEL_SIZE = 0.095
BOT_X_TITLE_OFFSET = 1.05

BOT_Y_TITLE_SIZE = 0.10
BOT_Y_LABEL_SIZE = 0.085
BOT_Y_TITLE_OFFSET = 0.55

LEGEND_TEXT_SIZE = 0.030


# ============================================================
# Titles
# ============================================================

def x_axis_title():
    """ROOT-formatted x-axis title."""

    return "Lepton |#eta|"


def observable_title():
    """Human-readable observable title."""

    return "Lepton pseudorapidity"


def make_pretty_title(pair_label):
    """
    Example:
        Pythia_plus
            ->
        Pythia plus: Lepton pseudorapidity
    """

    pretty_pair = pair_label.replace("_", " ")

    return f"{pretty_pair}: {observable_title()}"


# ============================================================
# Filename parsing
# ============================================================

def parse_csv_filename(csv_path):
    """
    Expected filenames:

        Pythia_plus_etalepton.csv
        Pythia_minus_etalepton.csv
        Herwig_plus_etalepton.csv
        Herwig_minus_etalepton.csv
    """

    stem = csv_path.stem

    suffix = f"_{OBSERVABLE}"

    if not stem.endswith(suffix):
        raise RuntimeError(
            f"Unexpected CSV filename: {csv_path.name}"
        )

    pair_label = stem[:-len(suffix)]

    return pair_label


# ============================================================
# CSV reader
# ============================================================

def load_csv_data(csv_path):
    """
    Read a correction-factor CSV.

    This version deliberately strips whitespace from BOTH
    the CSV column names and the values.

    Therefore a header such as:

        pair_label  , observable, bin, bin_low_edge, ...

    is correctly interpreted as:

        pair_label
        observable
        bin
        bin_low_edge
        ...
    """

    low_edges = []
    up_edges = []

    corr_vals = []

    total_unc = []
    stat_unc = []
    scale_unc = []
    pdf_unc = []
    shower_unc = []
    model_unc = []

    # utf-8-sig also safely handles a possible BOM at the
    # beginning of a CSV file.
    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        # ----------------------------------------------------
        # IMPORTANT FIX:
        #
        # Your CSV headers contain whitespace around some
        # column names, e.g.
        #
        #     " bin_low_edge"
        #
        # rather than
        #
        #     "bin_low_edge"
        #
        # Strip that whitespace before reading the rows.
        # ----------------------------------------------------

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header row: {csv_path}"
            )

        reader.fieldnames = [
            name.strip()
            for name in reader.fieldnames
        ]

        required_columns = [
            "bin_low_edge",
            "bin_up_edge",
            CORR_COLUMN,
            TOTAL_COLUMN,
            STAT_COLUMN,
            SCALE_COLUMN,
            PDF_COLUMN,
            SHOWER_COLUMN,
            MODEL_COLUMN,
        ]

        # Check that all required columns now exist.
        for column in required_columns:

            if column not in reader.fieldnames:
                raise RuntimeError(
                    f"Missing required column '{column}' in:\n"
                    f"  {csv_path}\n"
                    f"Columns found: {reader.fieldnames}"
                )

        # ----------------------------------------------------
        # Read rows
        # ----------------------------------------------------

        for row in reader:

            # Strip whitespace from string values as well.
            clean_row = {
                key: value.strip()
                if isinstance(value, str)
                else value
                for key, value in row.items()
            }

            low_edges.append(
                float(clean_row["bin_low_edge"])
            )

            up_edges.append(
                float(clean_row["bin_up_edge"])
            )

            corr_vals.append(
                float(clean_row[CORR_COLUMN])
            )

            total_unc.append(
                float(clean_row[TOTAL_COLUMN])
            )

            stat_unc.append(
                float(clean_row[STAT_COLUMN])
            )

            scale_unc.append(
                float(clean_row[SCALE_COLUMN])
            )

            pdf_unc.append(
                float(clean_row[PDF_COLUMN])
            )

            shower_unc.append(
                float(clean_row[SHOWER_COLUMN])
            )

            model_unc.append(
                float(clean_row[MODEL_COLUMN])
            )

    if not low_edges:
        raise RuntimeError(
            f"No data rows found in:\n"
            f"  {csv_path}"
        )

    # Basic consistency check.
    lengths = {
        len(low_edges),
        len(up_edges),
        len(corr_vals),
        len(total_unc),
        len(stat_unc),
        len(scale_unc),
        len(pdf_unc),
        len(shower_unc),
        len(model_unc),
    }

    if len(lengths) != 1:
        raise RuntimeError(
            f"Inconsistent column lengths in:\n"
            f"  {csv_path}"
        )

    return {
        "low_edges": low_edges,
        "up_edges": up_edges,
        "corr_vals": corr_vals,
        "total_unc": total_unc,
        "stat_unc": stat_unc,
        "scale_unc": scale_unc,
        "pdf_unc": pdf_unc,
        "shower_unc": shower_unc,
        "model_unc": model_unc,
    }


# ============================================================
# Build ROOT histogram from CSV
# ============================================================

def make_hist_from_csv(csv_path, data_dict):
    """
    Reconstruct the variable-bin-width correction-factor
    histogram.

    Bin content:
        correction_factor

    Bin error:
        total_unc
    """

    pair_label = parse_csv_filename(csv_path)

    title = make_pretty_title(pair_label)

    low_edges = data_dict["low_edges"]
    up_edges = data_dict["up_edges"]

    corr_vals = data_dict["corr_vals"]
    total_unc = data_dict["total_unc"]

    # Reconstruct complete bin-edge array.
    edges = [low_edges[0]] + up_edges

    edge_array = array("d", edges)

    hist_name = f"h_{pair_label}_{OBSERVABLE}"

    hist = ROOT.TH1D(
        hist_name,
        title,
        len(corr_vals),
        edge_array,
    )

    hist.SetDirectory(0)

    for i, (correction, uncertainty) in enumerate(
        zip(corr_vals, total_unc),
        start=1,
    ):

        hist.SetBinContent(
            i,
            correction,
        )

        hist.SetBinError(
            i,
            uncertainty,
        )

    hist.SetTitle(title)

    return hist, pair_label


# ============================================================
# E2 uncertainty band
# ============================================================

def make_band_hist(source_hist):
    """
    Make the filled total-uncertainty band.
    """

    band = source_hist.Clone(
        source_hist.GetName() + "_band"
    )

    band.SetDirectory(0)

    band.SetLineWidth(0)

    band.SetMarkerStyle(0)
    band.SetMarkerSize(0)

    band.SetFillStyle(1001)

    band.SetFillColorAlpha(
        ROOT.kBlue,
        0.25,
    )

    return band


# ============================================================
# E1 correction-factor points
# ============================================================

def make_point_hist(source_hist):
    """
    Make the correction-factor point histogram.

    This keeps the same behaviour as the original
    plot_csv script.
    """

    points = source_hist.Clone(
        source_hist.GetName() + "_points"
    )

    points.SetDirectory(0)

    points.SetLineColor(
        ROOT.kBlue + 2
    )

    points.SetLineWidth(2)

    points.SetMarkerColor(
        ROOT.kBlue + 2
    )

    points.SetMarkerStyle(20)

    points.SetMarkerSize(0.9)

    return points


# ============================================================
# Uncertainty TGraph
# ============================================================

def make_uncertainty_graph(
    low_edges,
    up_edges,
    values,
    name,
    color,
    marker_style,
):
    """
    Put each uncertainty value at the centre of its eta bin.
    """

    x_values = []
    y_values = []

    for low_edge, up_edge, value in zip(
        low_edges,
        up_edges,
        values,
    ):

        bin_centre = 0.5 * (
            low_edge + up_edge
        )

        x_values.append(
            bin_centre
        )

        y_values.append(
            value
        )

    graph = ROOT.TGraph(
        len(values),
        array("d", x_values),
        array("d", y_values),
    )

    graph.SetName(name)

    graph.SetLineColor(color)
    graph.SetMarkerColor(color)

    graph.SetLineWidth(2)

    graph.SetMarkerStyle(
        marker_style
    )

    graph.SetMarkerSize(
        0.85
    )

    return graph


# ============================================================
# Small helper
# ============================================================

def max_of_list(values):
    return max(values) if values else 0.0


# ============================================================
# Axis styling
# ============================================================

def style_top_axes(hist):

    hist.GetXaxis().SetTitle(
        x_axis_title()
    )

    hist.GetYaxis().SetTitle(
        "Correction factor"
    )

    hist.GetXaxis().SetTitleSize(
        TOP_X_TITLE_SIZE
    )

    hist.GetXaxis().SetLabelSize(
        TOP_X_LABEL_SIZE
    )

    hist.GetYaxis().SetTitleSize(
        TOP_Y_TITLE_SIZE
    )

    hist.GetYaxis().SetLabelSize(
        TOP_Y_LABEL_SIZE
    )

    hist.GetYaxis().SetTitleOffset(
        TOP_Y_TITLE_OFFSET
    )


def style_bottom_axes(frame):

    frame.GetXaxis().SetTitle(
        x_axis_title()
    )

    frame.GetYaxis().SetTitle(
        "Uncertainty"
    )

    frame.GetXaxis().SetTitleSize(
        BOT_X_TITLE_SIZE
    )

    frame.GetXaxis().SetLabelSize(
        BOT_X_LABEL_SIZE
    )

    frame.GetXaxis().SetTitleOffset(
        BOT_X_TITLE_OFFSET
    )

    frame.GetYaxis().SetTitleSize(
        BOT_Y_TITLE_SIZE
    )

    frame.GetYaxis().SetLabelSize(
        BOT_Y_LABEL_SIZE
    )

    frame.GetYaxis().SetTitleOffset(
        BOT_Y_TITLE_OFFSET
    )


# ============================================================
# Plot one CSV
# ============================================================

def save_hist_plots(
    hist,
    csv_path,
    output_dir,
    data_dict,
):

    base = csv_path.stem

    low_edges = data_dict["low_edges"]
    up_edges = data_dict["up_edges"]

    total_unc = data_dict["total_unc"]
    stat_unc = data_dict["stat_unc"]
    scale_unc = data_dict["scale_unc"]
    pdf_unc = data_dict["pdf_unc"]
    shower_unc = data_dict["shower_unc"]
    model_unc = data_dict["model_unc"]


    # ========================================================
    # Top-panel y range
    # ========================================================

    corr_high = max(
        hist.GetBinContent(i)
        + hist.GetBinError(i)
        for i in range(
            1,
            hist.GetNbinsX() + 1,
        )
    )

    corr_low = min(
        hist.GetBinContent(i)
        - hist.GetBinError(i)
        for i in range(
            1,
            hist.GetNbinsX() + 1,
        )
    )

    span = corr_high - corr_low

    if span <= 0:
        span = max(
            abs(corr_high),
            1.0,
        )

    top_y_min = (
        corr_low
        - 0.10 * span
    )

    top_y_max = (
        corr_high
        + 0.10 * span
    )


    # ========================================================
    # Bottom-panel y range
    # ========================================================

    unc_max = max(
        max_of_list(total_unc),
        max_of_list(stat_unc),
        max_of_list(scale_unc),
        max_of_list(pdf_unc),
        max_of_list(shower_unc),
        max_of_list(model_unc),
    )

    bot_y_min = 0.0

    bot_y_max = (
        1.15 * unc_max
        if unc_max > 0
        else 1.0
    )


    # ========================================================
    # Canvas
    # ========================================================

    canvas = ROOT.TCanvas(
        f"c_{base}",
        f"c_{base}",
        CANVAS_W,
        CANVAS_H,
    )


    # ========================================================
    # TOP PAD
    # ========================================================

    pad_top = ROOT.TPad(
        f"pad_top_{base}",
        "",
        0.0,
        TOP_PAD_YMIN,
        1.0,
        TOP_PAD_YMAX,
    )

    pad_top.SetLeftMargin(
        LEFT_MARGIN
    )

    pad_top.SetRightMargin(
        RIGHT_MARGIN
    )

    pad_top.SetTopMargin(
        TOP_MARGIN
    )

    pad_top.SetBottomMargin(
        0.02
    )

    pad_top.Draw()
    pad_top.cd()


    # --------------------------------------------------------
    # Correction factor
    # --------------------------------------------------------

    band_hist = make_band_hist(
        hist
    )

    point_hist = make_point_hist(
        hist
    )

    style_top_axes(
        band_hist
    )

    band_hist.SetMinimum(
        top_y_min
    )

    band_hist.SetMaximum(
        top_y_max
    )

    band_hist.Draw("E2")

    point_hist.Draw(
        "E1 SAME"
    )


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend = ROOT.TLegend(
        0.73,
        0.18,
        0.98,
        0.88,
    )

    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    legend.SetTextSize(
        LEGEND_TEXT_SIZE
    )

    legend.AddEntry(
        point_hist,
        "Correction factor",
        "lep",
    )


    # ========================================================
    # BOTTOM PAD
    # ========================================================

    canvas.cd()

    pad_bot = ROOT.TPad(
        f"pad_bot_{base}",
        "",
        0.0,
        BOT_PAD_YMIN,
        1.0,
        BOT_PAD_YMAX,
    )

    pad_bot.SetLeftMargin(
        LEFT_MARGIN
    )

    pad_bot.SetRightMargin(
        RIGHT_MARGIN
    )

    pad_bot.SetTopMargin(
        0.02
    )

    pad_bot.SetBottomMargin(
        BOTTOM_MARGIN + 0.12
    )

    pad_bot.Draw()
    pad_bot.cd()


    # --------------------------------------------------------
    # Bottom frame
    # --------------------------------------------------------

    x_min = (
        hist.GetXaxis()
        .GetXmin()
    )

    x_max = (
        hist.GetXaxis()
        .GetXmax()
    )

    frame = ROOT.TH1F(
        f"frame_{base}",
        "",
        1,
        x_min,
        x_max,
    )

    frame.SetDirectory(0)

    frame.SetMinimum(
        bot_y_min
    )

    frame.SetMaximum(
        bot_y_max
    )

    style_bottom_axes(
        frame
    )

    frame.Draw()


    # ========================================================
    # Uncertainty graphs
    # ========================================================

    g_total = make_uncertainty_graph(
        low_edges,
        up_edges,
        total_unc,
        f"g_total_{base}",
        ROOT.kBlack,
        24,
    )

    g_stat = make_uncertainty_graph(
        low_edges,
        up_edges,
        stat_unc,
        f"g_stat_{base}",
        ROOT.kRed + 1,
        20,
    )

    g_scale = make_uncertainty_graph(
        low_edges,
        up_edges,
        scale_unc,
        f"g_scale_{base}",
        ROOT.kGreen + 2,
        21,
    )

    g_pdf = make_uncertainty_graph(
        low_edges,
        up_edges,
        pdf_unc,
        f"g_pdf_{base}",
        ROOT.kMagenta + 1,
        22,
    )

    g_shower = make_uncertainty_graph(
        low_edges,
        up_edges,
        shower_unc,
        f"g_shower_{base}",
        ROOT.kOrange + 7,
        23,
    )

    g_model = make_uncertainty_graph(
        low_edges,
        up_edges,
        model_unc,
        f"g_model_{base}",
        ROOT.kCyan + 2,
        33,
    )


    # --------------------------------------------------------
    # Draw uncertainty curves
    # --------------------------------------------------------

    g_total.Draw(
        "LP SAME"
    )

    g_stat.Draw(
        "LP SAME"
    )

    g_scale.Draw(
        "LP SAME"
    )

    g_pdf.Draw(
        "LP SAME"
    )

    g_shower.Draw(
        "LP SAME"
    )

    g_model.Draw(
        "LP SAME"
    )


    # --------------------------------------------------------
    # Legend entries
    # --------------------------------------------------------

    legend.AddEntry(
        g_total,
        "Total uncertainty",
        "lp",
    )

    legend.AddEntry(
        g_stat,
        "Stat uncertainty",
        "lp",
    )

    legend.AddEntry(
        g_scale,
        "Scale uncertainty",
        "lp",
    )

    legend.AddEntry(
        g_pdf,
        "PDF uncertainty",
        "lp",
    )

    legend.AddEntry(
        g_shower,
        "Shower uncertainty",
        "lp",
    )

    legend.AddEntry(
        g_model,
        "Model uncertainty",
        "lp",
    )


    # --------------------------------------------------------
    # Draw legend on top pad
    # --------------------------------------------------------

    canvas.cd()
    pad_top.cd()

    legend.Draw()

    canvas.Update()


    # ========================================================
    # Save files
    # ========================================================

    saved_files = []

    if SAVE_PDF:

        pdf_path = (
            output_dir
            / f"{base}_E2E1.pdf"
        )

        canvas.SaveAs(
            str(pdf_path)
        )

        saved_files.append(
            pdf_path
        )

    if SAVE_PNG:

        png_path = (
            output_dir
            / f"{base}_E2E1.png"
        )

        canvas.SaveAs(
            str(png_path)
        )

        saved_files.append(
            png_path
        )

    canvas.Close()

    return saved_files


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("==============================================")
    print(" Lepton pseudorapidity plotting")
    print("==============================================")
    print()

    print("CSV input directory:")
    print(f"  {INPUT_DIR}")
    print()

    print("Plot output directory:")
    print(f"  {OUTPUT_DIR}")
    print()


    # --------------------------------------------------------
    # Check input directory
    # --------------------------------------------------------

    if not INPUT_DIR.is_dir():

        raise RuntimeError(
            f"INPUT_DIR does not exist:\n"
            f"  {INPUT_DIR}"
        )


    # --------------------------------------------------------
    # Find ONLY etalepton CSV files
    # --------------------------------------------------------

    csv_files = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == ".csv"
            and path.stem.endswith(
                f"_{OBSERVABLE}"
            )
        )
    )


    if not csv_files:

        raise RuntimeError(
            "No lepton pseudorapidity CSV files "
            "found in:\n"
            f"  {INPUT_DIR}"
        )


    print(
        f"Found {len(csv_files)} "
        f"lepton pseudorapidity CSV files."
    )

    print()


    # --------------------------------------------------------
    # Process every CSV
    # --------------------------------------------------------

    saved_count = 0

    for csv_path in csv_files:

        try:

            data_dict = load_csv_data(
                csv_path
            )

            (
                hist,
                pair_label,
            ) = make_hist_from_csv(
                csv_path,
                data_dict,
            )

            saved_files = save_hist_plots(
                hist=hist,
                csv_path=csv_path,
                output_dir=OUTPUT_DIR,
                data_dict=data_dict,
            )

            print(
                f"Plotted: {csv_path.name}"
            )

            for path in saved_files:

                print(
                    f"  -> {path}"
                )

            saved_count += 1

        except Exception as exc:

            print(
                f"Failed on {csv_path}"
            )

            print(
                f"  Reason: {exc}"
            )


    # ========================================================
    # Finished
    # ========================================================

    print()
    print("==============================================")
    print(" Finished")
    print("==============================================")
    print()

    print(
        f"Successfully plotted "
        f"{saved_count} CSV files."
    )

    print()

    print("Output directory:")
    print(f"  {OUTPUT_DIR}")

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()