#!/usr/bin/env python3

# ============================================================
# MTWcut_plot_csv_etalepton_interactive.py
#
# Creates TWO interactive 2x2 dashboards using the
# m_T^W > 40 GeV correction-factor dataset:
#
#   1. Lepton pseudorapidity correction factors
#   2. Lepton pseudorapidity uncertainty decomposition
#
# Input:
#   MTWcut_build_corrections_etalepton_csv/
#
# These CSVs were built from histograms in which the event
# selection already required m_T^W > 40 GeV. This plotting
# script does not re-apply the cut; it visualises those results.
#
#
# IMPORTANT AXIS BEHAVIOUR:
#
#   - Every quadrant has its OWN visible x-axis
#   - Every quadrant has its OWN visible y-axis
#   - Every quadrant has x and y tick labels
#   - Every quadrant has x and y axis titles
#
#   BUT:
#
#   - all x axes are synchronised
#   - all y axes are synchronised
#   - zooming/panning one quadrant updates all four
#
#
# Correction-factor style:
#
#   - isolated correction-factor markers
#   - NO connecting line
#   - vertical total-uncertainty bars only
#   - NO horizontal error bars
#   - rectangular / stepwise shaded total uncertainty
#
#
# Outputs:
#
#   plot_csv_etalepton_interactive_outputs/
#
#     etalepton_correction_factor_interactive_2x2.html
#     etalepton_uncertainties_interactive_2x2.html
#
# ============================================================


import csv
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# Directories
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_DIR = (
    SCRIPT_DIR
    / "MTWcut_build_corrections_etalepton_csv"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "MTWcut_plot_csv_etalepton_interactive_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Output HTML files
# ============================================================

CORRECTION_HTML = (
    OUTPUT_DIR
    / "MTWcut_etalepton_correction_factor_interactive_2x2.html"
)

UNCERTAINTY_HTML = (
    OUTPUT_DIR
    / "MTWcut_etalepton_uncertainties_interactive_2x2.html"
)


# ============================================================
# Observable
# ============================================================

OBSERVABLE = "etalepton"


# ============================================================
# CSV columns
# ============================================================

CORR_COLUMN = "correction_factor"

TOTAL_COLUMN = "total_unc"
STAT_COLUMN = "stat_unc"
SCALE_COLUMN = "scale_unc"
PDF_COLUMN = "pdf_unc"
SHOWER_COLUMN = "shower_unc"
MODEL_COLUMN = "model_unc"


# ============================================================
# Generator / charge ordering
# ============================================================

PAIR_ORDER = [
    "Pythia_plus",
    "Pythia_minus",
    "Herwig_plus",
    "Herwig_minus",
]


PAIR_TITLES = {
    "Pythia_plus": "Pythia · W⁺",
    "Pythia_minus": "Pythia · W⁻",
    "Herwig_plus": "Herwig · W⁺",
    "Herwig_minus": "Herwig · W⁻",
}


SUBPLOT_POSITIONS = {
    "Pythia_plus": (1, 1),
    "Pythia_minus": (1, 2),
    "Herwig_plus": (2, 1),
    "Herwig_minus": (2, 2),
}


# ============================================================
# Colours
# ============================================================

CORRECTION_COLOR = "#1D4ED8"

TOTAL_BAND_COLOR = (
    "rgba(99, 102, 241, 0.28)"
)

TOTAL_COLOR = "#111827"
STAT_COLOR = "#DC2626"
SCALE_COLOR = "#16A34A"
PDF_COLOR = "#9333EA"
SHOWER_COLOR = "#EA580C"
MODEL_COLOR = "#0891B2"


# ============================================================
# Parse CSV filename
# ============================================================

def parse_csv_filename(csv_path):

    stem = csv_path.stem

    suffix = f"_{OBSERVABLE}"

    if not stem.endswith(suffix):

        raise RuntimeError(
            f"Unexpected CSV filename: "
            f"{csv_path.name}"
        )

    return stem[:-len(suffix)]


# ============================================================
# Load CSV
# ============================================================

def load_csv_data(csv_path):

    data = {
        "low_edges": [],
        "up_edges": [],
        "corr_vals": [],
        "total_unc": [],
        "stat_unc": [],
        "scale_unc": [],
        "pdf_unc": [],
        "shower_unc": [],
        "model_unc": [],
    }


    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )


        if reader.fieldnames is None:

            raise RuntimeError(
                f"CSV contains no header:\n"
                f"  {csv_path}"
            )


        # ----------------------------------------------------
        # Strip whitespace from headers
        # ----------------------------------------------------

        reader.fieldnames = [
            field.strip()
            for field in reader.fieldnames
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


        for column in required_columns:

            if column not in reader.fieldnames:

                raise RuntimeError(
                    f"Missing required column "
                    f"'{column}' in:\n"
                    f"  {csv_path}\n\n"
                    f"Columns found:\n"
                    f"  {reader.fieldnames}"
                )


        # ----------------------------------------------------
        # Load values
        # ----------------------------------------------------

        for row in reader:

            clean_row = {
                key: (
                    value.strip()
                    if isinstance(value, str)
                    else value
                )
                for key, value in row.items()
            }


            data["low_edges"].append(
                float(
                    clean_row[
                        "bin_low_edge"
                    ]
                )
            )

            data["up_edges"].append(
                float(
                    clean_row[
                        "bin_up_edge"
                    ]
                )
            )

            data["corr_vals"].append(
                float(
                    clean_row[
                        CORR_COLUMN
                    ]
                )
            )

            data["total_unc"].append(
                float(
                    clean_row[
                        TOTAL_COLUMN
                    ]
                )
            )

            data["stat_unc"].append(
                float(
                    clean_row[
                        STAT_COLUMN
                    ]
                )
            )

            data["scale_unc"].append(
                float(
                    clean_row[
                        SCALE_COLUMN
                    ]
                )
            )

            data["pdf_unc"].append(
                float(
                    clean_row[
                        PDF_COLUMN
                    ]
                )
            )

            data["shower_unc"].append(
                float(
                    clean_row[
                        SHOWER_COLUMN
                    ]
                )
            )

            data["model_unc"].append(
                float(
                    clean_row[
                        MODEL_COLUMN
                    ]
                )
            )


    if not data["low_edges"]:

        raise RuntimeError(
            f"No data found in:\n"
            f"  {csv_path}"
        )


    # ========================================================
    # Derived quantities
    # ========================================================

    data["centres"] = [
        0.5 * (low + high)
        for low, high in zip(
            data["low_edges"],
            data["up_edges"],
        )
    ]


    data["upper_total"] = [
        correction + uncertainty
        for correction, uncertainty in zip(
            data["corr_vals"],
            data["total_unc"],
        )
    ]


    data["lower_total"] = [
        correction - uncertainty
        for correction, uncertainty in zip(
            data["corr_vals"],
            data["total_unc"],
        )
    ]


    return data


# ============================================================
# Make flat histogram-style coordinates
# ============================================================

def make_step_coordinates(
    low_edges,
    up_edges,
    values,
):

    x_values = []
    y_values = []


    for low, high, value in zip(
        low_edges,
        up_edges,
        values,
    ):

        x_values.extend(
            [
                low,
                high,
            ]
        )

        y_values.extend(
            [
                value,
                value,
            ]
        )


    return (
        x_values,
        y_values,
    )


# ============================================================
# Make stepwise uncertainty polygon
# ============================================================

def make_uncertainty_band_polygon(
    data,
):

    (
        upper_x,
        upper_y,
    ) = make_step_coordinates(
        data["low_edges"],
        data["up_edges"],
        data["upper_total"],
    )


    (
        lower_x,
        lower_y,
    ) = make_step_coordinates(
        data["low_edges"],
        data["up_edges"],
        data["lower_total"],
    )


    polygon_x = (
        upper_x
        + list(
            reversed(
                lower_x
            )
        )
    )


    polygon_y = (
        upper_y
        + list(
            reversed(
                lower_y
            )
        )
    )


    return (
        polygon_x,
        polygon_y,
    )


# ============================================================
# Correction hover information
# ============================================================

def correction_custom_data(
    data,
):

    return [
        list(values)
        for values in zip(
            data["low_edges"],
            data["up_edges"],
            data["total_unc"],
            data["stat_unc"],
            data["scale_unc"],
            data["pdf_unc"],
            data["shower_unc"],
            data["model_unc"],
        )
    ]


# ============================================================
# Add correction-factor subplot
# ============================================================

def add_correction_subplot(
    fig,
    pair_label,
    data,
    row,
    col,
    show_legend,
):

    # ========================================================
    # Stepwise shaded uncertainty
    # ========================================================

    (
        band_x,
        band_y,
    ) = make_uncertainty_band_polygon(
        data
    )


    fig.add_trace(

        go.Scatter(

            x=band_x,

            y=band_y,

            mode="lines",

            line=dict(
                width=0,
            ),

            fill="toself",

            fillcolor=(
                TOTAL_BAND_COLOR
            ),

            name=(
                "Total uncertainty"
            ),

            legendgroup=(
                "total_band"
            ),

            showlegend=(
                show_legend
            ),

            hoverinfo="skip",
        ),

        row=row,
        col=col,
    )


    # ========================================================
    # Correction factor points
    #
    # NO connecting line
    # NO horizontal uncertainty bars
    # ========================================================

    fig.add_trace(

        go.Scatter(

            x=data[
                "centres"
            ],

            y=data[
                "corr_vals"
            ],

            mode="markers",

            name=(
                "Correction factor"
            ),

            legendgroup=(
                "correction"
            ),

            showlegend=(
                show_legend
            ),

            marker=dict(

                color=(
                    CORRECTION_COLOR
                ),

                size=8,

                symbol="circle",

                line=dict(
                    color="white",
                    width=1,
                ),
            ),


            # ----------------------------------------------
            # Vertical error bars ONLY
            # ----------------------------------------------

            error_y=dict(

                type="data",

                array=data[
                    "total_unc"
                ],

                visible=True,

                color=(
                    CORRECTION_COLOR
                ),

                thickness=1.3,

                width=3,
            ),


            customdata=(
                correction_custom_data(
                    data
                )
            ),


            hovertemplate=(

                "<b>"
                + PAIR_TITLES[
                    pair_label
                ]
                + "</b>"

                "<br>"

                "|η<sub>ℓ</sub>| bin: "
                "%{customdata[0]:.2f}"
                " – "
                "%{customdata[1]:.2f}"

                "<br>"

                "Correction factor: "
                "<b>%{y:.6f}</b>"

                "<br>"

                "Total uncertainty: "
                "%{customdata[2]:.6f}"

                "<br>"

                "Statistical: "
                "%{customdata[3]:.6f}"

                "<br>"

                "Scale: "
                "%{customdata[4]:.6f}"

                "<br>"

                "PDF: "
                "%{customdata[5]:.6f}"

                "<br>"

                "Shower: "
                "%{customdata[6]:.6f}"

                "<br>"

                "Model: "
                "%{customdata[7]:.6f}"

                "<extra></extra>"
            ),
        ),

        row=row,
        col=col,
    )


# ============================================================
# Add uncertainty subplot
# ============================================================

def add_uncertainty_subplot(
    fig,
    data,
    row,
    col,
    show_legend,
):

    x = data[
        "centres"
    ]


    uncertainty_specs = [

        (
            "Total",
            data["total_unc"],
            TOTAL_COLOR,
            "solid",
            "total",
            8,
        ),

        (
            "Statistical",
            data["stat_unc"],
            STAT_COLOR,
            "dash",
            "stat",
            7,
        ),

        (
            "Scale",
            data["scale_unc"],
            SCALE_COLOR,
            "dash",
            "scale",
            7,
        ),

        (
            "PDF",
            data["pdf_unc"],
            PDF_COLOR,
            "dot",
            "pdf",
            7,
        ),

        (
            "Shower",
            data["shower_unc"],
            SHOWER_COLOR,
            "dashdot",
            "shower",
            7,
        ),

        (
            "Model",
            data["model_unc"],
            MODEL_COLOR,
            "longdash",
            "model",
            7,
        ),
    ]


    for (
        label,
        values,
        colour,
        dash,
        group,
        marker_size,
    ) in uncertainty_specs:


        fig.add_trace(

            go.Scatter(

                x=x,

                y=values,

                mode="lines+markers",

                name=label,

                legendgroup=group,

                showlegend=(
                    show_legend
                ),

                line=dict(

                    color=colour,

                    width=(
                        3
                        if label == "Total"
                        else 2.2
                    ),

                    dash=dash,
                ),

                marker=dict(
                    color=colour,
                    size=marker_size,
                ),

                customdata=list(
                    zip(
                        data[
                            "low_edges"
                        ],
                        data[
                            "up_edges"
                        ],
                    )
                ),

                hovertemplate=(

                    "<b>"
                    + label
                    + " uncertainty</b>"

                    "<br>"

                    "|η<sub>ℓ</sub>| bin: "
                    "%{customdata[0]:.2f}"
                    " – "
                    "%{customdata[1]:.2f}"

                    "<br>"

                    "Uncertainty: "
                    "<b>%{y:.6f}</b>"

                    "<extra></extra>"
                ),
            ),

            row=row,
            col=col,
        )


# ============================================================
# Global x range
# ============================================================

def get_global_x_range(
    all_data,
):

    lows = []
    highs = []


    for pair_label in PAIR_ORDER:

        lows.extend(
            all_data[
                pair_label
            ]["low_edges"]
        )

        highs.extend(
            all_data[
                pair_label
            ]["up_edges"]
        )


    return (
        min(lows),
        max(highs),
    )


# ============================================================
# Global correction-factor y range
# ============================================================

def get_global_correction_y_range(
    all_data,
):

    lowers = []
    uppers = []


    for pair_label in PAIR_ORDER:

        data = all_data[
            pair_label
        ]

        lowers.extend(
            data[
                "lower_total"
            ]
        )

        uppers.extend(
            data[
                "upper_total"
            ]
        )


    y_min = min(
        lowers
    )

    y_max = max(
        uppers
    )


    span = (
        y_max
        - y_min
    )


    if span <= 0:

        span = max(
            abs(y_max),
            1.0,
        )


    padding = (
        0.08
        * span
    )


    return (
        y_min - padding,
        y_max + padding,
    )


# ============================================================
# Global uncertainty y range
# ============================================================

def get_global_uncertainty_y_range(
    all_data,
):

    values = []


    for pair_label in PAIR_ORDER:

        data = all_data[
            pair_label
        ]

        values.extend(
            data["total_unc"]
        )

        values.extend(
            data["stat_unc"]
        )

        values.extend(
            data["scale_unc"]
        )

        values.extend(
            data["pdf_unc"]
        )

        values.extend(
            data["shower_unc"]
        )

        values.extend(
            data["model_unc"]
        )


    maximum = max(
        values
    )


    if maximum <= 0:

        maximum = 1.0


    return (
        0.0,
        1.10 * maximum,
    )


# ============================================================
# Configure ALL subplot axes
# ============================================================

def configure_subplot_axes(
    fig,
    x_range,
    y_range,
    y_title,
):

    """
    Every quadrant gets its own visible axes and labels.

    The axes remain synchronised through Plotly's 'matches'
    system rather than using shared_xaxes/shared_yaxes.
    """


    # ========================================================
    # Loop over every physical quadrant
    # ========================================================

    for row in [
        1,
        2,
    ]:

        for col in [
            1,
            2,
        ]:


            # =================================================
            # X axis
            # =================================================

            fig.update_xaxes(

                title_text=(
                    "Lepton |η|"
                ),

                range=[
                    x_range[0],
                    x_range[1],
                ],

                # Link all ranges to the first x axis
                matches="x",

                # IMPORTANT:
                # force visible labels on every quadrant
                showticklabels=True,

                ticks="outside",

                ticklen=5,

                tickwidth=1,

                showline=True,

                linewidth=1,

                linecolor=(
                    "rgba(40,40,40,0.65)"
                ),

                mirror=False,

                showgrid=True,

                gridcolor=(
                    "rgba(120,120,120,0.12)"
                ),

                zeroline=False,

                showspikes=True,

                spikemode="across",

                spikesnap="cursor",

                spikedash="dot",

                row=row,
                col=col,
            )


            # =================================================
            # Y axis
            # =================================================

            fig.update_yaxes(

                title_text=(
                    y_title
                ),

                range=[
                    y_range[0],
                    y_range[1],
                ],

                # Link all ranges to the first y axis
                matches="y",

                # IMPORTANT:
                # force visible labels on every quadrant
                showticklabels=True,

                ticks="outside",

                ticklen=5,

                tickwidth=1,

                showline=True,

                linewidth=1,

                linecolor=(
                    "rgba(40,40,40,0.65)"
                ),

                mirror=False,

                showgrid=True,

                gridcolor=(
                    "rgba(120,120,120,0.12)"
                ),

                zeroline=False,

                row=row,
                col=col,
            )


# ============================================================
# Correction-factor dashboard
# ============================================================

def make_correction_dashboard(
    all_data,
):

    x_range = (
        get_global_x_range(
            all_data
        )
    )

    y_range = (
        get_global_correction_y_range(
            all_data
        )
    )


    # ========================================================
    # IMPORTANT:
    #
    # Do NOT use shared_xaxes/shared_yaxes here.
    #
    # That is what caused Plotly to hide some of the
    # duplicated labels.
    #
    # Synchronisation is applied later using matches=.
    # ========================================================

    fig = make_subplots(

        rows=2,
        cols=2,

        subplot_titles=[
            PAIR_TITLES[
                "Pythia_plus"
            ],
            PAIR_TITLES[
                "Pythia_minus"
            ],
            PAIR_TITLES[
                "Herwig_plus"
            ],
            PAIR_TITLES[
                "Herwig_minus"
            ],
        ],

        horizontal_spacing=0.10,

        vertical_spacing=0.18,
    )


    # ========================================================
    # Add plots
    # ========================================================

    for index, pair_label in enumerate(
        PAIR_ORDER
    ):

        row, col = (
            SUBPLOT_POSITIONS[
                pair_label
            ]
        )


        add_correction_subplot(

            fig=fig,

            pair_label=pair_label,

            data=all_data[
                pair_label
            ],

            row=row,
            col=col,

            show_legend=(
                index == 0
            ),
        )


    # ========================================================
    # Give EVERY subplot its own axes + synchronise them
    # ========================================================

    configure_subplot_axes(

        fig=fig,

        x_range=x_range,

        y_range=y_range,

        y_title=(
            "Correction factor"
        ),
    )


    # ========================================================
    # Layout
    # ========================================================

    fig.update_layout(

        title=dict(

            text=(

                "<b>"
                "Lepton pseudorapidity "
                "correction factors"
                "</b>"

                "<br>"

                "<span style='font-size:16px;'>"

                "Pythia and Herwig "
                "particle-to-parton corrections · m<sub>T</sub><sup>W</sup> &gt; 40 GeV"

                "</span>"
            ),

            x=0.5,

            xanchor="center",

            y=0.98,
        ),


        template=(
            "plotly_white"
        ),


        width=1500,

        height=1100,


        margin=dict(

            l=100,

            r=75,

            t=150,

            b=90,
        ),


        hovermode=(
            "closest"
        ),


        dragmode=(
            "pan"
        ),


        legend=dict(

            title=dict(
                text="<b>Display</b>"
            ),

            orientation="h",

            yanchor="bottom",

            y=1.035,

            xanchor="center",

            x=0.5,

            bgcolor=(
                "rgba(255,255,255,0.92)"
            ),

            bordercolor=(
                "rgba(100,100,100,0.20)"
            ),

            borderwidth=1,

            groupclick=(
                "togglegroup"
            ),
        ),


        font=dict(

            family=(
                "Arial, Helvetica, sans-serif"
            ),

            size=14,
        ),


        paper_bgcolor=(
            "#F8FAFC"
        ),


        plot_bgcolor=(
            "#FFFFFF"
        ),
    )


    # ========================================================
    # Quadrant titles
    # ========================================================

    for annotation in (
        fig.layout.annotations
    ):

        annotation.font = dict(

            size=20,

            color="#111827",
        )


    return fig


# ============================================================
# Uncertainty dashboard
# ============================================================

def make_uncertainty_dashboard(
    all_data,
):

    x_range = (
        get_global_x_range(
            all_data
        )
    )

    y_range = (
        get_global_uncertainty_y_range(
            all_data
        )
    )


    # Again: independent axes visually, linked with matches.
    fig = make_subplots(

        rows=2,
        cols=2,

        subplot_titles=[
            PAIR_TITLES[
                "Pythia_plus"
            ],
            PAIR_TITLES[
                "Pythia_minus"
            ],
            PAIR_TITLES[
                "Herwig_plus"
            ],
            PAIR_TITLES[
                "Herwig_minus"
            ],
        ],

        horizontal_spacing=0.10,

        vertical_spacing=0.18,
    )


    # ========================================================
    # Add uncertainty plots
    # ========================================================

    for index, pair_label in enumerate(
        PAIR_ORDER
    ):

        row, col = (
            SUBPLOT_POSITIONS[
                pair_label
            ]
        )


        add_uncertainty_subplot(

            fig=fig,

            data=all_data[
                pair_label
            ],

            row=row,
            col=col,

            show_legend=(
                index == 0
            ),
        )


    # ========================================================
    # Give EVERY quadrant its own axes + synchronise
    # ========================================================

    configure_subplot_axes(

        fig=fig,

        x_range=x_range,

        y_range=y_range,

        y_title=(
            "Absolute uncertainty"
        ),
    )


    # ========================================================
    # Layout
    # ========================================================

    fig.update_layout(

        title=dict(

            text=(

                "<b>"
                "Lepton pseudorapidity "
                "uncertainty decomposition"
                "</b>"

                "<br>"

                "<span style='font-size:16px;'>"

                "m<sub>T</sub><sup>W</sup> &gt; 40 GeV · Statistical, scale, PDF, shower "
                "and modelling contributions"

                "</span>"
            ),

            x=0.5,

            xanchor="center",

            y=0.98,
        ),


        template=(
            "plotly_white"
        ),


        width=1500,

        height=1100,


        margin=dict(

            l=100,

            r=75,

            t=150,

            b=90,
        ),


        hovermode=(
            "closest"
        ),


        dragmode=(
            "pan"
        ),


        legend=dict(

            title=dict(

                text=(
                    "<b>"
                    "Uncertainty source"
                    "</b>"
                )
            ),

            orientation="h",

            yanchor="bottom",

            y=1.035,

            xanchor="center",

            x=0.5,

            bgcolor=(
                "rgba(255,255,255,0.92)"
            ),

            bordercolor=(
                "rgba(100,100,100,0.20)"
            ),

            borderwidth=1,

            groupclick=(
                "togglegroup"
            ),
        ),


        font=dict(

            family=(
                "Arial, Helvetica, sans-serif"
            ),

            size=14,
        ),


        paper_bgcolor=(
            "#F8FAFC"
        ),


        plot_bgcolor=(
            "#FFFFFF"
        ),
    )


    # ========================================================
    # Quadrant titles
    # ========================================================

    for annotation in (
        fig.layout.annotations
    ):

        annotation.font = dict(

            size=20,

            color="#111827",
        )


    return fig


# ============================================================
# Plotly browser configuration
# ============================================================

def get_plotly_config(
    image_filename,
):

    return {

        "scrollZoom": True,

        "responsive": True,

        "displaylogo": False,

        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
        ],

        "toImageButtonOptions": {

            "format": "png",

            "filename": (
                image_filename
            ),

            "height": 1100,

            "width": 1500,

            "scale": 2,
        },
    }


# ============================================================
# Save both HTML dashboards
# ============================================================

def save_dashboards(
    correction_fig,
    uncertainty_fig,
):

    correction_fig.write_html(

        str(
            CORRECTION_HTML
        ),

        config=(
            get_plotly_config(
                "MTWcut_etalepton_correction_factor_2x2"
            )
        ),

        include_plotlyjs=True,

        full_html=True,

        auto_open=False,
    )


    uncertainty_fig.write_html(

        str(
            UNCERTAINTY_HTML
        ),

        config=(
            get_plotly_config(
                "MTWcut_etalepton_uncertainties_2x2"
            )
        ),

        include_plotlyjs=True,

        full_html=True,

        auto_open=False,
    )


# ============================================================
# Main
# ============================================================

def main():

    print()

    print(
        "=============================================="
    )

    print(
        " Interactive lepton pseudorapidity plotting: m_T^W > 40 GeV"
    )

    print(
        "=============================================="
    )

    print()


    print(
        "CSV input directory:"
    )

    print(
        f"  {INPUT_DIR}"
    )

    print()


    print(
        "Interactive output directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()


    # ========================================================
    # Check input directory
    # ========================================================

    if not INPUT_DIR.is_dir():

        raise RuntimeError(

            "Input directory does not exist:\n"

            f"  {INPUT_DIR}"
        )


    # ========================================================
    # Find eta CSV files
    # ========================================================

    csv_files = {

        parse_csv_filename(
            path
        ): path

        for path in INPUT_DIR.iterdir()

        if (
            path.is_file()

            and

            path.suffix.lower()
            == ".csv"

            and

            path.stem.endswith(
                f"_{OBSERVABLE}"
            )
        )
    }


    # ========================================================
    # Require all four
    # ========================================================

    missing_pairs = [

        pair_label

        for pair_label in PAIR_ORDER

        if pair_label not in csv_files
    ]


    if missing_pairs:

        raise RuntimeError(

            "Missing CSV files for:\n  "

            + "\n  ".join(
                missing_pairs
            )
        )


    print(
        "Found all four lepton "
        "pseudorapidity CSV files."
    )

    print()


    # ========================================================
    # Load data
    # ========================================================

    all_data = {}


    for pair_label in PAIR_ORDER:

        csv_path = (
            csv_files[
                pair_label
            ]
        )


        all_data[
            pair_label
        ] = load_csv_data(
            csv_path
        )


        print(
            f"Loaded: "
            f"{csv_path.name}"
        )


    # ========================================================
    # Build dashboards
    # ========================================================

    print()

    print(
        "Building correction-factor dashboard..."
    )


    correction_fig = (
        make_correction_dashboard(
            all_data
        )
    )


    print(
        "Building uncertainty dashboard..."
    )


    uncertainty_fig = (
        make_uncertainty_dashboard(
            all_data
        )
    )


    # ========================================================
    # Save
    # ========================================================

    save_dashboards(
        correction_fig,
        uncertainty_fig,
    )


    # ========================================================
    # Finished
    # ========================================================

    print()

    print(
        "=============================================="
    )

    print(
        " Finished"
    )

    print(
        "=============================================="
    )

    print()


    print(
        "Created correction-factor dashboard:"
    )

    print()

    print(
        f"  {CORRECTION_HTML}"
    )

    print()


    print(
        "Created uncertainty dashboard:"
    )

    print()

    print(
        f"  {UNCERTAINTY_HTML}"
    )

    print()


    print(
        "Axis configuration:"
    )

    print(
        "  - every quadrant has its own x-axis"
    )

    print(
        "  - every quadrant has its own y-axis"
    )

    print(
        "  - every quadrant has x-axis labels"
    )

    print(
        "  - every quadrant has y-axis labels"
    )

    print(
        "  - all four x axes remain synchronised"
    )

    print(
        "  - all four y axes remain synchronised"
    )

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()