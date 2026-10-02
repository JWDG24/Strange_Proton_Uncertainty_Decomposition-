#!/usr/bin/env python3

# ============================================================
# plot_even_odd_interactive.py
#
# Interactive Plotly visualisation for the odd/even statistical
# uncertainty study.
#
# Expected location:
#
#   uncertainty_decomposition/
#   └── lepton_pseudorapidity_work/
#       └── even_odd_Uncertainties/
#           ├── even_odd_uncertainty.py
#           └── plot_even_odd_interactive.py
#
# Expected CSV inputs:
#
#   even_odd_Uncertainties/
#   └── outputs/
#       └── csv/
#           ├── even_odd_Pythia_plus.csv
#           ├── even_odd_Pythia_minus.csv
#           ├── even_odd_Herwig_plus.csv
#           └── even_odd_Herwig_minus.csv
#
# Output:
#
#   even_odd_Uncertainties/
#   └── outputs/
#       └── interactive_plots/
#           └── even_odd_uncertainty_interactive.html
#
# The HTML is standalone and opens in a web browser.
# ============================================================

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

CSV_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "csv"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "interactive_plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_HTML = (
    OUTPUT_DIR
    / "even_odd_uncertainty_interactive.html"
)


# ============================================================
# Pair definitions
# ============================================================

PAIR_FILES = {
    "Pythia_plus": (
        "Pythia · W⁺",
        CSV_DIR / "even_odd_Pythia_plus.csv",
    ),
    "Pythia_minus": (
        "Pythia · W⁻",
        CSV_DIR / "even_odd_Pythia_minus.csv",
    ),
    "Herwig_plus": (
        "Herwig · W⁺",
        CSV_DIR / "even_odd_Herwig_plus.csv",
    ),
    "Herwig_minus": (
        "Herwig · W⁻",
        CSV_DIR / "even_odd_Herwig_minus.csv",
    ),
}


# ============================================================
# Required CSV columns
# ============================================================

REQUIRED_COLUMNS = {
    "bin_low_edge",
    "bin_up_edge",
    "C_all",
    "C_odd",
    "C_even",
    "abs_odd_even_difference",
    "split_stat_uncertainty_half_difference",
    "relative_split_stat_uncertainty",
    "all_ratio_ROOT_stat_error",
    "odd_ratio_ROOT_stat_error",
    "even_ratio_ROOT_stat_error",
}


# ============================================================
# Helpers
# ============================================================

def load_pair_csv(path):
    """
    Load one odd/even CSV and add bin-centre / half-width columns.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing CSV file:\n  {path}\n\n"
            "Run even_odd_uncertainty.py first."
        )

    dataframe = pd.read_csv(
        path
    )

    missing = (
        REQUIRED_COLUMNS
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"CSV file is missing required columns:\n"
            f"  {path}\n"
            f"Missing: {sorted(missing)}"
        )

    dataframe[
        "eta_centre"
    ] = (
        0.5
        * (
            dataframe[
                "bin_low_edge"
            ]
            + dataframe[
                "bin_up_edge"
            ]
        )
    )

    dataframe[
        "eta_half_width"
    ] = (
        0.5
        * (
            dataframe[
                "bin_up_edge"
            ]
            - dataframe[
                "bin_low_edge"
            ]
        )
    )

    dataframe[
        "relative_split_percent"
    ] = (
        100.0
        * dataframe[
            "relative_split_stat_uncertainty"
        ]
    )

    dataframe[
        "relative_root_stat_percent"
    ] = (
        dataframe.apply(
            lambda row: (
                100.0
                * abs(
                    row[
                        "all_ratio_ROOT_stat_error"
                    ]
                )
                / abs(
                    row[
                        "C_all"
                    ]
                )
                if row[
                    "C_all"
                ] != 0.0
                else 0.0
            ),
            axis=1,
        )
    )

    return dataframe


def hover_text(
    dataframe,
    correction_column,
    stat_error_column=None,
):
    """
    Construct readable per-bin hover labels.
    """

    hover = []

    for _, row in dataframe.iterrows():

        text = (
            f"|ηℓ| bin: "
            f"{row['bin_low_edge']:.2f}–"
            f"{row['bin_up_edge']:.2f}"
            f"<br>Correction: "
            f"{row[correction_column]:.6f}"
        )

        if stat_error_column is not None:

            text += (
                f"<br>ROOT stat. error: "
                f"{row[stat_error_column]:.6f}"
            )

        hover.append(
            text
        )

    return hover


def add_pair_traces(
    figure,
    dataframe,
    pair_label,
    visible,
):
    """
    Add the full trace set for one generator / W-charge pair.

    Three panels:
      1. correction factors: all, odd, even
      2. absolute statistical comparison:
           split half-difference vs existing ROOT propagated error
      3. relative statistical comparison in percent
    """

    x = dataframe[
        "eta_centre"
    ]

    xerr = dataframe[
        "eta_half_width"
    ]

    # --------------------------------------------------------
    # Row 1: correction factors
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=x,
            y=dataframe[
                "C_all"
            ],
            mode="lines+markers",
            name="All events",
            legendgroup="all",
            visible=visible,
            error_x=dict(
                type="data",
                array=xerr,
                visible=True,
            ),
            error_y=dict(
                type="data",
                array=dataframe[
                    "all_ratio_ROOT_stat_error"
                ],
                visible=True,
            ),
            hovertext=hover_text(
                dataframe,
                "C_all",
                "all_ratio_ROOT_stat_error",
            ),
            hovertemplate=(
                "%{hovertext}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=x,
            y=dataframe[
                "C_odd"
            ],
            mode="lines+markers",
            name="Odd entries",
            legendgroup="odd",
            visible=visible,
            error_x=dict(
                type="data",
                array=xerr,
                visible=True,
            ),
            error_y=dict(
                type="data",
                array=dataframe[
                    "odd_ratio_ROOT_stat_error"
                ],
                visible=True,
            ),
            hovertext=hover_text(
                dataframe,
                "C_odd",
                "odd_ratio_ROOT_stat_error",
            ),
            hovertemplate=(
                "%{hovertext}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=x,
            y=dataframe[
                "C_even"
            ],
            mode="lines+markers",
            name="Even entries",
            legendgroup="even",
            visible=visible,
            error_x=dict(
                type="data",
                array=xerr,
                visible=True,
            ),
            error_y=dict(
                type="data",
                array=dataframe[
                    "even_ratio_ROOT_stat_error"
                ],
                visible=True,
            ),
            hovertext=hover_text(
                dataframe,
                "C_even",
                "even_ratio_ROOT_stat_error",
            ),
            hovertemplate=(
                "%{hovertext}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # Row 2: absolute uncertainties
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=x,
            y=dataframe[
                "split_stat_uncertainty_half_difference"
            ],
            mode="lines+markers",
            name="Odd/even split: |Codd − Ceven| / 2",
            legendgroup="split",
            visible=visible,
            hovertemplate=(
                "|ηℓ| = %{x:.3f}"
                "<br>Split uncertainty = %{y:.6f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=x,
            y=dataframe[
                "all_ratio_ROOT_stat_error"
            ],
            mode="lines+markers",
            name="Existing ROOT propagated stat. error",
            legendgroup="rootstat",
            visible=visible,
            hovertemplate=(
                "|ηℓ| = %{x:.3f}"
                "<br>ROOT stat. error = %{y:.6f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=x,
            y=dataframe[
                "abs_odd_even_difference"
            ],
            mode="lines",
            name="Raw |Codd − Ceven|",
            legendgroup="rawdifference",
            visible=visible,
            line=dict(
                dash="dot",
            ),
            hovertemplate=(
                "|ηℓ| = %{x:.3f}"
                "<br>|Codd − Ceven| = %{y:.6f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # --------------------------------------------------------
    # Row 3: relative uncertainties
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=x,
            y=dataframe[
                "relative_split_percent"
            ],
            mode="lines+markers",
            name="Relative split uncertainty",
            legendgroup="relative_split",
            visible=visible,
            hovertemplate=(
                "|ηℓ| = %{x:.3f}"
                "<br>Relative split uncertainty = %{y:.3f}%"
                "<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=x,
            y=dataframe[
                "relative_root_stat_percent"
            ],
            mode="lines+markers",
            name="Relative ROOT stat. error",
            legendgroup="relative_root",
            visible=visible,
            hovertemplate=(
                "|ηℓ| = %{x:.3f}"
                "<br>Relative ROOT stat. error = %{y:.3f}%"
                "<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )


# ============================================================
# Build interactive figure
# ============================================================

def build_figure():
    """
    Create one standalone interactive Plotly figure.

    A dropdown switches between:
        Pythia W+
        Pythia W-
        Herwig W+
        Herwig W-

    The legend remains interactive, so individual curves can be
    hidden/shown by clicking their names.
    """

    pair_data = {}

    for pair_key, (
        display_label,
        csv_path,
    ) in PAIR_FILES.items():

        pair_data[
            pair_key
        ] = {
            "label": display_label,
            "data": load_pair_csv(
                csv_path
            ),
        }

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Correction factors",
            "Absolute statistical comparison",
            "Relative statistical comparison",
        ),
        row_heights=[
            0.46,
            0.30,
            0.24,
        ],
    )

    pair_keys = list(
        PAIR_FILES.keys()
    )

    traces_per_pair = 8

    for pair_index, pair_key in enumerate(
        pair_keys
    ):

        add_pair_traces(
            figure=figure,
            dataframe=pair_data[
                pair_key
            ][
                "data"
            ],
            pair_label=pair_data[
                pair_key
            ][
                "label"
            ],
            visible=(
                pair_index == 0
            ),
        )

    # --------------------------------------------------------
    # Dropdown
    # --------------------------------------------------------

    buttons = []

    total_traces = (
        traces_per_pair
        * len(
            pair_keys
        )
    )

    for pair_index, pair_key in enumerate(
        pair_keys
    ):

        visible_mask = [
            False
        ] * total_traces

        first_trace = (
            pair_index
            * traces_per_pair
        )

        last_trace = (
            first_trace
            + traces_per_pair
        )

        for trace_index in range(
            first_trace,
            last_trace,
        ):
            visible_mask[
                trace_index
            ] = True

        display_label = (
            pair_data[
                pair_key
            ][
                "label"
            ]
        )

        buttons.append(
            dict(
                label=display_label,
                method="update",
                args=[
                    {
                        "visible": visible_mask,
                    },
                    {
                        "title.text": (
                            "Lepton pseudorapidity odd/even "
                            "statistical uncertainty"
                            f"<br><sup>{display_label} · "
                            "m<sub>T</sub><sup>W</sup> &gt; 40 GeV"
                            "</sup>"
                        ),
                    },
                ],
            )
        )

    first_label = (
        pair_data[
            pair_keys[0]
        ][
            "label"
        ]
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    figure.update_layout(
        title=dict(
            text=(
                "Lepton pseudorapidity odd/even statistical uncertainty"
                f"<br><sup>{first_label} · "
                "m<sub>T</sub><sup>W</sup> &gt; 40 GeV"
                "</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=1050,
        width=1150,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=90,
            r=40,
            t=170,
            b=80,
        ),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.16,
                yanchor="top",
            )
        ],
        annotations=[
            *figure.layout.annotations,
            dict(
                text="Generator / charge:",
                x=0.01,
                y=1.205,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                font=dict(
                    size=13,
                ),
            ),
        ],
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    figure.update_yaxes(
        title_text="Correction factor",
        row=1,
        col=1,
        showgrid=True,
        zeroline=False,
    )

    figure.update_yaxes(
        title_text="Absolute uncertainty",
        rangemode="tozero",
        row=2,
        col=1,
        showgrid=True,
        zeroline=True,
    )

    figure.update_yaxes(
        title_text="Relative uncertainty (%)",
        rangemode="tozero",
        row=3,
        col=1,
        showgrid=True,
        zeroline=True,
    )

    figure.update_xaxes(
        title_text="Lepton |η|",
        range=[
            ETA_MIN,
            ETA_MAX,
        ],
        row=3,
        col=1,
        showgrid=True,
    )

    # Keep the upper panels on exactly the same eta domain.
    figure.update_xaxes(
        range=[
            ETA_MIN,
            ETA_MAX,
        ],
        row=1,
        col=1,
        showgrid=True,
    )

    figure.update_xaxes(
        range=[
            ETA_MIN,
            ETA_MAX,
        ],
        row=2,
        col=1,
        showgrid=True,
    )

    return figure


# ============================================================
# Eta plotting range
# ============================================================

ETA_MIN = 0.0
ETA_MAX = 2.5


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "============================================================"
    )
    print(
        " Interactive odd/even uncertainty plot"
    )
    print(
        "============================================================"
    )
    print()

    print(
        f"Reading CSVs from:\n  {CSV_DIR}"
    )
    print()

    figure = build_figure()

    figure.write_html(
        str(
            OUTPUT_HTML
        ),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )

    print(
        "Saved interactive plot:"
    )
    print(
        f"  {OUTPUT_HTML}"
    )
    print()

    print(
        "Open the HTML file in your browser."
    )
    print()

    print(
        "Interactive controls:"
    )
    print(
        "  - dropdown: switch Pythia/Herwig and W+/W-"
    )
    print(
        "  - click legend entries: hide/show individual curves"
    )
    print(
        "  - hover: inspect exact bin values"
    )
    print(
        "  - drag: zoom"
    )
    print(
        "  - double-click plot: reset zoom"
    )
    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
