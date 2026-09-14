#!/usr/bin/env python3

"""
plot_relative_uncertainties_interactive.py

Interactive visualisation for supervisor part (a).

Expected location:

uncertainty_decomposition/
└── lepton_pseudorapidity_work/
    └── weights_relative_uncertainties/
        ├── build_relative_uncertainties.py
        └── plot_relative_uncertainties_interactive.py

This script reads the CSV files produced by
build_relative_uncertainties.py and creates two interactive figures:

1. Relative uncertainty decomposition
   A 2x2 layout showing:
       Pythia W+
       Pythia W-
       Herwig W+
       Herwig W-

   Each panel contains:
       scale
       PDF
       shower
       model
       total systematic

2. Individual relative weight responses
   Another 2x2 layout with the same four channels.
   A dropdown selects:
       scale
       PDF
       shower
       model

   For the selected source, every individual correlated weight
   variation is shown as

       100 * (C_weight - C_weight0) / C_weight0

   together with the final envelope or RMS uncertainty for that
   source.

The script writes:
    outputs/interactive/
        relative_uncertainty_decomposition.html
        relative_weight_responses.html
        relative_uncertainties_dashboard.html

The dashboard contains both figures behind two tabs.

Run from the project root with:

python3 lepton_pseudorapidity_work/weights_relative_uncertainties/plot_relative_uncertainties_interactive.py

Optional:
    add --open to open the dashboard automatically in the normal
    Windows browser when running inside WSL.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
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

INTERACTIVE_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "interactive"
)

INTERACTIVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_HTML = (
    INTERACTIVE_DIR
    / "relative_uncertainty_decomposition.html"
)

WEIGHTS_HTML = (
    INTERACTIVE_DIR
    / "relative_weight_responses.html"
)

DASHBOARD_HTML = (
    INTERACTIVE_DIR
    / "relative_uncertainties_dashboard.html"
)


# ============================================================
# Analysis channels
# ============================================================

PAIR_DEFINITIONS = {
    "Pythia_plus": {
        "title": "Pythia · W⁺",
        "row": 1,
        "col": 1,
    },
    "Pythia_minus": {
        "title": "Pythia · W⁻",
        "row": 1,
        "col": 2,
    },
    "Herwig_plus": {
        "title": "Herwig · W⁺",
        "row": 2,
        "col": 1,
    },
    "Herwig_minus": {
        "title": "Herwig · W⁻",
        "row": 2,
        "col": 2,
    },
}


# ============================================================
# Uncertainty source configuration
# ============================================================

SOURCES = {
    "scale": {
        "label": "Scale",
        "summary_column": "scale_unc_percent",
        "method": "Envelope",
        "colour": "#00A67D",
    },
    "pdf": {
        "label": "PDF",
        "summary_column": "pdf_unc_percent",
        "method": "RMS",
        "colour": "#8B5CF6",
    },
    "shower": {
        "label": "Shower",
        "summary_column": "shower_unc_percent",
        "method": "Envelope",
        "colour": "#F59E0B",
    },
    "model": {
        "label": "Model",
        "summary_column": "model_unc_percent",
        "method": "Envelope",
        "colour": "#0891B2",
    },
}

TOTAL_SOURCE = {
    "label": "Total systematic",
    "summary_column": "total_systematic_unc_percent",
    "colour": "#222222",
}


# ============================================================
# Input helpers
# ============================================================

SUMMARY_REQUIRED = {
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "nominal_correction_weight0",
    "scale_unc_percent",
    "pdf_unc_percent",
    "shower_unc_percent",
    "model_unc_percent",
    "total_systematic_unc_percent",
}

PER_WEIGHT_REQUIRED = {
    "weight_index",
    "weight_name",
    "category",
    "bin",
    "bin_low_edge",
    "bin_up_edge",
    "C_nominal_weight0",
    "C_weight",
    "C_weight_over_C_nominal",
    "signed_relative_shift",
    "absolute_relative_shift",
    "signed_relative_shift_percent",
}


def add_eta_columns(dataframe):
    """
    Add eta-bin centre and half-width.
    """

    dataframe = dataframe.copy()

    dataframe["eta_centre"] = (
        0.5
        * (
            dataframe["bin_low_edge"]
            + dataframe["bin_up_edge"]
        )
    )

    dataframe["eta_half_width"] = (
        0.5
        * (
            dataframe["bin_up_edge"]
            - dataframe["bin_low_edge"]
        )
    )

    return dataframe


def load_csv_checked(
    path,
    required_columns,
):
    """
    Load a CSV and verify that the columns needed by the plot exist.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Required CSV does not exist:\n"
            f"  {path}\n\n"
            "Run build_relative_uncertainties.py first."
        )

    dataframe = pd.read_csv(
        path
    )

    missing = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"CSV is missing required columns:\n"
            f"  {path}\n"
            f"Missing: {sorted(missing)}"
        )

    return add_eta_columns(
        dataframe
    )


def load_all_data():
    """
    Read the summary and per-weight CSV for all four channels.
    """

    all_data = {}

    for pair_label in PAIR_DEFINITIONS:

        summary_path = (
            CSV_DIR
            / f"{pair_label}_relative_uncertainties.csv"
        )

        per_weight_path = (
            CSV_DIR
            / f"{pair_label}_per_weight_relative_shifts.csv"
        )

        all_data[
            pair_label
        ] = {
            "summary": load_csv_checked(
                summary_path,
                SUMMARY_REQUIRED,
            ),
            "weights": load_csv_checked(
                per_weight_path,
                PER_WEIGHT_REQUIRED,
            ),
        }

    return all_data


# ============================================================
# Figure 1
# Relative uncertainty decomposition
# ============================================================

def build_summary_figure(
    all_data,
):
    """
    Create a 2x2 interactive decomposition for all four channels.
    """

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            PAIR_DEFINITIONS[
                pair_label
            ][
                "title"
            ]
            for pair_label
            in PAIR_DEFINITIONS
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    for pair_index, (
        pair_label,
        pair_info,
    ) in enumerate(
        PAIR_DEFINITIONS.items()
    ):

        dataframe = all_data[
            pair_label
        ][
            "summary"
        ]

        row = pair_info[
            "row"
        ]

        col = pair_info[
            "col"
        ]

        customdata = list(
            zip(
                dataframe[
                    "bin_low_edge"
                ],
                dataframe[
                    "bin_up_edge"
                ],
            )
        )

        for source_key, source_info in SOURCES.items():

            figure.add_trace(
                go.Scatter(
                    x=dataframe[
                        "eta_centre"
                    ],
                    y=dataframe[
                        source_info[
                            "summary_column"
                        ]
                    ],
                    mode="lines+markers",
                    name=source_info[
                        "label"
                    ],
                    legendgroup=source_key,
                    showlegend=(
                        pair_index == 0
                    ),
                    line=dict(
                        color=source_info[
                            "colour"
                        ],
                        width=2,
                    ),
                    marker=dict(
                        size=7,
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        "|ηℓ| bin: "
                        "%{customdata[0]:.2f}–"
                        "%{customdata[1]:.2f}"
                        "<br>"
                        + source_info[
                            "label"
                        ]
                        + ": %{y:.4f}%"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        figure.add_trace(
            go.Scatter(
                x=dataframe[
                    "eta_centre"
                ],
                y=dataframe[
                    TOTAL_SOURCE[
                        "summary_column"
                    ]
                ],
                mode="lines+markers",
                name=TOTAL_SOURCE[
                    "label"
                ],
                legendgroup="total",
                showlegend=(
                    pair_index == 0
                ),
                line=dict(
                    color=TOTAL_SOURCE[
                        "colour"
                    ],
                    width=3,
                ),
                marker=dict(
                    size=8,
                    symbol="circle-open",
                ),
                customdata=customdata,
                hovertemplate=(
                    "|ηℓ| bin: "
                    "%{customdata[0]:.2f}–"
                    "%{customdata[1]:.2f}"
                    "<br>Total systematic: %{y:.4f}%"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    figure.update_layout(
        title=dict(
            text=(
                "Relative systematic uncertainty decomposition"
                "<br><sup>"
                "Systematic weights relative to nominal weight 0 · "
                "m<sub>T</sub><sup>W</sup> &gt; 40 GeV"
                "</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=900,
        width=1250,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=80,
            r=40,
            t=150,
            b=70,
        ),
    )

    for pair_info in PAIR_DEFINITIONS.values():

        row = pair_info[
            "row"
        ]

        col = pair_info[
            "col"
        ]

        figure.update_xaxes(
            title_text="Lepton |η|",
            range=[
                0.0,
                2.5,
            ],
            showgrid=True,
            row=row,
            col=col,
        )

        figure.update_yaxes(
            title_text="Relative uncertainty (%)",
            rangemode="tozero",
            showgrid=True,
            row=row,
            col=col,
        )

    return figure


# ============================================================
# Figure 2
# Individual correlated weight responses
# ============================================================

def build_weight_response_figure(
    all_data,
):
    """
    Create a 2x2 figure with a dropdown selecting the systematic
    source.

    Each selected source shows all individual weight responses:

        100 * (C_weight - C_weight0) / C_weight0

    The final envelope or RMS uncertainty is overlaid as a bold
    positive and negative boundary.
    """

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            PAIR_DEFINITIONS[
                pair_label
            ][
                "title"
            ]
            for pair_label
            in PAIR_DEFINITIONS
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )

    source_trace_indices = {
        source_key: []
        for source_key
        in SOURCES
    }

    initial_source = "scale"

    for source_key, source_info in SOURCES.items():

        for pair_label, pair_info in PAIR_DEFINITIONS.items():

            dataframe = all_data[
                pair_label
            ][
                "weights"
            ]

            summary = all_data[
                pair_label
            ][
                "summary"
            ]

            subset = dataframe[
                dataframe[
                    "category"
                ] == source_key
            ].copy()

            row = pair_info[
                "row"
            ]

            col = pair_info[
                "col"
            ]

            weight_indices = sorted(
                subset[
                    "weight_index"
                ].unique()
            )

            # ------------------------------------------------
            # Every individual correlated weight variation
            # ------------------------------------------------

            for weight_index in weight_indices:

                weight_data = subset[
                    subset[
                        "weight_index"
                    ] == weight_index
                ].sort_values(
                    "bin"
                )

                if weight_data.empty:
                    continue

                weight_name = str(
                    weight_data.iloc[
                        0
                    ][
                        "weight_name"
                    ]
                )

                trace_index = len(
                    figure.data
                )

                source_trace_indices[
                    source_key
                ].append(
                    trace_index
                )

                figure.add_trace(
                    go.Scatter(
                        x=weight_data[
                            "eta_centre"
                        ],
                        y=weight_data[
                            "signed_relative_shift_percent"
                        ],
                        mode="lines+markers",
                        visible=(
                            source_key
                            == initial_source
                        ),
                        showlegend=False,
                        line=dict(
                            color=source_info[
                                "colour"
                            ],
                            width=1,
                        ),
                        marker=dict(
                            color=source_info[
                                "colour"
                            ],
                            size=4,
                        ),
                        opacity=0.28,
                        customdata=list(
                            zip(
                                weight_data[
                                    "bin_low_edge"
                                ],
                                weight_data[
                                    "bin_up_edge"
                                ],
                            )
                        ),
                        hovertemplate=(
                            f"Weight {weight_index}"
                            f"<br>{weight_name}"
                            "<br>|ηℓ| bin: "
                            "%{customdata[0]:.2f}–"
                            "%{customdata[1]:.2f}"
                            "<br>Relative shift: %{y:.4f}%"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

            # ------------------------------------------------
            # Final source uncertainty
            #
            # For envelope sources this is a positive/negative
            # boundary. For PDF this shows ±RMS.
            # ------------------------------------------------

            source_uncertainty = summary[
                source_info[
                    "summary_column"
                ]
            ]

            customdata = list(
                zip(
                    summary[
                        "bin_low_edge"
                    ],
                    summary[
                        "bin_up_edge"
                    ],
                )
            )

            for sign, boundary_name in (
                (
                    1.0,
                    "+ uncertainty",
                ),
                (
                    -1.0,
                    "− uncertainty",
                ),
            ):

                trace_index = len(
                    figure.data
                )

                source_trace_indices[
                    source_key
                ].append(
                    trace_index
                )

                figure.add_trace(
                    go.Scatter(
                        x=summary[
                            "eta_centre"
                        ],
                        y=(
                            sign
                            * source_uncertainty
                        ),
                        mode="lines",
                        visible=(
                            source_key
                            == initial_source
                        ),
                        name=(
                            f"{source_info['method']} boundary"
                        ),
                        legendgroup=(
                            f"{source_key}_boundary"
                        ),
                        showlegend=(
                            pair_label
                            == "Pythia_plus"
                            and sign > 0
                        ),
                        line=dict(
                            color="#111111",
                            width=3,
                            dash="dash",
                        ),
                        customdata=customdata,
                        hovertemplate=(
                            "|ηℓ| bin: "
                            "%{customdata[0]:.2f}–"
                            "%{customdata[1]:.2f}"
                            "<br>"
                            + source_info[
                                "method"
                            ]
                            + " boundary: %{y:.4f}%"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

    # --------------------------------------------------------
    # Dropdown visibility masks
    # --------------------------------------------------------

    buttons = []

    total_traces = len(
        figure.data
    )

    for source_key, source_info in SOURCES.items():

        visible = [
            False
        ] * total_traces

        for trace_index in source_trace_indices[
            source_key
        ]:
            visible[
                trace_index
            ] = True

        buttons.append(
            dict(
                label=source_info[
                    "label"
                ],
                method="update",
                args=[
                    {
                        "visible": visible,
                    },
                    {
                        "title.text": (
                            "Individual relative weight responses"
                            "<br><sup>"
                            + source_info[
                                "label"
                            ]
                            + " variations relative to nominal weight 0 · "
                            + source_info[
                                "method"
                            ]
                            + " uncertainty · "
                            "m<sub>T</sub><sup>W</sup> &gt; 40 GeV"
                            "</sup>"
                        ),
                    },
                ],
            )
        )

    figure.update_layout(
        title=dict(
            text=(
                "Individual relative weight responses"
                "<br><sup>"
                "Scale variations relative to nominal weight 0 · "
                "Envelope uncertainty · "
                "m<sub>T</sub><sup>W</sup> &gt; 40 GeV"
                "</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        height=900,
        width=1250,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
        annotations=[
            *figure.layout.annotations,
            dict(
                text="Uncertainty source:",
                x=0.01,
                y=1.20,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
            ),
        ],
        margin=dict(
            l=80,
            r=40,
            t=165,
            b=70,
        ),
    )

    for pair_info in PAIR_DEFINITIONS.values():

        row = pair_info[
            "row"
        ]

        col = pair_info[
            "col"
        ]

        figure.update_xaxes(
            title_text="Lepton |η|",
            range=[
                0.0,
                2.5,
            ],
            showgrid=True,
            row=row,
            col=col,
        )

        figure.update_yaxes(
            title_text=(
                "(C_weight − C_weight0) / C_weight0 (%)"
            ),
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="#777777",
            showgrid=True,
            row=row,
            col=col,
        )

    return figure


# ============================================================
# Dashboard HTML
# ============================================================

def write_dashboard(
    summary_figure,
    weight_figure,
):
    """
    Write one self-contained HTML dashboard with two tabs.
    """

    plotly_js = get_plotlyjs()

    summary_div = pio.to_html(
        summary_figure,
        full_html=False,
        include_plotlyjs=False,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )

    weight_div = pio.to_html(
        weight_figure,
        full_html=False,
        include_plotlyjs=False,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relative uncertainty analysis</title>
<script>{plotly_js}</script>
<style>
    body {{
        margin: 0;
        padding: 24px;
        font-family: Arial, Helvetica, sans-serif;
        background: #ffffff;
        color: #172033;
    }}

    .header {{
        max-width: 1280px;
        margin: 0 auto 18px auto;
    }}

    .header h1 {{
        margin: 0 0 6px 0;
        font-size: 24px;
        font-weight: 600;
    }}

    .header p {{
        margin: 0;
        color: #5f6b7a;
        line-height: 1.45;
    }}

    .tabs {{
        max-width: 1280px;
        margin: 0 auto 14px auto;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }}

    .tab-button {{
        border: 1px solid #cbd5e1;
        background: #ffffff;
        color: #26364a;
        border-radius: 6px;
        padding: 10px 14px;
        cursor: pointer;
        font-size: 14px;
    }}

    .tab-button.active {{
        background: #eef3f8;
        border-color: #718096;
        font-weight: 600;
    }}

    .panel {{
        display: none;
        max-width: 1280px;
        margin: 0 auto;
    }}

    .panel.active {{
        display: block;
    }}

    .note {{
        max-width: 1280px;
        margin: 18px auto 0 auto;
        padding: 12px 14px;
        background: #f7f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        color: #4a5568;
        line-height: 1.45;
        font-size: 14px;
    }}
</style>
</head>
<body>

<div class="header">
    <h1>Relative systematic uncertainty analysis</h1>
    <p>
        Old weighted samples, with every systematic variation expressed
        relative to nominal weight 0 from the same event sample.
    </p>
</div>

<div class="tabs">
    <button
        id="summary-button"
        class="tab-button active"
        type="button"
    >
        Relative uncertainty decomposition
    </button>

    <button
        id="weights-button"
        class="tab-button"
        type="button"
    >
        Individual weight responses
    </button>
</div>

<div
    id="summary-panel"
    class="panel active"
>
    {summary_div}
</div>

<div
    id="weights-panel"
    class="panel"
>
    {weight_div}
</div>

<div class="note">
    The first tab shows the final scale, PDF, shower, model and total
    systematic uncertainties in percent for all four generator and charge
    channels. The second tab shows the individual signed variations
    (C_weight - C_weight0) / C_weight0. Use its dropdown to switch between
    scale, PDF, shower and model variations. The black dashed curves show
    the final envelope or RMS boundary used for that source.
</div>

<script>
(function () {{
    const summaryButton = document.getElementById("summary-button");
    const weightsButton = document.getElementById("weights-button");
    const summaryPanel = document.getElementById("summary-panel");
    const weightsPanel = document.getElementById("weights-panel");

    function showPanel(which) {{
        const summaryActive = which === "summary";

        summaryButton.classList.toggle(
            "active",
            summaryActive
        );

        weightsButton.classList.toggle(
            "active",
            !summaryActive
        );

        summaryPanel.classList.toggle(
            "active",
            summaryActive
        );

        weightsPanel.classList.toggle(
            "active",
            !summaryActive
        );

        window.setTimeout(function () {{
            const visiblePanel = (
                summaryActive
                ? summaryPanel
                : weightsPanel
            );

            const plots = visiblePanel.querySelectorAll(
                ".plotly-graph-div"
            );

            plots.forEach(function (plot) {{
                if (
                    window.Plotly
                    && Plotly.Plots
                ) {{
                    Plotly.Plots.resize(plot);
                }}
            }});
        }}, 30);
    }}

    summaryButton.addEventListener(
        "click",
        function () {{
            showPanel("summary");
        }}
    );

    weightsButton.addEventListener(
        "click",
        function () {{
            showPanel("weights");
        }}
    );
}})();
</script>

</body>
</html>
"""

    DASHBOARD_HTML.write_text(
        html,
        encoding="utf-8",
    )


# ============================================================
# Optional WSL browser opening
# ============================================================

def open_in_windows_browser(
    path,
):
    """
    Open an HTML file in the default Windows browser from WSL.
    """

    try:
        result = subprocess.run(
            [
                "wslpath",
                "-w",
                str(
                    path
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        windows_path = (
            result.stdout.strip()
        )

        subprocess.run(
            [
                "cmd.exe",
                "/C",
                "start",
                "",
                windows_path,
            ],
            check=True,
        )

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ) as exc:

        print()
        print(
            "Could not automatically open the Windows browser:"
        )
        print(
            f"  {exc}"
        )
        print()
        print(
            "Open this file manually:"
        )
        print(
            f"  {path}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot the relative systematic uncertainty analysis."
        )
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help=(
            "Open the combined dashboard in the default "
            "Windows browser after creating it."
        ),
    )

    args = parser.parse_args()

    print()
    print(
        "============================================================"
    )
    print(
        " Interactive relative uncertainty plots"
    )
    print(
        "============================================================"
    )
    print()

    print(
        f"Reading CSV files from:\n  {CSV_DIR}"
    )
    print()

    all_data = load_all_data()

    print(
        "Building 2x2 relative uncertainty decomposition..."
    )

    summary_figure = build_summary_figure(
        all_data
    )

    print(
        "Building 2x2 individual weight-response plot..."
    )

    weight_figure = build_weight_response_figure(
        all_data
    )

    # Standalone figure 1.
    summary_figure.write_html(
        str(
            SUMMARY_HTML
        ),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )

    # Standalone figure 2.
    weight_figure.write_html(
        str(
            WEIGHTS_HTML
        ),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )

    # Combined tabbed dashboard.
    write_dashboard(
        summary_figure,
        weight_figure,
    )

    print()
    print(
        "Created:"
    )
    print(
        f"  {SUMMARY_HTML}"
    )
    print(
        f"  {WEIGHTS_HTML}"
    )
    print(
        f"  {DASHBOARD_HTML}"
    )
    print()

    print(
        "The combined dashboard is the easiest one to use."
    )
    print()

    print(
        "Interactive controls:"
    )
    print(
        "  • click legend entries to hide/show uncertainty sources"
    )
    print(
        "  • hover over points for exact values"
    )
    print(
        "  • drag to zoom"
    )
    print(
        "  • double-click to reset"
    )
    print(
        "  • use the dropdown on the weight-response plot "
        "to switch source"
    )
    print()

    if args.open:

        print(
            "Opening dashboard in the Windows browser..."
        )

        open_in_windows_browser(
            DASHBOARD_HTML
        )


if __name__ == "__main__":
    main()
