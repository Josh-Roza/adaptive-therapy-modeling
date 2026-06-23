import csv
import atexit
import pandas
import matplotlib.pyplot as plt
import numpy as np
import colorama
import re
from pathlib import Path
from matplotlib.lines import Line2D

colorama.init(autoreset=True)

_PLOT_HOLD_REGISTERED = False


def _ensure_plot_hold_on_exit() -> None:
    """Register a one-time exit hook that keeps figures open at script end."""
    global _PLOT_HOLD_REGISTERED
    if not _PLOT_HOLD_REGISTERED:
        atexit.register(keep_plots_open)
        _PLOT_HOLD_REGISTERED = True

def _expand_parameter_values(parameters):
    """Normalize parameter input into an explicit list of values.
    Supported:
    - Explicit list/array of values, e.g. [0, 10, 20]
    - Range triple [start, step, max], e.g. [0, 10, 200]
    """
    if parameters is None:
        return []

    # Accept numpy arrays, pandas series, etc.
    try:
        values = list(parameters)
    except TypeError:
        return [parameters]

    if len(values) == 3:
        start, step, max_value = values
        if isinstance(step, (int, float)) and step != 0 and isinstance(max_value, (int, float)):
            # Inclusive range: start, start+step, ... <= max_value
            count = int((max_value - start) / step)
            return [start + step * i for i in range(0, count + 1)]

    return values


def _compute_drug_administered_frame(df: pandas.DataFrame) -> pandas.Series:
    """Compute per-run drug administered using BehaviorSpace columns.

    Uses the user's requested formula:
        drugAdministered = drugMolecule * ([step] - startTime) / timeInterval

    If required columns are missing, raises KeyError.
    If timeInterval is 0, returns NaN for those rows.
    """
    required = {"drugMolecule", "[step]", "startTime", "timeInterval"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns for drugAdministered: {sorted(missing)}")

    # Convert to numeric defensively; BehaviorSpace outputs are often string-typed.
    drug_molecule = pandas.to_numeric(df["drugMolecule"], errors="coerce")
    step = pandas.to_numeric(df["[step]"], errors="coerce")
    start_time = pandas.to_numeric(df["startTime"], errors="coerce")
    time_interval = pandas.to_numeric(df["timeInterval"], errors="coerce")

    denom = time_interval.replace({0: np.nan})
    return drug_molecule * ((step - start_time) / denom)


def _dot_color_from_counts(orange_mean, black_mean) -> str:
    """Return a color string based on orange/black mean counts.

    Requested rule:
    - If orange+black == 0 -> red
    - If orange+black > 6600 -> orange or black (whichever is more prevalent)
    - If 0 < orange+black <= 6600 -> darker orange or grey (whichever is more prevalent)

    Deep pink (vessel invaded) is handled by callers as an override.
    """
    try:
        if pandas.isna(orange_mean) or pandas.isna(black_mean):
            return "red"
    except Exception:
        return "red"

    try:
        orange_val = float(orange_mean)
        black_val = float(black_mean)
    except Exception:
        return "red"

    total = orange_val + black_val
    if total == 0:
        return "red"
    if total > 6600:
        return "black" if black_val > orange_val else "chocolate"
    return "dimgray" if black_val > orange_val else "sandybrown"


def _save_interactive_scatter3d_html(
    html_path: str,
    pts: np.ndarray,
    colors: list[str] | None,
    *,
    x_label: str,
    y_label: str,
    z_label: str,
    title: str | None = None,
    sidebar_lines: list[str] | None = None,
    legend_labels: dict[str, str] | None = None,
) -> None:
    """Save a rotatable 3D scatter plot as a self-contained HTML file.

    This uses Plotly if available. The output is a portable file you can open in
    any browser and share (unlike a Matplotlib GUI window).
    """
    # Import lazily so the script still runs without plotly installed.
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Plotly is required for interactive HTML export. Install with: pip install plotly"
        ) from e

    out_path = Path(html_path)
    if out_path.suffix.lower() != ".html":
        out_path = out_path.with_suffix(out_path.suffix + ".html") if out_path.suffix else out_path.with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()

    # Use multiple traces so Plotly shows a legend (keys) like the Matplotlib legend.
    if colors is not None and len(colors) == len(pts) and len(colors) > 0:
        base_order = list((legend_labels or {}).keys()) or [
            "deeppink",
            "chocolate",
            "black",
            "sandybrown",
            "dimgray",
            "red",
        ]
        unique_colors = []
        for c in base_order:
            if c not in unique_colors:
                unique_colors.append(c)
        for c in colors:
            if c not in unique_colors:
                unique_colors.append(c)

        for c in unique_colors:
            idx = [i for i, cc in enumerate(colors) if cc == c]
            label = (legend_labels or {}).get(c, c)
            if idx:
                xs, ys, zs = pts[idx, 0], pts[idx, 1], pts[idx, 2]
            else:
                # Empty trace so the legend still shows the key.
                xs, ys, zs = [], [], []
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers",
                    name=label,
                    marker={"size": 4, "color": c},
                    showlegend=True,
                )
            )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker={"size": 4},
                name="Data",
                showlegend=False,
            )
        )

    right_margin = 0
    if sidebar_lines:
        sidebar_text = "Parameters (row 1)<br>" + "<br>".join(sidebar_lines)
        fig.add_annotation(
            x=1.02,
            y=1,
            xref="paper",
            yref="paper",
            text=sidebar_text,
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="top",
            font={"family": "monospace", "size": 11},
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.85)",
        )
        right_margin = 320

    # Wrap long titles so they don't collide with the legend.
    title_text = (title or "").replace(": ", ":<br>")
    top_margin = 160 if (title_text or legend_labels) else 0
    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 16},
        },
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
        ),
        margin=dict(l=0, r=right_margin, t=top_margin, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)


def _sanitize_filename_part(value: str) -> str:
    """Make a string safe to embed in a filename."""
    value = str(value).strip().lower()
    # Keep alnum, underscore, dash, and dot; convert anything else to underscore.
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or "var"


def _add_outcome_color_legend(ax, *, include_pink: bool) -> None:
    """Attach a legend describing the meaning of dot colors.

    User convention:
    - deeppink: invades
    - orange/black: orange+black > 6600, whichever is more prevalent
    - sandybrown/dimgray: 0 < orange+black <= 6600, whichever is more prevalent
    - red: orange+black == 0
    """
    handles = []
    if include_pink:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor="deeppink",
                markeredgecolor="deeppink",
                label="Blood Vessel Invaded",
            )
        )

    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor="chocolate",
                markeredgecolor="chocolate",
                label="Sum > 6600; more PRO than DP",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor="black",
                markeredgecolor="black",
                label="Sum > 6600; more DP than PRO",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor="sandybrown",
                markeredgecolor="sandybrown",
                label="0 < Sum ≤ 6600; more PRO than DP",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor="dimgray",
                markeredgecolor="dimgray",
                label="0 < Sum ≤ 6600; more DP than PRO",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor="red",
                markeredgecolor="red",
                label="Sum = 0 (all cells died)",
            ),
        ]
    )

    # Note: 3D Axes also supports legend().
    # Put the legend above the axes so it doesn't block the top of the plot.
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        borderaxespad=0.0,
    )


_SIDEBAR_PARAMETER_COLUMNS = [
    "max-drug",
    "resource",
    "proliferationP-DP",
    "blood-vessel-width",
    "drugMolecule",
    "adaptive-burden-threshold",
    "PRO-to-DP",
    "invasionP-DP",
    "adaptive-therapy",
    "cell-count",
    "starvingThreshold",
    "timeInterval",
    "startTime",
    "resourceRate",
    "proliferationP-PRO",
]


def _add_parameter_sidebar(fig, data: pandas.DataFrame, *, exclude: set[str]) -> None:
    """Add a right-side list of parameter names/values from the CSV's first data row."""
    if data is None or data.empty:
        return

    try:
        first_row = data.iloc[0]
    except Exception:
        return

    lines = []
    for col in _SIDEBAR_PARAMETER_COLUMNS:
        if col in exclude:
            continue
        if col not in data.columns:
            continue
        val = first_row[col]
        display = "NA" if pandas.isna(val) else str(val)
        lines.append(f"{col}: {display}")

    if not lines:
        return

    # Make room on the right for the sidebar text.
    fig.subplots_adjust(right=0.72)
    fig.text(
        0.74,
        0.85,
        "Parameters (row 1)\n" + "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

def getMedianPer2Options(
    file,
    x_name,
    x_parameters,
    y_name,
    y_parameters,
    orangeCount=False,
    blackCount=False,
    totalInvasion=False,
    plot=False,
    drugAdministered=False,
    drugAdministeredPerStep=False,
    save_html=False,
):
    """Compute median frame count per (x, y) parameter combination.

    When plot=True, produces a 3D scatter of (x, y, median_frame_count).

    If save_html is True, writes an interactive HTML file next to the input CSV
    (same directory). If save_html is a string path, writes to that path.
    """
    print("=" * 100)
    x_values = _expand_parameter_values(x_parameters)
    y_values = _expand_parameter_values(y_parameters)
    print(f'Testing {x_name} with {x_values}')
    print(f'Testing {y_name} with {y_values}')

    data = pandas.read_csv(file, skiprows=6)

    total_invasion_col = None
    if totalInvasion:
        def _norm_colname(name: str) -> str:
            return str(name).strip().lower().replace(" ", "-").replace("_", "-")

        total_invasion_col = next(
            (
                c
                for c in data.columns
                if _norm_colname(c) in {"total-invasion", "totalinvasion"}
            ),
            None,
        )

    points = []
    colors = []
    section_index = 0
    for x in x_values:
        for y in y_values:
            if section_index > 0:
                print("-" * 100)
            selected = data[(data[x_name] == x) & (data[y_name] == y)]
            median = selected["[step]"].median()
            std = selected["[step]"].std()

            output_color = ""
            orange_median = None
            black_median = None
            invasion_median = None
            if orangeCount is True:
                orange_median = pandas.to_numeric(
                    selected["count patches with [pcolor = orange]"], errors="coerce"
                ).median()
            if blackCount is True:
                black_median = pandas.to_numeric(
                    selected["count patches with [pcolor = black]"], errors="coerce"
                ).median()
            if total_invasion_col is not None:
                invasion_median = pandas.to_numeric(selected[total_invasion_col], errors="coerce").median()
            if orangeCount is True and blackCount is True and orange_median == 0 and black_median == 0:
                output_color = colorama.Fore.RED

            print(output_color + f'{x_name}={x}, {y_name}={y} median frame count: {median}')
            print(output_color + f'{x_name}={x}, {y_name}={y} Frame count standard deviation: {std}')
            if orangeCount is True:
                print(output_color + f'{x_name}={x}, {y_name}={y} median Orange Cell Count: {orange_median}')
            if blackCount is True:
                print(output_color + f'{x_name}={x}, {y_name}={y} median Black Cell Count: {black_median}')
            if total_invasion_col is not None:
                print(output_color + f'{x_name}={x}, {y_name}={y} median {total_invasion_col}: {invasion_median}')

            if drugAdministered or drugAdministeredPerStep:
                try:
                    da = _compute_drug_administered_frame(selected)
                except KeyError as e:
                    print(output_color + f"{x_name}={x}, {y_name}={y} drugAdministered unavailable: {e}")
                else:
                    if drugAdministered:
                        print(output_color + f"{x_name}={x}, {y_name}={y} median drugAdministered: {da.median()}")
                        print(output_color + f"{x_name}={x}, {y_name}={y} drugAdministered standard deviation: {da.std()}")
                    if drugAdministeredPerStep:
                        step_series = pandas.to_numeric(selected["[step]"], errors="coerce").replace({0: np.nan})
                        da_per_step = da / step_series
                        print(output_color + f"{x_name}={x}, {y_name}={y} median drugAdministered/[step]: {da_per_step.median()}")

            if plot is True:
                points.append((x, y, median))
                if totalInvasion:
                    if invasion_median is not None and not pandas.isna(invasion_median) and invasion_median >= 5:
                        colors.append("deeppink")
                    elif orangeCount is True and blackCount is True:
                        colors.append(_dot_color_from_counts(orange_median, black_median))
                    else:
                        colors.append("red")
                elif orangeCount is True and blackCount is True:
                    colors.append(_dot_color_from_counts(orange_median, black_median))

            section_index += 1

    if plot is True:
        pts = np.asarray(points, dtype=float)
        keep_mask = ~np.isnan(pts).any(axis=1)
        pts = pts[keep_mask]  # drop rows with NaN median
        colors_filtered = [c for c, keep in zip(colors, keep_mask) if keep]

        html_out_path = None
        if isinstance(save_html, (str, Path)):
            html_out_path = str(save_html)
        elif save_html is True:
            src = Path(str(file))
            html_name = (
                f"{src.stem}__median__{_sanitize_filename_part(x_name)}__{_sanitize_filename_part(y_name)}.html"
            )
            html_out_path = str(src.with_name(html_name))

        if html_out_path:
            try:
                sidebar_lines = []
                try:
                    first_row = data.iloc[0]
                    for col in _SIDEBAR_PARAMETER_COLUMNS:
                        if col in {x_name, y_name}:
                            continue
                        if col not in data.columns:
                            continue
                        val = first_row[col]
                        display = "NA" if pandas.isna(val) else str(val)
                        sidebar_lines.append(f"{col}: {display}")
                except Exception:
                    sidebar_lines = []

                legend_labels = None
                if totalInvasion or (orangeCount and blackCount):
                    legend_labels = {
                        "deeppink": "Blood Vessel Invaded",
                        "chocolate": "Sum > 6600; more PRO than DP",
                        "black": "Sum > 6600; more DP than PRO",
                        "sandybrown": "0 < Sum ≤ 6600; more PRO than DP",
                        "dimgray": "0 < Sum ≤ 6600; more DP than PRO",
                        "red": "Sum = 0 (all cells died)",
                    }

                _save_interactive_scatter3d_html(
                    html_out_path,
                    pts,
                    colors_filtered if len(colors_filtered) == len(pts) else None,
                    x_label=x_name,
                    y_label=y_name,
                    z_label="Median Frame Count",
                    title=f"{Path(str(file)).name}: median([step]) by {x_name} and {y_name}",
                    sidebar_lines=sidebar_lines,
                    legend_labels=legend_labels,
                )
                print(f"Saved interactive plot to: {html_out_path}")
            except Exception as e:
                print(f"Could not save interactive HTML plot ({html_out_path}): {e}")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if len(colors_filtered) == len(pts) and len(colors_filtered) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors_filtered, depthshade=False)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], depthshade=False)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel("median Frame Count")
        _add_parameter_sidebar(fig, data, exclude={x_name, y_name})
        if totalInvasion or (orangeCount and blackCount):
            _add_outcome_color_legend(ax, include_pink=bool(totalInvasion))
        _ensure_plot_hold_on_exit()
        plt.show(block=False)
        plt.pause(0.001)

    print("=" * 100)

def getMeanPer2Options(
    file,
    x_name,
    x_parameters,
    y_name,
    y_parameters,
    orangeCount=False,
    blackCount=False,
    totalInvasion=False,
    plot=False,
    drugAdministered=False,
    drugAdministeredPerStep=False,
    save_html=False,
):
    """Compute mean frame count per (x, y) parameter combination.

    When plot=True, produces a 3D scatter of (x, y, mean_frame_count).

    If save_html is True, writes an interactive HTML file next to the input CSV
    (same directory). If save_html is a string path, writes to that path.
    """
    print("=" * 100)
    x_values = _expand_parameter_values(x_parameters)
    y_values = _expand_parameter_values(y_parameters)
    print(f'Testing {x_name} with {x_values}')
    print(f'Testing {y_name} with {y_values}')

    data = pandas.read_csv(file, skiprows=6)

    total_invasion_col = None
    if totalInvasion:
        def _norm_colname(name: str) -> str:
            return str(name).strip().lower().replace(" ", "-").replace("_", "-")

        total_invasion_col = next(
            (
                c
                for c in data.columns
                if _norm_colname(c) in {"total-invasion", "totalinvasion"}
            ),
            None,
        )

    points = []
    colors = []
    section_index = 0
    for x in x_values:
        for y in y_values:
            if section_index > 0:
                print("-" * 100)
            selected = data[(data[x_name] == x) & (data[y_name] == y)]
            median = selected["[step]"].median()
            mean = selected["[step]"].mean()
            std = selected["[step]"].std()

            output_color = ""
            orange_mean = None
            black_mean = None
            invasion_mean = None
            if orangeCount is True:
                orange_mean = pandas.to_numeric(
                    selected["count patches with [pcolor = orange]"], errors="coerce"
                ).mean()
            if blackCount is True:
                black_mean = pandas.to_numeric(
                    selected["count patches with [pcolor = black]"], errors="coerce"
                ).mean()
            if total_invasion_col is not None:
                invasion_mean = pandas.to_numeric(selected[total_invasion_col], errors="coerce").mean()
            if orangeCount is True and blackCount is True and orange_mean == 0 and black_mean == 0:
                output_color = colorama.Fore.RED

            print(output_color + f'{x_name}={x}, {y_name}={y} Mean frame count: {mean}')
            print(output_color + f'{x_name}={x}, {y_name}={y} Frame count standard deviation: {std}')
            if orangeCount is True:
                print(output_color + f'{x_name}={x}, {y_name}={y} Mean Orange Cell Count: {orange_mean}')
            if blackCount is True:
                print(output_color + f'{x_name}={x}, {y_name}={y} Mean Black Cell Count: {black_mean}')
            if total_invasion_col is not None:
                print(output_color + f'{x_name}={x}, {y_name}={y} Mean {total_invasion_col}: {invasion_mean}')

            if drugAdministered or drugAdministeredPerStep:
                try:
                    da = _compute_drug_administered_frame(selected)
                except KeyError as e:
                    print(output_color + f"{x_name}={x}, {y_name}={y} drugAdministered unavailable: {e}")
                else:
                    if drugAdministered:
                        print(output_color + f"{x_name}={x}, {y_name}={y} Mean drugAdministered: {da.mean()}")
                        print(output_color + f"{x_name}={x}, {y_name}={y} drugAdministered standard deviation: {da.std()}")
                    if drugAdministeredPerStep:
                        step_series = pandas.to_numeric(selected["[step]"], errors="coerce").replace({0: np.nan})
                        da_per_step = da / step_series
                        print(output_color + f"{x_name}={x}, {y_name}={y} Mean drugAdministered/[step]: {da_per_step.mean()}")

            if plot is True:
                points.append((x, y, mean))
                if totalInvasion:
                    if invasion_mean is not None and not pandas.isna(invasion_mean) and invasion_mean >= 5:
                        colors.append("deeppink")
                    elif orangeCount is True and blackCount is True:
                        colors.append(_dot_color_from_counts(orange_mean, black_mean))
                    else:
                        colors.append("red")
                elif orangeCount is True and blackCount is True:
                    colors.append(_dot_color_from_counts(orange_mean, black_mean))

            section_index += 1

    if plot is True:
        pts = np.asarray(points, dtype=float)
        keep_mask = ~np.isnan(pts).any(axis=1)
        pts = pts[keep_mask]  # drop rows with NaN mean
        colors_filtered = [c for c, keep in zip(colors, keep_mask) if keep]

        html_out_path = None
        if isinstance(save_html, (str, Path)):
            html_out_path = str(save_html)
        elif save_html is True:
            src = Path(str(file))
            html_name = (
                f"{src.stem}__mean__{_sanitize_filename_part(x_name)}__{_sanitize_filename_part(y_name)}.html"
            )
            html_out_path = str(src.with_name(html_name))

        if html_out_path:
            try:
                sidebar_lines = []
                try:
                    first_row = data.iloc[0]
                    for col in _SIDEBAR_PARAMETER_COLUMNS:
                        if col in {x_name, y_name}:
                            continue
                        if col not in data.columns:
                            continue
                        val = first_row[col]
                        display = "NA" if pandas.isna(val) else str(val)
                        sidebar_lines.append(f"{col}: {display}")
                except Exception:
                    sidebar_lines = []

                legend_labels = None
                if totalInvasion or (orangeCount and blackCount):
                    legend_labels = {
                        "deeppink": "Blood Vessel Invaded",
                        "chocolate": "Sum > 6600; more PRO than DP",
                        "black": "Sum > 6600; more DP than PRO",
                        "sandybrown": "0 < Sum ≤ 6600; more PRO than DP",
                        "dimgray": "0 < Sum ≤ 6600; more DP than PRO",
                        "red": "Sum = 0 (all cells died)",
                    }

                _save_interactive_scatter3d_html(
                    html_out_path,
                    pts,
                    colors_filtered if len(colors_filtered) == len(pts) else None,
                    x_label=x_name,
                    y_label=y_name,
                    z_label="Mean Frame Count",
                    title=f"{Path(str(file)).name}: mean([step]) by {x_name} and {y_name}",
                    sidebar_lines=sidebar_lines,
                    legend_labels=legend_labels,
                )
                print(f"Saved interactive plot to: {html_out_path}")
            except Exception as e:
                print(f"Could not save interactive HTML plot ({html_out_path}): {e}")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if len(colors_filtered) == len(pts) and len(colors_filtered) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors_filtered, depthshade=False)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], depthshade=False)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel("Mean Frame Count")
        _add_parameter_sidebar(fig, data, exclude={x_name, y_name})
        if totalInvasion or (orangeCount and blackCount):
            _add_outcome_color_legend(ax, include_pink=bool(totalInvasion))
        _ensure_plot_hold_on_exit()
        plt.show(block=False)
        plt.pause(0.001)

    print("=" * 100)

def getMedianPerOption(
    file,
    string,
    parameters,
    orangeCount=False,
    blackCount=False,
    totalInvasion=False,
    plot=False,
    drugAdministered=False,
    drugAdministeredPerStep=False,
):
    print("=" * 100)
    values = _expand_parameter_values(parameters)
    print(f'Testing {string} with {values}')

    data = pandas.read_csv(file, skiprows=6)

    total_invasion_col = None
    if totalInvasion:
        def _norm_colname(name: str) -> str:
            return str(name).strip().lower().replace(" ", "-").replace("_", "-")

        total_invasion_col = next(
            (
                c
                for c in data.columns
                if _norm_colname(c) in {"total-invasion", "totalinvasion"}
            ),
            None,
        )

    points = []
    colors = []
    for index, i in enumerate(values):
        if index > 0:
            print("-" * 100)
        selected = data[data[string] == i]
        orange_median = None
        black_median = None
        invasion_median = None
        if orangeCount == True:
            orange_median = pandas.to_numeric(
                selected["count patches with [pcolor = orange]"], errors="coerce"
            ).median()
        if blackCount == True:
            black_median = pandas.to_numeric(
                selected["count patches with [pcolor = black]"], errors="coerce"
            ).median()

        color = colorama.Fore.CYAN if (index % 2) == 0 else colorama.Fore.LIGHTGREEN_EX
        if orangeCount == True and blackCount == True and orange_median == 0 and black_median == 0:
            color = colorama.Fore.RED

        median = selected["[step]"].median()
        print(color + f'{i} {string} median frame count: {median}')
        print(color + f'{i} {string} Frame count standard deviation: {selected["[step]"].std()}')
        if orangeCount == True:
            print(color + f'{i} {string} median Orange Cell Count: {orange_median}')
        if blackCount == True:
            print(color + f'{i} {string} median Black Cell Count: {black_median}')
        if total_invasion_col is not None:
            invasion_median = pandas.to_numeric(selected[total_invasion_col], errors="coerce").median()
            print(color + f'{i} {string} median {total_invasion_col}: {invasion_median}')

        if drugAdministered or drugAdministeredPerStep:
            try:
                da = _compute_drug_administered_frame(selected)
            except KeyError as e:
                print(color + f'{i} {string} drugAdministered unavailable: {e}')
            else:
                if drugAdministered:
                    print(color + f'{i} {string} median drugAdministered: {da.median()}')
                    print(color + f'{i} {string} drugAdministered standard deviation: {da.std()}')
                if drugAdministeredPerStep:
                    step_series = pandas.to_numeric(selected["[step]"], errors="coerce").replace({0: np.nan})
                    da_per_step = da / step_series
                    print(color + f'{i} {string} median drugAdministered/[step]: {da_per_step.median()}')
        if plot == True:
            points.append((i, median))
            if totalInvasion:
                # Pink overrides other color rules when median total invasion is high.
                if invasion_median is not None and not pandas.isna(invasion_median) and invasion_median >= 5:
                    colors.append("deeppink")
                elif orangeCount == True and blackCount == True:
                    colors.append(_dot_color_from_counts(orange_median, black_median))
                else:
                    colors.append("red")
            elif orangeCount == True and blackCount == True:
                colors.append(_dot_color_from_counts(orange_median, black_median))
    
    if plot == True:
        pts = np.asarray(points, dtype=float)
        keep_mask = ~np.isnan(pts).any(axis=1)
        pts = pts[keep_mask]  # drop rows with NaN median
        colors_filtered = [c for c, keep in zip(colors, keep_mask) if keep]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if len(colors_filtered) == len(pts) and len(colors_filtered) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=colors_filtered)
        else:
            ax.scatter(pts[:, 0], pts[:, 1])
        ax.set_xlabel(string)
        ax.set_ylabel("median Frame Count")
        _add_parameter_sidebar(fig, data, exclude={string})
        if totalInvasion or (orangeCount and blackCount):
            _add_outcome_color_legend(ax, include_pink=bool(totalInvasion))
        _ensure_plot_hold_on_exit()
        plt.show(block=False)
        plt.pause(0.001)

    print("=" * 100)


def getMeanPerOption(
    file,
    string,
    parameters,
    orangeCount=False,
    blackCount=False,
    totalInvasion=False,
    plot=False,
    drugAdministered=False,
    drugAdministeredPerStep=False,
    useMedian=False,
):
    print("=" * 100)
    values = _expand_parameter_values(parameters)
    print(f'Testing {string} with {values}')

    data = pandas.read_csv(file, skiprows=6)

    total_invasion_col = None
    if totalInvasion:
        def _norm_colname(name: str) -> str:
            return str(name).strip().lower().replace(" ", "-").replace("_", "-")

        total_invasion_col = next(
            (
                c
                for c in data.columns
                if _norm_colname(c) in {"total-invasion", "totalinvasion"}
            ),
            None,
        )

    points = []
    colors = []
    for index, i in enumerate(values):
        if index > 0:
            print("-" * 100)
        selected = data[data[string] == i]
        orange_mean = None
        black_mean = None
        invasion_mean = None
        if orangeCount == True:
            orange_mean = pandas.to_numeric(
                selected["count patches with [pcolor = orange]"], errors="coerce"
            ).mean()
        if blackCount == True:
            black_mean = pandas.to_numeric(
                selected["count patches with [pcolor = black]"], errors="coerce"
            ).mean()

        color = colorama.Fore.CYAN if (index % 2) == 0 else colorama.Fore.LIGHTGREEN_EX
        if orangeCount == True and blackCount == True and orange_mean == 0 and black_mean == 0:
            color = colorama.Fore.RED

        mean = selected["[step]"].mean()
        print(color + f'{i} {string} Mean frame count: {mean}')
        print(color + f'{i} {string} Frame count standard deviation: {selected["[step]"].std()}')
        if orangeCount == True:
            print(color + f'{i} {string} Mean Orange Cell Count: {orange_mean}')
        if blackCount == True:
            print(color + f'{i} {string} Mean Black Cell Count: {black_mean}')
        if total_invasion_col is not None:
            invasion_mean = pandas.to_numeric(selected[total_invasion_col], errors="coerce").mean()
            print(color + f'{i} {string} Mean {total_invasion_col}: {invasion_mean}')

        if drugAdministered or drugAdministeredPerStep:
            try:
                da = _compute_drug_administered_frame(selected)
            except KeyError as e:
                print(color + f'{i} {string} drugAdministered unavailable: {e}')
            else:
                if drugAdministered:
                    print(color + f'{i} {string} Mean drugAdministered: {da.mean()}')
                    print(color + f'{i} {string} drugAdministered standard deviation: {da.std()}')
                if drugAdministeredPerStep:
                    step_series = pandas.to_numeric(selected["[step]"], errors="coerce").replace({0: np.nan})
                    da_per_step = da / step_series
                    print(color + f'{i} {string} Mean drugAdministered/[step]: {da_per_step.mean()}')
        if plot == True:
            points.append((i, mean))
            if totalInvasion:
                # Pink overrides other color rules when mean total invasion is high.
                if invasion_mean is not None and not pandas.isna(invasion_mean) and invasion_mean >= 5:
                    colors.append("deeppink")
                elif orangeCount == True and blackCount == True:
                    colors.append(_dot_color_from_counts(orange_mean, black_mean))
                else:
                    colors.append("red")
            elif orangeCount == True and blackCount == True:
                colors.append(_dot_color_from_counts(orange_mean, black_mean))
    
    if plot == True:
        pts = np.asarray(points, dtype=float)
        keep_mask = ~np.isnan(pts).any(axis=1)
        pts = pts[keep_mask]  # drop rows with NaN mean
        colors_filtered = [c for c, keep in zip(colors, keep_mask) if keep]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if len(colors_filtered) == len(pts) and len(colors_filtered) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=colors_filtered)
        else:
            ax.scatter(pts[:, 0], pts[:, 1])
        ax.set_xlabel(string)
        ax.set_ylabel("Mean Frame Count")
        _add_parameter_sidebar(fig, data, exclude={string})
        if totalInvasion or (orangeCount and blackCount):
            _add_outcome_color_legend(ax, include_pink=bool(totalInvasion))
        _ensure_plot_hold_on_exit()
        plt.show(block=False)
        plt.pause(0.001)

    print("=" * 100)

def visualize(file, var1=None, var2=None, plotMedian=False, save_html=False):
    data = pandas.read_csv(file, skiprows=6, header=0)
    headings = data.columns

    possiblevaryerIndexes = ["max-drug","resource","proliferationP-DP","blood-vessel-width","drugMolecule","invasion-threshold","adaptive-burden-threshold","bolusAmount","PRO-to-DP","invasionP-DP","adaptive-therapy","cell-count","starvingThreshold","timeInterval","startTime","resourceRate","proliferationP-PRO","nextBolusTime",]
    varyerIndexes = []

    orangeOn = False
    blackOn = False
    invOn = False

    #find if the optional parameters are present
    if "count patches with [pcolor = orange]" in headings:
        orangeOn = True
    if "count patches with [pcolor = black]" in headings:
        blackOn = True
    if 'total-invasion' in headings:
        invOn = True
    
    #find the limits and step for an independent variable
    def getParameters(varyerName):
        if varyerName == 'adaptive-therapy':
            print("adaptive-therapy can't be used as a variable in this function")
        else:
            values = data[varyerName].drop_duplicates()
            values = values.astype(float)
            min = values.min()
            max = values.max()
            min2 = values.nsmallest(2).iloc[-1]
            step = float(min2) - float(min)
            return [min, step, max]
    
    #use uer specified variables if they are present
    if var1 != None or var2 != None:
        if var1 == "adaptive-therapy" or var2 == "adapative-therapy":
            print("adaptive-therapy can't be the variable")
        elif var2 == None:
            getMeanPerOption(file, var1, getParameters(var1), blackCount=blackOn, orangeCount=orangeOn, totalInvasion=invOn, plot=True)
            if plotMedian == True:
                getMedianPerOption(file, var1, getParameters(var1), blackCount=blackOn, orangeCount=orangeOn, totalInvasion=invOn, plot=True)
        elif var2 != None:
            getMeanPer2Options(file, var1, getParameters(var1), var2, getParameters(var2), blackCount=blackOn, orangeCount=orangeOn, totalInvasion=invOn, plot=True, save_html=save_html)
            if plotMedian == True:
                getMeanPer2Options(file, var1, getParameters(var1), var2, getParameters(var2), blackCount=blackOn, orangeCount=orangeOn, totalInvasion=invOn, plot=True, save_html=save_html)


    #find which columns vary
    else:
        for i in range(1,20):
            if i == 13:
                continue
            firstRow = data.iat[0,i]
            for j in range(1, int(len(data)/2) + 2):
                testRow = data.iat[j,i]
                if firstRow != testRow:
                    varyerIndexes.append(i)
                    break
        
        varyerIndexes = [possiblevaryerIndexes[varyerIndexes[i] - 1] for i in range(0, len(varyerIndexes) - 1)]

        #if 1 column varies call getMeanPerOption with that column
        if len(varyerIndexes) == 1:
            getMeanPerOption(file, varyerIndexes[0], getParameters(varyerIndexes[0]), blackCount=blackOn, orangeCount=orangeOn, totalInvasion=invOn, plot=True)
            if plotMedian == True:
                getMedianPerOption(file, varyerIndexes[0], getParameters(varyerIndexes[0]), blackCount=blackOn, orangeCount=orangeOn, totalInvasion=invOn, plot=True)
        #if 2 columns vary call get mean per option with those columns
        elif len(varyerIndexes) == 2:
            getMeanPer2Options(file, varyerIndexes[0], getParameters(varyerIndexes[0]), varyerIndexes[1], getParameters(varyerIndexes[1]), blackCount=blackOn, orangeCount=orangeOn, totalInvasion=invOn, plot=True)
            if plotMedian == True:
                getMedianPer2Options(file, varyerIndexes[0], getParameters(varyerIndexes[0]), varyerIndexes[1], getParameters(varyerIndexes[1]), blackCount=blackOn, orangeCount=orangeOn, totalInvasion=invOn, plot=True)

        #inform the users if there are too many or too few variables
        elif len(varyerIndexes) == 0:
            print('There are no independent variables')
        elif len(varyerIndexes) > 2:
            print("The visualize function can't handle more than 2 independent variables. \n If you would like to see a graph with more variables specify those variables with var1=<variable> and var2=<variable>")


def keep_plots_open() -> None:
    """Block until all Matplotlib figure windows are closed.

    Useful when calling plotting functions from a short-lived script.
    """
    if plt.get_fignums():
        plt.show(block=True)