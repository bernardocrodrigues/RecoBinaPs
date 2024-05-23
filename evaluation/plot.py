""" plot.py

This module implements all the functions used to plot the results of the benchmarking process.

Copyright 2023 Bernardo C. Rodrigues
See LICENSE file for license details
"""

import itertools
import multiprocessing
from typing import Tuple, List
from collections import defaultdict
import scipy
import pandas as pd
import numpy as np

import plotly.io as pio
import plotly.graph_objects as go
from surprise import Trainset


DPI = 300
WIDTH = 1200
HEIGHT = 800
FORMAT = "png"


pd.set_option("display.expand_frame_repr", False)


def customize_default_template():
    """
    Customize the default template with specific layout settings.

    This will give all figures the same look and feel.
    """

    # Access the default template
    default_template = pio.templates[pio.templates.default]

    # Customize font settings
    default_template.layout.font.family = "Latin Modern"
    default_template.layout.font.size = 16
    default_template.layout.font.color = "black"

    # Customize margin and width
    default_template.layout.margin = go.layout.Margin(t=50, b=50, l=50, r=50)
    default_template.layout.width = WIDTH
    default_template.layout.height = HEIGHT

    # Customize background color
    default_template.layout.plot_bgcolor = "rgb(245,245,245)"

    # Customize y-axis settings
    default_template.layout.yaxis = dict(
        mirror=True, ticks="outside", showline=True, linecolor="black", gridcolor="lightgrey"
    )

    # Customize x-axis settings
    default_template.layout.xaxis = dict(
        mirror=True, ticks="outside", showline=True, linecolor="black", gridcolor="lightgrey"
    )

    # Customize legend background color
    default_template.layout.legend = dict(bgcolor="rgb(245,245,245)")

    # Set the default renderer to JPEG
    pio.renderers.default = FORMAT


def coalesce_fold_results(raw_results: List) -> dict:
    """
    Coalesce the raw experiment results into a dictionary that maps recommender variations to
    metrics to folds to lists of results. This function is used to coalesce the results of a
    single fold.

    Args:
        raw_results: List of raw experiment results.
        metric_names: List of metric names to be coalesced.

    Returns:
        Dictionary that maps recommender names to metrics to folds to lists of results.

    Example:
        >>> raw_results = [
        ...     (0, "UBCF", {"mae": 0.1, "rmse": 0.2}),
        ...     (0, "UBCF", {"mae": 0.5, "rmse": 0.7}),
        ...     (1, "UBCF", {"mae": 0.1, "rmse": 0.4}),
        ...     (1, "UBCF", {"mae": 0.6, "rmse": 0.9}),
        ... ]
        >>> coalesce_raw_results(raw_results)
        {
            "UBCF": {
                "mae": {
                    0: [0.1, 0.5],
                    1: [0.1, 0.6],
                },
                "rmse": {
                    0: [0.2, 0.7],
                    1: [0.4, 0.9],
                },
            },
        }

    """
    coalesced_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for raw_experiment_result in raw_results:
        fold, recommender_variation, experiment_results = raw_experiment_result
        for metric_name in experiment_results.keys():
            coalesced_results[recommender_variation][metric_name][fold].append(
                experiment_results[metric_name]
            )
    return coalesced_results


def concatenate_fold_results(coalesced_results: dict) -> dict:
    """
    Concatenate the results of the folds into a single list. This function is used to concatenate
    the results of all folds.

    Args:
        coalesced_results: Dictionary that maps recommender variations to metrics to folds to lists
            of results.

    Returns:
        Dictionary that maps recommender variations to metrics to lists of results.

    Example:
        >>> coalesced_results = {
        ...     "UBCF": {
        ...         "mae": {
        ...             0: [0.1, 0.5],
        ...             1: [0.1, 0.6],
        ...         },
        ...         "rmse": {
        ...             0: [0.2, 0.7],
        ...             1: [0.4, 0.9],
        ...         },
        ...     },
        ... }
        >>> concatenate_fold_results(coalesced_results)
        {
            "UBCF": {
                "mae": [0.1, 0.5, 0.1, 0.6],
                "rmse": [0.2, 0.7, 0.4, 0.9],
            },
        }
    """
    concatenated_results = defaultdict(lambda: defaultdict(list))
    for recommender_name, metric_results in coalesced_results.items():
        for metric_name, fold_results in metric_results.items():
            concatenated_results[recommender_name][metric_name] = list(
                itertools.chain.from_iterable(fold_results.values())
            )
    return concatenated_results


def calculate_boxplot_values(series: List[float]):
    """
    Calculate boxplot values for a given data set.

    Args:
        series: The data set for which the boxplot values should be calculated.

    Returns:
        q_1: The first quartile.
        q_3: The third quartile.
        lower_fence: The lower fence.
        upper_fence: The upper fence.

    """

    assert isinstance(series, list)
    assert len(series) > 0
    assert all(isinstance(element, float) or isinstance(element, int) for element in series)

    q_1 = np.percentile(series, 25)
    q_3 = np.percentile(series, 75)
    iqr = q_3 - q_1

    lower_fence = q_1 - 1.5 * iqr
    upper_fence = q_3 + 1.5 * iqr

    return q_1, q_3, lower_fence, upper_fence


def plot_metric_box_plot(metric_name: str, concatenated_results: dict):
    """
    Plot a box plot for a given metric.

    Args:
        metric_name: The name of the metric to be plotted.
        concatenated_results: The results as given by concatenate_fold_results.
    """

    fig = go.Figure()
    for recommender_name, metric_results in concatenated_results.items():
        fig.add_trace(
            go.Box(
                y=metric_results[metric_name],
                name=recommender_name,
                fillcolor="gray",
                marker_color="black",
                showlegend=False,  # Add this line to hide the legend
            )
        )

    fig.update_layout(
        yaxis_title=metric_name.upper(),
        xaxis_title="Recommender",
        width=600,
        height=400,
        margin_l=60,
        margin_r=100,
        margin_b=100,
    )

    fig.show()


def get_result_table(metric_name: str, concatenated_results: dict):
    """
    Get a table with the results for a given metric.

    Args:
        metric_name: The name of the metric to be plotted.
        concatenated_results: The results as given by concatenate_fold_results.
    """
    results = []
    for recommender_name, metric_results in concatenated_results.items():
        metric_data = metric_results[metric_name]
        mean = np.mean(metric_data)
        median = np.median(metric_data)
        standard_deviation = np.std(metric_data)
        variance = np.var(metric_data)
        skewness = scipy.stats.skew(metric_data)
        kurtosis = scipy.stats.kurtosis(metric_data)
        min_val = np.min(metric_data)
        max_val = np.max(metric_data)
        Q1, Q3, lower_fence, upper_fence = calculate_boxplot_values(metric_data)

        results.append(
            {
                "Recommender": recommender_name,
                "Mean": mean,
                "Standard Deviation": standard_deviation,
                "Variance": variance,
                "Min": min_val,
                "Max": max_val,
                "Median": median,
                "Q1": Q1,
                "Q3": Q3,
                "Lower Fence": lower_fence,
                "Upper Fence": upper_fence,
                "Skewness": skewness,
                "Kurtosis": kurtosis,
            }
        )

    return pd.DataFrame(results)


def get_latex_boxplot_code(metric_name: str, concatenated_results: dict):
    """
    Get the latex code for a boxplot for a given metric.

    Args:
        metric_name: The name of the metric to be plotted.
        concatenated_results: The results as given by concatenate_fold_results.
    """

    names = ""
    plots = ""

    for recommender_name, metric_results in concatenated_results.items():
        metric_data = metric_results[metric_name]
        median = np.median(metric_data)
        Q1, Q3, lower_fence, upper_fence = calculate_boxplot_values(metric_data)

        names += f"{recommender_name}, "

        plots += f"""
\\addplot+[
    boxplot prepared=*
        median={median},
        upper quartile={Q3},
        lower quartile={Q1},
        upper whisker={upper_fence},
        lower whisker={lower_fence}
    $,
    solid,
    color=gray,
    fill=gray!30
] coordinates *$;
        """.replace(
            "*", "{"
        ).replace(
            "$", "}"
        )

    return names + plots


def get_latex_table_from_pandas_table(pandas_table: pd.DataFrame):
    """
    Get a latex table from a pandas table.

    Args:
        pandas_table: The pandas table to be converted to latex.
    """
    return pandas_table.to_latex(index=False, float_format="%.3f")
