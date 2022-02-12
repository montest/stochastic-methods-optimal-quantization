import os
import sys
import csv
import imageio

import numpy as np

from typing import List
from pathlib import Path
from bokeh.io import export_svg, export_png
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

Point = np.ndarray


def find_closest_centroid(centroids: List[Point], p: Point):
    index_closest_centroid = -1
    min_dist = sys.float_info.max
    for i, x_i in enumerate(centroids):
        dist = np.linalg.norm(x_i - p)
        if dist < min_dist:
            index_closest_centroid = i
            min_dist = dist
    return index_closest_centroid, min_dist


def make_plot_distortion(directory: str, method: str):
    with open(os.path.join(directory, f"distortion.txt"), "r") as f_distortion:
        distortion_per_step = csv.reader(f_distortion, delimiter=',')
        step = list()
        disto = list()
        next(distortion_per_step)
        for row in distortion_per_step:
            step.append(int(row[0]))
            disto.append(float(row[1]))

        plot = figure(plot_width=600, plot_height=400)
        if method == 'lloyd':
            plot.xaxis.axis_label = "Number of fixed-point iterations"
        elif method == 'clvq':
            plot.xaxis.axis_label = "Number of stochastic gradient descend steps"
        plot.yaxis.axis_label = "Distortion"

        source = ColumnDataSource(data=dict(step=step, disto=disto))
        plot.line(x='step', y='disto', source=source)
        return plot


def save_plot_distortion(N: int, M: int, method: str):
    directory = get_directory(N, M, method)

    if os.path.exists(directory):
        plot = make_plot_distortion(directory, method)
        export_svg(plot, filename=os.path.join(directory, f"distortion_N_{N}_random_{method}_{M}.svg"))
        # export_png(plot, filename=os.path.join(directory, f"distortion_N_{N}_random_{method}_{M}.png"))
    else:
        print(f"Could not save the distortion plot because the directory {directory} does not exist!")


def extract_step_number_from_filename(filename):
    prefix = 'step_'
    index_start = filename.find(prefix) + len(prefix)
    index_end = filename.rfind(".")
    x = filename[index_start:index_end]
    return int(x)


def make_gif(directory):
    filenames = [os.path.join(directory, filename) for filename in os.listdir(directory) if
                 filename.startswith('step_') and filename.endswith('.png')]
    filenames.sort(key=extract_step_number_from_filename)

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))

    splitted_directory_path = directory.split('/')
    gif_name = splitted_directory_path[-2] + '_' + splitted_directory_path[-1] + '.gif'
    imageio.mimsave(os.path.join(directory, gif_name), images)
    # imageio.mimsave(os.path.join(directory, f"my_gif.gif"), images, loop=1)


def get_directory(N, M, method):
    if method == 'lloyd':
        return os.path.join("_output", "gaussian", f"N_{N}", f"random_lloyd_{M}")
    if method == 'clvq':
        return os.path.join("_output", "gaussian", f"N_{N}", f"random_clvq_{M}")
    raise ValueError(f"method {method} is not handled")


def save_results(centroids, probas, distortion, step, M, method):
    N = len(centroids)
    directory = get_directory(N, M, method)
    if step == 0:
        Path.mkdir(Path(directory), parents=True, exist_ok=True)
        with open(os.path.join(directory, f"distortion.txt"), "w") as f_distortion:
            f_distortion.write("step,distortion\n")

    # first, save distortion in file and update plot
    with open(os.path.join(directory, f"distortion.txt"), "a+") as f_distortion:
        f_distortion.write(f"{step+1 if method=='lloyd' else (step+1)*M},{distortion}\n")
    save_plot_distortion(N, M, method)

    # second, plot quantizer
    from tessellation import VoronoiTessellation
    tessellation = VoronoiTessellation(centroids.tolist())
    tessellation.save_quantization_bokeh(os.path.join(directory, f"step_{step}"), probas, show_color_bar=False)
