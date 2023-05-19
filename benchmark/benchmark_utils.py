import itertools
import math
import os
import time

import chromedriver_autoinstaller
import numpy as np
import pandas as pd
import torch
from bokeh.io import export_svg, export_png
from bokeh.models import ColumnDataSource
from bokeh.palettes import Viridis
from bokeh.plotting import figure
from bokeh.transform import dodge

chromedriver_autoinstaller.install()


def check_existance(dict_of_values, df):
    v = df.iloc[:, 0] == df.iloc[:, 0]
    for key, value in dict_of_values.items():
        v &= (df[key] == value)
    return v.any()


def testing_method(fct_to_test, parameters_grid: dict, path_to_results: str):
    if os.path.exists(path_to_results) and os.path.getsize(path_to_results) > 0:
        df_results = pd.read_csv(path_to_results, index_col=0)
    else:
        df_results = pd.DataFrame()

    keys, values = zip(*parameters_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for permutations in permutations_dicts:
        dict_result = permutations.copy()
        dict_result["method_name"] = fct_to_test.__name__
        if len(df_results) > 0 and check_existance(dict_result, df_results):
            print(f"Skipping {dict_result}")
            continue
        torch.cuda.empty_cache()

        start_time = time.time()
        centroids, probabilities, distortion = fct_to_test(**permutations)
        if math.isnan(distortion):
            print(f"Results for following values {dict_result} were not saved "
                  f"because an nan was present in the centroids")
            continue
        elapsed_time = time.time() - start_time
        dict_result["elapsed_time"] = elapsed_time
        df_results = pd.concat(
            [df_results, pd.DataFrame(dict_result, index=[0])],
            ignore_index=True
        )
        df_results.to_csv(path_to_results)


def plot_results_lloyd(df_grouped, M, directory_path):
    grouped_by_values = df_grouped.groupby(["method", "M"]).agg(list).to_dict()
    elapsed_times_per_iter_per_method = grouped_by_values.get("elapsed_time_by_iter")
    Ns_per_method = grouped_by_values.get("N")

    source = ColumnDataSource(
        data=dict(
            Ns_numpy=Ns_per_method.get(("numpy_cpu", M)),
            numpy_cpu=elapsed_times_per_iter_per_method.get(("numpy_cpu", M)),
            Ns_pytorch_cpu=Ns_per_method.get(("pytorch_cpu", M)),
            pytorch_cpu=elapsed_times_per_iter_per_method.get(("pytorch_cpu", M)),
            Ns_pytorch_cuda=Ns_per_method.get(("pytorch_cuda", M)),
            pytorch_cuda=elapsed_times_per_iter_per_method.get(("pytorch_cuda", M))
        )
    )

    color_numpy_cpu = Viridis[3][1]
    color_pytorch_cpu = Viridis[3][2]
    color_pytorch_cuda = Viridis[3][0]
    general_font_size = '14pt'
    # general_font_size = '28pt'

    plot = figure(plot_width=600, plot_height=500)
    # plot = figure(plot_width=1200, plot_height=1000)

    plot.xaxis.axis_label = "Grid size (N)"
    plot.xaxis.axis_label_text_font_size = general_font_size

    plot.yaxis.axis_label = "Time elapsed per iter (in seconds)"
    plot.yaxis.axis_label_text_font_size = general_font_size

    plot.circle(x='Ns_numpy', y='numpy_cpu', source=source, fill_color=None, line_color=color_numpy_cpu, legend_label='numpy')
    plot.line(x='Ns_numpy', y='numpy_cpu', source=source, line_color=color_numpy_cpu, legend_label='numpy')

    plot.circle(x='Ns_pytorch_cpu', y='pytorch_cpu', source=source, fill_color=None, line_color=color_pytorch_cpu, legend_label='pytorch (cpu)')
    plot.line(x='Ns_pytorch_cpu', y='pytorch_cpu', source=source, line_color=color_pytorch_cpu, legend_label='pytorch (cpu)')

    plot.circle(x='Ns_pytorch_cuda', y='pytorch_cuda', source=source, fill_color=color_pytorch_cuda, line_color=color_pytorch_cuda, legend_label='pytorch (cuda: T4)')
    plot.line(x='Ns_pytorch_cuda', y='pytorch_cuda', source=source, line_color=color_pytorch_cuda, legend_label='pytorch (cuda: T4)')
    plot.legend.location = "top_left"
    plot.legend.label_text_font_size = general_font_size
    plot.xaxis.major_label_text_font_size = general_font_size
    plot.yaxis.major_label_text_font_size = general_font_size

    # show(plot)
    export_png(plot, filename=os.path.join(directory_path, f"stochastic_lloyd_1d_method_comparison_M_{M}.png"))
    export_svg(plot, filename=os.path.join(directory_path, f"stochastic_lloyd_1d_method_comparison_M_{M}.svg"))


def plot_ratios_lloyd(df_grouped, M, directory_path):
    color_numpy_cpu = Viridis[3][1]
    color_pytorch_cpu = Viridis[3][2]
    color_pytorch_cuda = Viridis[3][0]
    general_font_size = '14pt'
    # general_font_size = '28pt'

    grouped_by_values = df_grouped.groupby(["method", "M"]).agg(list).to_dict()
    elapsed_times_per_iter_per_method = grouped_by_values.get("elapsed_time_by_iter")
    Ns_per_method = grouped_by_values.get("N")
    if (Ns_per_method.get(('pytorch_cuda', M)) != Ns_per_method.get(('pytorch_cpu', M)) or Ns_per_method.get(
            ('numpy_cpu', M)) != Ns_per_method.get(('pytorch_cpu', M))):
        print(f"Cannot plot ratios for M equals {M} because N values does not match!!")
        return

    rescaled_comparisons = {
        (method, M): np.array(elapsed_times_per_iter_per_method.get((method, M))) / np.array(
            elapsed_times_per_iter_per_method.get(("pytorch_cuda", M)))
        for method, M in elapsed_times_per_iter_per_method
    }
    Ns = Ns_per_method.get(("pytorch_cuda", M))
    Ns = [str(N) for N in Ns]

    # plot = figure(x_range=Ns, plot_width=1200, plot_height=1000)
    plot = figure(x_range=Ns, plot_width=600, plot_height=500)

    plot.xaxis.axis_label = "Grid size (N)"
    plot.xaxis.axis_label_text_font_size = general_font_size

    plot.yaxis.axis_label = "Ratio over PyTorch cuda (T4) time"
    plot.yaxis.axis_label_text_font_size = general_font_size

    source = ColumnDataSource(data={
        'Ns': Ns,
        'pytorch_cuda': rescaled_comparisons.get(("pytorch_cuda", M)),
        'pytorch_cpu': rescaled_comparisons.get(("pytorch_cpu", M)),
        'numpy_cpu': rescaled_comparisons.get(("numpy_cpu", M))
    })

    plot.vbar(x=dodge('Ns', -0.25, range=plot.x_range), top='pytorch_cuda', source=source, width=0.2, color=color_pytorch_cuda, legend_label="pytorch (cuda: T4)")
    plot.vbar(x=dodge('Ns', 0.0, range=plot.x_range), top='pytorch_cpu', source=source, width=0.2, color=color_pytorch_cpu, legend_label="pytorch (cpu)")
    plot.vbar(x=dodge('Ns', 0.25, range=plot.x_range), top='numpy_cpu', source=source, width=0.2, color=color_numpy_cpu, legend_label="numpy")
    plot.legend.location = "top_left"
    plot.legend.label_text_font_size = general_font_size
    plot.xaxis.major_label_text_font_size = general_font_size
    plot.yaxis.major_label_text_font_size = general_font_size

    # show(plot)
    export_png(plot, filename=os.path.join(directory_path, f"stochastic_lloyd_1d_ratio_comparison_M_{M}.png"))
    export_svg(plot, filename=os.path.join(directory_path, f"stochastic_lloyd_1d_ratio_comparison_M_{M}.svg"))


def plot_results_clvq(df_grouped, M, directory_path):
    grouped_by_values = df_grouped.groupby(["method", "M"]).agg(list).to_dict()
    elapsed_times_per_epoch_per_method = grouped_by_values.get("elapsed_time_by_epoch")
    Ns_per_method = grouped_by_values.get("N")

    source = ColumnDataSource(
        data=dict(
            Ns_numpy=Ns_per_method.get(("numpy_cpu", M)),
            numpy_cpu=elapsed_times_per_epoch_per_method.get(("numpy_cpu", M)),
            Ns_pytorch_cpu=Ns_per_method.get(("pytorch_cpu", M)),
            pytorch_cpu=elapsed_times_per_epoch_per_method.get(("pytorch_cpu", M)),
            Ns_pytorch_cuda=Ns_per_method.get(("pytorch_cuda", M)),
            pytorch_cuda=elapsed_times_per_epoch_per_method.get(("pytorch_cuda", M)),
            Ns_pytorch_autograd_cpu=Ns_per_method.get(("pytorch_autograd_cpu", M)),
            pytorch_autograd_cpu=elapsed_times_per_epoch_per_method.get(("pytorch_autograd_cpu", M)),
            Ns_pytorch_autograd_cuda=Ns_per_method.get(("pytorch_autograd_cuda", M)),
            pytorch_autograd_cuda=elapsed_times_per_epoch_per_method.get(("pytorch_autograd_cuda", M))
        )
    )

    color_numpy_cpu = Viridis[3][1]
    color_pytorch_cpu = Viridis[3][2]
    color_pytorch_cuda = Viridis[3][0]
    # todo: verifier les couleurs ici (peut etre qu'il faut utiliser Viridis[5][0]
    color_pytorch_autograd_cpu = Viridis[3][3]
    color_pytorch_autograd_cuda = Viridis[3][4]
    general_font_size = '14pt'
    # general_font_size = '28pt'

    plot = figure(plot_width=600, plot_height=500)
    # plot = figure(plot_width=1200, plot_height=1000)

    plot.xaxis.axis_label = "Grid size (N)"
    plot.xaxis.axis_label_text_font_size = general_font_size

    plot.yaxis.axis_label = "Time elapsed per epoch (in seconds)"
    plot.yaxis.axis_label_text_font_size = general_font_size

    plot.circle(x='Ns_numpy', y='numpy_cpu', source=source, fill_color=None, line_color=color_numpy_cpu, legend_label='numpy')
    plot.line(x='Ns_numpy', y='numpy_cpu', source=source, line_color=color_numpy_cpu, legend_label='numpy')

    plot.circle(x='Ns_pytorch_cpu', y='pytorch_cpu', source=source, fill_color=None, line_color=color_pytorch_cpu, legend_label='pytorch (cpu)')
    plot.line(x='Ns_pytorch_cpu', y='pytorch_cpu', source=source, line_color=color_pytorch_cpu, legend_label='pytorch (cpu)')

    plot.circle(x='Ns_pytorch_cuda', y='pytorch_cuda', source=source, fill_color=color_pytorch_cuda, line_color=color_pytorch_cuda, legend_label='pytorch (cuda: T4)')
    plot.line(x='Ns_pytorch_cuda', y='pytorch_cuda', source=source, line_color=color_pytorch_cuda, legend_label='pytorch (cuda: T4)')

    plot.circle(x='Ns_pytorch_autograd_cpu', y='pytorch_autograd_cpu', source=source, fill_color=None, line_color=color_pytorch_autograd_cpu, legend_label='pytorch autograd (cpu)')
    plot.line(x='Ns_pytorch_autograd_cpu', y='pytorch_autograd_cpu', source=source, line_color=color_pytorch_autograd_cpu, legend_label='pytorch autograd (cpu)')

    plot.circle(x='Ns_pytorch_autograd_cuda', y='pytorch_autograd_cuda', source=source, fill_color=color_pytorch_autograd_cuda, line_color=color_pytorch_autograd_cuda, legend_label='pytorch autograd (cuda: T4)')
    plot.line(x='Ns_pytorch_autograd_cuda', y='pytorch_autograd_cuda', source=source, line_color=color_pytorch_autograd_cuda, legend_label='pytorch autograd (cuda: T4)')

    plot.legend.location = "top_left"
    plot.legend.label_text_font_size = general_font_size
    plot.xaxis.major_label_text_font_size = general_font_size
    plot.yaxis.major_label_text_font_size = general_font_size

    # show(plot)
    export_png(plot, filename=os.path.join(directory_path, f"stochastic_clvq_1d_method_comparison_M_{M}.png"))
    export_svg(plot, filename=os.path.join(directory_path, f"stochastic_clvq_1d_method_comparison_M_{M}.svg"))


def plot_ratios_clvq(df_grouped, M, directory_path):
    color_numpy_cpu = Viridis[3][1]
    color_pytorch_cpu = Viridis[3][2]
    color_pytorch_cuda = Viridis[3][0]
    # todo: verifier les couleurs ici (peut etre qu'il faut utiliser Viridis[5][0]
    color_pytorch_autograd_cpu = Viridis[3][3]
    color_pytorch_autograd_cuda = Viridis[3][4]
    general_font_size = '14pt'
    # general_font_size = '28pt'

    grouped_by_values = df_grouped.groupby(["method", "M"]).agg(list).to_dict()
    elapsed_times_per_epoch_per_method = grouped_by_values.get("elapsed_time_by_epoch")
    Ns_per_method = grouped_by_values.get("N")
    if (Ns_per_method.get(('pytorch_cuda', M)) != Ns_per_method.get(('pytorch_cpu', M))
            or Ns_per_method.get(('numpy_cpu', M)) != Ns_per_method.get(('pytorch_cpu', M))):
        print(f"Cannot plot ratios for M equals {M} because N values does not match!!")
        return

    rescaled_comparisons = {
        (method, M): np.array(elapsed_times_per_epoch_per_method.get((method, M))) / np.array(
            elapsed_times_per_epoch_per_method.get(("pytorch_cuda", M)))
        for method, M in elapsed_times_per_epoch_per_method
    }
    Ns = Ns_per_method.get(("pytorch_cuda", M))
    Ns = [str(N) for N in Ns]

    # plot = figure(x_range=Ns, plot_width=1200, plot_height=1000)
    plot = figure(x_range=Ns, plot_width=600, plot_height=500)

    plot.xaxis.axis_label = "Grid size (N)"
    plot.xaxis.axis_label_text_font_size = general_font_size

    plot.yaxis.axis_label = "Ratio over PyTorch cuda (T4) time"
    plot.yaxis.axis_label_text_font_size = general_font_size

    source = ColumnDataSource(data={
        'Ns': Ns,
        'pytorch_cuda': rescaled_comparisons.get(("pytorch_cuda", M)),
        'pytorch_cpu': rescaled_comparisons.get(("pytorch_cpu", M)),
        'pytorch_autograd_cuda': rescaled_comparisons.get(("pytorch_autograd_cuda", M)),
        'pytorch_autograd_cpu': rescaled_comparisons.get(("pytorch_autograd_cpu", M)),
        'numpy_cpu': rescaled_comparisons.get(("numpy_cpu", M)),
    })

    plot.vbar(x=dodge('Ns', -0.25, range=plot.x_range), top='pytorch_autograd_cuda', source=source, width=0.1, color=color_pytorch_autograd_cuda, legend_label="pytorch autograd (cuda: T4)")
    plot.vbar(x=dodge('Ns', -0.15, range=plot.x_range), top='pytorch_autograd_cpu', source=source, width=0.1, color=color_pytorch_autograd_cpu, legend_label="pytorch autograd (cpu)")
    plot.vbar(x=dodge('Ns', 0.0, range=plot.x_range), top='pytorch_cuda', source=source, width=0.1, color=color_pytorch_cuda, legend_label="pytorch (cuda: T4)")
    plot.vbar(x=dodge('Ns', 0.15, range=plot.x_range), top='pytorch_cpu', source=source, width=0.1, color=color_pytorch_cpu, legend_label="pytorch (cpu)")
    plot.vbar(x=dodge('Ns', 0.25, range=plot.x_range), top='numpy_cpu', source=source, width=0.1, color=color_numpy_cpu, legend_label="numpy")
    plot.legend.location = "top_left"
    plot.legend.label_text_font_size = general_font_size
    plot.xaxis.major_label_text_font_size = general_font_size
    plot.yaxis.major_label_text_font_size = general_font_size

    # show(plot)
    export_png(plot, filename=os.path.join(directory_path, f"stochastic_clvq_1d_ratio_comparison_M_{M}.png"))
    export_svg(plot, filename=os.path.join(directory_path, f"stochastic_clvq_1d_ratio_comparison_M_{M}.svg"))

