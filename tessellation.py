import numpy as np

from dataclasses import dataclass, field
from typing import List, Union
from scipy.spatial import Voronoi
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, Range1d
from bokeh.palettes import Viridis256
from bokeh.plotting import figure, show
from bokeh.io import export_png, export_svg
from utils import Point


@dataclass
class VoronoiCell:
    centroid: Point
    vertices: List[Point]


@dataclass
class VoronoiTessellation:
    centroids: List[Point]
    tessellation: List[VoronoiCell] = field(init=False)
    cells: List[VoronoiCell] = field(init=False)
    M: float = field(init=False, default=20.)
    outer_box: List[Point] = field(init=False)

    def __post_init__(self):
        self.outer_box = [np.array([0, -self.M]), np.array([0, self.M]), np.array([self.M, 0]), np.array([-self.M, 0])]
        # self.outer_box = [np.array([-self.M, -self.M]), np.array([-self.M, self.M]), np.array([self.M, -self.M]), np.array([self.M, self.M])]
        self.build_tessellation()

    def build_tessellation(self):
        pts = self.centroids.copy() + self.outer_box
        vor = Voronoi(pts)

        self.cells = []
        for i in range(len(self.centroids)):
            region_number = vor.point_region[i]
            vertices = [[vor.vertices[index][0], vor.vertices[index][1]] for index in vor.regions[region_number]]
            self.cells.append(VoronoiCell(centroid=self.centroids[i], vertices=vertices))

    def update_cells(self, centroids: List[Point]):
        self.centroids = centroids.copy()
        self.build_tessellation()

    def save_quantization_bokeh(self, name, probabilities, show_color_bar=True):
        plot = self.plot_quantization_bokeh(probabilities, show_color_bar=show_color_bar)
        export_svg(plot, filename=name+'.svg')
        export_png(plot, filename=name+'.png')

    def show_quantization_bokeh(self, probabilities: Union[List[float], None], show_color_bar=True):
        plot = self.plot_quantization_bokeh(probabilities, show_color_bar=show_color_bar)
        show(plot)

    def plot_quantization_bokeh(self, probabilities: List[float], show_color_bar=True):
        size_dots = 25 if len(self.cells) < 75 else 20

        width_color_bar = 20
        palette = Viridis256

        x_vertices = list()
        y_vertices = list()
        x_centroid = list()
        y_centroid = list()
        for i, cell in enumerate(self.cells):
            x_vertices.append([s[0] for s in cell.vertices])
            y_vertices.append([s[1] for s in cell.vertices])
            x_centroid.append(cell.centroid[0])
            y_centroid.append(cell.centroid[1])

        df = dict()
        df['x_vertices'] = x_vertices
        df['y_vertices'] = y_vertices
        df['x_centroid'] = x_centroid
        df['y_centroid'] = y_centroid
        # max_weight = max(probabilities)
        # df['ws'] = [w/max_weight for w in grid.cellsWeight]
        if probabilities is not None:
            df['ws'] = probabilities
        else:
            df['ws'] = range(0, len(self.cells))

        dfsource = ColumnDataSource(data=df)

        # palette.reverse()
        color_mapper = LinearColorMapper(palette=palette, low=min(df['ws']), high=max(df['ws']))
        plot = figure(
            plot_width=600+width_color_bar+30 if show_color_bar else 600,
            plot_height=600
        )
        left, right, bottom, top = -5, 5, -5, 5
        plot.x_range = Range1d(left, right)
        plot.y_range = Range1d(bottom, top)
        plot.patches(
            xs='x_vertices',
            ys='y_vertices',
            source=dfsource,
            line_width=2,
            fill_color={'field': 'ws', 'transform': color_mapper}
        )
        plot.dot(
            x='x_centroid',
            y='y_centroid',
            source=dfsource,
            size=size_dots,
            color='red'
        )
        if show_color_bar:
            color_bar = ColorBar(color_mapper=color_mapper, width=width_color_bar, location=(0, 0))
            plot.add_layout(color_bar, 'right')
        return plot
