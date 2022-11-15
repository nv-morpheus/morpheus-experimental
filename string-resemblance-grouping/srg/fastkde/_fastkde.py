# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from math import ceil
from math import exp
from math import factorial
from math import floor
from math import log
from math import pi
from math import sqrt
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from srg.utils import create_x_axis
from srg.utils import find_local_max
from srg.utils import find_local_min

import dask

import cudf
import dask_cudf


def _b(rex, rx, p, h):
    return min(rex, (rx + sqrt(pow(rx, 2.0) + 8 * p * pow(h, 2.0))) / 2)


def _p_check(r, p, rex, rx, h, eps):
    b = _b(rex, rx, p, h)
    return sqrt(factorial(r)) / factorial(p) * pow(rx * b / pow(h, 2.0), p) * exp(
        -pow(rx - b, 2.0) / pow(2.0 * h, 2.0)) <= eps


@dataclass
class _Params:
    bandwidth: Union[float, None]
    Xmin: float
    Xmax: float
    n: int
    std: float
    r: int
    error: float

    def __post_init__(self):
        if self.bandwidth is None:
            self.bandwidth = pow((4 * pow(self.std, 5)) / (3 * self.n), 0.2)
        if self.bandwidth > 0:

            self.scale = self.Xmax - self.Xmin
            self.h = self.bandwidth / self.scale
            self.q = pow(-1, self.r) / sqrt(2 * pi) * self.n * pow(self.h, self.r + 1)
            self.eps_prime = self.error / (self.n * abs(self.q))
            self.r_x = self.h / 2.0
            self.centers = [self.r_x + i * 2.0 * self.r_x for i in range(int(ceil(1.0 / self.h)))]
            self.boundaries = [i * self.h for i in range(int(ceil(1.0 / self.h)))]
            self.r_ex = self.r_x + 2 * self.h * sqrt(log(sqrt(factorial(self.r)) / self.eps_prime))
            self.p = 1
            while not _p_check(self.r, self.p, self.r_ex, self.r_x, self.h, self.eps_prime) and self.p <= 100:
                self.p += 1

        else:
            self.scale = 0
            self.h = 0
            self.q = 0
            self.eps_prime = 0
            self.r_x = 0
            self.centers = [0]
            self.boundaries = [0]
            self.r_ex = 0
            self.p = 0


class FastKDE:
    """
    Approximation method for univariate Gaussian kernel density estimation
    """
    def __init__(self, derivative: int = 0, error: float = 0.0001):

        self._derivative = derivative
        self._error = error
        self._params = None
        self._groups = None
        self._B = None
        self._a = None
        self._model_built = False
        self._flat_model = False
        """
        Parameters
        ----------
        derivative: int
            The derivative of the kernel density estimation function.
        error: float
            The specified precision of the kernel density estimation function.
        """

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def derivative(self):
        return self._derivative

    @property
    def error(self):
        return self._error

    def fit(self,
            X,
            delimiter=None,
            names=None,
            npartitions: int = 2,
            column: Union[str, None] = None,
            groupby: Union[str, None] = None,
            bandwidth: float = None):

        """
        Fits the kernel density estimation function to the provided univariate data.

        Parameters
        ----------
        X
            A pre-loaded collection of data or the path to a csv
        delimiter: str
            If loading a csv from disk, this is the delimiter of the csv. Leave as `None` to use the default of the
            loader.
        names: List[str]
            Manually define the csv column names
        column: str
            The column containing the data for the KDE
        groupby: str
            This allows multiple KDE's to be generated for a single data aggregated by a grouping column
        bandwidth: float
            Manually specify the desired bandwidth for the KDE. Otherwise a bandwidth is selected based on the standard
            deviation of the data.
        """
        if groupby is None:
            self._fit_series(X, delimiter, names, npartitions, column, bandwidth)
        else:
            self._fit_by_group(X, delimiter, names, npartitions, column, groupby, bandwidth)

    def _fit_series(self, X, delimiter, names, npartitions, column, bandwidth):
        if isinstance(X, (dask.dataframe.core.Series, dask.dataframe.core.DataFrame)) and \
         not isinstance(X, (dask_cudf.core.DataFrame, dask_cudf.core.Series)):
            X = dask_cudf.from_dask_dataframe(X)
        if isinstance(X, str):
            df = dask_cudf.read_csv(X)
            if column is None:
                dask_series = df.iloc[:, 0]
            else:
                dask_series = df[column]
        elif isinstance(X, (list, cudf.core.series.Series)):
            series = cudf.Series(X) if isinstance(X, list) else X
            dask_series = dask_cudf.from_cudf(series, npartitions=npartitions)
        elif isinstance(X, (dask_cudf.core.DataFrame, cudf.core.dataframe.DataFrame)):
            ddf = dask_cudf.from_cudf(X, npartitions=npartitions) if isinstance(X, cudf.core.dataframe.DataFrame) else X
            if column is None:
                dask_series = ddf.iloc[:, 0]
            else:
                dask_series = ddf[column]
        else:
            dask_series = X
        assert isinstance(dask_series, dask_cudf.core.Series), \
            'X must be a path to a csv, list, or a (dask-)cudf Series or DataFrame'

        description = dask_series.describe().compute()
        self._params = _Params(bandwidth=bandwidth,
                               Xmin=description['min'],
                               Xmax=description['max'],
                               n=description['count'],
                               std=description['std'],
                               r=self._derivative,
                               error=self._error)
        if self._params.Xmin == self._params.Xmax:
            self._model_built = True
            self._flat_model = True

            return self

        scaled = (dask_series - self._params.Xmin) / self._params.scale
        self._B = scaled.reduction(self._calculate_B, aggregate=self._combine_B, meta=('B', list)).compute()
        self._B = np.asarray(self._B[0])
        self._a = self._calculate_a()

        self._model_built = True

        return self

    def _fit_by_group(self, X, delimiter, names, npartitions, column, groupby, bandwidth):
        self._col = column or 'variable'

        if isinstance(X, (dask.dataframe.core.Series, dask.dataframe.core.DataFrame)) and \
           not isinstance(X, (dask_cudf.core.DataFrame, dask_cudf.core.Series)):
            X = dask_cudf.from_dask_dataframe(X)
        if isinstance(X, str):
            ddf = dask_cudf.read_csv(X, delimiter=delimiter, names=names)
            if column is None:
                ddf = ddf.rename(columns={0: 'variable'})
        elif isinstance(X, (list, cudf.core.series.Series)):
            series = cudf.Series(X) if isinstance(X, list) else X
            cdf = cudf.DataFrame({self._col: series})
            ddf = dask_cudf.from_cudf(cdf, npartitions=npartitions)
        elif isinstance(X, (dask_cudf.core.DataFrame, cudf.core.dataframe.DataFrame)):
            ddf = dask_cudf.from_cudf(X, npartitions=npartitions) if isinstance(X, cudf.core.dataframe.DataFrame) else X
            if column is None:
                ddf = ddf.rename(columns={0: 'variable'})
        else:
            ddf = X
        assert isinstance(ddf, dask_cudf.core.DataFrame), \
            'X must be a path to a csv, list, or a (dask-)cudf Series or DataFrame'
        assert groupby in ddf.columns, "Group column not found"
        ddf.persist()
        raw_groups = ddf[groupby].unique().compute()
        if isinstance(raw_groups, cudf.core.series.Series):
            self._groups = raw_groups.to_pandas().to_list()
        else:
            self._groups = raw_groups.to_list()
        raw_descriptions = ddf.groupby(groupby).agg({self._col: ['count', 'min', 'max', 'std']}).compute()
        if isinstance(raw_descriptions, cudf.core.dataframe.DataFrame):
            descriptions = raw_descriptions.to_pandas().to_dict('index')
        else:
            descriptions = raw_descriptions.to_dict('index')
        self._params = dict()
        self._flat_model = []
        for group in self._groups:
            xmin = descriptions[group][(self._col, 'min')]
            xmax = descriptions[group][(self._col, 'max')]
            params = _Params(bandwidth=bandwidth,
                             Xmin=xmin,
                             Xmax=xmax,
                             n=descriptions[group][(self._col, 'count')],
                             std=descriptions[group][(self._col, 'std')],
                             r=self._derivative,
                             error=self._error)
            self._params[group] = params
            if xmin == xmax:
                self._flat_model.append(group)
        scaled = ddf.map_partitions(lambda df: self._scale_df(df, groupby)).dropna(subset=['scaled'])
        raw_B = scaled.reduction(chunk=self._group_calculate_B,
                                 chunk_kwargs={
                                     'groupby': groupby
                                 },
                                 aggregate=self._group_combine_B,
                                 meta=dict).compute()
        self._B = dict()
        for group, B in raw_B.items():
            self._B[group] = np.asarray(B)
        self._a = self._calculate_a()

        self._model_built = True

    def _group_calculate_B(self, df, groupby):
        B = dict()
        for group in self._groups:
            K = self._params[group].p
            T = self._params[group].r + 1
            L = len(self._params[group].centers)
            B[group] = [[[0.0 for _ in range(T)] for _ in range(K)] for _ in range(L)]
        for row in df.to_dict('records'):
            group = row[groupby]
            K = self._params[group].p
            T = self._params[group].r + 1
            L = len(self._params[group].centers)
            x_i = row['scaled']
            little_l = sum([b <= x_i for b in self._params[group].boundaries]) - 1
            c_l = self._params[group].centers[little_l]
            for k in range(K):
                for t in range(T):
                    B[group][little_l][k][t] = B[group][little_l][k][t] + 1/factorial(k) * self._params[group].q * \
                        exp(-pow(abs(x_i-c_l), 2.0)/(2 * pow(self._params[group].h, 2.0))) * \
                        pow((x_i-c_l)/self._params[group].h, k+t)
        return B

    def _group_combine_B(self, s):
        B = dict()
        for group in self._groups:
            K = self._params[group].p
            T = self._params[group].r + 1
            L = len(self._params[group].centers)
            B[group] = [[[0.0 for _ in range(T)] for _ in range(K)] for _ in range(L)]
        for B_dict in s:
            for group, B_i in B_dict.items():
                for firstIdx, oneD in enumerate(B_i):
                    for secondIdx, twoD in enumerate(oneD):
                        for thirdIdx, threeD in enumerate(twoD):
                            B[group][firstIdx][secondIdx][thirdIdx] = B[group][firstIdx][secondIdx][thirdIdx] + threeD
        return B

    def _scale_df(self, df, group):
        if isinstance(df, cudf.core.dataframe.DataFrame):
            pdf = df.to_pandas().copy()
        else:
            pdf = df.copy()
        pdf['scaled'] = pdf.apply(lambda x: (x[self._col] - self._params[x[group]].Xmin) / self._params[x[group]].scale
                                  if self._params[x[group]].scale != 0 else None,
                                  axis=1)
        return pdf

    def transform(self, X, input_column=None, group=None, output_column='density'):
        """
        Calculates the KDE for the each point of the provided data

        Parameters
        ----------
        X
            A pre-loaded collection of data or a single value to apply the KDE function to.
        input_column: str
            The column containing the data for the KDE
        group: str
            The group column or specific group if multiple KDE's were generated.
        ouptut_column: str
            The name of the output column if a collection of data points was passed into the transform method.

        Returns
        -------
        float or List[float] or dask_cudf.core.DataFrame
            If a single point is passed into transform, a single float is returned. If a list is passed in then a list
            is returned. Otherwise, a dask_cudf DataFrame is returned if a Series or DataFrame is used.
        """
        assert self._model_built, 'Density not yet estimated'
        if isinstance(self._B, dict):
            return self._groupby_model_transform(X, input_column, group, output_column)
        else:
            return self._single_model_transform(X, input_column, output_column)

    def _groupby_model_transform(self, X, input_column, group, output_column):
        if isinstance(X, (dask.dataframe.core.Series, dask.dataframe.core.DataFrame)) and \
         not isinstance(X, (dask_cudf.core.DataFrame, dask_cudf.core.Series)):
            X = dask_cudf.from_dask_dataframe(X)
        if isinstance(X, (int, float)):
            assert group in self._groups, 'model for specified group not found'
            return self._single_group_transform(X, group)
        elif isinstance(X, list):
            assert group in self._groups, 'model for specified group not found'
            if isinstance(X[0], tuple):
                return [(group, self._single_group_transform(x, group)) for (group, x) in X]
            else:
                return [self._single_group_transform(x, group) for x in X]
        elif isinstance(X, cudf.core.series.Series):
            assert group in self._groups, 'model for specified group not found'
            return X.applymap(lambda x: self._single_transform(x, group))
        elif isinstance(X, dask_cudf.core.Series):
            assert group in self._groups, 'model for specified group not found'
            return X.map_partitions(lambda x: self._single_transform(x, group))
        elif isinstance(X, dask_cudf.core.DataFrame):
            return X.map_partitions(lambda df: self._group_df_transform(df, input_column, group, output_column))
        else:
            print("Unknown input type")
            return None

    def _single_model_transform(self, X, input_column, output_column):
        if isinstance(X, (dask.dataframe.core.Series, dask.dataframe.core.DataFrame)) and \
         not isinstance(X, (dask_cudf.core.DataFrame, dask_cudf.core.Series)):
            X = dask_cudf.from_dask_dataframe(X)
        if isinstance(X, (int, float)):
            return self._single_transform(X)
        elif isinstance(X, list):
            return [self._single_transform(x) for x in X]
        elif isinstance(X, cudf.core.series.Series):
            return X.applymap(self._single_transform)
        elif isinstance(X, dask_cudf.core.Series):
            return X.map_partitions(self._single_transform)
        elif isinstance(X, dask_cudf.core.DataFrame):
            return X.map_partitions(lambda df: self._df_transform(df, input_column, output_column))
        else:
            print("Unknown input type")
            return None

    def _x_axis(self, increments):
        start = self._params.Xmin
        stop = self._params.Xmax
        step = (stop - start) / increments
        return create_x_axis(start, stop, step)

    def get_local_extrema(self, increments=1000):
        """
        Determines the local minima and maxima of the KDE.

        Parameters
        ----------
        increments: int
            The number of points to use to fill out the KDE curve.

        Returns
        -------
        local_minima: List[float]
            List of floats containing the local minima
        local_maxima: List[float]
            List of floats containing the local maxima
        """
        assert self._model_built, 'Density not yet estimated'
        if isinstance(self._B, dict):
            local_min = dict()
            local_max = dict()
            for group in self._groups:
                start = self._params[group].Xmin
                stop = self._params[group].Xmax
                step = (stop - start) / increments
                if step == 0:
                    local_min[group] = [start]
                    local_max[group] = [start]
                else:
                    x_axis = create_x_axis(start, stop, step)
                    y_axis = self.transform(x_axis, group=group)
                    local_min[group] = find_local_min(x_axis, y_axis)
                    local_max[group] = find_local_max(x_axis, y_axis)
        else:
            x_axis = self._x_axis(increments)
            y_axis = self.transform(x_axis)
            local_min = find_local_min(x_axis, y_axis)
            local_max = find_local_max(x_axis, y_axis)
        return local_min, local_max

    def plot_density(self, increments=1000, width=10, height=10, title=None):
        """
        Return a plot of the KDE curve.

        Parameters
        ----------
        increments: int
            The number of points to use to fill out the KDE curve.
        width: int
            The width of the plot
        height: int
            The height of the plot
        title: str
            The title for the plot

        Returns
        -------
        matplotlib.pyplot.figure
            Figure object of the plot
        """
        assert self._model_built, 'Density not yet estimated'
        assert not isinstance(self._B, dict), 'groupby kde plotting not implemented'
        fig = plt.figure(figsize=(width, height))
        if title is None:
            title = 'Gaussian Kernel Approximation with bandwidth=%f' % self._params.bandwidth
        ax = fig.add_subplot(title=title)

        x_axis = self._x_axis(increments)
        y_axis = self.transform(x_axis)

        ax.plot(x_axis, y_axis)

        plt.close(fig)
        return fig

    def _df_transform(self, df, incol, outcol):
        pdf = df.to_pandas().copy()
        pdf[outcol] = pdf.apply(lambda x: self._single_transform(x[incol]), axis=1)
        return pdf

    def _group_df_transform(self, df, incol, groupby, outcol):
        pdf = df.to_pandas().copy()
        pdf[outcol] = pdf.apply(lambda x: self._single_group_transform(x[incol], x[groupby]), axis=1)
        return pdf

    def _single_transform(self, x):
        if self._params.scale == 0:
            return self._params.Xmin
        x = (x - self._params.Xmin) / self._params.scale
        int_centers = [(interval, center) for (interval, center) in enumerate(self._params.centers)
                       if abs(x - center) <= self._params.r_ex]
        kd = 0.0
        for little_l, center in int_centers:
            for k in range(self._params.p):
                for s in range(int(floor(self._params.r / 2) + 1)):
                    for t in range(self._params.r - 2 * s + 1):
                        kd = kd + self._a[s, t]*self._B[little_l, k, t] *\
                            exp(-pow(abs(x-center), 2.0)/(2*pow(self._params.h, 2.0))) *\
                            pow((x - center)/self._params.h, k+self._params.r-2*s-t)
        final_kd = kd / pow(self._params.scale, self._params.r + 1)
        return final_kd

    def _single_group_transform(self, x, group):
        if self._params[group].scale == 0:
            return self._params[group].n
        x = (x - self._params[group].Xmin) / self._params[group].scale
        int_centers = [(interval, center) for (interval, center) in enumerate(self._params[group].centers)
                       if abs(x - center) <= self._params[group].r_ex]
        kd = 0.0
        for little_l, center in int_centers:
            for k in range(self._params[group].p):
                for s in range(int(floor(self._params[group].r / 2) + 1)):
                    for t in range(self._params[group].r - 2 * s + 1):
                        kd = kd + self._a[s, t] * self._B[group][little_l, k, t] * \
                              exp(-pow(abs(x-center), 2.0)/(2*pow(self._params[group].h, 2.0))) * \
                              pow((x - center)/self._params[group].h, k+self._params[group].r-2*s-t)
        final_kd = kd / pow(self._params[group].scale, self._params[group].r + 1)
        return final_kd

    def _calculate_B(self, s):
        K = self._params.p
        T = self._params.r + 1
        L = len(self._params.centers)
        B = [[[0.0 for _ in range(T)] for _ in range(K)] for _ in range(L)]
        for x_i in s.values_host:
            little_l = sum([b <= x_i for b in self._params.boundaries]) - 1
            c_l = self._params.centers[little_l]
            for k in range(K):
                for t in range(T):
                    B[little_l][k][t] = B[little_l][k][t] + 1/factorial(k) * \
                        self._params.q * exp(-pow(abs(x_i-c_l), 2.0)/(2 * pow(self._params.h, 2.0))) * \
                        pow((x_i-c_l)/self._params.h, k+t)
        return B

    def _combine_B(self, s):
        K = self._params.p
        T = self._params.r + 1
        L = len(self._params.centers)
        B = [[[0.0 for _ in range(T)] for _ in range(K)] for _ in range(L)]
        for B_i in s:
            for firstIdx, oneD in enumerate(B_i):
                for secondIdx, twoD in enumerate(oneD):
                    for thirdIdx, threeD in enumerate(twoD):
                        B[firstIdx][secondIdx][thirdIdx] = B[firstIdx][secondIdx][thirdIdx] + threeD
        return B

    def _calculate_a(self) -> np.ndarray:
        if self._groups is not None:
            r = self._params[self._groups[0]].r
        else:
            r = self._params.r
        S = floor(r / 2) + 1
        T = r + 1
        a = np.zeros((S, T))
        for s in range(S):
            for t in range(T):
                a[s, t] = pow(
                    -1, s + t) * factorial(r) / (pow(2, s) * factorial(s) * factorial(t) * factorial(r - 2 * s - t))
        return a
