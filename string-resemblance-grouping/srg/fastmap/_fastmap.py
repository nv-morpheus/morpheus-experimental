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

import os
import random
from dataclasses import dataclass
from math import sqrt
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import dill as pickle
import numpy as np
from numpy.random import poisson
from srg.fastmap.distances._distance import Distance
from tqdm import tqdm_notebook

import dask

import cudf
import dask_cudf


class ModelError(Exception):
    def __init__(self, message):
        self.message = message


class FastMapObject:
    def __init__(self,
                 obj: object,
                 transform: Union[object, Callable[[Any], Any], None] = None,
                 hasher: Union[None, Callable[[Any], int]] = None):

        self._obj = obj
        self._transform = transform(obj) if isinstance(transform, Callable) else transform
        self._hasher = hasher

    def __hash__(self):
        if self._hasher is None:
            return hash(self._obj)
        else:
            return self._hasher(self._obj)

    @property
    def obj(self):
        return self._obj

    @property
    def transform(self):
        return self._transform or self._obj


@dataclass
class Pivots:
    """
    A helper class containing two anchor pivots for the FastMap method.
    """
    left: object
    left_proj: list
    right: object
    right_proj: list
    distance: float

    def _update_right(self, r, rp, d):
        self.right = r
        self.right_proj = rp
        self.distance = d

    def _swap(self):
        l_tmp = self.left
        lp_tmp = self.left_proj
        self.left = self.right
        self.left_proj = self.right_proj
        self.right = l_tmp
        self.right_proj = lp_tmp

    def get_left(self):
        return self.left, self.left_proj

    def get_right(self):
        return self.right, self.right_proj


class FastMapModel:

    def __init__(self,
                 dim: int,
                 distance: Distance,
                 dist_args: Dict[str, Any] = dict(),
                 obj_transformer: Callable[[Any], Any] = None):

        self._dim: int = dim
        self._pivots: List[Pivots] = []
        self._distance = distance(**dist_args)
        self._obj_transformer = obj_transformer
        self._groups = None

    @property
    def dim(self):
        return self._dim

    def _set_initial_pivots(self, obj, position):
        proj = self._i_proj(obj, position)
        self._pivots.insert(position, Pivots(obj, proj, obj, proj, 0.0))

    def _check_right_candidate(self, current, curr_proj, current_dist, candidate, position):
        cand_proj = self._i_proj(candidate, position)
        left, left_proj = self._pivots[position].get_left()
        d = self._dist(left, left_proj, candidate, cand_proj, position)
        if current is None:
            return candidate, cand_proj, d
        elif d > current_dist:
            return candidate, cand_proj, d
        else:
            return current, curr_proj, current_dist

    def _swap(self, position):
        self._pivots[position]._swap_and_reset()


class FastMap:
    """
    A multidimensional scaling technique [1]_ for projecting objects into a vector space designed to maintain the
    relative distance between objects for a given distance metric.

    .. _[1]:
        Faloutsos, C., & Lin, K. I. (1995, May). FastMap: A fast algorithm for indexing, data-mining and
        visualization of traditional and multimedia datasets. In Proceedings of the 1995 ACM SIGMOD international
        conference on Management of data (pp. 163-174).
    """
    def __init__(self,
                 dim: int,
                 distance: Distance,
                 dist_args: Dict[str, Any] = dict(),
                 num_models: int = 1,
                 iters: int = 5):
        """
        Parameters
        ----------
        dim: int
            The dimension of the resulting vector space embedding
        distance: Distance
            The distance metric to use to create the vector space embedding
        dist_args: Dict[str, Any]
            Any arguments to pass to the distance metric method
        num_models: int
            The number of simultaneous FastMap models to build over the input objects
        iters: int
            The number of iterations to find anchor points for each model
        """
        self._dim = dim
        self._dist_args = dist_args
        self._distance = distance(**dist_args)
        self._iters = iters
        self._num_models = num_models
        self._pivots = []
        self._model_built = False
        self._group_models = False

    @property
    def distance(self):
        return self._distance.get_name()

    def _pivot_distance(self, left, right, position, model_num):
        left_proj = self._i_proj(left, position, model_num)
        right_proj = self._i_proj(right, position, model_num)
        return self._dist(left, left_proj, right, right_proj, position)

    def _group_pivot_distance(self, left, right, position, model_num, group):
        left_proj = self._group_i_proj(left, position, model_num, group)
        right_proj = self._group_i_proj(right, position, model_num, group)
        return self._dist(left, left_proj, right, right_proj, position)

    def _candidate_distances(self, df, position, left_pivots):
        pdf = df.to_pandas().copy()
        for i in range(self._num_models):
            pdf['model_%i' % i] = pdf.apply(lambda x: self._pivot_distance(left_pivots[i], x[self._col], position, i),
                                            axis=1,
                                            result_type='expand')
        return pdf

    def _groupby_candidate_distances(self, df, groupby, position, left_pivots):
        pdf = df.to_pandas().copy()
        for i in range(self._num_models):
            pdf['model_%i' % i] = pdf.apply(
                lambda x: self._group_pivot_distance(left_pivots[x[groupby]][i], x[self._col], position, i, x[groupby]),
                axis=1,
                result_type='expand')
        return pdf

    def fit(self,
            X,
            delimiter=None,
            names=None,
            npartitions: int = 2,
            column: Union[str, None] = None,
            groupby: Union[str, None] = None):
        """
        Find the FastMap model for the given input data

        Parameters
        ----------
        X: str or dask_cudf.DataFrame or list
            The path to a csv to load or a collection of objects to project
        delimiter: str
            If a csv path is specified, this indicates the delimiter to use when reading the file
        names: List[str]
            Manually specify the column names for an input csv
        npartitions: int
            The number of partitions for the dask_cudf DataFrame
        column: str
            The column containing the data to be projected
        groupby: str
            Specify the column grouping for building submodels.
        """
        if groupby is None:
            self._full_fit(X, delimiter, names, npartitions, column)
        else:
            self._groupby_fit(X, delimiter, names, npartitions, column, groupby)

    def _groupby_fit(self, X, delimiter, names, npartitions, column, groupby):
        self._col = column or 'obj'
        if isinstance(X, (dask.dataframe.core.Series, dask.dataframe.core.DataFrame)) and\
           not isinstance(X, (dask_cudf.core.DataFrame, dask_cudf.core.Series)):
            X = dask_cudf.from_dask_dataframe(X)
        if isinstance(X, str):
            ddf = dask_cudf.read_csv(X, delimiter=delimiter, names=names)
            if column is None:
                ddf = ddf.rename(columns={0: 'obj'})
        elif isinstance(X, (list, cudf.core.series.Series)):
            series = cudf.Series(X) if isinstance(X, list) else X
            cdf = cudf.DataFrame({self._col: series})
            ddf = dask_cudf.from_cudf(cdf, npartitions=npartitions)
        elif isinstance(X, (dask_cudf.core.DataFrame, cudf.core.dataframe.DataFrame)):
            ddf = dask_cudf.from_cudf(X, npartitions=npartitions) if isinstance(X, cudf.core.dataframe.DataFrame) else X
            if column is None:
                ddf = ddf.rename(columns={0: 'obj'})
        else:
            ddf = X
        assert isinstance(ddf, dask_cudf.core.DataFrame),\
            'X must be a path to a csv, list, or a (dask-)cudf Series or DataFrame'
        if groupby is not None:
            assert groupby in ddf.columns

        ddf = ddf.persist()
        raw_groups = ddf[groupby].unique().compute()
        try:
            self._groups = raw_groups.to_pandas().to_list()
        except Exception as e:
            try:
                print(e)
                self._groups = raw_groups.to_list()
            except Exception as e1:
                print(e1)
                self._groups = None
        sample_fracs = {
            group: max(50, 2 * self._num_models) / group_count
            for group,
            group_count in ddf.groupby(groupby).count()[self._col].compute().to_pandas().to_dict().items()
        }
        left = ddf.head(1)[self._col].to_arrow().to_pylist()[0]
        self._pivots = {
            group: {
                model_num: {
                    0: {
                        'left': left,
                        'left_proj': [0.0 for _ in range(self._dim)],
                        'right': left,
                        'right_proj': [0.0 for _ in range(self._dim)],
                        'dist': 0.0
                    }
                }
                for model_num in range(self._num_models)
            }
            for group in self._groups
        }
        for k in tqdm_notebook(range(self._dim), desc='Current dimension'):
            initial_group_samples = ddf.reduction(chunk=self._groupby_initial_pivots_chunks,
                                                  chunk_kwargs={
                                                      'groupby': groupby, 'sample_fracs': sample_fracs
                                                  },
                                                  aggregate=self._groupby_initial_pivots_aggregate,
                                                  meta=dict).compute()

            initial_pivots = {
                group: random.sample(list(samples), self._num_models)
                for group,
                samples in initial_group_samples.items()
            }
            right_pivots = {group: [[left, 0.0] for left in initial_pivots[group]] for group in self._groups}
            for m in tqdm_notebook(range(self._iters), desc='Iteration', leave=False):
                left_pivots = {group: [new_left for [new_left, _] in right_pivots[group]] for group in self._groups}
                assigned_ddf = ddf.map_partitions(
                    lambda df: self._groupby_candidate_distances(df, groupby, k, left_pivots))
                right_pivots = assigned_ddf.reduction(chunk=self._groupby_chunk,
                                                      chunk_kwargs={
                                                          'groupby': groupby, 'lefts': left_pivots
                                                      },
                                                      aggregate=self._groupby_aggregate,
                                                      aggregate_kwargs={
                                                          'lefts': left_pivots
                                                      },
                                                      meta=dict).compute()
            for group in self._groups:
                for i in range(self._num_models):
                    left, right, dist = left_pivots[group][i], right_pivots[group][i][0], right_pivots[group][i][1]
                    final_pivots = {
                        'left': left,
                        'left_proj': self._i_proj(left, k, i),
                        'right': right,
                        'right_proj': self._i_proj(right, k, i),
                        'dist': dist
                    }
                    self._pivots[group][i][k] = final_pivots
        self._model_built = True
        self._group_models = True

    def _groupby_initial_pivots_chunks(self, df, groupby, sample_fracs):
        group_samples = {group: set() for group in self._groups}
        for row in df.to_pandas().to_dict('records'):
            group = row[groupby]
            frac = sample_fracs[group]
            if poisson(frac):
                group_samples[row[groupby]].add(row[self._col])
        pruned_samples = dict()
        for group, samples in group_samples.items():
            if len(samples) <= self._num_models:
                pruned_samples[group] = list(samples)
            else:
                pruned_samples[group] = random.sample(samples, self._num_models)
        return group_samples

    def _groupby_initial_pivots_aggregate(self, s):
        group_samples = {group: set() for group in self._groups}
        for gs in s:
            for group, samples in gs.items():
                group_samples[group].update(samples)
        pruned_samples = dict()
        for group, samples in group_samples.items():
            if len(samples) <= self._num_models:
                pruned_samples[group] = list(samples)
            else:
                pruned_samples[group] = random.sample(samples, self._num_models)
        return group_samples

    def _full_fit(self, X, delimiter, names, npartitions, column):
        self._col = column or 'obj'
        if isinstance(X, (dask.dataframe.core.Series, dask.dataframe.core.DataFrame)) and\
           not isinstance(X, (dask_cudf.core.DataFrame, dask_cudf.core.Series)):
            X = dask_cudf.from_dask_dataframe(X)
        if isinstance(X, str):
            ddf = dask_cudf.read_csv(X, delimiter=delimiter, names=names)
            if column is None:
                ddf = ddf.rename(columns={0: 'obj'})
        elif isinstance(X, (list, cudf.core.series.Series)):
            series = cudf.Series(X) if isinstance(X, list) else X
            cdf = cudf.DataFrame({self._col: series})
            ddf = dask_cudf.from_cudf(cdf, npartitions=npartitions)
        elif isinstance(X, (dask_cudf.core.DataFrame, cudf.core.dataframe.DataFrame)):
            ddf = dask_cudf.from_cudf(X, npartitions=npartitions) if isinstance(X, cudf.core.dataframe.DataFrame) else X
            if column is None:
                ddf = ddf.rename(columns={0: 'obj'})
        else:
            ddf = X
        assert isinstance(ddf, dask_cudf.core.DataFrame),\
            'X must be a path to a csv, list, or a (dask-)cudf Series or DataFrame'

        ddf = ddf.persist()
        n = ddf[self._col].count().compute()
        samp_frac = max(50, 3 * self._num_models) / n
        left = ddf.head(1)[self._col].to_arrow().to_pylist()[0]
        self._pivots = {
            model_num: {
                0: {
                    'left': left,
                    'left_proj': [0.0 for _ in range(self._dim)],
                    'right': left,
                    'right_proj': [0.0 for _ in range(self._dim)],
                    'dist': 0.0
                }
            }
            for model_num in range(self._num_models)
        }
        for k in tqdm_notebook(range(self._dim), desc='Current dimension'):
            init_pivs = random.sample(list(set(ddf[self._col].sample(frac=samp_frac).compute().to_arrow().to_pylist())),
                                      self._num_models)
            right_pivots = [[left, 0.0] for left in init_pivs]
            for m in tqdm_notebook(range(self._iters), desc='Iteration', leave=False):
                left_pivots = [new_left for [new_left, _] in right_pivots]
                assigned_ddf = ddf.map_partitions(lambda df: self._candidate_distances(df, k, left_pivots))
                right_pivots = assigned_ddf.reduction(chunk=self._chunk,
                                                      chunk_kwargs={
                                                          'lefts': left_pivots
                                                      },
                                                      aggregate=self._aggregate,
                                                      aggregate_kwargs={
                                                          'lefts': left_pivots
                                                      },
                                                      meta=list).compute()
            for i in range(self._num_models):
                left, right, dist = left_pivots[i], right_pivots[i][0], right_pivots[i][1]
                final_pivots = {
                    'left': left,
                    'left_proj': self._i_proj(left, k, i),
                    'right': right,
                    'right_proj': self._i_proj(right, k, i),
                    'dist': dist
                }
                self._pivots[i][k] = final_pivots
        self._model_built = True

    def transform(self,
                  X,
                  model: int = None,
                  delimiter=None,
                  names=None,
                  npartitions: int = 2,
                  column: Union[str, None] = None,
                  group: Union[str, None] = None):
        """
        Apply the model to an object or collection of objects and return the embedding.

        Parameters
        ----------
        X: str or list or dask_cudf.DataFrame
            The path to a csv or an individual object or collection of objects to project.
        model: int
            If multiple models were generated, specify the model to use when passing in a single object
        delimiter: str
            If a path is input, this manually sets the csv delimiter
        names: List[str]
            Manually specify the column names of an input csv
        npartitions: int
            The number of partitions for the dask_cudf DataFrame
        column: str
            The column containing the data to be projected
        group: str
            Specify the column containing the grouping if submodels were built.
        """
        if model is not None:
            if model >= self._num_models:
                print('Not a valid model. Transforming all models.')
                model = None
        assert self._model_built, 'Model not built'
        if self._group_models:
            return self._groupby_models_transform(X, model, delimiter, names, npartitions, column, group)
        else:
            return self._full_models_transform(X, model, delimiter, names, npartitions, column)

    def _groupby_models_transform(self, X, model, delimiter, names, npartitions, column, group):
        col = column or 'obj'
        if isinstance(X, (dask.dataframe.core.Series, dask.dataframe.core.DataFrame)) and\
           not isinstance(X, (dask_cudf.core.DataFrame, dask_cudf.core.Series)):
            X = dask_cudf.from_dask_dataframe(X)
        if isinstance(X, str):
            if os.path.isfile(X):
                ddf = dask_cudf.read_csv(X, delimiter=delimiter, names=names)
                if column is None:
                    ddf = ddf.rename(columns={0: 'obj'})
            else:
                projs = [self._group_i_proj(X, self._dim, m, group) for m in range(self._num_models)]
                return projs[0] if self._num_models == 1 else projs
        elif isinstance(X, (dask_cudf.core.DataFrame, cudf.core.dataframe.DataFrame)):
            ddf = dask_cudf.from_cudf(X, npartitions=npartitions) if isinstance(X, cudf.core.dataframe.DataFrame) else X
            if column is None:
                ddf = ddf.rename(columns={0: 'obj'})
        else:
            projs = [self._group_i_proj(X, self._dim, m, group) for m in range(self._num_models)]
            return projs[0] if self._num_models == 1 else projs
        assert isinstance(ddf, dask_cudf.core.DataFrame),\
            'X must be a path to a csv or a (dask-)cudf Series or DataFrame, or a projectable object'
        if self._num_models == 1 or model is not None:
            if model is None or model not in range(self._num_models):
                model = 0
            meta = {c: v for c, v in zip(ddf._meta, ddf._meta.dtypes)}
            for i in range(self._dim):
                meta['x_%i' % i] = 'float'
            projs = ddf.map_partitions(lambda df: self._group_assign_single_projs(df, col, group, model), meta=meta)
        else:
            projs = ddf.map_partitions(lambda df: self._group_assign_multi_projs(df, col, group, model))
        return projs

    def _full_models_transform(self, X, model, delimiter, names, npartitions, column):
        col = column or 'obj'
        if isinstance(X, (dask.dataframe.core.Series, dask.dataframe.core.DataFrame)) and\
           not isinstance(X, (dask_cudf.core.DataFrame, dask_cudf.core.Series)):
            X = dask_cudf.from_dask_dataframe(X)
        if isinstance(X, str):
            if os.path.isfile(X):
                ddf = dask_cudf.read_csv(X, delimiter=delimiter, names=names)
                if column is None:
                    ddf = ddf.rename(columns={0: 'obj'})
            else:
                projs = [self._i_proj(X, self._dim, m) for m in range(self._num_models)]
                return projs[0] if self._num_models == 1 else projs
        elif isinstance(X, (list, cudf.core.series.Series)):
            series = cudf.Series(X) if isinstance(X, list) else X
            cdf = cudf.DataFrame({col: series})
            ddf = dask_cudf.from_cudf(cdf, npartitions=npartitions)
        elif isinstance(X, (dask_cudf.core.DataFrame, cudf.core.dataframe.DataFrame)):
            ddf = dask_cudf.from_cudf(X, npartitions=npartitions) if isinstance(X, cudf.core.dataframe.DataFrame) else X
            if column is None:
                ddf = ddf.rename(columns={0: 'obj'})
        else:
            projs = [self._i_proj(X, self._dim, m) for m in range(self._num_models)]
            return projs[0] if self._num_models == 1 else projs
        assert isinstance(ddf, dask_cudf.core.DataFrame),\
            'X must be a path to a csv, list, or a (dask-)cudf Series or DataFrame, or a projectable object'
        if self._num_models == 1 or model is not None:
            if model is None:
                model = 0
            meta = {c: v for c, v in zip(ddf._meta, ddf._meta.dtypes)}
            for i in range(self._dim):
                meta['x_%i' % i] = 'float'
            projs = ddf.map_partitions(lambda df: self._assign_single_projs(df, col, model), meta=meta)
        else:
            projs = ddf.map_partitions(lambda df: self._assign_multi_projs(df, col, model))
        return projs

    def _assign_single_projs(self, df, col, model):
        pdf = df.to_pandas().copy()
        x_cols = ['x_%i' % i for i in range(self._dim)]
        pdf[x_cols] = pdf.apply(lambda x: self._i_proj(x[col], self._dim, model), axis=1, result_type='expand')
        return pdf

    def _assign_multi_projs(self, df, col, model):
        pdf = df.to_pandas().copy()
        for i in range(self._num_models):
            pdf['model_%i' % i] = pdf.apply(lambda x: self._i_proj(x[col], self._dim, i), axis=1, result_type='reduce')
        return pdf

    def _group_assign_single_projs(self, df, col, groupby, model):
        pdf = df.to_pandas().copy()
        x_cols = ['x_%i' % i for i in range(self._dim)]
        pdf[x_cols] = pdf.apply(lambda x: self._group_i_proj(x[col], self._dim, model, x[groupby]),
                                axis=1,
                                result_type='expand')
        return pdf

    def _group_assign_multi_projs(self, df, col, groupby, model):
        pdf = df.to_pandas().copy()
        for i in range(self._num_models):
            pdf['model_%i' % i] = pdf.apply(lambda x: self._group_i_proj(x[col], self._dim, i, x[groupby]),
                                            axis=1,
                                            result_type='reduce')
        return pdf

    def _compute_proj_i(self, index, model_num, i, obj, obj_proj):

        left_dist = self._dist(self._pivots[model_num][i]['left'],
                               self._pivots[model_num][i]['left_proj'],
                               obj,
                               obj_proj,
                               index)
        right_dist = self._dist(self._pivots[model_num][i]['right'],
                                self._pivots[model_num][i]['right_proj'],
                                obj,
                                obj_proj,
                                index)
        numer = pow(left_dist, 2) + pow(self._pivots[model_num][i]['dist'], 2) - pow(right_dist, 2)
        denom = 2 * self._pivots[model_num][i]['dist']
        if denom == 0:
            return 0
        else:
            return numer / denom

    def _i_proj(self, obj, index, model_num):
        assert len(self._pivots[model_num]) >= index
        x_proj = np.zeros(self._dim)
        for i in range(0, index):
            x_proj[i] = self._compute_proj_i(index, model_num, i, obj, x_proj)
        return x_proj

    def _group_compute_proj_i(self, index, model_num, group, i, obj, obj_proj):

        left_dist = self._dist(self._pivots[group][model_num][i]['left'],
                               self._pivots[group][model_num][i]['left_proj'],
                               obj,
                               obj_proj,
                               index)
        right_dist = self._dist(self._pivots[group][model_num][i]['right'],
                                self._pivots[group][model_num][i]['right_proj'],
                                obj,
                                obj_proj,
                                index)
        numer = pow(left_dist, 2) + pow(self._pivots[group][model_num][i]['dist'], 2) - pow(right_dist, 2)
        denom = 2 * self._pivots[group][model_num][i]['dist']
        if denom == 0:
            return 0
        else:
            return numer / denom

    def _group_i_proj(self, obj, index, model_num, group):
        assert len(self._pivots[group][model_num]) >= index
        x_proj = np.zeros(self._dim)
        for i in range(0, index):
            x_proj[i] = self._group_compute_proj_i(index, model_num, group, i, obj, x_proj)
        return x_proj

    def _dist(self, x, x_proj, y, y_proj, index):
        d_sq = pow(self._distance.calculate(x, y), 2)
        diff_sq = sum([pow(x_i - y_i, 2) for (x_i, y_i) in zip(x_proj[0:index], y_proj[0:index])])
        return sqrt(max(d_sq - diff_sq, 0))

    def _chunk(self, df, lefts):
        candidates = [['<EMPTY>', -1.0] for _ in range(self._num_models)]
        current_candidates = set(lefts)
        for row in df.to_dict('records'):
            for model_num in range(self._num_models):
                incoming_dist = row['model_%i' % model_num]
                if (incoming_dist > candidates[model_num][1]) and (row[self._col] not in current_candidates):
                    current_candidates.add(row[self._col])
                    current_candidates.discard(candidates[model_num][0])
                    candidates[model_num] = [row[self._col], incoming_dist]
        return candidates

    def _aggregate(self, chunks, lefts):
        candidates = [['<EMPTY>', -1.0] for _ in range(self._num_models)]
        current_candidates = set(lefts)
        for chunk in chunks:
            for model_num, chunk_candidates in enumerate(chunk):
                if (candidates[model_num][1] < chunk_candidates[1]) and (chunk_candidates[0] not in current_candidates):
                    current_candidates.add(chunk_candidates[0])
                    current_candidates.discard(candidates[model_num][0])
                    candidates[model_num] = chunk_candidates
        return candidates

    def _groupby_chunk(self, df, groupby, lefts):
        candidates = {group: [['<EMPTY>', -1.0] for _ in range(self._num_models)] for group in self._groups}
        current_candidates = {group: {'<EMPTY>'} for group in self._groups}
        for row in df.to_dict('records'):
            incoming_group = row[groupby]
            for model_num in range(self._num_models):
                incoming_dist = row['model_%i' % model_num]
                if (incoming_dist > candidates[incoming_group][model_num][1]) and\
                   (row[self._col] not in current_candidates[incoming_group]):
                    current_candidates[incoming_group].add(row[self._col])
                    current_candidates[incoming_group].discard(candidates[incoming_group][model_num][0])
                    candidates[incoming_group][model_num] = [row[self._col], incoming_dist]
        return candidates

    def _groupby_aggregate(self, chunks, lefts):
        candidates = {group: [['<EMPTY>', -1.0] for _ in range(self._num_models)] for group in self._groups}
        current_candidates = {group: set('<EMPTY>') for group in self._groups}
        for chunk in chunks:
            for group, candidate_distances in chunk.items():
                for model_num, chunk_candidates in enumerate(candidate_distances):
                    if (candidates[group][model_num][1] < chunk_candidates[1]) and\
                       (chunk_candidates[0] not in current_candidates[group]):
                        current_candidates[group].add(chunk_candidates[0])
                        current_candidates[group].discard(candidates[group][model_num][0])
                        candidates[group][model_num] = chunk_candidates
        return candidates

    def save(self, path):
        """
        Pickle the FastMap model(s).

        Parameters
        ----------
        path: str
            The path to save the pickle of the model.
        """
        assert self._model_built, "Model has not been built"
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load a pre-built FastMap model(s).

        Parameters
        ----------
        path: str
            The path to the pre-built model pickle
        """
        with open(path, 'rb') as f:
            fastmap = pickle.load(f)
        return fastmap
