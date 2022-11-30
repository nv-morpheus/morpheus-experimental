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
from typing import Union

import dill as pickle
from srg.fastkde import FastKDE
from srg.fastmap import FastMap
from srg.fastmap.distances import Jaccard

import dask

import cudf
import dask_cudf

_PRESORT_COLUMN = 'presort'
_PRESORT_LABEL_COLUMN = 'presort_label'
_COMBINED_LABEL = 'combined_label'


class SRG:
    """
    A technique for syntactically grouping strings and electing a group representative for each group.
    See `SRG README`_ for more details on how the technique is designed.

    .. _SRG README:
        /string-resemblance-grouping/README.md
    """
    def __init__(self):
        self._model_built = False
        self._presort = None
        self._presort_kde = None
        self._presort_minima = None
        self._fastmap = None
        self._proj_kde = None
        self._proj_minima = None
        self._proj_maxima = None
        self._groups = None
        self._reps = None

    def fit(self,
            X,
            delimiter=None,
            names=None,
            npartitions=2,
            column=None,
            num_fastmaps=5,
            iters=5,
            shingle_size=4,
            bandwidth=None,
            presort='length'):
        """
        Find the groups and representatives for the provided data set.

        Parameters
        ----------
        X: str or List[str] or (dask_)cudf Series/DataFrame
            A path to a csv to load, a list of strings, or a pre-loaded (dask_)cudf Series or DataFrame
        delimiter: str
            If a csv path is specified, this indicates the delimiter to use when reading the file
        names: List[str]
            Manually specify the column names for an input csv
        npartitions: int
            The number of partitions for the dask_cudf DataFrame
        column: str
            The column containing the data to be grouped
        num_fastmaps: int
            The number of simultaneous FastMap models to build.
        iters: int
            The number of passes to find the FastMap pivots.
        shingle_size: int
            The size of the n-gram character shingles created over the string data.
        bandwidth: float
            Manually specify the bandwidth for the kernel density estimators.
        presort: str
            The desired method for pre-sorting the data. Default is to create subgroups based on the lengths of the
            strings.
        """
        self._presort = presort
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

        Xpresort = ddf.map_partitions(lambda df: self._lengths(df))
        self._presort_kde = FastKDE()
        self._presort_kde.fit(Xpresort, column=_PRESORT_COLUMN)
        self._presort_minima, _ = self._presort_kde.get_local_extrema()
        Xpresort_labeled = Xpresort.map_partitions(lambda df: self._presort_label(df))

        self._fastmap = FastMap(dim=1,
                                num_models=1,
                                distance=Jaccard,
                                dist_args={'shingle_size': shingle_size},
                                iters=iters)
        self._fastmap.fit(Xpresort_labeled, column=self._col, groupby=_PRESORT_LABEL_COLUMN)

        X_proj = self._fastmap.transform(Xpresort_labeled, column=self._col, group=_PRESORT_LABEL_COLUMN)

        X_proj.persist()

        self._proj_kde = FastKDE()
        self._proj_kde.fit(X_proj, column='x_0', groupby=_PRESORT_LABEL_COLUMN)

        self._proj_minima, self._proj_maxima = self._proj_kde.get_local_extrema()

        all_groups = []
        for ps in range(len(self._presort_minima) + 1):
            maxima = self._proj_maxima[ps]
            if len(maxima) == 0:
                all_groups.append((ps, 0))
            else:
                for maxima_idx, _ in enumerate(self._proj_maxima[ps]):
                    all_groups.append((ps, maxima_idx))
        self._groups = {labels: idx for idx, labels in enumerate(all_groups)}

        self._reps = X_proj.reduction(chunk=self._rep_chunks, aggregate=self._rep_agg, meta=dict).compute()

        self._model_built = True

    def transform(self,
                  X,
                  delimiter=None,
                  names=None,
                  npartitions: int = 2,
                  column: Union[str, None] = None):
        """
        Find the group and representative for a specific string or each string in a collection.

        Parameters
        ----------
        X: str or List[str] or (dask_)cudf Series/DataFrame
            A single string, a path to a csv to load, a list of strings,
            or a pre-loaded (dask_)cudf Series or DataFrame
        delimiter: str
            If a csv path is specified, this indicates the delimiter to use when reading the file
        names: List[str]
            Manually specify the column names for an input csv
        npartitions: int
            The number of partitions for the dask_cudf DataFrame
        column: str
            The column containing the data to be grouped

        Returns
        -------
        Tuple or DataFrame
            The group and representative if a single string is passed or a DataFrame with the group and representative
            for each string added to the input DataFrame
        """
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
                return self._single_transform(X)
        elif isinstance(X, (list, cudf.core.series.Series)):
            series = cudf.Series(X) if isinstance(X, list) else X
            cdf = cudf.DataFrame({col: series})
            ddf = dask_cudf.from_cudf(cdf, npartitions=npartitions)
        elif isinstance(X, (dask_cudf.core.DataFrame, cudf.core.dataframe.DataFrame)):
            ddf = dask_cudf.from_cudf(X, npartitions=npartitions) if isinstance(X, cudf.core.dataframe.DataFrame) else X
            if column is None:
                ddf = ddf.rename(columns={0: 'obj'})
        else:
            ddf = X
        assert isinstance(ddf, dask_cudf.core.DataFrame),\
            'X must be a path to a csv, list, or a (dask-)cudf Series or DataFrame, or a projectable object'
        labeled = ddf.map_partitions(lambda df: self._df_transform(df, col))
        return labeled

    def _df_transform(self, df, col):
        if isinstance(df, cudf.core.dataframe.DataFrame):
            pdf = df.to_pandas().copy()
        else:
            pdf = df.copy()
        pdf[['srg_label', 'srg_rep']] = pdf.apply(lambda x: self._single_transform(x[col]),
                                                  axis=1,
                                                  result_type='expand')
        return pdf

    def _single_transform(self, x):
        length = len(x)
        presort_label = sum([1 if length > minimal_value else 0 for minimal_value in self._presort_minima])
        proj = self._fastmap.transform(x, group=presort_label)
        maxima_dists = [(idx, abs(proj - maxima)) for idx, maxima in enumerate(self._proj_maxima[presort_label])]
        maxima_dists.sort(key=lambda x: x[1])
        rep_group = maxima_dists[0][0]
        label = self._groups[(presort_label, rep_group)]
        rep = self._reps[(presort_label, rep_group)][0]
        return [label, rep]

    def _rep_chunks(self, df):
        candidates = dict()
        for row in df.to_dict('records'):
            presort_label = row[_PRESORT_LABEL_COLUMN]
            if (len(self._proj_maxima[presort_label]) == 0) and ((presort_label, 0) not in candidates):
                candidates[(presort_label, 0)] = [row[self._col], 0]
            else:
                for idx, maxima in enumerate(self._proj_maxima[presort_label]):
                    incoming_group = (presort_label, idx)
                    if incoming_group not in candidates:
                        candidates[incoming_group] = [row[self._col], abs(maxima - row['x_0'])]
                    else:
                        current_dist = candidates[incoming_group][1]
                        if current_dist > abs(maxima - row['x_0']):
                            candidates[incoming_group] = [row[self._col], abs(maxima - row['x_0'])]
        return candidates

    def _rep_agg(self, chunks):
        candidates = dict()
        for chunk in chunks:
            for group, chunk_candidate in chunk.items():
                if group not in candidates:
                    candidates[group] = chunk_candidate
                else:
                    current_frontrunner = candidates[group]
                    if current_frontrunner[1] > chunk_candidate[1]:
                        candidates[group] = chunk_candidate
        return candidates

    def _lengths(self, df):
        if isinstance(df, cudf.core.dataframe.DataFrame):
            pdf = df.to_pandas().copy()
        else:
            pdf = df.copy()
        pdf[_PRESORT_COLUMN] = pdf.apply(lambda x: len(x[self._col]), axis=1)
        return pdf

    def _presort_label(self, df):
        if isinstance(df, cudf.core.dataframe.DataFrame):
            pdf = df.to_pandas().copy()
        else:
            pdf = df.copy()
        pdf[_PRESORT_LABEL_COLUMN] = pdf.apply(
            lambda x: sum([1 if x[_PRESORT_COLUMN] > minimal_value else 0 for minimal_value in self._presort_minima]),
            axis=1)
        return pdf

    def _combined_label(self, df):
        if isinstance(df, cudf.core.dataframe.DataFrame):
            pdf = df.to_pandas().copy()
        else:
            pdf = df.copy()
        pdf[_COMBINED_LABEL] = pdf.apply(lambda x: '%i::%i' %
                                         (x[_PRESORT_LABEL_COLUMN],
                                          sum([
                                              1 if x[self._col] > minimal_value else 0
                                              for minimal_value in self._proj_minima[x[_PRESORT_LABEL_COLUMN]]
                                          ])),
                                         axis=1)

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
            srg = pickle.load(f)
        return srg
