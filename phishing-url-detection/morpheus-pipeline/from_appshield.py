# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import glob
import io
import json
import logging
import os
import queue
import re
import typing
from functools import partial

import neo
import pandas as pd
from neo.core import operators as ops
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from morpheus._lib.common import FiberQueue
from morpheus.config import Config
from morpheus.pipeline.file_types import FileTypes
from morpheus.pipeline.messages import MessageMeta
from morpheus.pipeline.pipeline import SingleOutputSource
from morpheus.pipeline.pipeline import StreamPair
from morpheus.utils.producer_consumer_queue import Closed

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SourceMessageMeta(MessageMeta):

    source: str


class AppShieldFileHandler(PatternMatchingEventHandler):

    def __init__(self, q: FiberQueue, match_pattern):
        PatternMatchingEventHandler.__init__(self,
                                             patterns=[match_pattern],
                                             ignore_patterns=None,
                                             ignore_directories=False,
                                             case_sensitive=True)
        self._q = q

    def on_created(self, event):
        self._q.put(([event.src_path], True))


class AppShieldSourceStage(SingleOutputSource):
    """
    Source stage is used to load messages from a file and dumping the contents into the pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    input_glob : str
        Input glob pattern to match files to read. For example, `./input_dir/*/snapshot-*/*.json` would read all files with the
        'json' extension in the directory input_dir.
    watch_directory : bool, default = False
        The watch directory option instructs this stage to not close down once all files have been read. Instead it will
        read all files that match the 'input_glob' pattern, and then continue to watch the directory for additional
        files. Any new files that are added that match the glob will then be processed.
    max_files: int, default = -1
        Max number of files to read. Useful for debugging to limit startup time. Default value of -1 is unlimited.
    file_type : FileSourceTypes, default = 'auto'
        Indicates what type of file to read. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'json', 'csv'
    repeat: int, default = 1
        How many times to repeat the dataset. Useful for extending small datasets in debugging.
    raw_features: List[str], default = None
        Raw features to extract from appshield plugins data.
    required_plugins: List[str], default = None
        Plugins for appshield to be extracted.
    """

    def __init__(self,
                 c: Config,
                 input_glob: str,
                 watch_directory: bool = False,
                 max_files: int = -1,
                 file_type: FileTypes = FileTypes.Auto,
                 repeat: int = 1,
                 raw_feature_columns: typing.List[str] = None,
                 required_plugins: typing.List[str] = None):

        super().__init__(c)

        self._input_glob = input_glob
        self._file_type = file_type
        self._max_files = max_files
        self._exclude_columns = ["SHA256"]
        self._raw_feature_columns = raw_feature_columns
        self._required_plugins = required_plugins

        self._input_count = None

        # Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode
        # is good for interleaving source stages. Non-iterative is better for dask (uploads entire dataset in one call)
        self._repeat_count = repeat
        self._watch_directory = watch_directory

        # Will be a watchdog observer if enabled
        self._watcher = None

    @property
    def name(self) -> str:
        return "from-appshield"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    def stop(self):

        if (self._watcher is not None):
            self._watcher.stop()

        return super().stop()

    async def join(self):

        if (self._watcher is not None):
            self._watcher.join()

        return await super().join()

    def _get_filename_queue(self) -> FiberQueue:
        """
        Returns an async queue with tuples of `([files], is_event)` where `is_event` indicates if this is a file changed
        event (and we should wait for potentially more changes) or if these files were read on startup and should be
        processed immediately
        """
        q = FiberQueue(512)

        if (self._watch_directory):
            match_pattern = '*.json'
            event_handler = AppShieldFileHandler(q, match_pattern)
            self._watcher = Observer()
            glob_split = self._input_glob.split("*", 1)
            if (len(glob_split) == 1):
                raise RuntimeError(("When watching directories, input_glob must have a wildcard. "
                                    "Otherwise no files will be matched."))
            dir_to_watch = os.path.abspath(os.path.dirname(glob_split[0]))
            self._watcher.schedule(event_handler, dir_to_watch, recursive=True)
            self._watcher.start()
            self.join()

        # Load the glob once and return
        file_list = glob.glob(self._input_glob)
        if (self._max_files > 0):
            file_list = file_list[:self._max_files]

        logger.info("Found %d Appshield files in glob. Loading...", len(file_list))

        # Push all to the queue and close it
        q.put((file_list, False))

        if (not self._watch_directory):
            # Close the queue
            q.close()

        return q

    def _generate_filenames(self):

        # Gets a queue of filenames as they come in. Returns list[str]
        file_queue: FiberQueue = self._get_filename_queue()

        batch_timeout = 2.0

        files_to_process = []

        while True:

            try:
                files, is_event = file_queue.get(timeout=batch_timeout)

                if (is_event):
                    # We may be getting files one at a time from the folder watcher, wait a bit
                    files_to_process = files_to_process + files
                    continue

                # We must have gotten a group at startup, process immediately
                if len(files) > 0:
                    yield files

                # df_queue.task_done()

            except queue.Empty:
                # We timed out, if we have any items in the queue, push those now
                if (len(files_to_process) > 0):
                    yield files_to_process
                    files_to_process = []

            except Closed:
                # Just in case there are any files waiting
                if (len(files_to_process) > 0):
                    yield files_to_process
                    files_to_process = []
                break

    @staticmethod
    def fill_feature_cols(plugin_df: pd.DataFrame, feature_columns: typing.List[str]):
        """
        Fill missing feature columns.

        Parameters
        ----------
        plugin_df : pd.DataFrame
            Snapshot plugin dataframe
        feature_columns : typing.List[str]
            Columns that needs to be included

        Returns
        -------
        pd.DataFrame
            The columns added dataframe
        """
        cols_exists = plugin_df.columns
        for col in feature_columns:
            if col not in cols_exists:
                plugin_df[col] = None
        plugin_df = plugin_df[feature_columns]

        return plugin_df

    @staticmethod
    def read_file_to_df(file: io.TextIOWrapper, exclude_columns: typing.List[str]):
        data = json.load(file)
        features_plugin = data["titles"]
        features_plugin = [col for col in data['titles'] if col not in exclude_columns]
        plugin_df = pd.DataFrame(columns=features_plugin, data=data["data"])
        return plugin_df

    @staticmethod
    def load_df(filepath: str, exclude_columns: typing.List[str]) -> pd.DataFrame:
        """
        Reads a file into a dataframe

        Parameters
        ----------
        filepath : str
            Path to a file.
        exclude_columns : typing.List[str]
            Columns that needs to exclude

        Returns
        -------
        pd.DataFrame
            The parsed dataframe

        Raises
        ------
        RuntimeError
            If an unsupported file type is detected
        """

        try:
            with open(filepath, encoding="latin1") as file:
                plugin_df = AppShieldSourceStage.read_file_to_df(file, exclude_columns)
        except Exception as e:
            try:
                with open(filepath, encoding='utf8') as file:
                    plugin_df = AppShieldSourceStage.read_file_to_df(file, exclude_columns)
            except Exception as e:
                raise Exception("Unable to parse json file: " + filepath) from e

        return plugin_df

    @staticmethod
    def load_meta_cols(filepath_split: typing.List[str], plugin: str, plugin_df: pd.DataFrame) -> pd.DataFrame:
        """
        Loads meta columns to dataframe

        Parameters
        ----------
        filepath_split : typing.List[str]
            Splits of file path.
        plugin : str
            Plugin name to which the data belongs to

        Returns
        -------
        pd.DataFrame
            The parsed dataframe
        """

        source = filepath_split[-3]

        snapshot_id = int(filepath_split[-2].split('-')[1])
        timestamp = re.search('[a-z]+_([0-9\-\_\.]+).json', filepath_split[-1]).group(1)

        plugin_df['snapshot_id'] = snapshot_id
        plugin_df['timestamp'] = timestamp
        plugin_df['source'] = source
        plugin_df['plugin'] = plugin
        return plugin_df

    @staticmethod
    def batch_sources_split(x: typing.List[pd.DataFrame], source_column_name: str) -> typing.Dict[str, pd.DataFrame]:
        """
        Combines plugin dataframes from multiple snapshot and split dataframe per source.

        Parameters
        ----------
        x : typing.List[str]
            Dataframes from multiple sources
        source_column_name : str
            Source column name to group it

        Returns
        -------
        typing.Dict[str, pd.DataFrame]
            Grouped dataframes by source
        """

        combined_df = pd.concat(x)

        # Get the sources in this DF
        unique_sources = combined_df[source_column_name].unique()

        source_dfs = {}

        if len(unique_sources) > 1:
            for source_name in unique_sources:
                source_dfs[source_name] = combined_df[combined_df[source_column_name] == source_name]
        else:
            source_dfs[unique_sources[0]] = combined_df

        return source_dfs

    @staticmethod
    def files_to_dfs(x: typing.List[str],
                     feature_columns: typing.List[str],
                     exclude_columns: typing.List[str],
                     required_plugins: typing.List[str]) -> pd.DataFrame:

        # Using pandas to parse nested JSON until cuDF adds support
        # https://github.com/rapidsai/cudf/issues/8827
        plugin_dfs = []
        for filepath in x:
            try:
                filepath_split = filepath.split('/')
                plugin = filepath_split[-1].split('_')[0]
                if plugin in required_plugins:
                    plugin_df = AppShieldSourceStage.load_df(filepath, exclude_columns)
                    plugin_df = AppShieldSourceStage.fill_feature_cols(plugin_df, feature_columns)
                    plugin_df = AppShieldSourceStage.load_meta_cols(filepath_split, plugin, plugin_df)
                    plugin_dfs.append(plugin_df)
            except Exception as e:
                print(e)

        df_per_sources = AppShieldSourceStage.batch_sources_split(plugin_dfs, 'source')

        return df_per_sources

    def _build_source_metadata(self, x: typing.Dict[str, pd.DataFrame]):

        source_metas = []

        for source, source_df in x.items():

            # Now make a SourceMessageMeta with the source name
            meta = SourceMessageMeta(source_df, source)

            source_metas.append(meta)

        return source_metas

    def _build_source(self, seg: neo.Segment) -> StreamPair:

        # The first source just produces filenames
        filename_source = seg.make_source(self.unique_name, self._generate_filenames())

        out_type = typing.List[str]

        # Supposed to just return a source here
        return filename_source, out_type

    def _post_build_single(self, seg: neo.Segment, out_pair: StreamPair) -> StreamPair:

        out_stream = out_pair[0]
        out_type = out_pair[1]

        def node_fn(input: neo.Observable, output: neo.Subscriber):
            input.pipe(
                # At this point, we have batches of filenames to process. Make a node for processing batches of
                # filenames into batches of dataframes
                ops.map(
                    partial(self.files_to_dfs,
                            feature_columns=self._raw_feature_columns,
                            exclude_columns=self._exclude_columns,
                            required_plugins=self._required_plugins)),
                ops.map(self._build_source_metadata),
                # Finally flatten to single meta
                ops.flatten()).subscribe(output)

        post_node = seg.make_node_full(self.unique_name + "-post", node_fn)
        seg.make_edge(out_stream, post_node)

        out_stream = post_node
        out_type = SourceMessageMeta

        return super()._post_build_single(seg, (out_stream, out_type))
