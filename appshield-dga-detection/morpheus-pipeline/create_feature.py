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
import typing

import numpy as np
import pandas as pd
import srf
import tldextract
from dask.distributed import Client
from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.appshield_source_stage import AppShieldMessageMeta
from srf.core import operators as ops

MAX_DOMAIN = 75


def get_domain(url):
    domain = tldextract.extract(url).domain
    if domain in ["ddns", "bazar", "onion"]:
        url = url.strip("https://").strip("http://")
        urls = url.split(".")
        urls_i = urls.index(domain)
        if urls_i == 0:
            return domain
        return urls[urls_i - 1]
    return domain


def get_domain_space(domain):
    try:
        return " ".join(domain)
    except:
        return ""


@dataclasses.dataclass
class FeatureConfig:
    required_plugins: typing.List[str]
    full_memory_address: int
    file_extn_list: typing.List[str]
    protections: typing.Dict[str, str]
    features_dummy_data: typing.Dict[str, int]


def _build_features(snapshot_df: pd.DataFrame, word_index) -> pd.DataFrame:

    snapshot_df["Domain"] = snapshot_df["URL"].apply(get_domain)
    data_chars = snapshot_df["Domain"].apply(get_domain_space)

    chars_df = data_chars.str.split(" ", expand=True)
    for col in chars_df.columns:
        chars_df[col] = chars_df[col].map(word_index)

    chars_df = chars_df.fillna(0)
    pad_width = MAX_DOMAIN - chars_df.shape[1]

    padded_np_array = np.pad(
        chars_df.to_numpy(), ((0, 0), (0, pad_width)), mode="constant"
    )

    df_features = pd.DataFrame(
        columns=["char_" + str(i) for i in range(MAX_DOMAIN)], data=padded_np_array
    )
    df_features["domain"] = snapshot_df["Domain"]

    return df_features


def _combined_features(x: typing.List[pd.DataFrame]) -> pd.DataFrame:

    return pd.concat(x)


class CreateFeatureDGAStage(MultiMessageStage):
    def __init__(
        self,
        c: Config,
        required_plugins: typing.List[str],
        feature_columns: typing.List[str],
        tokenizer_path,
    ):
        self._required_plugins = required_plugins
        self._feature_columns = feature_columns
        self._features_dummy_data = dict.fromkeys(self._feature_columns, 0)

        self._word_index = (
            pd.read_csv(tokenizer_path).set_index("keys")["values"].to_dict()
        )

        self._client = Client(threads_per_worker=2, n_workers=6)

        super().__init__(c)

    @property
    def name(self) -> str:
        return "create-features"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (AppShieldMessageMeta,)

    def supports_cpp_node(self):
        return False

    def _build_single(self, seg: srf.Builder, input_stream: StreamPair) -> StreamPair:
        stream = input_stream[0]

        def node_fn(input: srf.Observable, output: srf.Subscriber):
            def on_next(x: AppShieldMessageMeta):
                to_send = []
                snapshot_fea_dfs = []

                df = x.df
                df["PID"] = df["PID"].astype(str)
                df["PID_Process"] = df.PID + "_" + df.Process
                snapshot_ids = x.df.snapshot_id.unique()

                all_dfs = [
                    df[df.snapshot_id == snapshot_id] for snapshot_id in snapshot_ids
                ]
                snapshot_fea_dfs = self._client.map(
                    _build_features, all_dfs, word_index=self._word_index
                )

                features_df = self._client.submit(_combined_features, snapshot_fea_dfs)

                features_df = features_df.result()

                features_df["source"] = x.source
                features_df["pid_process"] = df["PID_Process"]
                features_df["snapshot_id"] = df["snapshot_id"]
                features_df = features_df.sort_values(
                    by=["pid_process", "snapshot_id"]
                ).reset_index(drop=True)
                x = AppShieldMessageMeta(features_df, x.source)

                unique_pid_processes = features_df.pid_process.unique()

                # Create multi messaage per pid_process
                for pid_process in unique_pid_processes:
                    start = features_df.pid_process[
                        features_df.pid_process == pid_process
                    ].index.min()
                    stop = (
                        features_df.pid_process[
                            features_df.pid_process == pid_process
                        ].index.max()
                        + 1
                    )
                    multi_message = MultiMessage(
                        meta=x, mess_offset=start, mess_count=stop - start
                    )
                    to_send.append(multi_message)
                return to_send

            def on_completed():
                # Close dask client when pipeline initiates shutdown
                self._client.close()

            input.pipe(ops.map(on_next), ops.on_completed(on_completed), ops.flatten()).subscribe(output)

        node = seg.make_node_full(self.unique_name, node_fn)
        seg.make_edge(stream, node)
        stream = node
        return stream, MultiMessage
