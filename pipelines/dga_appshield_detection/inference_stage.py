from abc import abstractmethod

import cupy as cp

from morpheus.config import Config
from morpheus.pipeline.messages import MultiInferenceMessage
from morpheus.pipeline.messages import MultiResponseProbsMessage
from morpheus.pipeline.messages import ResponseMemory
from morpheus.pipeline.messages import ResponseMemoryProbs
from morpheus.pipeline.pipeline import MultiMessageStage
from morpheus.pipeline.inference.inference_stage import InferenceStage

class DGAInferenceStage(InferenceStage):
    def __init__(self, c: Config):
        super().__init__(c)

    @staticmethod
    def _convert_one_response(memory: ResponseMemory, inf: MultiInferenceMessage, res: ResponseMemoryProbs):
        # Make sure we have a continuous list
        # assert inf.mess_offset == saved_offset + saved_count

        probs = memory.get_output("probs")

        # Two scenarios:
        if (inf.mess_count == inf.count):
            # In message and out message have same count. Just use probs as is
            probs[inf.offset:inf.count + inf.offset, :] = res.get_output("probs")
        else:
            assert inf.count == res.count

            mess_ids = inf.seq_ids[:, 0].get().tolist()

            # Out message has more reponses, so we have to do key based blending of probs
            for i, idx in enumerate(mess_ids):
                probs[idx, :] = cp.maximum(probs[idx, :], res.probs[i, :])

        return MultiResponseProbsMessage(meta=inf.meta,
                                         mess_offset=inf.mess_offset,
                                         mess_count=inf.mess_count,
                                         memory=memory,
                                         offset=inf.offset,
                                         count=inf.count)
