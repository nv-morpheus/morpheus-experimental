ARG MORPHEUS_CONTAINER=nvcr.io/nvidia/morpheus/morpheus
ARG MORPHEUS_CONTAINER_VERSION=24.10-runtime

FROM ${MORPHEUS_CONTAINER}:${MORPHEUS_CONTAINER_VERSION} as base

COPY docker/start-jupyter.sh /workspace/docker/start-jupyter.sh
COPY docker/entrypoint.sh /workspace/docker/entrypoint.sh

RUN chmod +x /workspace/docker/start-jupyter.sh
RUN chmod +x /workspace/docker/entrypoint.sh

# Install cuML
RUN source activate morpheus \
  && mamba install -y -c rapidsai -c conda-forge -c nvidia cuml=24.10

# Install the jupyter specific requirements
RUN source activate morpheus \
    && pip install \
        torch-geometric==2.6.1 \
        jupyterlab==4.3.3 \
    && conda clean -afy \
    && rm -rf /var/lib/apt/lists/*

# Fix the entrypoint to work with different WORKDIR
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "/workspace/docker/entrypoint.sh" ]
