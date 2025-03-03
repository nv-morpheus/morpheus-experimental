#!/bin/bash

# Activate "morpheus" conda environment
. /opt/conda/etc/profile.d/conda.sh
conda activate morpheus

# Start jupyter server
# /workspace/docker/start-jupyter.sh > /dev/null
# echo "There was a jupyter-lab instance started on port 8888, http://127.0.0.1:8888"

# Run whatever user wants
exec "$@"
