#!/bin/bash

# Configurations
readonly NAME="%%NAME%%"
readonly HOME='%%HOME%%'
readonly JOB_ID='%%JOB_ID%%'

readonly PROJECT_PATH="${HOME}/${NAME}"
readonly CONDA_PATH="${HOME}/miniconda"
readonly CONDA_ENV="${NAME}_env"

%%ENVIRONMENT_VARIABLES%%
# Text colors
# 31 red 32 green 33 yellow 34 blue

run_python() {
  source "${CONDA_PATH}/etc/profile.d/conda.sh"
  if [ "$?" != 0 ]; then
    return 1
  fi

  conda activate "${CONDA_ENV}"
  if [ "$?" != 0 ]; then
    return 1
  fi

  cd "${PROJECT_PATH}"
  if [ "$?" != 0 ]; then
    return 1
  fi

  export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"

  if [ ! -d ".jobs" ]; then
    mkdir .jobs
  fi

  mkdir ".jobs/${JOB_ID}"

  nohup %%RUN_COMMAND%% > ".jobs/${JOB_ID}/job.out" 2> ".jobs/${JOB_ID}/job.err" &
  local pid=$!
  echo $pid > ".jobs/${JOB_ID}/job.pid"
  echo $pid

  if [ "$?" != 0 ]; then
    return 1
  fi

  return 0
}

run_python
