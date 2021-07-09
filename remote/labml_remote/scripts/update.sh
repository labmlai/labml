#!/bin/bash

# Configurations
readonly NAME="%%NAME%%"
readonly HOME='%%HOME%%'
readonly HAS_PIPFILE='%%HAS_PIPFILE%%'
readonly HAS_REQUIREMENTS='%%HAS_REQUIREMENTS%%'

readonly PROJECT_PATH="${HOME}/${NAME}"
readonly CONDA_PATH="${HOME}/miniconda"
readonly CONDA_ENV="${NAME}_env"

# Text colors
# 31 red 32 green 33 yellow 34 blue

update_pipenv() {
  if [ "${HAS_PIPFILE}" != "True" ]; then
    return 0
  fi

  printf "Updating pipenv...\n"
  pipenv install
  if [ "$?" == 0 ]; then
    printf "Updating pipenv... \x1B[0;32m[DONE]\x1B[0m\n"
    return 0
  else
    printf "Updating pipenv... \x1B[0;31m[FAILED]\x1B[0m\n"
    return 1
  fi
}

update_requirements() {
  if [ "${HAS_REQUIREMENTS}" != "True" ]; then
    return 0
  fi

  printf "Updating pip...\n"
  pip install -r requirements.txt
  if [ "$?" == 0 ]; then
    printf "Updating pip... \x1B[0;32m[DONE]\x1B[0m\n"
    return 0
  else
    printf "Updating pip... \x1B[0;31m[FAILED]\x1B[0m\n"
    return 1
  fi
}

update() {
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

  update_pipenv
  if [ "$?" != 0 ]; then
    return 1
  fi

  update_requirements
  if [ "$?" != 0 ]; then
    return 1
  fi

  return 0
}

update
