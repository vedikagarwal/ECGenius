#!/bin/bash

set -e

function incorrect_usage() {
    echo "ERROR - Usage: ./setup_env.sh --train-dir <train_dir> --validation-dir <validation_dir> --codebook <codebook>"
    echo "Where <train_dir> and <validation_dir> contain .dcm files, and <codebook> is the path to the codebook.csv file"
    exit 1
}

if [ "$#" -eq 6 ]; then
    echo "Setting up environment"
    SCRIPT=$(readlink -f "$0")
    SCRIPTPATH=$(dirname "$SCRIPT")

    if [ "$1" != "--train-dir" ] || [ "$3" != "--validation-dir" ] || [ "$5" != "--codebook" ]; then
        incorrect_usage
    fi
    train_dir=$(realpath "$2")
    validation_dir=$(realpath "$4")
    codebook=$(realpath "$6")

    if [ ! -d "$train_dir" ] || [ ! -d "$validation_dir" ]; then
        incorrect_usage
    fi

    mkdir -p "$SCRIPTPATH/data"
    ln -sfn "$train_dir" "$SCRIPTPATH/data/train"
    ln -sfn "$validation_dir" "$SCRIPTPATH/data/validation"
    ln -sfn "$codebook" "$SCRIPTPATH/data/codebook.csv"
else
    incorrect_usage
fi

# generate environment.json
python3 "$SCRIPTPATH/setup/generate_env_json.py"
