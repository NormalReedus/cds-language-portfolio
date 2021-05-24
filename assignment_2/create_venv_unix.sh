#!/usr/bin/env bash

# no third party modules used

VENVNAME=assignment_2_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate

mkdir -p data/
mkdir -p output/

echo "build $VENVNAME"
