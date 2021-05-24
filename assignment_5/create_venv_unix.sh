#!/usr/bin/env bash

VENVNAME=assignment_5_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

test -f requirements.txt && pip install -r requirements.txt

python -m spacy download en_core_web_sm

mkdir -p data/
mkdir -p output/

echo "build $VENVNAME"
