#!/usr/bin/env bash

VENVNAME=assignment_3_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

# 2.5.3 of spacy is required, but this can conflict with spacytextblob if spacytextblob version is not unspecified
test -f requirements.txt && pip install -r requirements.txt

python -m spacy download en_core_web_sm

mkdir -p data/
mkdir -p output/

echo "build $VENVNAME"
