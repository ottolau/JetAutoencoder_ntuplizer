#!/bin/bash
python3 -m venv --copies venv
source venv/bin/activate
python3 -m pip install setuptools pip wheel --upgrade
python3 -m pip install xxhash coffea
python3 -m pip install zstandard
