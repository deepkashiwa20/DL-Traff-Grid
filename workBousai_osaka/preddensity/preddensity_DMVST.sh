#!/usr/bin/env bash
# generate data(only for first run)
python preprocess.py
cd line
chmod 777 line_DMVST.sh
./line_DMVST.sh
cd ..

# predict
python preddensity_DMVST.py