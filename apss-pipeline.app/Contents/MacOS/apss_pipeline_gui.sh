#!/bin/bash

BASEDIR=$(dirname "$0")

export PATH=/Users/paolo/Software/miniconda3/bin:$PATH
#export PATH=/Users/silviosarubbo/CLINT/Software/miniconda3/bin:$PATH
export PYTHONPATH=${BASEDIR}:PYTHONPATH

${BASEDIR}/pipeline_gui.py

#cd ${BASEDIR}/../../..
#ROOTDIR=`pwd`
#python ${ROOTDIR}/pipeline.app/Contents/MacOS/pipeline-gui
