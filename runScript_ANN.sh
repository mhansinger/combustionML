#!/bin/bash

cd FPV_ANN_pureRes_newGPU
ipython3 FPV_resnet_4D.py
cd ..

cd FPV_ANN_NoRes_TF2
ipython3 FPV_NoResNet.py
cd ..