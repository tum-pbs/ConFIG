#usr/bin/python3
# -*- coding: UTF-8 -*-
import numpy as np
RE=40
NU=1/RE
LAMBDA=1/(2*NU)-np.sqrt(1/(4*NU**2)+4*np.pi**2)
X_START=-0.5
X_END=1.0
Y_START=-0.5
Y_END=1.5
N_INTERNAL=20000
N_BOUNDARY=1000
