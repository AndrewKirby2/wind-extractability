"""Tests for functions analysing NWP data"""

import numpy as np
import numpy.testing as npt
from read_NWP_data import *

def test_dummy():
    var_dict = load_NWP_data('DS5',20)
    npt.assert_array_equal(np.array([0]),np.array([0]))