import functions
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import scipy as sp

# m1 är massan längst vänster (initialt)
# m2 är massan längst höger (initialt)
test = functions.parser(filename = "./raw_data/data.tsv", m1 = 1.0, m2 = 1.0, fps = 100, testnr = 0, offset = 10, headers = 10, rotation = False)