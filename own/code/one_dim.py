import functions
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import scipy as sp

# m1 är massan längst vänster (initialt)
# m2 är massan längst höger (initialt)
test = functions.parser(filename = "own/raw_data/daniel.tsv", m1 = 1.0, m2 = 1.0, fps = 100, testnr = 0, offset = 5, headers = 10, rotation = False)
plt.subplot(4, 2, 1)
plt.plot(test.time[:test.impact_start], test.pos1xbefore(test.time)[:test.impact_start], label = "pos1xbefore")
plt.plot(test.time[test.impact_stopp:], test.pos1xafter(test.time)[test.impact_stopp:], label = "pos1xafter")
plt.plot(test.time[:test.impact_start], test.pos2xbefore(test.time)[:test.impact_start], label = "pos2xbefore")
plt.plot(test.time[test.impact_stopp:], test.pos2xafter(test.time)[test.impact_stopp:], label = "pos2xafter")
plt.legend()

plt.subplot(4, 2, 2)
plt.plot(test.time[:test.impact_start], test.v1xbefore(test.time)[:test.impact_start], label = "v1xbefore")
plt.plot(test.time[test.impact_stopp:], test.v1xafter(test.time)[test.impact_stopp:], label = "v1xafter")
plt.legend()

plt.plot(test.time[:test.impact_start], test.v2xbefore(test.time)[:test.impact_start], label = "v2xbefore")
plt.plot(test.time[test.impact_stopp:], test.v2xafter(test.time)[test.impact_stopp:], label = "v2xafter")
plt.legend()

plt.subplot(4, 2, 3)
plt.plot(test.time[:test.impact_start], test.E_kin1_before + test.E_kin2_before, label = "E_kin_before")
plt.plot(test.time[test.impact_stopp:], test.E_kin1_after + test.E_kin2_after, label = "E_kin_after")
plt.legend()
print("E_kin_before = ", (test.E_kin1_before[test.E_kin1_before.size-1] + test.E_kin2_before[test.E_kin1_before.size-1]))
print("E_kin_after = ", (test.E_kin1_after[0] + test.E_kin2_after[0]))
print("E_kin_bevar = ", (test.E_kin1_after[0] + test.E_kin2_after[0]) / (test.E_kin1_before[test.E_kin1_before.size-1] + test.E_kin2_before[test.E_kin2_before.size-1]))

plt.subplot(4, 2, 4)
plt.plot(test.time[:test.impact_start], (test.ln_mom_1x_before) + (test.ln_mom_2x_before), label = "ln_mom_before")
plt.plot(test.time[test.impact_stopp:], (test.ln_mom_1x_after) + (test.ln_mom_2x_after), label = "ln_mom_after")
plt.legend()
print("ln_mom_before = ", (test.ln_mom_1x_before[test.ln_mom_1x_before.size-1]) + (test.ln_mom_2x_before[test.ln_mom_2x_before.size-1]))
print("ln_mom_after = ", (test.ln_mom_1x_after[0]) + (test.ln_mom_2x_after[0]))
print("ln_mom_bevar = ", ((test.ln_mom_1x_after[0]) + (test.ln_mom_2x_after[0])) / ((test.ln_mom_1x_before[test.ln_mom_1x_before.size-1]) + (test.ln_mom_2x_before[test.ln_mom_2x_before.size-1])))

plt.subplot(4, 2, 5)
plt.plot(test.time[:test.impact_start], (test.ln_mom_1x_before), label = "ln_1x_mom_before")
plt.plot(test.time[test.impact_stopp:], (test.ln_mom_1x_after), label = "ln_1x_mom_after")
plt.legend()

plt.subplot(4, 2, 6)
plt.plot(test.time[:test.impact_start], (test.ln_mom_2x_before), label = "ln_2x_mom_before")
plt.plot(test.time[test.impact_stopp:], (test.ln_mom_2x_after), label = "ln_2x_mom_after")
plt.legend()

plt.show()