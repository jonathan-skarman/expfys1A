import numpy as np
import matplotlib.pyplot as plt

plotline = False
titles = False

datan = np.genfromtxt("mätvärden/luftbord/datan.tsv", np.float64, delimiter="\t", skip_header=1, filling_values=1)

#NEDAN: 5:10 innebär experiment 6, 7, 8, 9, 10
data = datan[np.r_[0:22, 23:50, 51:60], :]
data = datan[:,:]

testnr = data[:,0]
m1 = data[:,1]
m2 = data[:,2]
vx_innan = data[:,3]
vx2_innan = data[:,4]
vx_efter = data[:,5]
vx2_efter = data[:,6]
vy_innan = data[:,7]
vy2_innan = data[:,8]
vy_efter = data[:,9]
vy2_efter = data[:,10]
px_tot_innan = data[:,11]
px_tot_efter = data[:,12]
py_tot_innan = data[:,13]
py_tot_efter = data[:,14]
px_diff = data[:,15]
py_diff = data[:,16]
K_px = data[:,17]
K_py = data[:,18]
E_innan = data[:,19]
E_efter = data[:,20]
E_diff = data[:,21]
K_E = data[:,22]

#variera nedan funktioner för att se olika data. plot(x-axel, y-axel)
plt.plot(np.hypot(vx_innan - vx2_innan, vy_innan - vy2_innan), np.abs(px_diff) + np.abs(py_diff), 'o')
#plt.plot(np.abs(vx_innan  - vx2_innan) + np.abs(vy_innan - vy2_innan), K_E, 'bo', label="Datapunkter")

plotline = True

if plotline==True:
    y = np.polyfit(np.hypot(vx_innan - vx2_innan, vy_innan - vy2_innan), np.abs(px_diff) + np.abs(py_diff), deg=1)
    xs = np.linspace(0.2, 1.2, 50)
    plt.plot(xs, np.polyval(y, xs), 'r', label=f"Linjär regression med k={y[0]:.2g}")

titles = True

if titles == True:
    plt.xlabel("Summan av relativ hastighet i båda led (m/s)")
    plt.ylabel(f"${{{"\Delta p"}}}$ (kgm/s)")
    plt.title("Impuls under stöt som en funktion av relativ hastighet")
    plt.legend()

plt.show()