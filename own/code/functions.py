import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import scipy as sp

class storheter:
	def trans_energy(speed, weight):
		return weight * speed * speed * (1/2)

	def rotat_energy(angle_speed, troghet):
		return troghet * angle_speed * angle_speed * (1/2)

	def kin_energy(speed, weight, angle_speed, troghet, rotation = False):
		if rotation == False:
			return storheter.trans_energy(speed, weight) + storheter.rotat_energy(angle_speed, troghet)
		elif rotation == True:
			return storheter.trans_energy(speed, weight) + storheter.rotat_energy(angle_speed, troghet)

	def momentum(weight, speed):
		return weight * speed

	def angular_momentum(angle_speed, troghet):
		return angle_speed * troghet

class parser:
	def __init__(self, filename = "data.tsv", m1 = 1.0, m2 = 1.0, fps = 100, testnr = 0, offset = 10, headers = 10, rotation = False):
		"""m1 respektive m2 är massorna, m1 längst till vänster. 
		fps är bilder per sekund, dt =1/fps
		headers är antal rader att skippa i början av dokuentet"""
		self.fps = fps
		dt = 1/fps
		self.dt = dt
		self.m1 = m1
		self.m2 = m2
		self.testnr = testnr
		self.offset = offset
		self.resid = 0

		#"data.tsv" är filnamn, skip_header bör varieras beroende på filformat
		data = np.genfromtxt(filename, np.float64, delimiter="\t", skip_header=headers, filling_values=1)

		self.pos1 = (10**-3) * data[:, 2:5] #pos1 är rad 2-4, rad 2 är 5-8, mäter i mm
		self.pos2 = (10**-3) * data[:, 5:8]

		self.antal_pkter = data.shape[0]
		self.time = data[:, 1]

		#polynomial fit längs alla 3 dimensioner. använd p_x.deriv()
		def fit(self, pos, indx_start = 0, indx_stopp = None, deg = 2, dimension = 2):
			"""pos = (n, 3) matris av positioner av det valda objektet.
			t_start och t_stopp bestämmer vart datapunkterna för anpassningen slutar vara tillåtna."""
			if indx_stopp == None:
				indx_stopp = self.antal_pkter
		
			t_snitt = self.time[indx_start:indx_stopp]
			pos_snitt = pos[indx_start:indx_stopp]

			p_x, resid = Polynomial.fit(t_snitt, pos_snitt[:, 0], deg=deg, full=True)
			p_y = Polynomial.fit(t_snitt, pos_snitt[:, 1], deg=deg)
			p_z = Polynomial.fit(t_snitt, pos_snitt[:, 2], deg=deg)

			self.resid += resid[0][0]

			return p_x, p_y, p_z

		def hastighet(self, p_x, p_y, p_z):
			v_x = p_x.deriv()
			v_y = p_y.deriv()
			v_z = p_z.deriv()
			vx = v_x(self.time)
			vy = v_y(self.time)
			vz = v_z(self.time)
			vb = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

			return vb, v_x, v_y, v_z

		def finn_stöt(self, pos):
			v = np.diff(pos, axis=0) / self.dt

			a = np.diff(v, axis=0) / self.dt

			a_belopp = np.linalg.norm(a, axis=1)

			plt.plot(self.time[1:-1], a_belopp)
			plt.show()

			max_a_indx = np.argmax(a_belopp)

			return max_a_indx-self.offset, max_a_indx + 2 + self.offset

		def storheter(self, p_x, p_y, p_z, v_x, v_y, v_z):
			"""beräknar kinetisk energi och rörelsemängd för de två massorna"""
			vx = v_x(self.time)
			vy = v_y(self.time)
			vz = v_z(self.time)
			vb = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

			E_kin1 = storheter.kin_energy(vb, self.m1, 0, 0, rotation=self.rotation)
			E_kin2 = storheter.kin_energy(vb, self.m2, 0, 0, rotation=self.rotation)

			p1x = storheter.momentum(self.m1, vx)
			p1y = storheter.momentum(self.m1, vy)
			p1z = storheter.momentum(self.m1, vz)
			p1 = np.matrix([p1x, p1y, p1z])
			p2x = storheter.momentum(self.m2, vx)
			p2y = storheter.momentum(self.m2, vy)
			p2z = storheter.momentum(self.m2, vz)
			p2 = np.matrix([p2x, p2y, p2z])

			return E_kin1, E_kin2, p1, p2