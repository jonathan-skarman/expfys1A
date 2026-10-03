import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import scipy as sp
import csv
import math

def trans_energy(speed, weight):
	return (weight * speed * speed * (1/2))

def rotat_energy(angle_speed, troghet):
	return (troghet * angle_speed * angle_speed * (1/2))

def kin_energy(speed, weight, angle_speed=0, troghet=0, rotation = False):
	if rotation == False:
		return (trans_energy(speed, weight) + rotat_energy(angle_speed, troghet))
	elif rotation == True:
		return (trans_energy(speed, weight) + rotat_energy(angle_speed, troghet))

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
		self.rotation = rotation
		self.offset = offset
		self.resid = 0

		with open(filename, mode='r', encoding='utf-8') as file:
			reader = csv.reader(file, delimiter='\t')
			for i, row in enumerate(reader):
				if i == 8: #row 9
					self.marker_names = row[1:]
					#print (self.marker_names)
					break

		#"data.tsv" är filnamn, skip_header bör varieras beroende på filformat
		data = np.genfromtxt(filename, np.float64, delimiter="\t", skip_header=headers, filling_values=1)

		assigned = set()
		for i, marker_name in enumerate(self.marker_names):
			if "_" not in marker_name:
				raise ValueError(f"Unexpected marker name '{marker_name}' at index {i}")

			particle, marker_type = marker_name.split("_", 1)
			target = (10**-3) * data[:, 2 + i*3:5 + i*3]

			if particle == "1":
				if marker_type == "edge1":
					self.pos1_edge1 = target
					assigned.add("1_edge1")
				elif marker_type == "edge2":
					self.pos1_edge2 = target
					assigned.add("1_edge2")
				elif marker_type == "mid":
					self.pos1_mid = target
					assigned.add("1_mid")
			else:
				if marker_type == "edge1":
					self.pos2_edge1 = target
					assigned.add("2_edge1")
				elif marker_type == "edge2":
					self.pos2_edge2 = target
					assigned.add("2_edge2")
				elif marker_type == "mid":
					self.pos2_mid = target
					assigned.add("2_mid")
				else:
					raise ValueError(f"Marker type '{marker_type}' is not supported for '{marker_name}'")

		required = {"1_edge1", "1_edge2", "1_mid", "2_edge1", "2_edge2", "2_mid"}
		missing = sorted(required - assigned)
		if missing:
			raise ValueError(f"Marker names not found in file: {missing}")
		#print(self.pos1_edge1, self.pos1_edge2, self.pos1_mid, self.pos2_edge1, self.pos2_edge2, self.pos2_mid)


		self.antal_pkter = data.shape[0]
		self.time = data[:, 1]

		def find_start_and_stop(self, pos):
			if pos[0, 0] == 0.0:
				start_found = False
				#print("Start not first")
			else:
				start = 0
				start_found = True
			if pos[:,0][len(pos[:, 0]) - 1] == 0.0:
				stop_found = False
				#print("Stop not last")
			else:
				stop = len(pos[:, 0]) - 1
				stop_found = True
			if start_found == True and stop_found == True:
				return start, stop
			for i in range(self.antal_pkter):
				if start_found == False and pos[i, 0] != 0.0:
					start = i
					start_found = True
				elif start_found == True and pos[i, 0] == 0.0:
					stop = i-1
					stop_found = True
					break
			#print(f"Start: {start}, Stop: {stop}")
			return start, stop
		self.start1, self.stop1 = find_start_and_stop(self, self.pos1_mid)
		self.start2, self.stop2 = find_start_and_stop(self, self.pos2_mid)

		def find_impact(self, pos, start, stop):
			v = np.diff(pos[start:stop], axis=0) / self.dt

			a = np.diff(v, axis=0) / self.dt

			a_belopp = np.linalg.norm(a, axis=1)

			#plt.plot(self.time[1:-1], a_belopp)
			#plt.show()

			max_a_indx = np.argmax(a_belopp)

			return max_a_indx-self.offset, max_a_indx + 2 + self.offset
		self.impact_start, self.impact_stopp = find_impact(self, self.pos1_mid, self.start1, self.stop1)
		print(self.start1, self.impact_start, self.impact_stopp, self.stop1)

		#polynomial fit längs alla 3 dimensioner. använd p_x.deriv()
		def fit(self, pos, indx_start = 0, indx_stopp = None, deg = 2, dimension = 2):
			"""pos = (n, 3) matris av positioner av det valda objektet.
			t_start och t_stopp bestämmer vart datapunkterna för anpassningen slutar vara tillåtna."""
			if indx_stopp == None:
				indx_stopp = self.antal_pkter
		
			t_snitt = self.time[indx_start:indx_stopp]
			pos_snitt = np.asarray(pos[indx_start:indx_stopp])

			if pos_snitt.ndim == 1:
				p = Polynomial.fit(t_snitt, pos_snitt, deg=deg)
				return p

			p_x, resid = Polynomial.fit(t_snitt, pos_snitt[:, 0], deg=deg, full=True)
			p_y = Polynomial.fit(t_snitt, pos_snitt[:, 1], deg=deg)
			p_z = Polynomial.fit(t_snitt, pos_snitt[:, 2], deg=deg)

			self.resid += resid[0][0]

			return p_x, p_y, p_z
		self.pos1xbefore, self.pos1ybefore, self.pos1zbefore = fit(self, self.pos1_mid, indx_start = self.start1, indx_stopp = self.impact_start, deg = 2)
		self.pos2xbefore, self.pos2ybefore, self.pos2zbefore = fit(self, self.pos2_mid, indx_start = self.start2, indx_stopp = self.impact_start, deg = 2)
		self.pos1xafter, self.pos1yafter, self.pos1zafter = fit(self, self.pos1_mid, indx_start = self.impact_stopp, indx_stopp = self.stop1, deg = 2)
		self.pos2xafter, self.pos2yafter, self.pos2zafter = fit(self, self.pos2_mid, indx_start = self.impact_stopp, indx_stopp = self.stop2, deg = 2)

		def hastighet(self, pos_x, pos_y, pos_z):
			v_x = pos_x.deriv()
			v_y = pos_y.deriv()
			v_z = pos_z.deriv()
			vx = v_x(self.time)
			vy = v_y(self.time)
			vz = v_z(self.time)
			vtot = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

			return vtot, v_x, v_y, v_z
		self.v1totbefore, self.v1xbefore, self.v1ybefore, self.v1zbefore = hastighet(self, self.pos1xbefore, self.pos1ybefore, self.pos1zbefore)
		self.v2totbefore, self.v2xbefore, self.v2ybefore, self.v2zbefore = hastighet(self, self.pos2xbefore, self.pos2ybefore, self.pos2zbefore)
		self.v1totafter, self.v1xafter, self.v1yafter, self.v1zafter = hastighet(self, self.pos1xafter, self.pos1yafter, self.pos1zafter)
		self.v2totafter, self.v2xafter, self.v2yafter, self.v2zafter = hastighet(self, self.pos2xafter, self.pos2yafter, self.pos2zafter)



		def angle(self, pos_x, pos_y, pos_z, pos_x_edge1, pos_y_edge1, pos_z_edge1, pos_x_edge2, pos_y_edge2, pos_z_edge2):
			"""
			Beräknar vinkeln mot x-axeln i radianer för alla tidssteg samtidigt.

			Förväntar sig NumPy-arrayer som indata (t.ex. shape (N,)).
			"""
			# Skillnader mellan edge2 och edge1 i x- och y-led
			dx = pos_x_edge2 - pos_x_edge1
			dy = pos_y_edge2 - pos_y_edge1

			# Vektoriserad arctan2 – returnerar en array med vinklar i radianer [-pi, pi]
			return np.arctan2(dy, dx)
		self.angles1 =angle(self, self.pos1_mid[:, 0], self.pos1_mid[:, 1], self.pos1_mid[:, 2], self.pos1_edge1[:, 0], self.pos1_edge1[:, 1], self.pos1_edge1[:, 2], self.pos1_edge2[:, 0], self.pos1_edge2[:, 1], self.pos1_edge2[:, 2])
		self.angles2 =angle(self, self.pos2_mid[:, 0], self.pos2_mid[:, 1], self.pos2_mid[:, 2], self.pos2_edge1[:, 0], self.pos2_edge1[:, 1], self.pos2_edge1[:, 2], self.pos2_edge2[:, 0], self.pos2_edge2[:, 1], self.pos2_edge2[:, 2])

		self.angles_fit1_before = fit(self, self.angles1, indx_start = self.start1, indx_stopp = self.impact_start, deg = 2)
		self.angles_fit1_after = fit(self, self.angles1, indx_start = self.impact_stopp, indx_stopp = self.stop1, deg = 2)
		self.angles_fit2_before = fit(self, self.angles2, indx_start = self.start2, indx_stopp = self.impact_start, deg = 2)
		self.angles_fit2_after = fit(self, self.angles2, indx_start = self.impact_stopp, indx_stopp = self.stop2, deg = 2)

		def angle_speed(angle_fit, timespan=None):
			if timespan is None:
				timespan = self.time
			angle_speed = angle_fit.deriv()
			angle_speed_values = angle_speed(timespan)
			return angle_speed_values, angle_speed
		self.angle_speed1_values_before, self.angle_speed1_before = angle_speed(self.angles_fit1_before, self.time[:self.impact_start])
		self.angle_speed1_values_after, self.angle_speed1_after = angle_speed(self.angles_fit1_after, self.time[self.impact_stopp:])
		self.angle_speed2_values_before, self.angle_speed2_before = angle_speed(self.angles_fit2_before, self.time[:self.impact_start])
		self.angle_speed2_values_after, self.angle_speed2_after = angle_speed(self.angles_fit2_after, self.time[self.impact_stopp:])

		def troghet(self, pos_x_edge1, pos_y_edge1, pos_z_edge1, pos_x_edge2, pos_y_edge2, pos_z_edge2, mass):
			troghet = mass * 1/8 * ((pos_x_edge2 - pos_x_edge1) ** 2 + (pos_y_edge2 - pos_y_edge1) ** 2 + (pos_z_edge2 - pos_z_edge1) ** 2)
			return troghet
		self.troghet1 = troghet(self, self.pos1_edge1[:, 0], self.pos1_edge1[:, 1], self.pos1_edge1[:, 2], self.pos1_edge2[:, 0], self.pos1_edge2[:, 1], self.pos1_edge2[:, 2], self.m1)
		self.troghet2 = troghet(self, self.pos2_edge1[:, 0], self.pos2_edge1[:, 1], self.pos2_edge1[:, 2], self.pos2_edge2[:, 0], self.pos2_edge2[:, 1], self.pos2_edge2[:, 2], self.m2)

		def storheter(self, mass, p_x, p_y, p_z, v_x, v_y, v_z, timespan, angle_speed, troghet):
			"""beräknar kinetisk energi och rörelsemängd"""
			vx = v_x(timespan)
			vy = v_y(timespan)
			vz = v_z(timespan)
			vtot = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

			E_kin = kin_energy(vtot, mass, angle_speed, troghet, rotation=True)

			ln_mom_x = momentum(mass, vx)
			ln_mom_y = momentum(mass, vy)
			ln_mom_z = momentum(mass, vz)
			rot_mom = angular_momentum(angle_speed, troghet)
			#ln_mom = np.matrix([ln_mom_x, ln_mom_y, ln_mom_z])

			return E_kin, ln_mom_x, ln_mom_y, ln_mom_z, rot_mom
		self.E_kin1_before, self.ln_mom_1x_before, self.ln_mom_1y_before, self.ln_mom_1z_before, self.rot_mom_1_before = storheter(self, self.m1, self.pos1xbefore, self.pos1ybefore, self.pos1zbefore, self.v1xbefore, self.v1ybefore, self.v1zbefore, self.time[:self.impact_start], self.angle_speed1_values_before, self.troghet1[:self.impact_start])
		self.E_kin1_after, self.ln_mom_1x_after, self.ln_mom_1y_after, self.ln_mom_1z_after, self.rot_mom_1_after = storheter(self, self.m1, self.pos1xafter, self.pos1yafter, self.pos1zafter, self.v1xafter, self.v1yafter, self.v1zafter, self.time[self.impact_stopp:], self.angle_speed1_values_after, self.troghet1[self.impact_stopp:])
		self.E_kin2_before, self.ln_mom_2x_before, self.ln_mom_2y_before, self.ln_mom_2z_before, self.rot_mom_2_before = storheter(self, self.m2, self.pos2xbefore, self.pos2ybefore, self.pos2zbefore, self.v2xbefore, self.v2ybefore, self.v2zbefore, self.time[:self.impact_start], self.angle_speed2_values_before, self.troghet2[:self.impact_start])
		self.E_kin2_after, self.ln_mom_2x_after, self.ln_mom_2y_after, self.ln_mom_2z_after, self.rot_mom_2_after = storheter(self, self.m2, self.pos2xafter, self.pos2yafter, self.pos2zafter, self.v2xafter, self.v2yafter, self.v2zafter, self.time[self.impact_stopp:], self.angle_speed2_values_after, self.troghet2[self.impact_stopp:])