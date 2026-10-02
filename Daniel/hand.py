import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

#antag att det inte är nödvändigt att projicera ner på ett plan, räkna på x och y och i värsta fall får du imorgon fixa det.

class bearbeta:
	def __init__(self, filnamn = "data.tsv", m1 = 1.0, m2 = 1.0, dt = 1/100, headers=10, testnr=1, offset=10):
		"""m1 respektive m2 är massorna, m1 längst till vänster. dt är 1/uppdateringsfrekvensen.
		headers är antal rader att skippa i början av dokuentet"""
		self.dt = dt
		self.m1 = m1
		self.m2 = m2
		self.testnr = testnr
		self.offset = offset
		self.resid = 0

		#"data.tsv" är filnamn, skip_header bör varieras beroende på filformat
		data = np.genfromtxt(filnamn, np.float64, delimiter="\t", skip_header=headers, filling_values=1)

		self.pos1 = (10**-3) * data[:, 2:5]
		self.pos2 = (10**-3) * data[:, 5:8]

		self.antal_pkter = data.shape[0]
		#self.t = np.arange(self.antal_pkter) * self.dt
		self.t = data[:, 1]

		#polynomial fit längs alla 3 dimensioner. använd p_x.deriv()
		def fit(self, pos, indx_start = 0, indx_stopp = None):
				"""pos = (n, 3) matris av positioner av det valda objektet.
				t_start och t_stopp bestämmer vart datapunkterna för anpassningen slutar vara tillåtna."""
				if indx_stopp == None:
						indx_stopp = self.antal_pkter
		
				t_snitt = self.t[indx_start:indx_stopp]
				pos_snitt = pos[indx_start:indx_stopp]
				#print(self.t)
				#print(pos_snitt)

				p_x, resid = Polynomial.fit(t_snitt, pos_snitt[:, 0], deg=2, full=True)
				p_y = Polynomial.fit(t_snitt, pos_snitt[:, 1], deg=2)
				p_z = Polynomial.fit(t_snitt, pos_snitt[:, 2], deg=2)

				self.resid += resid[0][0]
				

				return p_x, p_y, p_z

		def test_plot(self, indx_start = 0, indx_stopp = None, two=False, stöt = True, ovrd_i = None, ovrd_e = None, prinnt=False, b_längd=50):

				if stöt == True:
						if ovrd_i != None:
								indx_innan = ovrd_i
								indx_efter = ovrd_e
						else:
								indx_innan, indx_efter = self.finn_stöt(self.pos1)

						self.indx_innan = indx_innan
						self.indx_efter = indx_efter
				else:
						indx_innan = indx_stopp
						indx_efter = indx_start


				p_x_i, p_y_i, p_z_i = self.fit(self.pos1, indx_innan -b_längd, indx_innan)
				p_x_e, p_y_e, p_z_e = self.fit(self.pos1, indx_efter, indx_efter +b_längd)
				#print(p_x)

				if two == True:
						p_x2_i, p_y2_i, p_z2_i = self.fit(self.pos2, indx_innan -b_längd, indx_innan)
				
						p_x2_e, p_y2_e, p_z2_e = self.fit(self.pos2, indx_efter, indx_efter +b_längd)
						
						col = 2
						row = 4

						plt.subplot(row, col, 5)
						plt.plot(self.t, p_x2_i(self.t))
						plt.plot(self.t, p_x2_e(self.t))
						plt.plot(self.t, self.pos2[:, 0])
						plt.title("X2")

						plt.subplot(row, col, 6)
						plt.plot(self.t, p_y2_i(self.t))
						plt.plot(self.t, p_y2_e(self.t))
						plt.plot(self.t, self.pos2[:, 1])
						plt.title("Y2")

						plt.subplot(row, col, 7)
						plt.plot(self.t, p_z2_i(self.t))
						plt.plot(self.t, p_z2_e(self.t))
						plt.plot(self.t, self.pos2[:, 2])
						plt.title("Z2")

						v2_i, vx2_i, vy2_i, vz2_i = self.hastighet(p_x2_i, p_y2_i, p_z2_i)

						v2_e, vx2_e, vy2_e, vz2_e = self.hastighet(p_x2_e, p_y2_e, p_z2_e)

						plt.subplot(row, col, 8)
						plt.plot(self.t, v2_i)
						plt.plot(self.t, v2_e)
						plt.title("V2")
						

				else:
						col = 2
						row = 2


				plt.subplot(row, col, 1)
				plt.plot(self.t, p_x_i(self.t))
				plt.plot(self.t, p_x_e(self.t))
				plt.plot(self.t, self.pos1[:, 0])
				plt.title("X")

				plt.subplot(row, col, 2)
				plt.plot(self.t, p_y_i(self.t))
				plt.plot(self.t, p_y_e(self.t))
				plt.plot(self.t, self.pos1[:, 1])
				plt.title("Y")

				plt.subplot(row, col, 3)
				plt.plot(self.t, p_z_i(self.t))
				plt.plot(self.t, p_z_e(self.t))
				plt.plot(self.t, self.pos1[:, 2])
				plt.title("Z")

				v_i, vx_i, vy_i, vz_i = self.hastighet(p_x_i, p_y_i, p_z_i)

				v_e, vx_e, vy_e, vz_e = self.hastighet(p_x_e, p_y_e, p_z_e)

				plt.subplot(row, col, 4)
				plt.plot(self.t, v_i)
				plt.plot(self.t, v_e)
				plt.title("V")

				plt.tight_layout()

				plt.show()

				#print(self.pos1[indx_innan, 0])

				numder = (self.pos2[self.indx_efter +30, 0] - self.pos2[self.indx_efter, 0])/ (30* self.dt)
				numder_kort = (self.pos2[self.indx_efter +2, 0] - self.pos2[self.indx_efter, 0]) / (2* self.dt)
				print(f"Numerisk derivata av X2 efter stöt:{numder:.4f}")
				print(f"Kort numerisk derivata av X2 efter stöt: {numder_kort:.4f}\n")

				vx_innan, vx2_innan, vx_efter, vx2_efter, vy_innan, vy2_innan, vy_efter, vy2_efter, px_tot_innan, px_tot_efter, py_tot_innan, py_tot_efter, px_diff, py_diff, K_px, K_py = self.rörelsemängd(vx_i, vy_i, vz_i, vx_e, vy_e, vz_e, vx2_i, vy2_i, vz2_i, vx2_e, vy2_e, vz2_e)

				E_innan, E_efter, E_diff, K_E = self.energi(vx_i, vy_i, vz_i, vx_e, vy_e, vz_e, vx2_i, vy2_i, vz2_i, vx2_e, vy2_e, vz2_e)
				#t_innan = self.t[0: indx_innan]
				#t_efter = self.t[indx_efter: self.antal_pkter]
				#plt.plot(t_innan, p_tot_belopp_i, label="total rörelsemängd innan")
				#plt.plot(t_efter, p_tot_belopp_e, label="total rörelsemängd efter")
				#plt.legend()

				#plt.show()

				print(f"The residual of all 4 x functions is: {self.resid:.4g}")

				if prinnt == True:
						datan = np.atleast_2d([self.testnr, self.m1, self.m2, 
																	 vx_innan, vx2_innan, vx_efter, vx2_efter, vy_innan, vy2_innan, vy_efter, vy2_efter, 
																	 px_tot_innan, px_tot_efter, py_tot_innan, py_tot_efter, px_diff, py_diff, K_px, K_py, 
																	 E_innan, E_efter, E_diff, K_E])

						with open("mätvärden/luftbord/datan.tsv", mode="a", encoding="utf-8") as f:
								np.savetxt(f, datan, delimiter="\t", fmt="%s")

						print(f"====================================================\n NR {self.testnr} PRINTED TO DATAN.TSV \n====================================================")
				
		
		def hastighet(self, p_x, p_y, p_z):
				v_x = p_x.deriv()
				v_y = p_y.deriv()
				v_z = p_z.deriv()
				vx = v_x(self.t)
				vy = v_y(self.t)
				vz = v_z(self.t)
				vb = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

				return vb, v_x, v_y, v_z


		def finn_stöt(self, pos):
				v = np.diff(pos, axis=0) / self.dt

				a = np.diff(v, axis=0) / self.dt

				a_belopp = np.linalg.norm(a, axis=1)

				plt.plot(self.t[1:-1], a_belopp)
				plt.show()

				max_a_indx = np.argmax(a_belopp)

				return max_a_indx-self.offset, max_a_indx + 2 + self.offset


		def rörelsemängd(self, vx_i, vy_i, vz_i, vx_e, vy_e, vz_e, vx2_i, vy2_i, vz2_i, vx2_e, vy2_e, vz2_e):
				t_innan = (self.indx_innan +1 + self.offset) * self.dt
				t_efter = (self.indx_innan +1 + self.offset) * self.dt

				#print(t_innan, t_efter)
				#bestämmer t_innan = t_efter = exakta tidpunkten av stöten.

				vx_innan = vx_i(t_innan)
				vy_innan = vy_i(t_innan)
				vz_innan = vz_i(t_innan)
				#print(vx_innan)

				vx_efter = vx_e(t_efter)
				vy_efter = vy_e(t_efter)
				vz_efter = vz_e(t_efter)

				vx2_innan = vx2_i(t_innan)
				vy2_innan = vy2_i(t_innan)
				vz2_innan = vz2_i(t_innan)

				vx2_efter = vx2_e(t_efter)
				vy2_efter = vy2_e(t_efter)
				vz2_efter = vz2_e(t_efter)
				
				print(f"Analytisk derivata av X2 efter stöt: {vx2_efter:.4f}\n")
				#print("\n")
				#print(vx2_innan)
				#print(vx2_efter)

				#antag att rörelsemängd i varje led borde bevaras innan och efter stöt

				px_innan = self.m1 * vx_innan
				px2_innan = self.m2 * vx2_innan

				py_innan = self.m1 * vy_innan
				py2_innan = self.m2 * vy2_innan

				pz_innan = self.m1 * vz_innan
				pz2_innan = self.m2 * vz2_innan

				px_tot_innan = px_innan + px2_innan
				py_tot_innan = py_innan + py2_innan
				pz_tot_innan = pz_innan + pz2_innan


				px_efter = self.m1 * vx_efter
				px2_efter = self.m2 * vx2_efter

				py_efter = self.m1 * vy_efter
				py2_efter = self.m2 * vy2_efter

				pz_efter = self.m1 * vz_efter
				pz2_efter = self.m2 * vz2_efter

				px_tot_efter = px_efter + px2_efter
				py_tot_efter = py_efter + py2_efter
				pz_tot_efter = pz_efter + pz2_efter

				px_diff = px_tot_innan - px_tot_efter
				py_diff = py_tot_innan - py_tot_efter
				pz_diff = pz_tot_innan - pz_tot_efter

				print(f"\nHASTIGHETER -----------------------------------------")
				print(f"v1xi: {vx_innan:.3g}, v2xi: {vx2_innan:.3g}")
				print(f"v1xe: {vx_efter:.3g}, v2xe: {vx2_efter:.3g}")
				print(f"v1yi: {vy_innan:.3g}, v2yi: {vy2_innan:.3g}")
				print(f"v1ye: {vy_efter:.3g}, v2ye: {vy2_efter:.3g}")

				print(f"\nRÖRELSEMÄNGD ----------------------------------------")
				print(f" pxi: {px_tot_innan:.3g} \n pxe: {px_tot_efter:.3g} \n")
				print(f" pyi: {py_tot_innan} \n pye: {py_tot_efter} \n")
				#print(f" Z innan: {pz_tot_innan} \n Z efter: {pz_tot_efter} \n")
				print(f" Delta px: {px_diff:.3g} \n Delta py: {py_diff:.3g}")# \n Delta Z: {pz_diff:.3g} \n")
				print(f"K_px: \n {(px_tot_efter / px_tot_innan):.3g}\n")
				print(f"K_py \n {(py_tot_efter / py_tot_innan):.3f}\n")
				#print(f"Efter = skillnad: \n Z: {(pz_tot_efter / pz_tot_innan):.3f}\n")
				K_px = px_tot_efter / px_tot_innan
				K_py = py_tot_efter / py_tot_innan
				#print(K_px, K_py)

				return vx_innan, vx2_innan, vx_efter, vx2_efter, vy_innan, vy2_innan, vy_efter, vy2_efter, px_tot_innan, px_tot_efter, py_tot_innan, py_tot_efter, px_diff, py_diff, K_px, K_py


		def energi(self, vx_i, vy_i, vz_i, vx_e, vy_e, vz_e, vx2_i, vy2_i, vz2_i, vx2_e, vy2_e, vz2_e):
				t_innan = (self.indx_innan +1 + self.offset) * self.dt
				t_efter = (self.indx_innan +1 + self.offset) * self.dt

				#print(t_innan, t_efter)
				#bestämmer t_innan = t_efter = exakta tidpunkten av stöten.

				vx_innan = vx_i(t_innan)
				vy_innan = vy_i(t_innan)
				vz_innan = vz_i(t_innan)
				#print(vx_innan)

				vx_efter = vx_e(t_efter)
				vy_efter = vy_e(t_efter)
				vz_efter = vz_e(t_efter)
				print(f"Analytisk derivata av X efter stöt: {vx_efter:.3g}\n")

				vx2_innan = vx2_i(t_innan)
				vy2_innan = vy2_i(t_innan)
				vz2_innan = vz2_i(t_innan)

				vx2_efter = vx2_e(t_efter)
				vy2_efter = vy2_e(t_efter)
				vz2_efter = vz2_e(t_efter)
				#print("\n")
				#print(vx2_innan)
				#print(vx2_efter)

				#antag att rörelsemängd i varje led borde bevaras innan och efter stöt

				E_innan = ((self.m1 / 2) * (vx_innan ** 2 + vy_innan ** 2)) + ((self.m2 / 2) * (vx2_innan ** 2 + vy2_innan ** 2))
				E_efter = ((self.m1 / 2) * (vx_efter ** 2 + vy_efter ** 2)) + ((self.m2 / 2) * (vx2_efter ** 2 + vy2_efter ** 2))

				#print(f"\n\n{vx_innan}\n{E_innan}")
				#print(f"{vx2_innan}\n{Ex2_innan}\n{Ex2_innan + Ex_innan}")

				#print(f"\n\n{vx_efter}\n{Ex_efter}")
				#print(f"{vx2_efter}")#\n{Ex2_efter}\n{Ex2_efter + Ex_efter}")

				E_diff = E_innan - E_efter

				print(f"ENERGI -------------------------------------------------")
				print(f" Ei: {E_innan:.3g} \n Ee: {E_efter:.3g} \n")
				#print(f" Y innan: {py_tot_innan} \n Y efter: {py_tot_efter} \n")
				#print(f" Z innan: {pz_tot_innan} \n Z efter: {pz_tot_efter} \n")
				print(f" Delta E: {E_diff:.3g}")
				#print("Delta Y: {py_diff} \n Delta Z: {pz_diff} \n")
				print(f"K_E: \n {(E_efter / E_innan):.3g}\n")
				#print(f"Efter = skillnad: \n Y: {(py_tot_efter / py_tot_innan):.3f}\n")
				#print(f"Efter = skillnad: \n Z: {(pz_tot_efter / pz_tot_innan):.3f}\n")
				K_E = E_efter / E_innan

				return E_innan, E_efter, E_diff, K_E



t = np.linspace(0, 4, 2000)

test = bearbeta(filnamn="mätvärden/luftbord/22.tsv", testnr=22, dt = 1/200, m1=0.028, m2=0.028, offset=10)

test.test_plot(stöt=True, two=True, ovrd_i=None, ovrd_e=None, prinnt=False, b_längd=40)