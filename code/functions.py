class storheter:
  def trans_energy(speed, weight):
    return weight * speed * speed * (1/2)

  def rotat_energy(angle_speed, troghet):
    return troghet * angle_speed * angle_speed * (1/2)

  def kin_energy(speed, weight, angle_speed, troghet):
    return storheter.trans_energy(speed, weight) + storheter.rotat_energy(angle_speed, troghet)

  def momentum(weight, speed):
    return weight * speed

  def angular_momentum(angle_speed, troghet):
    return angle_speed * troghet

class parser:
  