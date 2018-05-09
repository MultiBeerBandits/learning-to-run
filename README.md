# learning-to-run
A project for the Deep Learning course held at Politecnico di Milano (A.Y. 2018)

# Observation vector
Values in the observation vector
- y, vx, vy, ax, ay, rz, vrz, arz of pelvis (8 values)
- x, y, vx, vy, ax, ay, rz, vrz, arz of head, torso, toes_l, toes_r, talus_l, talus_r (9*6 values)
- rz, vrz, arz of ankle_l, ankle_r, back, hip_l, hip_r, knee_l, knee_r (7*3 values)
- activation, fiber_len, fiber_vel for all muscles (3*18)
- x, y, vx, vy, ax, ay ofg center of mass (6)
- 8 + 9*6 + 8*3 + 3*18 + 6 = 146