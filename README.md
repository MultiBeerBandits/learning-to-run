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

# Installation
```
conda create -n opensim-rl -c kidzik opensim python=3.6.1
source activate opensim-rl
conda install -c conda-forge lapack git
pip install git+https://github.com/stanfordnmbl/osim-rl.git
```

# Baseline installation
Install the baseline version inside this repo:
```
cd baselines
pip install -e .
```

# Running
Using the script:
```
cp osim/run.sh.template osim/run.sh
chmod +x osim/run.sh
```

Or manually:
```
python ${ROOT}/osim/main.py --batch-size 200 \
							--nb-epochs 1000 \
                            --nb-epoch-cycles 1000 \
                            --nb-episodes 5 \
                            --episode-length 1000 \
                            --nb-train-steps 50 \
                            --eval-freq 1 \
                            --save-freq 1 \
                            --nb-eval-episodes 1 \
                            --action-repeat 5 \
                            --reward-scale 10 \
                            --flip-state \
                            --num-processes 5 
```

## Environment Wrapper
Basic observation vector:
41 dimensions: position and velocities

Full observation vector:
143 dimensions: positions, velocities and accelerations

# Authors

- Emiliano Gagliardi
- Leonardo Arcari
- Emanuele Ghelfi