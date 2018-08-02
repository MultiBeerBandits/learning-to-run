# learning-to-run
A project for the Deep Learning course held at Politecnico di Milano (A.Y. 2018).
Implementation taken from Reason8 group.
Video: https://youtu.be/HVOrhxypOGg

# Documentation
You can find the paper describing our approach and our results in the "doc" folder inside this repository.

# Observation vector
Values in the full observation vector (full_state branch)
- x, y, vx, vy of pelvis (4 values)
- x, y, vx, vy, ax, ay of head, torso, toes_l, toes_r, talus_l, talus_r (6*6 values)
- rz, vrz, arz of ankle_l, ankle_r, back, hip_l, hip_r, knee_l, knee_r, ground_pelvis (8*3 values)
- x, y, vx, vy of center of mass (4)
- 4 + 6*6 + 8*3 + 4 = 68

Values in the basic observation vector:
- x, y of pelvis (2 values)
- x, y  of head, torso, toes_l, toes_r, talus_l, talus_r (2*6 values)
- rz, vrz of ankle_l, ankle_r, hip_l, hip_r, knee_l, knee_r (2*6 values)
- r, vr of ground pelvis (2)
- x, y, vx, vy of center of mass (4)
- vx, vy of pelvis (2)
- 2 + 2*6 + 2*6 + 2 + 4 + 2 = 34

Body pose's `x` coordinates are centered with respect to the pelvis `x` coordinate.
It is possible to remove the pelvis `x` coordinate from the observation vector by setting `--exclude-centering-frame`.

Muscles strength is fixed to 1.

No obstacles.

# DDPG improvements
- Parameter noise
- Layer Normalization
- State and action flip
- State centered

# Our Implementation
- Parallel sampling
- Linear decay for learning rates

# Installation
```sh
conda create -n opensim-rl -c kidzik opensim python=3.6.1
source activate opensim-rl
conda install -c conda-forge lapack git
pip install git+https://github.com/stanfordnmbl/osim-rl.git
```

# Baseline installation
Install the baseline version inside this repo:
```sh
cd baselines
pip install -e .
```

# Running
Using the script:
```sh
cp osim/run.sh.template osim/run.sh
chmod +x osim/run.sh
```

Or manually:
```sh
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

# Generate video from frames
```sh
cd video_folder
ffmpeg -framerate 15 -i "Frame%04d.png" -vf format=yuv420p -preset veryslow l2run_video.mp4
```

# Authors

- Leonardo Arcari
- Emiliano Gagliardi
- Emanuele Ghelfi
