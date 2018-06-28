# Ask git the project root folder
ROOT=`git rev-parse --show-toplevel`

PYTHONPATH=${ROOT} python ${ROOT}/l2run/pois/runcbpois.py \
                           --horizon 200 \
                           --num_episodes 2 \
                           --seed 1234 \
                           --action-repeat 5 \
                           --no-full \
                           --exclude-centering-frame \
                           --reward-scale 10.0 \
                           --fail-reward -0.2 \
                           --integrator-accuracy 1e-3 \
                           --njobs 1