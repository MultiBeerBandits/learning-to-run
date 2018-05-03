# Ask git the project root folder
ROOT=`git rev-parse --show-toplevel`

python ${ROOT}/osim/main.py --nb_epochs 10 \
                            --nb_episodes 1000 \
                            --episode_length 1000 \
                            --nb_train_steps 50 \
                            --eval_freq 1 \
                            --nb_eval_episodes 1