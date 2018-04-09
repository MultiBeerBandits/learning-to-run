# Ask git the project root folder
ROOT=`git rev-parse --show-toplevel`

ENV_ID='MountainCarContinuous-v0'
python ${ROOT}/macchininarossarossa_example/main.py --env-id ${ENV_ID} \
                                                    --nb_epochs 10 \
                                                    --nb_episodes 1000 \
                                                    --episode_length 1000 \
                                                    --nb_train_steps 50 \
                                                    --eval_freq 100 \
                                                    --nb_eval_episodes 1