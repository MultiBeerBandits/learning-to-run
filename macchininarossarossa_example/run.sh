# Ask git the project root folder
ROOT=`git rev-parse --show-toplevel`

ENV_ID='MountainCarContinuous-v0'
python ${ROOT}/macchininarossarossa_example/main.py --env-id ${ENV_ID} \
                                                    --render-eval \
                                                    --no-render \
                                                    --nb-rollout-steps 300 \
                                                    --nb-eval-steps 300