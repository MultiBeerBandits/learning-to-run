# Ask git the project root folder
ROOT=`git rev-parse --show-toplevel`

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
                            --num-processes 5 \
                            --no-full \
                            --no-exclude-centering-frame
                            