mkdir -p ./save
mkdir -p ./trainlogs

method=famo
seed=42
gamma=0.001

python trainer.py --method=$method --seed=$seed --num_tasks=40 --gamma=$gamma > trainlogs/famo-gamma$gamma-$seed.log 2>&1 &
# for M-ConFIG, you can also set --num_updates as the number of momentum updates in each iteration