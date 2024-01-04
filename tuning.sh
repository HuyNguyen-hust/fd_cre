# DATA=FewRel
DATA=TACRED

for LR in 5e-6 1e-5 2e-5
do
    for BS in 16 32 64
    do
        python run_continual.py --dataname=$DATA --mode=fd --batch_size=$BS --learning_rate=$LR
    done
done