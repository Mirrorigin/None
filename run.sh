python main.py --dim 40 --model 'transE' --dataset 'NELL-995-h100' --p_norm 2

python main.py --dim 120 --model 'poincare' --lr 50 --p_norm 1
python main.py --dim 160 --model 'poincare' --lr 50 --p_norm 1
python main.py --dim 200 --model 'poincare' --lr 50 --p_norm 1

python main.py --dim 120 --model 'transE' --p_norm 1
python main.py --dim 160 --model 'transE' --p_norm 1
python main.py --dim 200 --model 'transE' --p_norm 1

python main.py --dim 120 --model 'transE' --p_norm 2
python main.py --dim 160 --model 'transE' --p_norm 2
python main.py --dim 200 --model 'transE' --p_norm 2

python main.py --dim 120 --model 'distmult' --p_norm 2
python main.py --dim 160 --model 'distmult' --p_norm 2
python main.py --dim 200 --model 'distmult' --p_norm 2