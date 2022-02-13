export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

#python train_rnn.py with "split_type='trial'"
#
#python train_rnn.py with "split_type='trial'" "n_out_layers=2"
#python train_rnn.py with "split_type='trial'" "n_out_layers=3"
#
#python train_rnn.py with "split_type='trial'" "n_out_layers=2" "out_act_fcn='ReLU'"
#python train_rnn.py with "split_type='trial'" "n_out_layers=3" "out_act_fcn='ReLU'"
#
#python train_rnn.py with "split_type='trial'" "normalization=False"

python train_rnn.py with "split_type='subject'"

python train_rnn.py with "split_type='subject'" "n_out_layers=2"
python train_rnn.py with "split_type='subject'" "n_out_layers=3"

python train_rnn.py with "split_type='subject'" "n_out_layers=2" "out_act_fcn='ReLU'"
python train_rnn.py with "split_type='subject'" "n_out_layers=3" "out_act_fcn='ReLU'"

python train_rnn.py with "split_type='subject'" "normalization=False"
