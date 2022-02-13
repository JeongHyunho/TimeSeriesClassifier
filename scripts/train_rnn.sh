export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."


# total split
python train_rnn.py

python train_rnn.py with "hidden_dim=256"
python train_rnn.py with "hidden_dim=512"
python train_rnn.py with "hidden_dim=2048"

python train_rnn.py with "n_layers=2"
python train_rnn.py with "n_layers=3"

python train_rnn.py with "lr=1e-4"
python train_rnn.py with "lr=1e-3"

python train_rnn.py with "p_drop=0.4"
python train_rnn.py with "p_drop=0.6"
python train_rnn.py with "p_drop=0.8"

# trial split
python train_rnn.py with "split_type='trial'"

python train_rnn.py with "split_type='trial'" "hidden_dim=256"
python train_rnn.py with "split_type='trial'" "hidden_dim=512"
python train_rnn.py with "split_type='trial'" "hidden_dim=2048"

python train_rnn.py with "split_type='trial'" "n_layers=2"
python train_rnn.py with "split_type='trial'" "n_layers=3"

python train_rnn.py with "split_type='trial'" "lr=1e-4"
python train_rnn.py with "split_type='trial'" "lr=1e-3"

python train_rnn.py with "split_type='trial'" "p_drop=0.4"
python train_rnn.py with "split_type='trial'" "p_drop=0.6"
python train_rnn.py with "split_type='trial'" "p_drop=0.8"

# subject split
python train_rnn.py with "split_type='subject'"

python train_rnn.py with "split_type='subject'" "hidden_dim=256"
python train_rnn.py with "split_type='subject'" "hidden_dim=512"
python train_rnn.py with "split_type='subject'" "hidden_dim=2048"

python train_rnn.py with "split_type='subject'" "n_layers=2"
python train_rnn.py with "split_type='subject'" "n_layers=3"

python train_rnn.py with "split_type='subject'" "lr=1e-4"
python train_rnn.py with "split_type='subject'" "lr=1e-3"

python train_rnn.py with "split_type='subject'" "p_drop=0.4"
python train_rnn.py with "split_type='subject'" "p_drop=0.6"
python train_rnn.py with "split_type='subject'" "p_drop=0.8"
