export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."


# amputee trial split
python train_rnn.py with "split_type='mech_trial'" -c "split_type=mech_trial"

python train_rnn.py with "split_type='mech_trial'" "hidden_dim=256" -c "split_type=mech_trial, hidden_dim=256"
python train_rnn.py with "split_type='mech_trial'" "hidden_dim=512" -c "split_type=mech_trial, hidden_dim=512"
python train_rnn.py with "split_type='mech_trial'" "hidden_dim=2048" -c "split_type=mech_trial, hidden_dim=2048"

python train_rnn.py with "split_type='mech_trial'" "n_layers=2" -c "split_type=mech_trial, n_layers=2"
python train_rnn.py with "split_type='mech_trial'" "n_layers=3" -c "split_type=mech_trial, n_layers=3"

python train_rnn.py with "split_type='mech_trial'" "lr=1e-4" -c "split_type=mech_trial, lr=le-4"
python train_rnn.py with "split_type='mech_trial'" "lr=1e-3" -c "split_type=mech_trial, lr=le-3"

python train_rnn.py with "split_type='mech_trial'" "p_drop=0.4" -c "split_type=mech_trial, p_drop=0.4"
python train_rnn.py with "split_type='mech_trial'" "p_drop=0.6" -c "split_type=mech_trial, p_drop=0.6"
python train_rnn.py with "split_type='mech_trial'" "p_drop=0.8" -c "split_type=mech_trial, p_drop=0.8"

# amputee subject split
python train_rnn.py with "split_type='mech_subject'" -c "split_type=mech_subject"

python train_rnn.py with "split_type='mech_subject'" "hidden_dim=256" -c "split_type=mech_subject, hidden_dim=256"
python train_rnn.py with "split_type='mech_subject'" "hidden_dim=512" -c "split_type=mech_subject, hidden_dim=512"
python train_rnn.py with "split_type='mech_subject'" "hidden_dim=2048" -c "split_type=mech_subject, hidden_dim=2048"

python train_rnn.py with "split_type='mech_subject'" "n_layers=2" -c "split_type=mech_subject, n_layers=2"
python train_rnn.py with "split_type='mech_subject'" "n_layers=3" -c "split_type=mech_subject, n_layers=3"

python train_rnn.py with "split_type='mech_subject'" "lr=1e-4" -c "split_type=mech_subject, lr=le-4"
python train_rnn.py with "split_type='mech_subject'" "lr=1e-3" -c "split_type=mech_subject, lr=le-3"

python train_rnn.py with "split_type='mech_subject'" "p_drop=0.4" -c "split_type=mech_subject, p_drop=0.4"
python train_rnn.py with "split_type='mech_subject'" "p_drop=0.6" -c "split_type=mech_subject, p_drop=0.6"
python train_rnn.py with "split_type='mech_subject'" "p_drop=0.8" -c "split_type=mech_subject, p_drop=0.8"
