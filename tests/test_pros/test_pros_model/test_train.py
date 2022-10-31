from models.gait_phase_classifier import CNNClassifier, LSTMClassifier


def test_cnn_estimator_train(make_session_fcn, data_loader_fcn, cnn_basic_kwargs, tmp_path):
    make_session_fcn(session_name='test', output_dir=tmp_path)
    dl = data_loader_fcn(log_dir=tmp_path / 'test' / 'log')
    model = CNNClassifier(**cnn_basic_kwargs)

    for epoch in range(10):
        model.train_model(epoch, dl)
        model.calc_acc(dl.dataset.samples, dl.dataset.labels)


def test_lstm_estimator_train(make_session_fcn, data_loader_fcn, lstm_basic_kwargs, tmp_path):
    make_session_fcn(session_name='test', output_dir=tmp_path)
    dl = data_loader_fcn(log_dir=tmp_path / 'test' / 'log')
    model = LSTMClassifier(**lstm_basic_kwargs)

    for epoch in range(10):
        model.train_model(epoch, dl)
        model.calc_acc(dl.dataset.samples, dl.dataset.labels)
