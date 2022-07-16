import os
import torch
import numpy as np
import sacred
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sacred.stflow import LogFileWriter
from sacred.observers import MongoObserver
from torch.utils.data import DataLoader

from data.snuh_dataset import SnuhGaitPhase
from models.phase_classifiers import LSTMClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cur_dir, _ = os.path.split(__file__)
log_dir = os.path.join(cur_dir, 'log', 'ankle')


ex = sacred.Experiment('ANKLE', save_git_info=False)
observer = MongoObserver(url='mongodb://user:6079@143.248.66.79', db_name='ANKLE')
ex.observers.append(observer)


@ex.config
def config():
    split_type = 'total'
    epochs = 250
    batch_size = 128
    hidden_dim = 1024
    n_lstm_layers = 1
    n_out_nodes = 256
    n_out_layers = 1
    out_act_fcn = 'Tanh'
    lr = 5e-4
    p_drop = 0.2
    normalization = True


@ex.command
@LogFileWriter(ex)
def train(split_type,
          epochs,
          batch_size,
          hidden_dim,
          n_lstm_layers,
          n_out_layers,
          n_out_nodes, out_act_fcn,
          lr,
          p_drop,
          normalization):
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    kwargs = {'pin_memory': True} if cuda else {}

    train_dataset = SnuhGaitPhase(split_type=split_type, validation=False, normalization=normalization)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, **kwargs)
    val_dataset = SnuhGaitPhase(split_type=split_type, validation=True, normalization=normalization)

    input_dim = train_dataset.sample_dim
    output_dim = train_dataset.label_dim
    out_act = torch.nn.Tanh if out_act_fcn == 'Tanh' else torch.nn.ReLU
    model = LSTMClassifier(input_dim=input_dim,
                           output_dim=output_dim,
                           hidden_dim=hidden_dim,
                           n_lstm_layers=n_lstm_layers,
                           n_out_layers=n_out_layers,
                           n_out_nodes=n_out_nodes,
                           out_act_fcn=out_act,
                           lr=lr,
                           p_drop=p_drop,
                           device=device)

    val_samples, val_labels = val_dataset[:]
    val_samples = model.fit_data(val_samples)
    val_labels = model.fit_data(val_labels)

    """ path for save model and results """
    run_num = observer.run_entry['_id']
    run_path = os.path.join(log_dir, f'run_{run_num}')
    os.makedirs(run_path, exist_ok=True)

    loss, total_acc, trans_acc = 0., 0., 0.
    for epoch in range(epochs):
        """ Train """
        loss, total_acc, trans_acc = model.train_classifier(epoch, train_loader)
        ex.log_scalar('loss', loss, epoch)
        ex.log_scalar('total_acc', total_acc, epoch)
        ex.log_scalar('trans_acc', trans_acc, epoch)

        """ Validation """
        with torch.no_grad():
            model.eval()
            val_loss = model.calc_loss(val_labels, val_samples).item()
            val_total_acc = 100 * model.calc_correct_ratio(val_labels, val_samples)
            n_trans, n_correct = model.calc_correct_transient(val_labels, val_samples)
            val_trans_acc = 100 * n_correct / n_trans
            ex.log_scalar('val_loss', val_loss, epoch)
            ex.log_scalar('val_total_acc', val_total_acc, epoch)
            ex.log_scalar('val_trans_acc', val_trans_acc, epoch)

            """ Save confusion matrix in observer """
            if (epoch + 1) % 50 == 0:
                p_pred = model.forward(val_samples)
                y_pred = torch.argmax(p_pred, -1).cpu().flatten()
                y_true = torch.argmax(val_labels, -1).cpu().flatten()
                conf_mat = confusion_matrix(y_true, y_pred)
                conf_mat = conf_mat / np.sum(conf_mat, -1)[..., np.newaxis]

                fig = plt.figure()
                ax = fig.gca()
                cax = ax.matshow(conf_mat)
                cax.set_clim(vmin=0., vmax=1.)
                fig.colorbar(cax)

                cats = ['R Stance', 'L Swing', 'L Stance', 'R Swing']
                ax.set_xticklabels([''] + cats)
                ax.set_yticklabels([''] + cats)

                for idx_r, row in enumerate(conf_mat):
                    for idx_c, el in enumerate(row):
                        ax.text(idx_c, idx_r, f'{100 * el:.1f}',
                                va='center', ha='center', fontsize='large')

                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                buf = canvas.buffer_rgba()
                img = Image.fromarray(np.array(buf))

                img.save(run_path + f'/conf_mat_{epoch}.png')
                ex.add_artifact(run_path + f'/conf_mat_{epoch}.png', f'conf_mat_{epoch}.png')

    torch.save(model, run_path + '/model.pth')
    return loss, total_acc


@ex.automain
def main():
    loss, _ = train()

    return loss
