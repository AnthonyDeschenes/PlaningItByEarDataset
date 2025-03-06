import datetime
import os
import warnings

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from joblib import dump, load
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve
from sklearn.svm import OneClassSVM
import random

from models.AutoEncoder import AutoEncoder
from models.ConvolutionalAutoEncoder import ConvolutionalAutoEncoder
from models.ConvolutionalAutoEncoderWithSkip import ConvolutionalAutoEncoderWithSkip
from models.ConvolutionalAutoEncoderWithSkipAndTransformer import ConvolutionalAutoEncoderWithSkipAndTransformer
from trainer.dataloader import CustomDataset
from trainer.trainer import ModelTrainer

warnings.filterwarnings("ignore", category=FutureWarning)
matplotlib.use('TkAgg')

# Change this value to True to train the models and to False to evaluate the models on the evaluation dataset and
# generate the ROC curves
train = True

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

v_transformer_cae = ModelTrainer(num_input=401, batch_size=32,
                                 num_input_channels=80,
                                 name="cae-transformer",
                                 network_class=ConvolutionalAutoEncoderWithSkipAndTransformer,
                                 lr=1e-3,
                                 loss=F.mse_loss)

v_skip_cae = ModelTrainer(num_input=401, batch_size=32,
                          num_input_channels=80,
                          name="skip-cae",
                          network_class=ConvolutionalAutoEncoderWithSkip,
                          lr=1e-3,
                          loss=F.mse_loss)

v_cae = ModelTrainer(num_input=401, batch_size=32,
                     num_input_channels=80,
                     name="cae-duman",
                     network_class=ConvolutionalAutoEncoder, lr=1e-3,
                     loss=F.mse_loss)

v_ae = ModelTrainer(num_input=401, batch_size=32,
                    num_input_channels=80,
                    name="ae-DCASE",
                    network_class=AutoEncoder, lr=1e-3,
                    loss=F.mse_loss)

if train:
    d = CustomDataset([
        "data/training/2x3/",
        "data/training/2x4/",
        "data/training/2x6/",
        "data/training/machine_start_stop/"
    ])
else:
    d = CustomDataset([
        "data/evaluation/normal/",
    ])

d_anomalies = CustomDataset([
    "data/evaluation/anomalies/broken_board/",
    "data/evaluation/anomalies/board_stuck/",
    "data/evaluation/anomalies/Uneven_thick_wood/",
])

print(len(d))
y_true = np.zeros(len(d))
y_true_anomalies = np.ones(len(d_anomalies))

y_true = np.concatenate((y_true, y_true_anomalies))

sub_files = [f.split("\\")[-1].split("_") for f in d.files]
date_per_file = [datetime.datetime(year=int(s[1].split("-")[0]) + 2000,
                                   month=int(s[1].split("-")[1]),
                                   day=int(s[1].split("-")[2]),
                                   hour=int(s[2][:2]),
                                   minute=int(s[2][2:]),
                                   second=(int(s[4][0]) - 1) * 10) for s in sub_files]

val_size = int(0.1 * len(d))
train_set, val_set = torch.utils.data.random_split(d, [len(d) - val_size, val_size])

print("Transformer")
v_transformer_cae.train(train_set, val_set)
print("AE")
v_ae.train(train_set, val_set)
print("CAE")
v_cae.train(train_set, val_set)
print("CAE skip")
v_skip_cae.train(train_set, val_set)

if not os.path.exists(os.path.join(os.path.dirname(__file__), f'one_class_svm.joblib')):
    oneclass_svm = OneClassSVM(gamma='auto').fit(d[[i for i in range(len(d))]].reshape(-1, 80 * 401))
    dump(oneclass_svm, os.path.join(os.path.dirname(__file__), f'one_class_svm.joblib'))
else:
    oneclass_svm = load(os.path.join(os.path.dirname(__file__), f'one_class_svm.joblib'))

if not os.path.exists(os.path.join(os.path.dirname(__file__), f'isolation_forest.joblib')):
    isolation_forest = IsolationForest(random_state=42).fit(d[[i for i in range(len(d))]].reshape(-1, 80 * 401))
    dump(isolation_forest, os.path.join(os.path.dirname(__file__), f'isolation_forest.joblib'))
else:
    isolation_forest = load(os.path.join(os.path.dirname(__file__), f'isolation_forest.joblib'))

if not train:
    datas = []
    isolation_forest_scores = []
    one_class_scores = []
    print("Scoring normal data")
    for i, data in enumerate(d):
        if i % 100 == 0:
            print(i, "/", len(d))
        datas.append(data.reshape(80, -1).tolist())
        isolation_forest_scores.append(-isolation_forest.score_samples(data.reshape(-1, 80 * 401)).item())
        one_class_scores.append(-oneclass_svm.score_samples(data.reshape(-1, 80 * 401)).item())

    datas = torch.tensor(datas)
    losses_ae = v_ae.get_loss(datas, reduction='none').mean(dim=(1, 2)).tolist()
    losses_cae = v_cae.get_loss(datas, reduction='none').mean(dim=(1, 2)).tolist()
    losses_transformer = v_transformer_cae.get_loss(datas, reduction='none').mean(dim=(1, 2)).tolist()
    losses_cae_skip = v_skip_cae.get_loss(datas, reduction='none').mean(dim=(1, 2)).tolist()

    print("Scoring anomalous data")
    datas_anomalies = []
    for i, data in enumerate(d_anomalies):
        datas_anomalies.append(data.reshape(80, -1).tolist())

    datas_anomalies = torch.tensor(datas_anomalies)
    losses_ae_anomalies = v_ae.get_loss(datas_anomalies, reduction='none').mean(dim=(1, 2)).tolist()
    losses_cae_anomalies = v_cae.get_loss(datas_anomalies, reduction='none').mean(dim=(1, 2)).tolist()
    losses_transformer_anomalies = v_transformer_cae.get_loss(datas_anomalies, reduction='none').mean(dim=(1, 2)).tolist()
    losses_cae_skip_anomalies = v_skip_cae.get_loss(datas_anomalies, reduction='none').mean(dim=(1, 2)).tolist()
    isolation_forest_scores_anomalies = list(-isolation_forest.score_samples(datas_anomalies.flatten(1, 2)))
    one_class_scores_anomalies = list(-oneclass_svm.score_samples(datas_anomalies.flatten(1, 2)))

    losses_ae.extend(losses_ae_anomalies)
    losses_cae.extend(losses_cae_anomalies)
    losses_transformer.extend(losses_transformer_anomalies)
    losses_cae_skip.extend(losses_cae_skip_anomalies)
    isolation_forest_scores.extend(isolation_forest_scores_anomalies)
    one_class_scores.extend(one_class_scores_anomalies)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}

    matplotlib.rc('font', **font)

    print("Generating ROC curves")
    for legend, scores, line in [
        ("Skip-CAE-Transformer", losses_transformer, 'solid'),
        ("Skip-CAE", losses_cae_skip, 'dotted'),
        ("CAE of Duman", losses_cae, 'dashed'),
        ("OneClassSVM", one_class_scores, 'dashdot'),
        ("DCASE Baseline", losses_ae, (0, (3, 1, 1, 1))),
        ("Isolation Forest", isolation_forest_scores, (0, (3, 5, 1, 5))),
    ]:
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = metrics.auc(fpr, tpr)
        roc_pauc = metrics.roc_auc_score(y_true, scores, max_fpr=0.1)

        plt.plot(fpr, tpr, label=f"{legend} (auc: %0.3f, pauc: %0.3f)" % (roc_auc, roc_pauc),
                 linewidth=2.0, linestyle=line)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of the different models')
    plt.legend(loc="lower right")
    plt.show()
