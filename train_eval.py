
import os
import pickle
import time
import random
import shutil
import argparse
import units as units
from units import FocalLoss, adjust_input
from transformer import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch

# Pick mps (Apple‑GPU) if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")



def train_model(training_file='training_file',
                validation_file='validation_file',
                testing_file='testing_file',
                n_diagnosis_codes=10000,
                n_labels=2,
                output_file='output_file',
                batch_size=100,
                dropout_rate=0.5,
                L2_reg=0.001,
                n_epoch=1000,
                log_eps=1e-8,
                visit_size=512,
                hidden_size=256,
                use_gpu=False,
                model_name='',
                disease = 'hf',
                code2id = None,
                running_data='',
                gamma=0.5,
                model_file = None,
                layer=1,
                log_handle=None):
    options = locals().copy()

    print('building the model ...')
    model = model_file(n_diagnosis_codes, batch_size, options)
    focal_loss = FocalLoss(2, gamma=gamma)
    print('constructing the optimizer ...')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = options['L2_reg'])
    print('done!')

    print('loading data ...')
    train, validate, test = units.load_data(training_file, validation_file, testing_file)
    n_batches = int(np.ceil(float(len(train[0])) / float(batch_size)))

    print('training start')
    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    best_test_cost = 0.0
    epoch_duaration = 0.0
    best_epoch = 0.0
    max_len = 50
    best_parameters_file = ''
    if use_gpu:
        model.to(device)          # mps  ➜  GPU   |   cpu ➜  CPU
    else:
        model.to("cpu")
    model.train()
    train_losses, val_losses = [], []
    for epoch in range(n_epoch):
        iteration = 0
        cost_vector = []
        start_time = time.time()
        samples = random.sample(range(n_batches), n_batches)
        counter = 0

        for index in samples:
            batch_diagnosis_codes = train[0][batch_size * index: batch_size * (index + 1)]
            batch_time_step = train[2][batch_size * index: batch_size * (index + 1)]
            batch_diagnosis_codes, batch_time_step = adjust_input(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes)
            batch_labels = train[1][batch_size * index: batch_size * (index + 1)]
            lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
            maxlen = np.max(lengths)
            predictions, labels, self_attention = model(batch_diagnosis_codes, batch_time_step, batch_labels, options, maxlen)
            optimizer.zero_grad()

            loss = focal_loss(predictions, labels)
            loss.backward()
            optimizer.step()


            cost_vector.append(loss.cpu().data.numpy())

            if (iteration % 50 == 0):
                print('epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, loss.cpu().data.numpy()))
                #print(self_attention[:,0,0].squeeze().cpu().data.numpy())
                #print(time_weight[:, 0])
                #print(prior_weight[:, 0])
                #print(model.time_encoder.time_weight[0:10])
                #print(self_weight[:, 0])
            iteration += 1

        duration  = time.time() - start_time
        mean_cost = np.mean(cost_vector)
        print(f"epoch:{epoch}, mean_cost:{mean_cost:.6f}, duration:{duration:.2f}s")

        # 2) compute validation & test costs
        validate_cost = units.calculate_cost_tran(model, validate, options, max_len, focal_loss)
        test_cost     = units.calculate_cost_tran(model, test,    options, max_len, focal_loss)
        print(f"epoch:{epoch}, validate_cost:{validate_cost:.6f}, test_cost:{test_cost:.6f}")

        # 3) now it’s safe to log (validate_cost & test_cost definitely exist)
        if log_handle is not None:
            log_handle.write(
                f"  Epoch {epoch:2d}: "
                f"Train={mean_cost:.4f}, "
                f"Val={validate_cost:.4f}, "
                f"Test={test_cost:.4f}, "
                f"Dur={duration:.2f}s\n"
            )

        train_cost = np.mean(cost_vector)
        validate_cost = units.calculate_cost_tran(model, validate, options, max_len, focal_loss)
        test_cost = units.calculate_cost_tran(model, test, options, max_len, focal_loss)
        print('epoch:%d, validate_cost:%f, duration:%f' % (epoch, validate_cost, duration))
        epoch_duaration += duration

        train_cost = np.mean(cost_vector)
        epoch_duaration += duration
        if validate_cost > (best_validate_cost + 0.04) and epoch > 19:
            print(validate_cost)
            print(best_validate_cost)
            break
        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_test_cost = test_cost
            best_epoch = epoch

            shutil.rmtree(output_file)
            os.mkdir(output_file)

            torch.save(model.state_dict(), output_file + model_name + '.' + str(epoch))
            best_parameters_file = output_file + model_name + '.' + str(epoch)
        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (
        best_epoch, best_train_cost, best_validate_cost, best_test_cost)
        print(buf)
    # testing
    #best_parameters_file = output_file + model_name + '.' + str(8)
    model.load_state_dict(torch.load(best_parameters_file))
    model.eval()
    n_batches = int(np.ceil(float(len(test[0])) / float(batch_size)))
    y_true = np.array([])
    y_pred = np.array([])
    y_probs = np.array([]) 
    for index in range(n_batches):
        batch_diagnosis_codes = test[0][batch_size * index: batch_size * (index + 1)]
        batch_time_step = test[2][batch_size * index: batch_size * (index + 1)]
        batch_diagnosis_codes, batch_time_step = adjust_input(batch_diagnosis_codes, batch_time_step, max_len, n_diagnosis_codes)
        batch_labels = test[1][batch_size * index: batch_size * (index + 1)]
        lengths = np.array([len(seq) for seq in batch_diagnosis_codes])
        maxlen = np.max(lengths)
        logit, labels, self_attention = model(batch_diagnosis_codes, batch_time_step, batch_labels, options, maxlen)

        # get class‐1 probability
        probs = torch.softmax(logit, dim=1)[:,1].detach().cpu().numpy()
        prediction = np.argmax(logit.detach().cpu().numpy(), axis=1)
        labels     = labels.detach().cpu().numpy() if use_gpu else labels.numpy()

        y_true  = np.concatenate((y_true,  labels))
        y_pred  = np.concatenate((y_pred,  prediction))
        y_probs = np.concatenate((y_probs, probs))

    accuary = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    print(accuary, precision, recall, f1, roc_auc)
    return (accuary,
            precision,
            recall,
            f1,
            roc_auc,
            train_losses,
            val_losses,
            y_true,
            y_probs)


# For real data set please contact the corresponding author


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train HiTANet TransformerTime")
    parser.add_argument('--batch_size',   type=int,   default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--L2_reg',       type=float, default=5e-3)
    parser.add_argument('--log_eps',      type=float, default=1e-7)
    parser.add_argument('--n_epoch',      type=int,   default=50)
    parser.add_argument('--n_labels',     type=int,   default=2)
    parser.add_argument('--visit_size',   type=int,   default=256)
    parser.add_argument('--hidden_size',  type=int,   default=256)
    parser.add_argument('--gamma',        type=float, default=0.0)
    parser.add_argument('--use_gpu',      action='store_true')
    parser.add_argument('--layer',        type=int,   default=2)
    parser.add_argument('--diseases',     nargs='+',  default=['hf_sample'])
    parser.add_argument('--model_choice', type=str,   default='TransformerTime')
    args = parser.parse_args()

    # copy into locals for backwards compatibility
    batch_size   = args.batch_size
    dropout_rate = args.dropout_rate
    L2_reg       = args.L2_reg
    log_eps      = args.log_eps
    n_epoch      = args.n_epoch
    n_labels     = args.n_labels
    visit_size   = args.visit_size
    hidden_size  = args.hidden_size
    gamma        = args.gamma
    use_gpu      = args.use_gpu
    layer        = args.layer
    model_choice = args.model_choice
    disease_list = args.diseases

    y_true_runs  = []
    y_score_runs = []
    y_probs = np.array([])
    
    model_file = eval(model_choice)
    for disease in disease_list:
        model_name   = f"tran_{model_choice}_{disease}_L{layer}_wt_1e-4_focal{gamma:.2f}"
        log_file     = f"results/{model_name}.txt"
        path         = f"data/{disease}/model_inputs/"
        trianing_file   = path + f"{disease}_training_new.pickle"
        validation_file = path + f"{disease}_validation_new.pickle"
        testing_file    = path + f"{disease}_testing_new.pickle"

        dict_file = f"data/{disease}/{disease}_code2idx_new.pickle"
        code2id   = pickle.load(open(dict_file, 'rb'))
        n_diagnosis_codes = len(code2id) + 1

        output_file_path = f"cache/{model_choice}_outputs/"
        os.makedirs(output_file_path, exist_ok=True)

        results = []
        with open(log_file, 'w') as f:
            f.write(model_name + '\n')
            f.write('=' * len(model_name) + '\n\n')
            for run_idx in range(1, 6):
                f.write(f"Run {run_idx}:\n")
                acc, prec, rec, f1, auc, train_hist, val_hist, y_true, y_probs = train_model(
                    trianing_file, validation_file, testing_file,
                    n_diagnosis_codes, n_labels,
                    output_file_path, batch_size, dropout_rate,
                    L2_reg, n_epoch, log_eps, visit_size,
                    hidden_size, use_gpu, model_name,
                    disease=disease, code2id=code2id,
                    gamma=gamma, layer=layer,
                    model_file=model_file,
                    log_handle=f
                )
                y_true_runs.append(y_true)
                y_score_runs.append(y_probs)
                results.append([acc, prec, rec, f1, auc])
                f.write('\n')

            results = np.array(results)
            f.write('Final metrics per run (acc, prec, rec, f1, auc):\n')
            for i, run in enumerate(results, 1):
                f.write(f" Run {i}: " + ', '.join(f"{x:.4f}" for x in run) + '\n')

            f.write('\nSummary:\n')
            f.write(' Mean: ' + ', '.join(f"{x:.4f}" for x in results.mean(axis=0)) + '\n')
            f.write('  Std: ' + ', '.join(f"{x:.4f}" for x in results.std(axis=0)) + '\n')
            


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to the results file uploaded
file_path = 'results/tran_TransformerTime_hf_sample_L2_wt_1e-4_focal0.00.txt'

# Read lines from the file
with open(file_path, 'r') as f:
    lines = f.readlines()

# Parse per-epoch histories for each run
runs_hist = []
i = 0
while i < len(lines):
    if lines[i].startswith('Run'):
        hist = {'train': [], 'val': [], 'test': []}
        i += 1
        while i < len(lines) and lines[i].strip().startswith('Epoch'):
            m = re.match(r'\s*Epoch\s*\d+:\s*Train=([\d.]+),\s*Val=([\d.]+),\s*Test=([\d.]+)', lines[i])
            if m:
                hist['train'].append(float(m.group(1)))
                hist['val'].append(float(m.group(2)))
                hist['test'].append(float(m.group(3)))
            i += 1
        runs_hist.append(hist)
    else:
        i += 1

# Convert histories to DataFrames for easier plotting
n_runs = len(runs_hist)
n_epochs = len(runs_hist[0]['train'])
train_array = np.array([run['train'] for run in runs_hist])
val_array   = np.array([run['val']   for run in runs_hist])
test_array  = np.array([run['test']  for run in runs_hist])

# Plot average train vs validation loss
mean_train = train_array.mean(axis=0)
mean_val   = val_array.mean(axis=0)

plt.figure()
plt.plot(range(n_epochs), mean_train, label='Avg Train Loss')
plt.plot(range(n_epochs), mean_val,   label='Avg Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Train & Validation Loss Over Epochs')
plt.legend()
plt.show()

# Parse final metrics
metrics = []
i = 0
# Find "Final metrics" section
for idx, line in enumerate(lines):
    if line.strip().startswith('Final metrics per run'):
        i = idx + 1
        break

while i < len(lines) and lines[i].strip().startswith('Run'):
    m = re.match(r'\s*Run\s*\d+:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)', lines[i])
    if m:
        metrics.append([float(m.group(j)) for j in range(1,6)])
    i += 1

metrics = np.array(metrics)
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC']
means = metrics.mean(axis=0)
stds  = metrics.std(axis=0)

# Plot bar chart of final metrics with error bars
plt.figure()
x = np.arange(len(metric_names))
plt.bar(x, means)
plt.errorbar(x, means, yerr=stds, fmt='none')
plt.xticks(x, metric_names, rotation=45)
plt.ylabel('Score')
plt.title('Final Metrics (Mean ± Std)')
plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ROC curves
plt.figure()
for t, s in zip(y_true_runs, y_score_runs):
    fpr, tpr, _ = roc_curve(t, s)
    plt.plot(fpr, tpr, alpha=0.3)
# mean ROC
all_fpr = np.unique(np.concatenate([roc_curve(t, s)[0] for t, s in zip(y_true_runs, y_score_runs)]))
mean_tpr = np.zeros_like(all_fpr)
for t, s in zip(y_true_runs, y_score_runs):
    fpr, tpr, _ = roc_curve(t, s)
    mean_tpr += np.interp(all_fpr, fpr, tpr)
mean_tpr /= len(y_true_runs)
plt.plot(all_fpr, mean_tpr, lw=2, label=f"Mean ROC (AUC={auc(all_fpr, mean_tpr):.3f})", color='black')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curves'); plt.legend()
plt.show()

# PR curves
plt.figure()
for t, s in zip(y_true_runs, y_score_runs):
    p, r, _ = precision_recall_curve(t, s)
    plt.plot(r, p, alpha=0.3)
# mean PR
all_rec = np.unique(np.concatenate([precision_recall_curve(t, s)[1] for t, s in zip(y_true_runs, y_score_runs)]))
mean_prec = np.zeros_like(all_rec)
for t, s in zip(y_true_runs, y_score_runs):
    p, r, _ = precision_recall_curve(t, s)
    mean_prec += np.interp(all_rec, r[::-1], p[::-1])
mean_prec /= len(y_true_runs)
ap = average_precision_score(np.concatenate(y_true_runs), np.concatenate(y_score_runs))
plt.plot(all_rec, mean_prec, lw=2, label=f"Mean PR (AP={ap:.3f})", color='black')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision–Recall Curves'); plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Pick one run’s predictions
y_true = y_true_runs[0]
y_pred = (y_score_runs[0] >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Run 1, thresh=0.5)')
plt.show()

from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(
    np.concatenate(y_true_runs),
    np.concatenate(y_score_runs),
    n_bins=10
)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.show()






