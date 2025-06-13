import csv
import numpy as np
import eval.ranking as rk
import ml_metrics as metrics
import matplotlib.pyplot as plt
import eval.load_dblp_data as dblp
import eval.evaluator as evaluator
import argparse

# === Parse arguments ===
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['dblp', 'dota2'], required=True)
parser.add_argument('--model1', type=int, required=True)
parser.add_argument('--model2', type=int, required=True)
args = parser.parse_args()

# === Configuration ===
filter_zero = True
at_k_mx = 10
at_k_set = range(1, at_k_mx + 1)

# === Load data ===
if args.dataset == 'dblp':
    file_names = [
        'Predictions/Rad_output_DBLP.csv',
        'Predictions/Sapienza_output_DBLP.csv',
        'Predictions/GERF_output_DBLP.csvv',
        'Predictions/DBLP_512_45k.csv'
    ]
    dataset_path = 'dataset/dblp/dblp_preprocessed_dataset_V2.3.pkl'
    index_path = 'dataset/dblp/Train_Test_Indices_V2.3.pkl'
    result_name = "Metrics/HelpHurt_{}_{}_DBLP.csv"
else:
    file_names = [
        'Predictions/DOTA2_rad_output.csv',
        'Predictions/DOTA2_sapienza_output.csv',
        'Predictions/DOTA2_GERF_output.csv',
        'Predictions/Dota2_1024.csv'
    ]
    dataset_path = 'dataset/dblp_preprocessed_dataset_V2.3.pkl'
    index_path = 'dataset/Train_Test_Indices_V2.3.pkl'
    result_name = "Metrics/HelpHurt_{}_{}_Dota2.csv"

model1 = args.model1 - 1
model2 = args.model2 - 1
file_name1 = file_names[model1]
file_name2 = file_names[model2]

foldIDsampleID_strata_dict = dblp.get_foldIDsampleID_stata_dict(
    data=dblp.load_preprocessed_dataset(file_path=dataset_path),
    train_test_indices=dblp.load_train_test_indices(file_path=index_path),
    kfold=10
)

# === Load outputs ===
method_name1, pred_indices1, true_indices1, _, _, k_fold, _ = evaluator.load_output_file(file_name1, foldIDsampleID_strata_dict)
method_name2, pred_indices2, true_indices2, _, _, _, _ = evaluator.load_output_file(file_name2, foldIDsampleID_strata_dict)
fold_set = np.arange(1, k_fold + 1)

# === Metric holders ===
holder_ndcg = evaluator.init_eval_holder(at_k_set)
holder_map = evaluator.init_eval_holder(at_k_set)
holder_mrr = evaluator.init_eval_holder(at_k_set)
if args.dataset == 'dblp':
    holder_recall = evaluator.init_eval_holder(at_k_set)

# === Compute differences ===
for k in at_k_set:
    for fold in fold_set:
        truth1 = true_indices1[fold]
        pred1 = pred_indices1[fold]
        truth2 = true_indices2[fold]
        pred2 = pred_indices2[fold]

        print(f'{method_name1} vs {method_name2} | Fold {fold} @K={k}')

        if args.dataset == 'dblp':
            holder_recall[k].extend([
                evaluator.r_at_k([p1], [t1], k=k) - evaluator.r_at_k([p2], [t2], k=k)
                for p1, t1, p2, t2 in zip(pred1, truth1, pred2, truth2)
            ])

        holder_ndcg[k].extend([
            rk.ndcg_at([p1], [t1], k=k) - rk.ndcg_at([p2], [t2], k=k)
            for p1, t1, p2, t2 in zip(pred1, truth1, pred2, truth2)
        ])
        holder_map[k].extend([
            metrics.mapk([p1], [t1], k=k) - metrics.mapk([p2], [t2], k=k)
            for p1, t1, p2, t2 in zip(pred1, truth1, pred2, truth2)
        ])
        holder_mrr[k].extend([
            evaluator.mean_reciprocal_rank(evaluator.cal_relevance_score([p1], [t1], k=k)) -
            evaluator.mean_reciprocal_rank(evaluator.cal_relevance_score([p2], [t2], k=k))
            for p1, t1, p2, t2 in zip(pred1, truth1, pred2, truth2)
        ])

# === Save metrics ===
output_path = result_name.format(method_name1, method_name2)
with open(output_path, 'w') as f:
    writer = csv.writer(f)
    header = ['@K']
    if args.dataset == 'dblp':
        header.append('Recall')
    header += ['NDCG', 'MAP', 'MRR']
    writer.writerow(header)

    for i, k in enumerate(at_k_set):
        row = [k]
        if args.dataset == 'dblp':
            row.append(np.mean(holder_recall[k]))
        row.append(np.mean(holder_ndcg[k]))
        row.append(np.mean(holder_map[k]))
        row.append(np.mean(holder_mrr[k]))
        writer.writerow(row)

print("File saved:", output_path)

# === Help-Hurt Plot ===
plt.title('Help-Hurt Plot')
plt.xlabel('Sample #')
if args.dataset == 'dblp':
    diff = np.sort(np.mean(list(holder_recall.values()), axis=0))[::-1]
else:
    diff = np.sort(np.mean(list(holder_ndcg.values()), axis=0))[::-1]

if filter_zero:
    diff = diff[diff != 0]

plt.bar(range(len(diff)), diff, color='b', width=1)
plt.plot(range(len(diff)), np.zeros(len(diff)), color='peachpuff')
plt.xlim(0, len(diff))
plt.ylim(-1.3 * max(abs(diff)), 1.3 * max(abs(diff)))
plt.grid()
plt.show()
