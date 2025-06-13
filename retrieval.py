# Retrieval.py
import argparse
import csv
import time
import torch
import numpy as np
import os
import sys
import random

# Local imports
import dal.load_dblp_data as dblp
import eval.evaluator_updated as evaluator
import ml_metrics as metrics
import eval.ranking as rk
from DiffTF.eval.utils import *
from loss import collaboration_score
import wandb
from team2vec import Team2Vec

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
t2v_model = Team2Vec()
t2v_model = load_T2V_model(t2v_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['dblp', 'dota2'], required=True)
    parser.add_argument('--method_name', type=str, default="Method_128")
    parser.add_argument('--k_fold', type=int, default=10)
    parser.add_argument('--k_max', type=int, default=100)
    args = parser.parse_args()

    # Paths
    if args.dataset == 'dblp':
        ae_path = 'Diffusion_Outputs/DBLP_128_45k.pkl'
        indices_path = 'dataset/dblp/Train_Test_indices_V2.3.pkl'
        preproc_path = 'dataset/dblp/dblp_preprocessed_dataset_V2.3.pkl'
    else:  # dota2
        ae_path = 'Diffusion_Outputs/Dota2_128_45k.pkl'
        indices_path = 'dataset/dota2/dota2_train_test_indices.pkl'
        preproc_path = 'dataset/dota2/dota2_dataset.pkl'

    result_output_name = f"ablation/predictions/{args.method_name}_output.csv"
    os.makedirs(os.path.dirname(result_output_name), exist_ok=True)

    with open(result_output_name, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Method Name', '# Total Folds', '# Fold Number',
                         '# Predictions', '# Truth', 'Computation Time (ms)',
                         'Prediction Indices', 'True Indices'])

    for fold_counter in range(1, args.k_fold + 1):
        print(f"Evaluating on test data fold #{fold_counter}")
        test_dataset = dblp.load_ae_dataset(file_path=ae_path)
        train_test_indices = dblp.load_train_test_indices(file_path=indices_path)
        _, _, _, x_test_user = dblp.get_fold_data(fold_counter, test_dataset, train_test_indices)

        preprocessed_dataset = dblp.load_preprocessed_dataset(file_path=preproc_path)
        test_index = train_test_indices[fold_counter]['Test']
        y_test = [sample[2].todense() for sample in preprocessed_dataset if sample[0] in test_index]
        y_test = np.asarray(y_test).reshape(len(y_test), -1)

        true_indices = []
        pred_indices = []

        with open(result_output_name, 'a+') as file:
            writer = csv.writer(file)
            for sample_x, sample_y in zip(x_test_user, y_test):
                start_time = time.time()
                sample_prediction = [[int(candidate[0]) for candidate in t2v_model.get_member_most_similar_by_vector(sample_x, args.k_max)]]
                elapsed_time = (time.time() - start_time) * 1000

                pred_index, true_index = evaluator.find_indices([sample_x], [sample_y])
                true_indices.append(true_index[0])
                pred_indices.append(sample_prediction[0])

                writer.writerow([args.method_name, args.k_fold, fold_counter,
                                 len(sample_prediction[0][:args.k_max]), len(true_index[0]),
                                 elapsed_time,
                                 *sample_prediction[0][:args.k_max],
                                 *true_index[0]])

    print("Evaluation completed. Now running aggregator...")

    method_name, pred_indices_dict, true_indices_dict, _, _, k_fold, _ = evaluator.load_output_file(result_output_name, None)
    evaluation_k_set = np.arange(1, args.k_max + 1)
    fold_set = np.arange(1, k_fold + 1)

    Coverage = evaluator.init_eval_holder(evaluation_k_set)
    nDCG = evaluator.init_eval_holder(evaluation_k_set)
    MAP = evaluator.init_eval_holder(evaluation_k_set)
    MRR = evaluator.init_eval_holder(evaluation_k_set)
    F1 = evaluator.init_eval_holder(evaluation_k_set)

    eval_output_name = f"Metrics/eval_results/{method_name}.csv"
    os.makedirs(os.path.dirname(eval_output_name), exist_ok=True)

    with open(eval_output_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['@K', 'Coverage Mean', 'Coverage STDev',
                         'nDCG Mean', 'nDCG STDev',
                         'MAP Mean', 'MAP STDev',
                         'MRR Mean', 'MRR STDev',
                         'F1 Mean', 'F1 Std'])

        for fold_id in fold_set:
            truth = true_indices_dict[fold_id]
            pred = pred_indices_dict[fold_id]

            for k_val in evaluation_k_set:
                coverage, _ = evaluator.r_at_k(pred, truth, k=k_val)
                Coverage[k_val].append(coverage)
                nDCG[k_val].append(rk.ndcg_at(pred, truth, k=k_val))
                MAP[k_val].append(metrics.mapk(truth, pred, k=k_val))
                MRR[k_val].append(evaluator.mean_reciprocal_rank(
                    evaluator.cal_relevance_score(pred, truth, k=k_val)))
                f1_score, _ = evaluator.f1_at_k(pred, truth, k=k_val)
                F1[k_val].append(f1_score)

        for k_val in evaluation_k_set:
            writer.writerow([
                k_val,
                np.mean(Coverage[k_val]), np.std(Coverage[k_val]),
                np.mean(nDCG[k_val]), np.std(nDCG[k_val]),
                np.mean(MAP[k_val]), np.std(MAP[k_val]),
                np.mean(MRR[k_val]), np.std(MRR[k_val]),
                np.mean(F1[k_val]), np.std(F1[k_val])
            ])

    print("Aggregator finished. Results saved.")
