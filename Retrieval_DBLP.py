import argparse
import csv
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys

# Local imports
import dal.load_dblp_data as dblp
sys.path.append("C:/Users/mhuot/Desktop/Coherent/eval/")
import eval.evaluator_updated as dblp_eval
import ml_metrics as metrics
import eval.ranking as rk
from cmn.utils import *
from loss import collaboration_score
import wandb

import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed(42)
'''
This project has 5 ablation study models. THIS IS MODEL 1

1. SpEp input with Skill conditioning
2. Ep input with Skill conditioning
3. SpEp input without Skill conditioning
4. Sp input with Skill conditioning
5. Sp input without Skill conditioning
'''

from team2vec import Team2Vec

t2v_model = Team2Vec()
t2v_model = load_T2V_model(t2v_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-method_name', type=str, default="DBLP_128")
    parser.add_argument('-k_fold', type=int, default=10)
    parser.add_argument('-k_max', type=int, default=100)
    args = parser.parse_args()

    # Prepare output file
    result_output_name = f"./output/predictions/{args.method_name}_output.csv"
    os.makedirs(os.path.dirname(result_output_name), exist_ok=True)

    with open(result_output_name, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Method Name', '# Total Folds', '# Fold Number',
            '# Predictions', '# Truth', 'Computation Time (ms)',
            'Prediction Indices', 'True Indices'
        ])

    for fold_counter in range(1, args.k_fold + 1):
        print(f"Evaluating on test data fold #{fold_counter}")

        # Load dataset
        test_dataset = dblp.load_ae_dataset(file_path='ablation/Diffusion_Outputs/DBLP_128_45k.pkl')
        train_test_indices = dblp.load_train_test_indices(file_path='dataset/Train_Test_indices_V2.3.pkl')
        _, _, x_test_skill, x_test_user = dblp.get_fold_data(fold_counter, test_dataset, train_test_indices)

        preprocessed_dataset = dblp.load_preprocessed_dataset(file_path='dataset/dblp_preprocessed_dataset_V2.3.pkl')
        #gt_dataset = dblp.load_ae_dataset(file_path='dataset/dblp_preprocessed_dataset_V2.3.pkl')
        test_index = train_test_indices[fold_counter]['Test']
        #_,_,_,y_test = dblp.get_fold_data(fold_counter, gt_dataset, train_test_indices)

        y_test = [sample[2].todense() for sample in preprocessed_dataset if sample[0] in test_index]
        y_test = np.asarray(y_test).reshape(len(y_test), -1)

        # Evaluation
        true_indices = []
        pred_indices = []

        with open(result_output_name, 'a+') as file:
            writer = csv.writer(file)
            for sample_x, sample_y in zip(x_test_user, y_test):
                start_time = time.time()
                #print(sample_x.shape)
                # Directly use expert embeddings for retrieval
                sample_prediction = [[int(candidate[0]) for candidate in t2v_model.get_member_most_similar_by_vector(sample_x, args.k_max)]]
                # Flatten the nested list before applying max() and min()
                flat_predictions = [item for sublist in sample_prediction for item in sublist]

                # Get max and min values
                max_value = max(flat_predictions)
                min_value = min(flat_predictions)

                #print(f"Max value: {max_value}")
                #print(f"Min value: {min_value}")

                #print("Sample pred shape is",sample_prediction)
                
                elapsed_time = (time.time() - start_time) * 1000

                # Get true and predicted indices
                pred_index, true_index = dblp_eval.find_indices([sample_x], [sample_y])
                true_indices.append(true_index[0])
                pred_indices.append(sample_prediction[0])

                writer.writerow([
                    args.method_name,
                    args.k_fold,
                    fold_counter,
                    len(sample_prediction[0][:args.k_max]),
                    len(true_index[0]),
                    elapsed_time,
                    *sample_prediction[0][:args.k_max],
                    *true_index[0]
                ])

    print("Evaluation completed. Now running aggregator...")

    file_names = [result_output_name]
    for file_name in file_names:
        method_name, pred_indices, true_indices, _, calc_skill_time, k_fold, _ = dblp_eval.load_output_file(
            file_name, None
        )
        k_max = args.k_max
        evaluation_k_set = np.arange(1, k_max + 1)
        fold_set = np.arange(1, k_fold + 1)

        Coverage = dblp_eval.init_eval_holder(evaluation_k_set)
        nDCG = dblp_eval.init_eval_holder(evaluation_k_set)
        MAP = dblp_eval.init_eval_holder(evaluation_k_set)
        MRR = dblp_eval.init_eval_holder(evaluation_k_set)
        F1 = dblp_eval.init_eval_holder(evaluation_k_set)

        eval_output_name = f"ablation/Metrics/eval_results/{method_name}.csv"
        os.makedirs(os.path.dirname(eval_output_name), exist_ok=True)

        with open(eval_output_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                '@K', 'Coverage Mean', 'Coverage STDev',
                'nDCG Mean', 'nDCG STDev',
                'MAP Mean', 'MAP STDev',
                'MRR Mean', 'MRR STDev',
                'F1 Mean', 'F1 Std'
            ])

            for fold_id in fold_set:
                truth = true_indices[fold_id]
                pred = pred_indices[fold_id]

                for k_val in evaluation_k_set:
                    coverage_overall, _ = dblp_eval.r_at_k(pred, truth, k=k_val)
                    Coverage[k_val].append(coverage_overall)

                    nDCG_val = rk.ndcg_at(pred, truth, k=k_val)
                    nDCG[k_val].append(nDCG_val)

                    MAP_val = metrics.mapk(truth, pred, k=k_val)
                    MAP[k_val].append(MAP_val)

                    MRR_val = dblp_eval.mean_reciprocal_rank(
                        dblp_eval.cal_relevance_score(pred, truth, k=k_val)
                    )
                    MRR[k_val].append(MRR_val)

                    f1_overall, _ = dblp_eval.f1_at_k(pred, truth, k=k_val)
                    F1[k_val].append(f1_overall)

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