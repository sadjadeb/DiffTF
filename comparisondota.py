import os
import csv
import numpy as np
import eval.ranking as rk
from dal import graph_util_dota
from dal.graph_util_dota import *
import ml_metrics as metrics
import eval.evaluator_updated as dblp_eval

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# user_HIndex = dblp.get_preprocessed_user_HIndex()
user_skill_dict = dblp.get_user_skill_dict(
    dblp.load_preprocessed_dataset(file_path='dataset/dota2/dota2_dataset.pkl'))
foldIDsampleID_strata_dict = dblp.get_foldIDsampleID_stata_dict(
    data=dblp.load_preprocessed_dataset(file_path='dataset/dota2/dota2_dataset.pkl'),
    train_test_indices=dblp.load_train_test_indices(file_path='dataset/dota2/dota2_train_test_indices.pkl'),
    kfold=10)
preprocessed_authorNameId_dict = dblp.get_preprocessed_dotaAuthorNameID_dict(file_path='authordota.txt')

graph_handler = DBLPGraph()
graph_handler.load_files(path_edgeHash=None)  # Skip edgeHash loading

# === START: BUILD GRAPH FROM TEAM DATA ===
def build_graph_from_team_data(handler, team_data):
    handler.g = nx.Graph()
    for team in team_data:
        for i in range(len(team)):
            for j in range(i + 1, len(team)):
                src, trg = team[i], team[j]
                if handler.g.has_edge(src, trg):
                    handler.g[src][trg]['weight'] += 1
                else:
                    handler.g.add_edge(src, trg, weight=1)

# Combine predicted and true teams
graph_handler.nameID = preprocessed_authorNameId_dict
# Graph will be overwritten below after loading prediction files
# === END: GRAPH BUILDER ===

file_names = ['ablation/predictions/DOTA2_stacked_output.csv']

for file_name in file_names:
    method_name, pred_indices, true_indices, calc_user_time, calc_skill_time, k_fold, k_max = \
        dblp_eval.load_output_file(file_name, foldIDsampleID_strata_dict)
    k_max = 100
    all_teams = []
    for teams in pred_indices.values():
        all_teams.extend(teams)
    for teams in true_indices.values():
        all_teams.extend(teams)
    evaluation_k_set = np.arange(1, k_max + 1, 1)
    fold_set = np.arange(1, k_fold + 1, 1)

    Coverage = dblp_eval.init_eval_holder(evaluation_k_set)
    Sensitivity = dblp_eval.init_eval_holder(evaluation_k_set)
    nDCG = dblp_eval.init_eval_holder(evaluation_k_set)
    MAP = dblp_eval.init_eval_holder(evaluation_k_set)
    MRR = dblp_eval.init_eval_holder(evaluation_k_set)
    Quality = dblp_eval.init_eval_holder(evaluation_k_set)
    team_communication = dblp_eval.init_eval_holder(evaluation_k_set)
    team_skill_cover = dblp_eval.init_eval_holder(evaluation_k_set)
    f1 = dblp_eval.init_eval_holder(evaluation_k_set)

    # === BUILD GRAPH FROM PREDICTED AND TRUE TEAMS ===
    all_teams = []
    for teams in pred_indices.values():
        all_teams.extend(teams)
    for teams in true_indices.values():
        all_teams.extend(teams)
    build_graph_from_team_data(graph_handler, all_teams)

    result_output_name = f"output/eval_results/dota2_{method_name}.csv"
    with open(result_output_name, 'w') as file:
        writer = csv.writer(file)

        writer.writerow(['User Quantity Strata Computation Time'])
        for strata in sorted(calc_user_time.keys()):
            writer.writerow(['Strata:', strata, 'Average:',
                             np.mean(calc_user_time[strata]), 'STDev:', np.std(calc_user_time[strata])])

        writer.writerow(['Skill Quantity Strata Computation Time'])
        for strata in sorted(calc_skill_time.keys()):
            writer.writerow(['Strata:', strata, 'Average:',
                             np.mean(calc_skill_time[strata]), 'STDev:', np.std(calc_skill_time[strata])])

        writer.writerow(['@K',
                         'Coverage Mean', 'Coverage STDev',
                         'nDCG Mean', 'nDCG STDev',
                         'MAP Mean', 'MAP STDev',
                         'MRR Mean', 'MRR STDev',
                         'Quality Mean', 'Quality STDev',
                         'Team Communication Mean', 'Team Communication STDev',
                         'Team Skill Cover Mean', 'Team Skill Cover STDev',
                         'F1 Mean', 'F1 STDev'])

        for i in fold_set:
            truth = true_indices[i]
            pred = pred_indices[i]
            for j in evaluation_k_set:
                print('{}, fold {}, @ {}'.format(method_name, i, j))
                coverage_overall, _ = dblp_eval.r_at_k(pred, truth, k=j)
                Coverage[j].append(coverage_overall)
                nDCG[j].append(rk.ndcg_at(pred, truth, k=j))
                MAP[j].append(metrics.mapk(truth, pred, k=j))
                MRR[j].append(dblp_eval.mean_reciprocal_rank(dblp_eval.cal_relevance_score(pred, truth, k=j)))
                Quality[j].append(dblp_eval.team_formation_feasibility(pred, truth, user_skill_dict, k=j))
                team_communication[j].append(dblp_eval.team_formation_feasibility(pred, truth,
                                                                                   user_x_dict=preprocessed_authorNameId_dict,
                                                                                   mode='communication',
                                                                                   graph=graph_handler, k=j))
                team_skill_cover[j].append(dblp_eval.team_formation_feasibility(pred, truth,
                                                                                 user_x_dict=user_skill_dict,
                                                                                 mode='skill_cover', k=j))
                f1_overall, _ = dblp_eval.f1_at_k(pred, truth, k=j)
                f1[j].append(f1_overall)

        for j in evaluation_k_set:
            writer.writerow([j,
                             np.mean(Coverage[j]), np.std(Coverage[j]),
                             np.mean(nDCG[j]), np.std(nDCG[j]),
                             np.mean(MAP[j]), np.std(MAP[j]),
                             np.mean(MRR[j]), np.std(MRR[j]),
                             np.mean(Quality[j]), np.std(Quality[j]),
                             np.mean(team_communication[j]), np.std(team_communication[j]),
                             np.mean(team_skill_cover[j]), np.std(team_skill_cover[j]),
                             np.mean(f1[j]), np.std(f1[j])])

        file.close()
