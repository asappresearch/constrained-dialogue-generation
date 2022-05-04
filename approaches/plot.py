import argparse
import os
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

today = date.today()

def convert_to_int(s):
    try:
        return int(s)
    except ValueError:
        return None

def convert_to_float(s):
    try:
        return float(s)
    except ValueError:
        return None

def get_str_list(s):
    if pd.isna(s):
        return []
    return s.split()

def compute_corpus_metrics(df):
    correct, predicted, actual = 0, 0, 0
    for index, row in df.iterrows():
        correct += len(get_str_list(row['true+generated_keywords']))
        predicted += len(get_str_list(row['generated_resp_keywords']))
        actual += len(get_str_list(row['true_resp_keywords']))
    if correct == 0 or predicted == 0:
        return {'precision':0, 'recall':0, 'f1-score':0}
    results = {"precision": correct/predicted, "recall": correct/actual}
    results["f1-score"] = (2*results['precision']*results['recall']) / (results['precision']+results['recall'])
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--eval_type", type=str)
    args = parser.parse_args()

    colors = {'Retrieval': 'C1',
              'Prompting': 'C2',
              'DirectedBeamSearch': 'C3',
              'WindowFuturesOfThePast': 'C4',
              'CGMH': 'C6',
              'PromptingWithTraining': 'C5',
              'AllControlWords':'C7',
              'WFirst':'C8',
              'FinetunedModel':'C9'}

    plot_labels = {"DirectedBeamSearch":"DBS","FuturesOfThePast":"FOP-guided", "WindowFuturesOfThePast": "FOP-guided", "Retrieval":"FOP-retrieval", "Prompting":"Prompt", "CGMH": "CGMH", "PromptingWithTraining": "Prompt+Train", "WFirst":"W-first", "FinetunedModel":"Finetuned"}
    fixed_params = {"lambda": [10, 15, 20, None], "keywords": [9], "generations": [10, None]}
    eval_types = [args.eval_type] #['real','simulated']
    num_df_examples=[]
    for pass_num in [0,1]: # The first pass identifies the minimum number of examples that all approaches have run and the second pass plots the figure for the same number of examples across all approaches
        for eval_type in eval_types:
            if eval_type == "real":
                metrics = ['perplexity', 'BLEU-score', 'BERT-score', 'precision', 'recall', 'f1-score']
                plot_types = ['keywords', 'lambda', 'generations']
            elif eval_type == "simulated":
                metrics = ['long-term-success-rate']
                plot_types = ['keywords']
            elif eval_type == "trainprompt":
                metrics = ['BLEU-score', 'BERT-score', 'precision', 'recall', 'f1-score']
                plot_types = ['percent-of-preprocess_data']
            elif eval_type == "percentdata":
                metrics = ['long-term-success-rate']
                plot_types = ['percent-of-historical-preprocess_data']
            for plot_type in plot_types:
                nrows, ncols = 2, 3
                figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,5))
                for index in range(len(metrics)):
                    metric = metrics[index]
                    results = defaultdict(list)
                    x_range = []
                    conditions = []
                    for f in os.listdir(args.save_dir):
                        if ".csv" not in f or (eval_type in ['real','simulated'] and eval_type.upper() not in f):
                            continue
                        values = f.split("_")
                        if "SIMULATED" in f or "REAL" in f:
                            condition = values[1]
                        else:
                            condition = values[0]
                        #if condition == "PromptingWithTraining" and "REAL" not in f:
                        #    continue
                        if condition not in colors:
                            continue
                        if condition not in conditions:
                            conditions.append(condition)
                        if eval_type == "trainprompt":
                            if condition == "PromptingWithTraining":
                                x_param = convert_to_float(f.split("prompt")[1].split("percent")[0])
                                if x_param is None:
                                    continue
                            else:
                                x_param = 1.0
                        elif plot_type == "percent-of-historical-preprocess_data":
                            x_param = convert_to_float(f.split("percent_datastore")[0].split("_")[-1])
                        else:
                            x_param = [convert_to_int(v.split(plot_type)[0]) for v in values if plot_type in v][0]
                        current_params = {'lambda': convert_to_int(f.split('lambda')[0].split("_")[-1]),
                                          'keywords': convert_to_int(f.split('keywords')[0].split("_")[-1]),
                                          'generations': convert_to_int(f.split('generations')[0].split("_")[-1])}
                        df = pd.read_csv(f"{args.save_dir}/{f}")
                        if metric in ['precision', 'recall', 'f1-score']:
                            results_dict = compute_corpus_metrics(df)
                            metric_value = (results_dict[metric], 0)
                        elif metric == "perplexity":
                            filtered_df = df["perplexity"].dropna()
                            filtered_df = filtered_df[filtered_df != 100000]
                            metric_value = (np.mean(filtered_df), stats.sem(filtered_df))
                        else:
                            filtered_df = df[metric].dropna()
                            filtered_df = filtered_df[filtered_df != 100000]
                            metric_value = (np.mean(filtered_df), stats.sem(filtered_df))
                        if pass_num == 0:
                            num_df_examples.append(len(filtered_df))
                            continue
                        else:
                            min_num = min(num_df_examples)
                            filtered_df = filtered_df[0:min_num]
                        num_examples = convert_to_int(f.split('examples')[0].split("_")[-1])
                        skip = False
                        for param in current_params:
                            if param == plot_type:
                                continue
                            if current_params[param] not in fixed_params[param]: # or num_examples != num_examples_dict[condition]:
                                skip = True
                        if skip:
                            continue
                        results[condition].append((x_param, metric_value))
                        if (isinstance(x_param, int) or isinstance(x_param, float)) and x_param not in x_range:
                            x_range.append(x_param)
                    x_range = sorted(x_range)
                    for c in conditions:
                        results[c] = sorted(results[c], key=lambda x: x[0])
                        x = np.array([k[0] for k in results[c]])
                        y_mean = np.array([k[1][0] for k in results[c]])
                        y_std = np.array([k[1][1] for k in results[c]])
                        if len(x) == 1 and len(y_mean) == 1:
                            x = x_range
                            y_mean = np.repeat(y_mean, len(x))
                            y_std = np.repeat(y_std, len(x))
                        row = int(index / (nrows+1))
                        col = int(index % (nrows+1))
                        axes[row,col].plot(x, y_mean, '-x', label=plot_labels[c], color=colors[c])
                        axes[row,col].fill_between(x, y_mean-y_std, y_mean+y_std, alpha=0.2, edgecolor=colors[c], facecolor=colors[c])
                    axes[row,col].set_xlabel(f"{plot_type}",fontsize=14)
                    if plot_type in ['trainprompt','percentdata']: #['training-checkpoint','percent-of-historical-preprocess_data']:
                        axes[row,col].set_xticks(x_range)
                        axes[row,col].set_yticks([x for x in np.arange(0,1.0,0.2)])
                        plt.setp(axes[row,col].get_xticklabels(), rotation=45, horizontalalignment='center')
                        axes[row,col].set_ylabel(f"{metric}",fontsize=12)
                        axes[row,col].set_xlabel(f"{plot_type}",fontsize=12)
                    else:
                        axes[row,col].set_title(metric, fontsize=14)
                    torch.save(results, f"{args.save_dir}/{plot_type}_{metric}_results_dict.pkl")

                if eval_type in ["simulated", "percentdata"]:
                    if eval_type == "simulated":
                        figure.legend([plot_labels[c] for c in conditions], bbox_to_anchor=(0.37,0.5,0.135,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=1, fontsize=12)
                        for r in range(nrows):
                            for c in range(ncols):
                                if not (r == 0 and c == 0):
                                    axes[r, c].axis('off')
                    elif eval_type in ["trainprompt", "percentdata"]:
                        figure.legend([plot_labels[c] for c in conditions], bbox_to_anchor=(0.37,0.7,0.18,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=1, fontsize=12)
                else:
                    figure.legend([plot_labels[c] for c in conditions], bbox_to_anchor=(0,-0.1,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4, fontsize=12)
                #if eval_type == "trainprompt":
                    #figure.set_xticks(rotation=45)
                    #figure.setp(ax.get_xticklabels(), rotation=70, horizontalalignment='right')
                figure.suptitle("Varying "+plot_type, fontsize=14, weight="bold")
                date = today.strftime("%b-%d-%Y")
                figure.tight_layout()
                figure.savefig(f"{args.save_dir}/final_{eval_type}_{plot_type}.png",facecolor='w',dpi=300, bbox_inches="tight")
