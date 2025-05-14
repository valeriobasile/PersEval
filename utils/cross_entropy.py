from collections import Counter
import statistics
import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
from scipy.spatial import distance

def prepare_df (test_set, label, dataset, model, trait, lamp=False):
    if not lamp:
        user_ids, text_ids, labels = [], [], []
        for annotation in test_set.annotation:
            user_ids.append(annotation[0])
            text_ids.append(annotation[1])
            labels.append(test_set.annotation[annotation[0], annotation[1]][label])
        gold_annotations = pd.DataFrame({"user_id":user_ids, 
                                        "text_id": text_ids, 
                                        "gold":labels})
            

        predictions = pd.read_csv(f"./predictions_{model}/predictions_{dataset}_True_train_False_{trait}.csv")
        predictions = predictions[["user_id", "text_id", "label"]]
        predictions = predictions.rename(columns={"label":"pred"})
        predictions["pred"] = predictions["pred"].astype(str).str.extract(r'(-?\d+)').astype(float).astype(int)

        # Assert predictions do not contain duplicates
        assert len(predictions) == len(predictions[["user_id", "text_id"]].drop_duplicates()), "The prediction file contains duplicates"
        # Assert the predictions has the same ids as the test set
        assert set(predictions[["user_id", "text_id"]]) == set(gold_annotations[["user_id", "text_id"]]), "The prediction file does not contain the same instances as in the test set"
        
        df = pd.merge(gold_annotations, predictions,  how='left', left_on=["user_id", "text_id"], right_on=["user_id", "text_id"])
    
    else: 
        predictions = pd.read_csv(f"./predictions_{model}/edited_{dataset}_{trait}_True.csv")
        predictions["predictions"] = predictions["predictions"].astype(str).str.extract(r'(-?\d+)').astype(float).astype(int)

        df = predictions.rename(columns={"predictions": "pred"})
    
    return df 


def compute_CE (dataset, dict_datasets, test_set, label, model, store_traits, lamp=False):
    for ds, list_traits in dict_datasets.items():
        if ds == dataset:
            for trait in list_traits:
   
                df = prepare_df(test_set, label, dataset, model, trait, lamp=lamp)
                print(trait, df.shape)

                user_info_df = df['user_id'].map(store_traits).apply(pd.Series)
                df = pd.concat([df, user_info_df], axis=1)

                values_per_text = df.groupby('text_id')[trait].nunique()
                text_ids_with_multiple_values = values_per_text[values_per_text > 1].index #gives the ids of texts annotated by more than one value_trait
                filtered_df = df[df['text_id'].isin(text_ids_with_multiple_values)]
                print("Filtered", trait, filtered_df.shape)
                all_labels = (sorted(set(df['gold'].unique()).union(set(df['pred'].unique()))))
                print("All labels:", all_labels)


                text_value = {}
                for idx,row in filtered_df.iterrows():
                    text_id = row["text_id"]
                    pred = row["pred"]
                    gold = row["gold"]
                    value = row[trait]

                    key = (text_id, value)

                    if key not in text_value:
                        text_value[key] = {"gold":[], "pred":[]}
                    text_value[key]["gold"].append(gold)
                    text_value[key]["pred"].append(pred)

                result = {}

                for key, labels in text_value.items():
                    gold = labels["gold"]
                    pred = list(set(labels["pred"]))

                    tot_gold = len(gold)
                    # tot_pred = len(pred)

                    gold_count = Counter(gold)
                    # pred_count = Counter(pred)


                    # dist_gold = [gold_count.get(0,0)/tot_gold, gold_count.get(1,0)/tot_gold] #get the first value (label 0 or label 1), if it doesn't exist returns value 0 
                    # dist_pred = [pred_count.get(0,0)/tot_pred, pred_count.get(1,0)/tot_pred]
                    dist_gold = [gold_count.get(label, 0) / tot_gold for label in all_labels]
                    # dist_pred = [pred_count.get(label, 0) / tot_pred for label in all_labels]
                    result[key] = (dist_gold, pred)

                store_score = {}
                for key,value in result.items():
                    value_trait = key[1]
                    dist_gold=value[0]
                    pred = value[1]

                    if value_trait not in store_score:
                        store_score[value_trait] = []
                    score = log_loss([pred],[dist_gold],labels=all_labels, normalize=True)
                    store_score[value_trait].append(score)

                print("====CROSS ENTROPY====")
                for k,v in store_score.items():
                    print(f"{k}: {statistics.mean(v)}")
                print("="*50)
                print()
                print()


def compute_JSD (dataset, dict_datasets, test_set, label, model, store_traits, lamp=False):
    for ds, list_traits in dict_datasets.items():
        if ds == dataset:
            for trait in list_traits:
   
                df = prepare_df(test_set, label, dataset, model, trait, lamp=lamp)
                print(trait, df.shape)

                user_info_df = df['user_id'].map(store_traits).apply(pd.Series)
                df = pd.concat([df, user_info_df], axis=1)

                values_per_text = df.groupby('text_id')[trait].nunique()
                text_ids_with_multiple_values = values_per_text[values_per_text > 1].index #gives the ids of texts annotated by more than one value_trait
                filtered_df = df[df['text_id'].isin(text_ids_with_multiple_values)]
                print("Filtered", trait, filtered_df.shape)
                all_labels = (sorted(set(df['gold'].unique()).union(set(df['pred'].unique()))))
                print("All labels:", all_labels)


                text_value = {}
                for idx,row in filtered_df.iterrows():
                    text_id = row["text_id"]
                    pred = row["pred"]
                    gold = row["gold"]
                    value = row[trait]

                    key = (text_id, value)

                    if key not in text_value:
                        text_value[key] = {"gold":[], "pred":[]}
                    text_value[key]["gold"].append(gold)
                    text_value[key]["pred"].append(pred)

                result = {}

                for key, labels in text_value.items():
                    gold = labels["gold"]
                    pred = labels["pred"]

                    tot_gold = len(gold)
                    tot_pred = len(pred)

                    gold_count = Counter(gold)
                    pred_count = Counter(pred)


                    # dist_gold = [gold_count.get(0,0)/tot_gold, gold_count.get(1,0)/tot_gold] #get the first value (label 0 or label 1), if it doesn't exist returns value 0 
                    # dist_pred = [pred_count.get(0,0)/tot_pred, pred_count.get(1,0)/tot_pred]
                    dist_gold = [gold_count.get(label, 0) / tot_gold for label in all_labels]
                    dist_pred = [pred_count.get(label, 0) / tot_pred for label in all_labels]
                    result[key] = (dist_gold, dist_pred)

                # print(text_value)
                # print("---")
                # print(result)
                store_score = {}
                for key,value in result.items():
                    value_trait = key[1]
                    dist_gold = value[0]
                    dist_pred = value[1]

                    if value_trait not in store_score:
                        store_score[value_trait] = []
                    score = distance.jensenshannon(dist_gold, dist_pred)
                    store_score[value_trait].append(score)
                print("====JSD====")
                for k,v in store_score.items():
                    print(f"{k}: {statistics.mean(v)}")
                print("=" * 50 + "\n")
                print()
                print()