import pandas as pd 
import os 

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
        predictions = predictions[["user_id", "text_id", "predictions"]]
        predictions["predictions"] = predictions["predictions"].astype(str).str.extract(r'(-?\d+)').astype(float).astype(int)

        # Assert predictions do not contain duplicates
        assert len(predictions) == len(predictions[["user_id", "text_id"]].drop_duplicates()), "The prediction file contains duplicates"
        # Assert the predictions has the same ids as the test set
        assert set(predictions[["user_id", "text_id"]]) == set(gold_annotations[["user_id", "text_id"]]), "The prediction file does not contain the same instances as in the test set"
        
        df = pd.merge(gold_annotations, predictions,  how='left', left_on=["user_id", "text_id"], right_on=["user_id", "text_id"])
    
    else: 
        df = pd.read_csv(f"./predictions_{model}/edited_{dataset}_{trait}_True.csv")
        df["predictions"] = df["predictions"].astype(str).str.extract(r'(-?\d+)').astype(float).astype(int)

    
    return df 



def prediction_change (dataset, dict_datasets, test_set, label, model, store_traits, lamp=False):
    for ds, list_traits in dict_datasets.items():

        if ds == dataset:
            for trait in list_traits:
                count_pred_change = 0
                text_pred = {}
                df = prepare_df(test_set, label, dataset, model, trait, lamp=lamp)
                print(trait, df.shape)

                user_info_df = df['user_id'].map(store_traits).apply(pd.Series)
                df = pd.concat([df, user_info_df], axis=1)

                len_trait = len(df[trait].unique().tolist())

                values_per_text = df.groupby('text_id')[trait].nunique()
                text_ids_with_multiple_values = values_per_text[values_per_text > 1].index #gives the ids of texts annotated by more than one value_trait
                filtered_df = df[df['text_id'].isin(text_ids_with_multiple_values)]
                print("Filtered", trait, filtered_df.shape)

                for idx,row in filtered_df.iterrows():
                    text_id = row["text_id"]
                    pred = row["predictions"]

                    if text_id not in text_pred:
                        text_pred[text_id] = list()
                    text_pred[text_id].append(pred)

                for id, labels in text_pred.items(): 
                    if len(set(labels)) > 1:
                        # print(labels)
                        # print(id)
                        count_pred_change+=1

                
                n_sample = len(filtered_df)
                c_norm = count_pred_change/n_sample
                
                print("Number of texts where the prediction changed: ",count_pred_change)
                print("Number of possible traits: ", len_trait)
                print("Normalized count of label change: ", c_norm)
                # print("Number of texts where the annotation changed: ",count_gold_change)
                print()