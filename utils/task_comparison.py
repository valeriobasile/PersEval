from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd 


def prepare_df_vanilla (test_set, label, datasets_dict, dataset, model): 
    list_traits = []
    for k,v in datasets_dict.items(): 
        if k == dataset: 
            list_traits = v

        user_ids, text_ids, labels = [], [], []
        for annotation in test_set.annotation:
            user_ids.append(annotation[0])
            text_ids.append(annotation[1])
            labels.append(test_set.annotation[annotation[0], annotation[1]][label])
        gold_annotations = pd.DataFrame({"user_id":user_ids, 
                                              "text_id": text_ids, 
                                              "label":labels})
        

    pred_zero = pd.read_csv(f"./predictions_{model}/predictions_{dataset}_False_train_False_zero.csv")
    pred_zero = pred_zero[["user_id", "text_id", "label"]]

    # Assert predictions do not contain duplicates
    assert len(pred_zero) == len(pred_zero[["user_id", "text_id"]].drop_duplicates()), "The prediction file contains duplicates"
    # Assert the predictions has the same ids as the test set
    assert set(pred_zero[["user_id", "text_id"]]) == set(gold_annotations[["user_id", "text_id"]]), "The prediction file does not contain the same instances as in the test set"
    
    baseline = pd.merge(gold_annotations, pred_zero,  how='left', left_on=["user_id", "text_id"], right_on=["user_id", "text_id"])
    baseline.columns = ["user_id", "text_id", "gold", "baseline"]

    df = baseline.copy()
    for trait in list_traits:
        df_ = pd.read_csv(f"./predictions_{model}/predictions_{dataset}_True_train_False_{trait}.csv")
        new_col_name = f"model_{trait}"
        df_ = df_.rename(columns={"label": new_col_name})
        df_ = df_[['user_id', 'text_id', new_col_name]]
        df = df.merge(df_[['user_id', 'text_id', new_col_name]], on=['user_id', 'text_id'])

    return df 



def prepare_df_lamp (datasets_dict, dataset, model):
    list_traits = []
    for k,v in datasets_dict.items(): 
        if k == dataset: 
            list_traits = v

    baseline = pd.read_csv(f"./predictions_{model}/edited_{dataset}_False.csv")
    baseline = baseline.rename(columns={"predictions": "baseline"})

    df = baseline.copy()
    for trait in list_traits:
        df_ = pd.read_csv(f"./predictions_{model}/edited_{dataset}_{trait}_True.csv")
        new_col_name = f"model_{trait}"
        df_ = df_.rename(columns={"predictions": new_col_name})
        df_ = df_[['user_id', 'text_id', new_col_name]]
        df = df.merge(df_[['user_id', 'text_id', new_col_name]], on=['user_id', 'text_id'])

    return df 


def prepare_list_tasks (dataset, datasets_dict): 
    list_tasks = ["baseline"]
    for k,v in datasets_dict.items(): 
        if k == dataset: 
            for trait in v: 
                task = "model_"+trait
                list_tasks.append(task)
    return list_tasks



def CM (df, list_tasks):
  for task in list_tasks:
    cm = confusion_matrix(df['gold'], df[task])
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(task)


def task_disagreements(df, list_tasks):

    #count the number of unique predictions. 
    # If they all agree the count will be 1, if there is a disagreement will be >1
    df["n_unique_predictions"] = df[list_tasks].nunique(axis=1) 

    print("Result explanation:")
    print("# 1 = All tasks agree")
    print("> 2 = At least one task disagree", "\n")
    
    print(df['n_unique_predictions'].value_counts(normalize=True), "\n")


    disagreements = df[df['n_unique_predictions'] > 1]

    # For each task, calculate accuracy on disagreement rows only
    for task in list_tasks:
        acc = (disagreements[task] == disagreements['gold']).mean()
        print(f"{task} accuracy (on disagreements): {acc:.2%}")




def task_agreements(df, list_tasks):
    # Check if all tasks agree on the label
    df["All_agree"] = df[list_tasks].eq(df["baseline"], axis=0).all(axis=1)

    print("Result explanation (All_agree):")
    print("# True  - All tasks gave the same label")
    print("# False - At least one task gave a different label\n")
    print(df["All_agree"].value_counts())

    print("\n\n")

    df_agreed = df.copy()
    df_agreed = df_agreed[df_agreed["All_agree"] == True]

    # Check if all agreed tasks are wrong (i.e., all disagree with the gold label)
    df_agreed["All_wrong"] = df_agreed[list_tasks].ne(df_agreed["gold"], axis=0).all(axis=1)

    print("Result explanation (All_wrong):")
    print("# True  - All agreed labels are wrong (different from gold)")
    print("# False - Agreed label matches the gold label\n")
    print(df_agreed["All_wrong"].value_counts(), "\n")

    print("Number of times tasks were wrong and right on each gold label")
    print(df_agreed[["All_wrong","gold"]].value_counts())




def comparison_task0(df, list_tasks):
    task_cols = [task for task in list_tasks if task != "baseline"]
    print(task_cols)
    total_rows = len(df)
    comparison_percentages = {}

    for task in task_cols:
        matches = (df[task] == df["baseline"]).sum()
        comparison_percentages[task] = (matches / total_rows) * 100

    print("How many times each task give the same label as baseline: ")
    for k,v in comparison_percentages.items():
      print(k, ": ", v)