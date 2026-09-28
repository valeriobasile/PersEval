import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

class Evaluator():
    def __init__(self, prediction_table, test_set, label):
        self.test_set = test_set
        if isinstance(prediction_table, pd.DataFrame):
            self.predictions = prediction_table
        else: 
            self.predictions = pd.read_csv(prediction_table)
        self.predictions = self.predictions[["user_id", "text_id", "predictions"]]
        self.label = label
        
        user_ids, text_ids, labels = [], [], []
        for annotation in test_set.annotation:
            user_ids.append(annotation['user'])
            text_ids.append(annotation['text'])
            labels.append(annotation['label'][label])
        self.gold_annotations = pd.DataFrame({"user_id":user_ids, 
                                              "text_id": text_ids, 
                                              "gold":labels})
        
        # Assert predictions do not contain duplicates
        assert len(self.predictions) == len(self.predictions[["user_id", "text_id"]].drop_duplicates()), "The prediction file contains duplicates"
        # Assert the predictions has the same ids as the test set
        assert set(self.predictions[["user_id", "text_id"]]) == set(self.gold_annotations[["user_id", "text_id"]]), "The prediction file does not contain the same instances as in the test set"

        # Join the two datasets. 
        # Predictions do not need to be in the same order as in the test set
        self.ordered_pred = pd.merge(self.gold_annotations, self.predictions,  how='left', left_on=["user_id", "text_id"], right_on=["user_id", "text_id"])
        self.ordered_pred.columns = ["user_id", "text_id", "gold", "predictions"]

        if self.ordered_pred["predictions"].isnull().any():
            nan_rows = self.ordered_pred.loc[self.ordered_pred["predictions"].isnull(), ["user_id", "text_id"]]
            nan_users = nan_rows["user_id"].unique()
            
            print(f"User {list(nan_users)} did not provide target demographic information.")
            self.ordered_pred = self.ordered_pred.fillna(-1)


    def print_metrics(self, dict_metrics):

        # Same row order as the sklearn report: classes (sorted), then the summary rows.
        # The dict order cannot be trusted: for the annotator-level average it follows
        # the first annotator, so a class that annotator lacks would end up last.
        summary_rows = [k for k in ('accuracy', 'micro avg', 'macro avg', 'weighted avg') if k in dict_metrics]
        class_rows = [k for k in dict_metrics if k not in summary_rows]
        try:
            class_rows = sorted(class_rows, key=float)
        except ValueError:
            pass # non-numeric labels: keep the order of the report

        dm = {}
        for k in class_rows + summary_rows:
            v = dict_metrics[k]
            if isinstance(v,dict):
                dm[k] = v
            else:
                dm[k] = {'f1-score' : v}

        df = pd.DataFrame(dm).T
        print(df.to_string(
            na_rep='--',
            float_format="{:.3f}".format,
            formatters={"support": lambda v_support: f"{v_support:.1f}"}))
        
            
        
    def global_metrics(self):
        print("\n----- Global metrics -----")
        self.global_metrics_dic = classification_report(
            self.ordered_pred["gold"], 
            self.ordered_pred["predictions"], 
            zero_division=0.0,
            output_dict=True)
        self.print_metrics(self.global_metrics_dic)
        return self.global_metrics_dic


    def annotator_level_metrics(self):
        print("\n----- Annotator-level metrics -----")
        self.annotator_level_metrics_dic = {}
        all_annotator_level_metrics = {}
        for annotator in list(set(self.ordered_pred["user_id"])):
            df_annotator = self.ordered_pred[self.ordered_pred["user_id"]==annotator]
            self.annotator_level_metrics_dic[annotator] = classification_report(
                                        df_annotator["gold"], 
                                        df_annotator["predictions"], 
                                        zero_division=0.0,
                                        output_dict=True)
            
            for label in self.annotator_level_metrics_dic[annotator]:
                if not isinstance(self.annotator_level_metrics_dic[annotator][label], float):
                    if not label in all_annotator_level_metrics:
                        all_annotator_level_metrics[label] = {}
                    for metric in self.annotator_level_metrics_dic[annotator][label]:
                        if not metric in all_annotator_level_metrics[label]:
                            all_annotator_level_metrics[label][metric] = [self.annotator_level_metrics_dic[annotator][label][metric]]
                        else:
                            all_annotator_level_metrics[label][metric].append(self.annotator_level_metrics_dic[annotator][label][metric])
                else:
                    if not label in all_annotator_level_metrics:
                        all_annotator_level_metrics[label] = [self.annotator_level_metrics_dic[annotator][label]]
                    else:
                        all_annotator_level_metrics[label].append(self.annotator_level_metrics_dic[annotator][label])

        print("\nAnnotator-level macro average")
        self.annotator_based_macro_avg = {}        
        for label in all_annotator_level_metrics:
            if not isinstance(all_annotator_level_metrics[label], list):
                if not label in self.annotator_based_macro_avg:
                    self.annotator_based_macro_avg[label] = {}
                for metric in all_annotator_level_metrics[label]:
                    if not metric in self.annotator_based_macro_avg[label]:
                        self.annotator_based_macro_avg[label][metric] = np.mean(all_annotator_level_metrics[label][metric])
                    # print("%s, %s --- %.3f" % (label, metric, self.annotator_based_macro_avg[label][metric]))
            else:    
                self.annotator_based_macro_avg[label] = np.mean(all_annotator_level_metrics[label])
                # print("%s --- %.3f" % (label, self.annotator_based_macro_avg[label]))
        self.print_metrics(self.annotator_based_macro_avg)
        return self.annotator_based_macro_avg
        

    def text_level_metrics(self):
        print("\n----- Text-level metrics -----")
        self.text_level_metrics_dic = {}
        all_text_level_metrics = {}
        for text in list(set(self.ordered_pred["text_id"])):
            df_text = self.ordered_pred[self.ordered_pred["text_id"]==text]
            self.text_level_metrics_dic[text] = classification_report(
                                        df_text["gold"], 
                                        df_text["predictions"], 
                                        zero_division=0.0,
                                        output_dict=True)
            
            for label in self.text_level_metrics_dic[text]:
                if not isinstance(self.text_level_metrics_dic[text][label], float):
                    if not label in all_text_level_metrics:
                        all_text_level_metrics[label] = {}
                    for metric in self.text_level_metrics_dic[text][label]:
                        if not metric in all_text_level_metrics[label]:
                            all_text_level_metrics[label][metric] = [self.text_level_metrics_dic[text][label][metric]]
                        else:
                            all_text_level_metrics[label][metric].append(self.text_level_metrics_dic[text][label][metric])
                else:
                    if not label in all_text_level_metrics:
                        all_text_level_metrics[label] = [self.text_level_metrics_dic[text][label]]
                    else:
                        all_text_level_metrics[label].append(self.text_level_metrics_dic[text][label])
        
        print("\nText-level macro average")
        self.text_based_macro_avg = {}        
        for label in all_text_level_metrics:
            if not isinstance(all_text_level_metrics[label], list):
                if not label in self.text_based_macro_avg:
                    self.text_based_macro_avg[label] = {}
                for metric in all_text_level_metrics[label]:
                    self.text_based_macro_avg[label][metric] = np.mean(all_text_level_metrics[label][metric])
                    # print("%s, %s --- %.3f" % (label, metric, np.mean(all_text_level_metrics[label][metric])))
            else:
                self.text_based_macro_avg[label] = np.mean(all_text_level_metrics[label])
                # print("%s --- %.3f" % (label, np.mean(all_text_level_metrics[label])))
        self.print_metrics(self.text_based_macro_avg)
        return self.text_based_macro_avg
    
    def trait_level_metrics(self):
        print("\n----- Trait-level metrics -----")        
        self.trait_level_metrics_dic = {}
        all_trait_level_metrics = {}

        trait_to_annotator = {}
        for annotator in self.test_set.users:
            a = self.test_set.users[annotator]
            for dim in a.traits:
                if dim not in trait_to_annotator:
                    trait_to_annotator[dim] = {}
                    trait_to_annotator[dim][a.traits[dim][0]] = [a.id]
                else:
                    if not a.traits[dim][0] in trait_to_annotator[dim]:
                        trait_to_annotator[dim][a.traits[dim][0]] = [a.id]
                    else:
                        trait_to_annotator[dim][a.traits[dim][0]].append(a.id)
        
        
        
        for dim in trait_to_annotator:
            if dim not in self.trait_level_metrics_dic:
                self.trait_level_metrics_dic[dim] = {} 
            if dim not in all_trait_level_metrics:
                all_trait_level_metrics[dim] = {}

            for trait in trait_to_annotator[dim]:
                if trait != "UNK":
                    df_trait = self.ordered_pred[self.ordered_pred["user_id"].isin(trait_to_annotator[dim][trait])]
                    self.trait_level_metrics_dic[dim][trait] = classification_report(
                                            df_trait["gold"], 
                                            df_trait["predictions"], 
                                            zero_division=0.0,
                                            output_dict=True)
                    
                    for label in self.trait_level_metrics_dic[dim][trait]:
                        if not isinstance(self.trait_level_metrics_dic[dim][trait][label], float):
                            if not label in all_trait_level_metrics[dim]:
                                all_trait_level_metrics[dim][label] = {}
                            for metric in self.trait_level_metrics_dic[dim][trait][label]:
                                if not metric in all_trait_level_metrics[dim][label]:
                                    all_trait_level_metrics[dim][label][metric] = [self.trait_level_metrics_dic[dim][trait][label][metric]]
                                else:
                                    all_trait_level_metrics[dim][label][metric].append(self.trait_level_metrics_dic[dim][trait][label][metric])
                        else:
                            if not label in all_trait_level_metrics[dim]:
                                all_trait_level_metrics[dim][label] = [self.trait_level_metrics_dic[dim][trait][label]]
                            else:
                                all_trait_level_metrics[dim][label].append(self.trait_level_metrics_dic[dim][trait][label])

        self.trait_based_macro_avg = {}

        if not trait_to_annotator:
            print("\nNo trait-level metrics: the test set has no annotator traits "
                  "(the task was generated with named=False, or the dataset has no annotator metadata).")
        else:
            print("\nTrait-level macro averages")
            for dim in all_trait_level_metrics:
                print("\n--- %s ---" % dim)
                if not dim in self.trait_based_macro_avg:
                    self.trait_based_macro_avg[dim] = {}
                for label in all_trait_level_metrics[dim]:
                    if not isinstance(all_trait_level_metrics[dim][label], list):
                        if not label in self.trait_based_macro_avg[dim]:
                            self.trait_based_macro_avg[dim][label] = {}
                        for metric in all_trait_level_metrics[dim][label]:
                            self.trait_based_macro_avg[dim][label][metric] = np.mean(all_trait_level_metrics[dim][label][metric])
                            # print("%s, %s --- %.3f" % (label, metric, np.mean(all_trait_level_metrics[dim][label][metric])))
                    else:
                        self.trait_based_macro_avg[dim][label] = np.mean(all_trait_level_metrics[dim][label])
                        # print("%s --- %.3f" % (label, np.mean(all_trait_level_metrics[dim][label])))
                self.print_metrics(self.trait_based_macro_avg[dim])

        return self.trait_based_macro_avg

                    
