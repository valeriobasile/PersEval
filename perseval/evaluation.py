import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

class Evaluator():
    def __init__(self, prediction_path, test_set, label):
        self.predictions = pd.read_csv(prediction_path)
        self.label = label
        
        user_ids, text_ids, labels = [], [], []
        for annotation in test_set.annotation:
            user_ids.append(annotation[0])
            text_ids.append(annotation[1])
            labels.append(test_set.annotation[annotation[0], annotation[1]][label])
        self.gold_annotations = pd.DataFrame({"user_id":user_ids, 
                                              "text_id": text_ids, 
                                              "label":labels})
        
        # Assert predictions do not contain duplicates
        assert len(self.predictions) == len(self.predictions[["user_id", "text_id"]].drop_duplicates()), "The prediction file contains duplicates"
        # Assert the predictions has the same ids as the test set
        assert set(self.predictions[["user_id", "text_id"]]) == set(self.gold_annotations[["user_id", "text_id"]]), "The prediction file does not contain the same instances as in the test set"

        # Join the two datasets. 
        # Predictions do not need to be in the same order as in the test set
        self.ordered_pred = pd.merge(self.gold_annotations, self.predictions,  how='left', left_on=["user_id", "text_id"], right_on=["user_id", "text_id"])
        self.ordered_pred.columns = ["user_id", "text_id", "gold", "predictions"]         


    def global_metrics(self):
        print("----- Global metrics -----")
        self.global_metrics = classification_report(
            self.ordered_pred["gold"], 
            self.ordered_pred["predictions"], 
            zero_division=0.0,
            output_dict=True)
        print(classification_report(
            self.ordered_pred["gold"], 
            self.ordered_pred["predictions"], 
            digits=3, 
            zero_division=0.0).replace("0.", "."))
    

    def annotator_level_metrics(self):
        print("----- Annotator-level metrics -----")
        self.annotator_level_metrics = {}
        all_annotator_level_metrics = {}
        for annotator in list(set(self.ordered_pred["user_id"])):
            df_annotator = self.ordered_pred[self.ordered_pred["user_id"]==annotator]
            self.annotator_level_metrics[annotator] = classification_report(
                                        df_annotator["gold"], 
                                        df_annotator["predictions"], 
                                        zero_division=0.0,
                                        output_dict=True)
            
            for label in self.annotator_level_metrics[annotator]:
                if not isinstance(self.annotator_level_metrics[annotator][label], float):
                    if not label in all_annotator_level_metrics:
                        all_annotator_level_metrics[label] = {}
                    for metric in self.annotator_level_metrics[annotator][label]:
                        if not metric in all_annotator_level_metrics[label]:
                            all_annotator_level_metrics[label][metric] = [self.annotator_level_metrics[annotator][label][metric]]
                        else:
                            all_annotator_level_metrics[label][metric].append(self.annotator_level_metrics[annotator][label][metric])
                else:
                    if not label in all_annotator_level_metrics:
                        all_annotator_level_metrics[label] = [self.annotator_level_metrics[annotator][label]]
                    else:
                        all_annotator_level_metrics[label].append(self.annotator_level_metrics[annotator][label])
        
        print("Annotator-level macro average")
        self.annotator_based_macro_avg = {}        
        for label in all_annotator_level_metrics:
            if not isinstance(all_annotator_level_metrics[label], list):
                if not label in self.annotator_based_macro_avg:
                    self.annotator_based_macro_avg[label] = {}
                for metric in all_annotator_level_metrics[label]:
                    self.annotator_based_macro_avg[label] = np.mean(all_annotator_level_metrics[label][metric])
                    print("%s, %s --- %.3f" % (label, metric, np.mean(all_annotator_level_metrics[label][metric])))
            else:
                self.annotator_based_macro_avg[label] = np.mean(all_annotator_level_metrics[label])
                print("%s --- %.3f" % (label, np.mean(all_annotator_level_metrics[label])))
        

    def text_level_metrics(self):
        print("----- Text-level metrics -----")
        self.text_level_metrics = {}
        all_text_level_metrics = {}
        for text in list(set(self.ordered_pred["text_id"])):
            df_text = self.ordered_pred[self.ordered_pred["text_id"]==text]
            self.text_level_metrics[text] = classification_report(
                                        df_text["gold"], 
                                        df_text["predictions"], 
                                        zero_division=0.0,
                                        output_dict=True)
            
            for label in self.text_level_metrics[text]:
                if not isinstance(self.text_level_metrics[text][label], float):
                    if not label in all_text_level_metrics:
                        all_text_level_metrics[label] = {}
                    for metric in self.text_level_metrics[text][label]:
                        if not metric in all_text_level_metrics[label]:
                            all_text_level_metrics[label][metric] = [self.text_level_metrics[text][label][metric]]
                        else:
                            all_text_level_metrics[label][metric].append(self.text_level_metrics[text][label][metric])
                else:
                    if not label in all_text_level_metrics:
                        all_text_level_metrics[label] = [self.text_level_metrics[text][label]]
                    else:
                        all_text_level_metrics[label].append(self.text_level_metrics[text][label])
        
        print("Text-level macro average")
        self.text_based_macro_avg = {}        
        for label in all_text_level_metrics:
            if not isinstance(all_text_level_metrics[label], list):
                if not label in self.text_based_macro_avg:
                    self.text_based_macro_avg[label] = {}
                for metric in all_text_level_metrics[label]:
                    self.text_based_macro_avg[label] = np.mean(all_text_level_metrics[label][metric])
                    print("%s, %s --- %.3f" % (label, metric, np.mean(all_text_level_metrics[label][metric])))
            else:
                self.text_based_macro_avg[label] = np.mean(all_text_level_metrics[label])
                print("%s --- %.3f" % (label, np.mean(all_text_level_metrics[label])))
        