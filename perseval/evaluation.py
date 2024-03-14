import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

class Predictions():
    def __init__(self):
        self.predictions = {
            "instance_id":[],
            "user":[],
            "label":[],
        }

    def add(self, instance_id, user, label):
        self.predictions["instance_id"].append(instance_id)
        self.predictions["user"].append(user)
        self.predictions["label"].append(label)

    def save(self, filename):
        df_out = pd.DataFrame(self.predictions)
        df_out.to_csv(filename, index=False)

    def load(self, filename):
        df = pd.read_csv(filename)
        self.predictions = df.reset_index().to_dict(orient='list')

    def evaluate(self, dataset):
        predicted_classes = self.predictions["label"]
        gold_standard = []
        for i, instance_id in enumerate(self.predictions["instance_id"]):
            gold_standard.append(
                dataset.annotation[self.predictions["user"][i], instance_id])

        print(classification_report(gold_standard, predicted_classes))    

def load_predictions(filename):
    predictions = Predictions()
    predictions.load(filename)
    return predictions
