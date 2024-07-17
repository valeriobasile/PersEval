from perseval.data import *
from perseval.model import *
from perseval.evaluation import *

'''
brexit_f_f = Epic()
brexit_f_f.get_splits(user_adaptation=False, strict=False)

model = PerspectivistEncoder("roberta-base", 
                             brexit_f_f, 
                            "irony", 
                            named=True)

#trainer = model.train()
#model.predict(trainer) # <-- Predictions are saved in the "predictions" folder, 
                       #     in a predictions.csv file.
                       #     The file must contain three columns:
                       #     "user_id", "text_id", "label"

evaluator = Evaluator(prediction_path="predictions/predictions.csv",
                      test_set=brexit_f_f.test_set,
                      label="irony")
evaluator.global_metrics()
evaluator.annotator_level_metrics()

# You can also access the metrics from
# print(evaluator.global_metrics)
# print(evaluator.annotator_level_metrics)
# print(evaluator.annotator_based_macro_avg)

'''





'''
# Check splits work for all splits

brexit_f_f = Brexit()
print("user_adaptation=False, strict=False")
brexit_f_f.get_splits(user_adaptation=False, strict=False)
brexit_f_f.describe_splits()
print()
print()

brexit_f_t = Brexit()
print("====user_adaptation=False, strict=True====")
brexit_f_t.get_splits(user_adaptation=False, strict=True)
brexit_f_t.describe_splits()
print()
print()


brexit_tr_f = Brexit()
print("====user_adaptation='train', strict=False====")
brexit_tr_f.get_splits(user_adaptation="train", strict=False)
brexit_tr_f.describe_splits()
print()
print()


brexit_tr_t = Brexit()
print("====user_adaptation='train', strict=True====")
brexit_tr_t.get_splits(user_adaptation="train", strict=True)
brexit_tr_t.describe_splits()
print()
print()

brexit_te_t = Brexit()
print("====user_adaptation='test', strict=True====")
brexit_te_t.get_splits(user_adaptation="test", strict=True)
brexit_te_t.describe_splits()
print()
print()

brexit_te_f = Brexit()
print("====user_adaptation='test', strict=False====")
brexit_te_f.get_splits(user_adaptation="test", strict=False)
brexit_te_f.describe_splits()
print()
print()

# All test sets are equal
assert brexit_f_f.test_set.users == brexit_f_t.test_set.users == brexit_tr_t.test_set.users == brexit_tr_f.test_set.users == brexit_te_t.test_set.users == brexit_tr_f.test_set.users
assert brexit_f_f.test_set.texts == brexit_f_t.test_set.texts == brexit_tr_t.test_set.texts == brexit_tr_f.test_set.texts == brexit_te_t.test_set.texts == brexit_tr_f.test_set.texts
assert brexit_f_f.test_set.annotation == brexit_f_t.test_set.annotation == brexit_tr_t.test_set.annotation == brexit_tr_f.test_set.annotation == brexit_te_t.test_set.annotation == brexit_tr_f.test_set.annotation
assert brexit_f_f.test_set.annotation_by_text == brexit_f_t.test_set.annotation_by_text == brexit_tr_t.test_set.annotation_by_text == brexit_tr_f.test_set.annotation_by_text == brexit_te_t.test_set.annotation_by_text == brexit_tr_f.test_set.annotation_by_text
'''

epic_f_f = Epic()
print("user_adaptation=False, strict=False")
epic_f_f.get_splits(user_adaptation=False, strict=False)
epic_f_f.describe_splits()
print()
print()

epic_f_t = Epic()
print("====user_adaptation=False, strict=True====")
epic_f_t.get_splits(user_adaptation=False, strict=True)
epic_f_t.describe_splits()
print()
print()


epic_tr_f = Epic()
print("====user_adaptation='train', strict=False====")
epic_tr_f.get_splits(user_adaptation="train", strict=False)
epic_tr_f.describe_splits()
print()
print()


epic_tr_t = Epic()
print("====user_adaptation='train', strict=True====")
epic_tr_t.get_splits(user_adaptation="train", strict=True)
epic_tr_t.describe_splits()
print()
print()

epic_te_t = Epic()
print("====user_adaptation='test', strict=True====")
epic_te_t.get_splits(user_adaptation="test", strict=True)
epic_te_t.describe_splits()
print()
print()

epic_te_f = Epic()
print("====user_adaptation='test', strict=False====")
epic_te_f.get_splits(user_adaptation="test", strict=False)
epic_te_f.describe_splits()
print()
print()

# All test sets are equal
assert epic_f_f.test_set.users == epic_f_t.test_set.users == epic_tr_t.test_set.users == epic_tr_f.test_set.users == epic_te_t.test_set.users == epic_tr_f.test_set.users
assert epic_f_f.test_set.texts == epic_f_t.test_set.texts == epic_tr_t.test_set.texts == epic_tr_f.test_set.texts == epic_te_t.test_set.texts == epic_tr_f.test_set.texts
assert epic_f_f.test_set.annotation == epic_f_t.test_set.annotation == epic_tr_t.test_set.annotation == epic_tr_f.test_set.annotation == epic_te_t.test_set.annotation == epic_tr_f.test_set.annotation
assert epic_f_f.test_set.annotation_by_text == epic_f_t.test_set.annotation_by_text == epic_tr_t.test_set.annotation_by_text == epic_tr_f.test_set.annotation_by_text == epic_te_t.test_set.annotation_by_text == epic_tr_f.test_set.annotation_by_text