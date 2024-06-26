from perseval.data_v2 import *

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
