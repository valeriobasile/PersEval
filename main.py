from perseval.data_v2 import *

epic = Epic()
print("user_adaptation=False, strict=False")
epic.get_splits(user_adaptation=False, strict=False)
epic.describe_splits()
print()
print()

epic = Epic()
print("====user_adaptation=False, strict=True====")
epic.get_splits(user_adaptation=False, strict=True)
epic.describe_splits()
print()
print()


epic = Epic()
print("====user_adaptation='train', strict=False====")
epic.get_splits(user_adaptation="train", strict=False)
epic.describe_splits()
print()
print()


epic = Epic()
print("====user_adaptation='train', strict=True====")
epic.get_splits(user_adaptation="train", strict=True)
epic.describe_splits()
print()
print()

epic = Epic()
print("====user_adaptation='test', strict=True====")
epic.get_splits(user_adaptation="test", strict=True)
epic.describe_splits()
print()
print()

epic = Epic()
print("====user_adaptation='test', strict=False====")
epic.get_splits(user_adaptation="test", strict=False)
epic.describe_splits()
print()
print()