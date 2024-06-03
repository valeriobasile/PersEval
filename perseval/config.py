seed = 42
dataset_filename_suffix = "_data.plk"
dataset_specific_splits = {
    "EPIC": {
        "user_based_split_percentage" : 0.2,
        "text_based_split_percentage" : 0.05,
    },

    "BREXIT": {
        "user_based_split_percentage" : 0.2,
        "text_based_split_percentage_train" : 0.7,
        "text_based_split_percentage_dev" : 0.05,
    },

    "MHS" : {
        "user_based_split_percentage" : 0.2,
        "text_based_split_percentage" : 0.05,
    }
}
