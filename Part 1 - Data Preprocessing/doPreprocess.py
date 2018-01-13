from preprocess import Preprocess

preprocess = Preprocess()

def quick_preprocess(source, independent_index, dependent_index):
    preprocess.create_dataset(source, independent_index, dependent_index)
    preprocess.create_train_test_matrix()

def complete_preprocess(source, independent_index, dependent_index):
    preprocess.create_dataset(source, independent_index, dependent_index)
    preprocess.fill_missing_data()
    preprocess.encode_data()
    preprocess.create_train_test_matrix()
    preprocess.scale_features()

quick_preprocess('Data.csv', -1, 3)