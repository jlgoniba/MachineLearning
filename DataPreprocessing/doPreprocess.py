from preprocess import Preprocess
import pandas

preprocess = Preprocess()

def quick_preprocess(source, independent_index, dependent_index):
    print("doing quick pre-process of data")
    preprocess.create_dataset(source, independent_index, dependent_index)
    preprocess.create_train_test_matrix()
    print("quick pre-process finished")

def complete_preprocess(source, independent_index, dependent_index):
    print("doing quick pre-process of data")
    preprocess.create_dataset(source, independent_index, dependent_index)
    preprocess.fill_missing_data()
    preprocess.encode_data()
    preprocess.create_train_test_matrix()
    preprocess.scale_features()
    print("doing quick pre-process of data")


quick_preprocess('Data.csv', -1, 3)

dataset = preprocess.dataset
dependant_matrix = pandas.DataFrame(preprocess.dependent_matrix)
dependant_train_matrix = pandas.DataFrame(preprocess.dependent_train_matrix)
dependant_test_matrix = pandas.DataFrame(preprocess.dependent_test_matrix)
independant_matrix = pandas.DataFrame(preprocess.independent_matrix)
independant_train_matrix = pandas.DataFrame(preprocess.independent_train_matrix)
independant_test_matrix = pandas.DataFrame(preprocess.independent_test_matrix)
