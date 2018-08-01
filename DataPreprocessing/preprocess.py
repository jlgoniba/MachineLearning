import numpy
import matplotlib.pyplot as pyplot
import pandas

class Preprocess:

    def __init__(self):
        self.dataset = None
        self.independent_matrix = None
        self.independent_train_matrix = None
        self.independent_test_matrix = None
        self.dependent_matrix = None
        self.dependent_train_matrix = None
        self.dependent_test_matrix = None

    def create_dataset(self, source_file,  independent_index, dependant_index):
        self.dataset = pandas.read_csv(source_file)
        self.independent_matrix = self.dataset.iloc[:, :independent_index].values
        self.dependent_matrix = self.dataset.iloc[:, dependant_index].values

    def fill_missing_data(self):
        from sklearn.preprocessing import Imputer
        imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        index = 0
        for column in self.independent_matrix[0]:
            if type(column) is int or type(column) is float:
                imputer = imputer.fit(self.independent_matrix[:, index:])
                self.independent_matrix[:, index:] = imputer.transform(self.independent_matrix[:, index:])
            index = index + 1

    def encode_data(self, categorical_features=0):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        independent_label_encoder = LabelEncoder()
        index = 0
        for column in self.independent_matrix[0]:
            if type(column) is str:
                self.independent_matrix[:, index] = independent_label_encoder.fit_transform(
                    self.independent_matrix[:, index])
            index = index + 1
        independent_onehotencoder = OneHotEncoder(categorical_features=[categorical_features])
        self.independent_matrix = independent_onehotencoder.fit_transform(self.independent_matrix).toarray()

        dependent_label_encoder = LabelEncoder()
        self.dependent_matrix = dependent_label_encoder.fit_transform(self.dependent_matrix)

    def create_train_test_matrix(self, test_size = 0.2, random_state = 0):
        from sklearn.model_selection import train_test_split
        self.independent_train_matrix, self.independent_test_matrix= train_test_split(self.independent_matrix,
                                                                                      test_size=test_size,
                                                                                      random_state=random_state)
        self.dependent_train_matrix, self.dependent_test_matrix= train_test_split(self.dependent_matrix,
                                                                                  test_size=test_size,
                                                                                  random_state=random_state)

    def scale_features(self):
        from sklearn.preprocessing import StandardScaler
        independent_scaler = StandardScaler()
        self.independent_train_matrix = independent_scaler.fit_transform(self.independent_train_matrix)
        self.independent_test_matrix = independent_scaler.transform(self.independent_test_matrix)
        dependent_scaler = StandardScaler()
        self.dependent_train_matrix = dependent_scaler.fit_transform(self.dependent_train_matrix)

    def quick_preprocess(self, source, independent_index, dependent_index):
        print("doing quick pre-process of data")
        self.create_dataset(source, independent_index, dependent_index)
        self.create_train_test_matrix()
        print("quick pre-process finished")

    def complete_preprocess(self, source, independent_index, dependent_index):
        print("doing quick pre-process of data")
        self.create_dataset(source, independent_index, dependent_index)
        self.fill_missing_data()
        self.encode_data()
        self.create_train_test_matrix()
        self.scale_features()
        print("doing quick pre-process of data")

    def doPreprocess(self, source, independent_index, dependent_index, complete=False):
        if complete:
            self.complete_preprocess(source, independent_index, dependent_index)
        else:
            self.quick_preprocess(source, independent_index, dependent_index)
