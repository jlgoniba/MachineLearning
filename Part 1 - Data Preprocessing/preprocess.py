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
        self.independent_matrix = self.dataset.iloc[:, :independent_index]
        self.dependent_matrix = self.dataset.iloc[:, :dependant_index]

    def fill_missing_data(self):
        from sklearn.preprocessing import Imputer
        imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imputer = imputer.fit(self.independent_matrix[:, :])
        self.independent_matrix[:, :] = imputer.transform(self.independent_matrix[:, :])

    def encode_data(self, column, categorical_features=0):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        independent_label_encoder = LabelEncoder()
        self.independent_matrix[:, column] = independent_label_encoder.fit_transform(self.independent_matrix[:, column])
        independet_onehotencoder = OneHotEncoder(categorical_features=[categorical_features])
        self.independent_matrix = independet_onehotencoder.fit_transform(self.independent_matrix).toarray()

        dependent_label_encoder = LabelEncoder()
        self.dependent_matrix = dependent_label_encoder.fit_transform(self.dependent_matrix)

    def create_train_test_matrix(self, test_size = 0.2, random_state = 0):
        from sklearn.cross_validation import train_test_split
        self.independent_train_matrix, self.independent_test_matrix= train_test_split(self.independent_matrix, test_size=test_size, random_state=random_state)
        self.dependent_train_matrix, self.dependent_test_matrix= train_test_split(self.dependent_matrix, test_size=test_size, random_state=random_state)

    def scale_features(self):
        from sklearn.preprocessing import StandardScaler
        independent_scaler = StandardScaler()
        self.independent_train_matrix = independent_scaler.fit_transform(self.independent_train_matrix)
        self.independent_test_matrix = independent_scaler.transform(self.independent_test_matrix)
        dependent_scaler = StandardScaler()
        self.dependent_train_matrix = dependent_scaler.fit_transform(self.dependent_train_matrix)