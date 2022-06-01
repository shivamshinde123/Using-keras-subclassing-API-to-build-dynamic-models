from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataHandler:

    def __init__(self):
        self.housing = fetch_california_housing()

    def getTrainTest(self):

        """
        Description: This function is used to split the data into training set and testing set

        Parameters: None 
        
        Returns:
            X_train_full: Training dataset (Independent features)
            X_test: Testing dataset (Independent features)
            y_train_full: Training dataset (Target features)
            y_test: Testing dataset (Target features)
        """

        X_train_full, X_test, y_train_full, y_test = train_test_split(self.housing.data, self.housing.target)
        return X_train_full, X_test, y_train_full, y_test

    def splitTrainIntoTrainAndValid(self):

        """
        Description: This function is used to split the full training data into training and validation data

        Parameters: None

        Returns:
            X_train: Training dataset (Independent features)
            X_valid: Validation dataset (Independent features)
            y_train: Training dataset (Target features)
            y_valid: Validation dataset (Target features)
        """

        X_train_full, X_test, y_train_full, y_test = self.getTrainTest()
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
        return X_train, X_valid, y_train, y_valid

    def IndependentFeatureScaler(self):

        """
        Description: This function is used to scale the independent features of the training, validation and testing data

        Parameters: None

        Returns:
            X_train: Training dataset (Independent features)
            X_valid: Validation dataset (Independent features)
            X_test: Testing dataset (Independent features)
            y_train: Training dataset (Target features)
            y_valid: Validation dataset (Target features)
            y_test: Testing dataset (Target features)
        """

        scaler = StandardScaler()
        X_train_full, X_test, y_train_full, y_test = self.getTrainTest()
        X_train, X_valid, y_train, y_valid = self.splitTrainIntoTrainAndValid()

        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

        return X_train, X_valid, X_test, y_train, y_valid, y_test
