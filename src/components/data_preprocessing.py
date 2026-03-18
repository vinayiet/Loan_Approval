from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataProcessing:
    def PreProcessData(self, data):
        X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
        y = data['Loan_Status']
        cat_col = X.select_dtypes(include='object').columns
        num_col = X.select_dtypes(exclude='object').columns

        num_pipeline = Pipeline([
            ('missing_value_treatment', SimpleImputer(strategy='median')),
            ('scaling', StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ('missing_value', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer([
            ('num_col', num_pipeline, num_col),
            ('cat_col', cat_pipeline, cat_col)
        ])

        return X, y, preprocessor

