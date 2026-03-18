import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

class ModelTrainer:

    def TrainModel(self, X, y, preprocessor):

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        # pipeline (preprocessing + model)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", GradientBoostingClassifier())
            ]
        )
        pipeline.fit(X_train, y_train)
        # prediction
        y_pred = pipeline.predict(X_test)
        # accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Model Accuracy:", accuracy)
        # create models folder
        os.makedirs("models", exist_ok=True)

        # save model
        joblib.dump(pipeline, "models/model.pkl")

        print("Model saved successfully!")
        return pipeline