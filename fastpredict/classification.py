"""This module includes fastpredict processes
"""
import sklearn.pipeline
import sklearn.base
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.svm
import sklearn.multiclass
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.utils
import sklearn.datasets
import sklearn.metrics
import numpy


class Settings:
    """This is setting class for classification algorithms.
    """

    def __init__(self):
        self.preprocessing = {'CategoricalNB':[('minmaxscaler',sklearn.preprocessing.MinMaxScaler())],
                        'ClassifierChain':[('',sklearn.datasets.make_multilabel_classification)],
                        'ComplementNB':[('minmaxscaler',sklearn.preprocessing.MinMaxScaler())],
                        'MultinomialNB':[('minmaxscaler',sklearn.preprocessing.MinMaxScaler())]}

        self.arguments = {'ClassifierChain': {'base_estimator':sklearn.linear_model.LogisticRegression(solver = 'lbfgs'),},
                    'MultiOutputClassifier': {'estimator':sklearn.linear_model.LogisticRegression(solver = 'lbfgs'),},
                    'OneVsOneClassifier': {'estimator':sklearn.svm.LinearSVC(dual="auto"),},
                    'OneVsRestClassifier': {'estimator':sklearn.svm.SVC(),},
                    'OutputCodeClassifier': {'estimator':sklearn.ensemble.RandomForestClassifier(),},
                    'StackingClassifier': {'estimators':[('rf', sklearn.ensemble.RandomForestClassifier(n_estimators=10)),
                                                        ('svr', sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                                                                               sklearn.svm.LinearSVC(dual="auto")))
                                                    ],},
                    'VotingClassifier': {'estimators':[
                    ('lr', sklearn.linear_model.LogisticRegression(multi_class='multinomial')),
                    ('rf', sklearn.ensemble.RandomForestClassifier(n_estimators=50)),
                    ('gnb',sklearn.naive_bayes.GaussianNB())
                ],'voting':'hard'},
                'AdaBoostClassifier':{'algorithm':'SAMME'}}
    def get_arguments(self):
        """Return used model arguments.

        Returns
        -------
        dict
            All arguments that belongs to used models.
        """
        return self.arguments
    def get_preprocessing(self):
        """Return used preprocessing steps.

        Returns
        -------
        dict
            All preprocessing steps that belongs to used models.
        """
        return self.preprocessing
class EmptyTransform(sklearn.base.TransformerMixin):
    """This class is used to pass data to next step without any transform in sklearn pipeline
    """
    def __init__(self,
                 **kwargs) -> None:
        pass

    def fit(self,
            x: numpy.ndarray,
            y: numpy.ndarray = numpy.array([])) -> sklearn.base.TransformerMixin:
        """Return itself

        Parameters
        ----------
        x : numpy.ndarray
            Train data
        y : numpy.ndarray, optional
            Train data labels, by default numpy.array([])

        Returns
        -------
        sklearn.base.TransformerMixin
            Itself
        """
        return self
    def transform(self,
                  x: numpy.ndarray) -> numpy.ndarray:
        """Return given data.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Input data.
        """
        return x
    def __repr__(self) -> str:
        return 'EmptyTransform()'

class FastPredict:
    """FastPredict class includes all possible classification algorithms to train, predict, and evaluate. 
    """
    def __init__(self,
                 verbose: bool = False,
                 preprocessing: dict = {},
                 arguments: dict = {}) -> None:
        # https://stackoverflow.com/questions/41844311/list-of-all-classification-algorithms
        self.settings = Settings()
        self.pipelines = {name: sklearn.pipeline.Pipeline(
            steps=list(step for step in self.settings.preprocessing.get(name, [('empty', EmptyTransform())]))
            + [('classifier', classifier(**self.settings.arguments.get(name, {})))])
                            for name, classifier in sklearn.utils.all_estimators()
                            if issubclass(classifier, sklearn.base.ClassifierMixin)}
    def fit(self,
            x_train:numpy.ndarray,
            y_train:numpy.ndarray) -> None:
        """Train all classifiers with given data.

        Parameters
        ----------
        x_train : numpy.ndarray
            Train data.
        y_train : numpy.ndarray
            Train data labels.
        """

        for pipeline_name, pipeline in self.pipelines.items():
            pipeline.fit(x_train, y_train)

    def predict(self,
                x_test:numpy.ndarray) -> dict:
        """Predict classes for given test data.

        Parameters
        ----------
        x_test : numpy.ndarray
            Test dataset to predict.

        Returns
        -------
        dict
            Obtained predictions of test dataset. 
        """
        all_predictions = {}
        for pipeline_name, pipeline in self.pipelines.items():
            predictions = pipeline.predict(x_test)
            all_predictions[pipeline_name] = predictions

        return all_predictions
    def evaluate(self,
                 x_test:numpy.ndarray,
                 y_test:numpy.ndarray,
                 metrics: list = []) -> dict:
        """Evaluate test dataset performance with given or default metrics.

        Parameters
        ----------
        x_test : numpy.ndarray
            Test dataset.
        y_test : numpy.ndarray
            Test dataset labels.
        metrics : list, optional
            Performance metrics to evaluate given test dataset, by default []

        Returns
        -------
        dict
            Performance metrics of given test dataset for all trained models. 
        """
        all_predictions = self.predict(x_test)
        return {model_name: {metric.__name__: metric(y_true,y_pred) for metric in metrics}
          for model_name, y_true, y_pred in zip(all_predictions.keys(),
                                            y_test[None,:].repeat(len(all_predictions.keys()),0),
                                            all_predictions.values())}
    def remove_classifier(self,
                          classifier_name: str) -> None:
        """Remove given model from FastPredict class.

        Parameters
        ----------
        classifier_name : str
            Classifier name to remove. 
        """
        self.pipelines.pop(classifier_name)
