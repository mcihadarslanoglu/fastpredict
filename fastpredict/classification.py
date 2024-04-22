"""This module includes fastpredict processes
"""
import multiprocessing
import itertools
import math
import collections.abc
import numpy
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
                    'OneVsOneClassifier': {'estimator':sklearn.svm.LinearSVC(dual='auto'),},
                    'OneVsRestClassifier': {'estimator':sklearn.svm.SVC(),},
                    'OutputCodeClassifier': {'estimator':sklearn.ensemble.RandomForestClassifier(),},
                    'StackingClassifier': {'estimators':[('rf', sklearn.ensemble.RandomForestClassifier(n_estimators=10)),
                                                        ('svr', sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                                                                               sklearn.svm.LinearSVC(dual='auto')))
                                                    ],},
                    'VotingClassifier': {'estimators':[
                    ('lr', sklearn.linear_model.LogisticRegression(multi_class='multinomial')),
                    ('rf', sklearn.ensemble.RandomForestClassifier(n_estimators=50)),
                    ('gnb',sklearn.naive_bayes.GaussianNB())
                ],'voting':'hard'},
                'AdaBoostClassifier':{'algorithm':'SAMME'},
                'LinearSVC':{'dual':'auto'}}
        """
        Some classification algorithms takes more times than others. 
        If we fit these kind of models in same core for multiprocessing,
        all training time is not decreased effectively. 
        We give a complexity order for each model that the less order 
        the more fitting time. We distribute these models into different cores. 
        """
        self.complexity_order = {'GaussianProcessClassifier': 1,
         'GradientBoostingClassifier': 2, 
         'LabelSpreading': 3, 
         'MLPClassifier': 4, 
         'LabelPropagation': 5, 
         'NuSVC': 6, 
         'RandomForestClassifier': 7, 
         'OutputCodeClassifier': 8, 
         'ExtraTreesClassifier': 9, 
         'StackingClassifier': 10, 
         'HistGradientBoostingClassifier': 11, 
         'VotingClassifier': 12, 
         'LogisticRegressionCV': 13, 
         'OneVsRestClassifier': 14, 
         'SVC': 15, 
         'AdaBoostClassifier': 16, 
         'BaggingClassifier': 17, 
         'CalibratedClassifierCV': 18, 
         'DecisionTreeClassifier': 19, 
         'SGDClassifier': 20, 
         'LogisticRegression': 21, 
         'LinearSVC': 22, 
         'QuadraticDiscriminantAnalysis': 23, 
         'OneVsOneClassifier': 24, 
         'RidgeClassifierCV': 25, 
         'PassiveAggressiveClassifier': 26, 
         'CategoricalNB': 27, 
         'RidgeClassifier': 28, 
         'LinearDiscriminantAnalysis': 29, 
         'Perceptron': 30, 
         'ExtraTreeClassifier': 31, 
         'GaussianNB': 32, 
         'ComplementNB': 33, 
         'BernoulliNB': 34, 
         'MultinomialNB': 35, 
         'NearestCentroid': 36, 
         'RadiusNeighborsClassifier': 37, 
         'KNeighborsClassifier': 38, 
         'DummyClassifier': 39}
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
    def get_complexity_orders(self):
        """Return complexity order of models.

        Returns
        -------
        dict
            Complexity order of models.
        """
        return self.complexity_order
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


def to_batch(pipelines: dict, n_core: int) -> collections.abc.Generator[tuple, None, None]:
    """Split all pipelines to batches. 

    Parameters
    ----------
    pipelines : dict
        Target pipelines to split._
    n_core : int
        Total batch or used core size. 

    Yields
    ------
    collections.abc.Generator[tuple]
        One batch of pipelines
    """
    iter_pipelines = iter(pipelines.items())
    while True:
        p = tuple(itertools.islice(iter_pipelines, n_core))
        if not p:
            break
        yield p

class FastPredict:
    """FastPredict class includes all possible classification algorithms to train, predict, and evaluate. 
    """
    def __init__(self,
                 verbose: bool = False,
                 preprocessing: dict = {},
                 arguments: dict = {},
                 n_core: int = 1) -> None:
        # https://stackoverflow.com/questions/41844311/list-of-all-classification-algorithms
        self.n_core = n_core
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
        processes = []
        for pipelines in to_batch(self.pipelines,
                                  len(self.pipelines)//self.n_core):
            process = multiprocessing.Process(target = self._fit,
                                              args = (pipelines,x_train,y_train))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

    def _fit(self,
            pipelines: dict,
            x_train: numpy.ndarray,
            y_train:numpy.ndarray) -> None:
        """Fit given pipelines with train data.

        Parameters
        ----------
        pipelines : dict
            Desired pipelines to fit with train data._
        x_train : numpy.ndarray
            Train data.
        y_train : numpy.ndarray
            Train data labels.
        """
        for pipeline_name, pipeline in pipelines:
            pipeline.fit(x_train,y_train)

    def order_pipelines(self):
        """Order pipelines for efficient multiprocessing.
        """
        complexity_order = self.settings.complexity_order
        complexity_order = sorted(complexity_order.items(),
                                  key = lambda item: item[1])
        tmp_pipelines = [[complexity_order.pop(0)]
                          for _ in range(math.ceil(len(complexity_order)/self.n_core))]
        complexity_order = to_batch(dict(complexity_order),
                                    math.ceil(len(complexity_order)/self.n_core))
        for index, tmp in enumerate(complexity_order):
            tmp_pipelines[index].extend(tmp)
        tmp_pipelines = [x for xs in tmp_pipelines
                         for x in xs ]
        self.pipelines = {name:self.pipelines[name]
                          for (name, order) in tmp_pipelines}

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

    def get_model(self, model_name: str) -> sklearn.pipeline.Pipeline:
        """Return desired model.

        Parameters
        ----------
        model_name : str
            Desired model name

        Returns
        -------
        sklearn.pipeline.Pipeline
            Sklearn pipeline that belongs to desired model.
        """
        assert model_name in self.pipelines.keys(), 'This model is not existed'
        return self.pipelines.get(model_name)
