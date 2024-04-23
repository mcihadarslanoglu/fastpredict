import sklearn.metrics
import sklearn.preprocessing
import fastpredict.classification
import sklearn.datasets


# You should use this if block on windows operation systems because of multiprocessing library.
if __name__ == '__main__':
    x, y = sklearn.datasets.load_iris(return_X_y=True)
    fastpredict = fastpredict.classification.FastPredict(n_core=2, # how many core will be used
                                                         warning_level= 'ignore')
    available_models = fastpredict.list_models()
    # See which models ar available ot fit.
    print(available_models)

    #-----------------------------------------------------------------------------------------------------#

    # Remove some classifiers if you desire.
    fastpredict.remove_classifier('VotingClassifier')
    fastpredict.remove_classifier('LogisticRegression')
    fastpredict.remove_classifier('LogisticRegressionCV')

    #-----------------------------------------------------------------------------------------------------#

    # you can specify model_name as 'all' to add given preprocessing step to all models.
    # Do not forget to build pipeline after do something like
    # removing classifier, adding argument or adding preprocessing to a pipeline
    # It is possible to get a specific model with its preprocessing steps,  i.e pipeline.
    fastpredict.add_preprocessing(model_name='AdaBoostClassifier',
                                  preprocessing_name='min_max_scalar',
                                  preprocessing= sklearn.preprocessing.MinMaxScaler())
    fastpredict.build_pipelines()
    a_model = fastpredict.get_model('AdaBoostClassifier')
    print(fastpredict.get_model('AdaBoostClassifier'))
    #-----------------------------------------------------------------------------------------------------#

    # You can add specific parameters to use when the model is initialized.
    fastpredict.add_arguments('LinearSVC', {'dual':True})
    a_model = fastpredict.get_model('LinearSVC')
    fastpredict.build_pipelines()
    print(fastpredict.get_model('LinearSVC'))

    #-----------------------------------------------------------------------------------------------------#

    # Fit the models
    fastpredict.fit(x,y)
    #-----------------------------------------------------------------------------------------------------#

    # Evaluation and predicting steps
    # Predefined evaulation metrics are f1-score, accuracy, precision and recall.
    # However, you can override them giving metrics as input to fastpredict.evaluate function
    # Do not forget to give metric_parameters even they are empty.
    evaluation = fastpredict.evaluate(x,y, metrics = [sklearn.metrics.accuracy_score, sklearn.metrics.f1_score],
                                      metric_parameters=[{}, {'average':'macro'}])

    print(evaluation)
    preds = fastpredict.predict(x)
    print(preds)

    #-----------------------------------------------------------------------------------------------------#
