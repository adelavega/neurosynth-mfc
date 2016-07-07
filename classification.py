import numpy as np
from tools import ProgressBar, mask_diagonal
from joblib import Parallel, delayed

# Replace with a more basic sklearn function
from neurosynth.analysis.classify import classify


def classify_parallel(classifier, scoring, region_data, importance_function):
    """ Parallel classification function. Used to classify for each region if study 
    was activated or not (typically based on neurosynth features)
    classifier: sklearn classifier
    scoring: sklearn scoring function
    region_data: contains (X, y) data for a given region
    importance function: function to format importance vector (i.e. what to pull out from fitted classifier)

    returns summary dictionary with score, importance, preditions and importance vectors """

    X, y = region_data

    output = classify(
        X, y, classifier=classifier, cross_val='4-Fold', scoring=scoring)
    output['importance'] = importance_function(output['clf'].clf)
    return output


def log_odds_ratio(clf):
    """ Extracts log odds-ratio from naive bayes classifier """
    return np.log(clf.theta_[1] / clf.theta_[0])


class RegionalClassifier(object):

    """" Object used to classify on a region by region basis (from a cluster solution) 
    if studies activated a region using Neurosynth features (e.g. topics) 
    as classification features """

    def __init__(self, dataset, mask_img, classifier=None, cv='4-Fold',
                 thresh=0.05, thresh_low=0):
        """
        dataset - Neurosynth dataset
        mask_img - Path to Nifti image containing discrete regions coded as levels 
        classifier - sklearn classifier
        cv - cross validation strategy
        thresh - Threshold used to determine if a study is considered to have activated a region
        thresh_low - Threshold used to determine if a study is considered to be inactivate in a region

        """
        self.mask_img = mask_img
        self.classifier = classifier
        self.dataset = dataset
        self.thresh = thresh
        self.thresh_low = thresh_low
        self.cv = cv
        self.data = None

    def load_data(self):
        """ Loads data to set up classificaiton problem. Most importantly self.data is filled in, which consists
        of a Numpy array (length = number of regions) with X and y data for each region """
        from neurosynth.analysis.reduce import average_within_regions

        all_ids = self.dataset.image_table.ids

        high_thresh = average_within_regions(
            self.dataset, self.mask_img, threshold=self.thresh)
        low_thresh = average_within_regions(
            self.dataset, self.mask_img, threshold=self.thresh_low)

        self.data = np.empty(high_thresh.shape[0], dtype=np.object)
        for i, on_mask in enumerate(high_thresh):
            on_data = self.dataset.get_feature_data(
                ids=np.array(all_ids)[np.where(on_mask == True)[0]]).dropna()

            off_mask = low_thresh[i]
            off_ids = list(
                set(all_ids) - set(np.array(all_ids)[np.where(off_mask == True)[0]]))
            off_data = self.dataset.feature_table.get_feature_data(
                ids=off_ids).dropna()

            y = np.array([0] * off_data.shape[0] + [1] * on_data.shape[0])
            X = np.vstack((np.array(off_data), np.array(on_data)))

            from sklearn.preprocessing import scale
            X = scale(X, with_mean=False)
            self.data[i] = (X, y)

        self.feature_names = self.dataset.get_feature_data().columns.tolist()
        self.n_regions = self.data.shape[0]

    def initalize_containers(self):
        """ Makes all the containers that will hold results from classificaiton """
        self.class_score = np.zeros(self.n_regions)
        self.predictions = np.empty(self.n_regions, np.object)
        self.importance = mask_diagonal(
            np.ma.masked_array(np.zeros((self.n_regions, len(self.feature_names)))))
        self.fit_clfs = np.empty(self.n_regions, np.object)

    def classify(self, scoring='accuracy', n_jobs=1, importance_function=None):
        """
        scoring -  scoring function or type (str)
        n_jobs - Number of parallel jobs
        importance_function - Function to extract importance vectors from classifiers (differs by algorithm)
        """
        if importance_function is None:
            importance_function = log_odds_ratio

        if self.data is None:
            self.load_data()
            self.initalize_containers()

        print("Classifying...")
        pb = ProgressBar(self.n_regions, start=True)

        for index, output in enumerate(Parallel(n_jobs=n_jobs)(
                delayed(classify_parallel)(
                    self.classifier, scoring, region_data, importance_function) for region_data in self.data)):
            self.class_score[index] = output['score']
            self.importance[index] = output['importance']
            self.predictions[index] = output['predictions']
            pb.next()

    def get_formatted_importances(self, feature_names=None):
        """ Returns a pandas table of importances for each feature for each region. 
        Optionally takes new names for each feature (i.e. nickanames) """
        import pandas as pd
        if feature_names is None:
            feature_names = self.feature_names

        o_fi = pd.DataFrame(self.importance, columns=feature_names)

        # Melt feature importances, and add top_words for each feeature
        o_fi['region'] = range(1, o_fi.shape[0] + 1)
        return pd.melt(o_fi, var_name='feature', value_name='importance', id_vars=['region'])
