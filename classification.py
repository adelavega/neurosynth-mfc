import numpy as np
from tools import ProgressBar, mask_diagonal
from joblib import Parallel, delayed

## Replace with a more basic sklearn function
from neurosynth.analysis.classify import classify

def classify_parallel(classifier, scoring, region_data):
    X, y = region_data

    output = classify(
        X, y, classifier=classifier, cross_val='4-Fold', scoring=scoring, class_weight=None)
    output['odds_ratio'] = np.log(output['clf'].clf.theta_[1] / output['clf'].clf.theta_[0])
    return output

class RegionalClassifier(object):
    def __init__(self, dataset, mask_img, classifier=None,
                 thresh=0.05, cv='4-Fold', thresh_low=0):
        """
        thresh_low - if OvA then this is the threshold for the rest of the brain
        """
        self.mask_img = mask_img
        self.classifier = classifier
        self.dataset = dataset
        self.thresh = thresh
        self.thresh_low = thresh_low
        self.cv = cv
        self.data = None

    def load_data(self):
        """ Loads ids and data for each individual mask """
        from neurosynth.analysis.reduce import average_within_regions

        all_ids = self.dataset.image_table.ids

        high_thresh = average_within_regions(self.dataset, self.mask_img, threshold=self.thresh)
        low_thresh = average_within_regions(self.dataset, self.mask_img, threshold=self.thresh_low)

        self.data = np.empty(high_thresh.shape[0], dtype=np.object)
        for i, on_mask in enumerate(high_thresh):
            on_data = self.dataset.get_feature_data(ids=np.array(all_ids)[np.where(on_mask == True)[0]]).dropna()

            off_mask = low_thresh[i]
            off_ids = list(set(all_ids) - set(np.array(all_ids)[np.where(off_mask == True)[0]]))
            off_data = self.dataset.feature_table.get_feature_data(ids=off_ids).dropna()

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
        self.odds_ratio = mask_diagonal(np.ma.masked_array(np.zeros((self.n_regions, len(self.feature_names)))))
        self.fit_clfs = np.empty(self.n_regions, np.object)

    ### MAKE THIS FUNCTION more like when I do bootstrapping / permutation test
    def classify(self, scoring='accuracy', n_jobs=1):
        if self.data is None:
            self.load_data()
            self.initalize_containers()

        print("Classifying...")
        pb = ProgressBar(self.n_regions, start=True)

        for index, output in enumerate(Parallel(n_jobs=n_jobs)(
			delayed(classify_parallel)(self.classifier, scoring, region_data) for region_data in self.data)):
                self.class_score[index] = output['score']
                self.odds_ratio[index] = output['odds_ratio']
                self.predictions[index] = output['predictions']
                pb.next()