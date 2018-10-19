#!/home/thomas/anaconda3/bin/python

"""
    # Author : Thomas Neuer (tneuer)
    # File Name : pipeliner.py
    # Creation Date : Don 22 Feb 2018 20:44:18 CET
    # Last Modified : Sam 12 Mai 2018 18:26:11 CEST
    # Description : All relevant pipeline utilities are programmed here.
"""
#==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import h5py

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
# from keras.preprocessing.sequence import pad_sequences
from baUtils import get_sg_and_bg

Start = 0
End = 0

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select certain features to handle in the Pipeline.

    The main purpose of this class is to enable easy pipelining with scikit-learn.
    """
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X):
        return self

    def transform(self, X):
        # print("Select features...")
        return X[self.attributes]


class InputPT(BaseEstimator, TransformerMixin):
    """Calculate PT of tracks

    Is used in testSplit.py to insert the PT right at the beginning.
    """
    def fit(self, X):
        return self

    def transform(self, X):
        print("Calculate PT...")
        pt = (X["tracks_PX"]**2 + X["tracks_PY"]**2)**0.5
        inser_id = int(np.where(X.columns.values=="nTrack")[0][0])+1
        X.insert(inser_id, "tracks_PT", pt)

        return X


class GlobalStandard(BaseEstimator, TransformerMixin):
    """Some track variables need to be scaled globally.

    To reduce variance but still conserve information some tracks variables (e.g.
    tracks_PT) are Standardized not per column (per track) but over all tracks.
    """

    def __init__(self, global_var):
        self.global_var = global_var

    def fit(self, X):
        self.means = {}
        self.stds = {}
        for col in self.global_var:
            S = sum(X[col].apply(sum))
            N = sum(X[col].apply(len))
            self.means[col] = S/N

            center = X[col].apply(lambda x: x-self.means[col])
            S = sum(center.apply(lambda x: sum(x**2)))
            self.stds[col] = np.sqrt(S/N)

        return self

    def transform(self, X):
        for col in self.global_var:
            print("Standardizing ", col, " ...")
            center = X[col].apply(lambda x: x-self.means[col])

            X.loc[:, col] = center/self.stds[col]

        return X


class TrackFiller(BaseEstimator, TransformerMixin):
    """Fill all tracks with the same amount of track information.

    Neural networks need the same number of input feature for every instance, but events
    have different amounts of tracks. This function equalizes all.
    """
    def __init__(self, maxTracks = 150):
        self.maxTracks = maxTracks

    def fit(self, X):
        return self


    def transform(self, X):
        Start = time.clock()
        # print("Filling...")
        # start, from where the arrays need to be filled in (all after nTracks)
        start = np.where(X.columns.values == "nTrack")[0][0]+1
        end = X.shape[1]

        nr_examples = X.shape[0]
        nr_features = start + (end-start) * self.maxTracks
        difference_to_full = (nr_features - (end-start)*X["nTrack"]).values

        # all arrays are stored in here where the first column is dummy
        model_matrix = np.zeros((nr_examples, nr_features))
        X.reset_index(inplace=True, drop=True)

        # determine indices which already have entries
        s = time.clock()
        ix_mask = np.matrix([np.pad(np.concatenate([np.arange(0, row["nTrack"])+k*self.maxTracks for k in range(len(row)-1)]), (0, difference_to_full[i]), "constant", constant_values = -1)+1 for i, row in X.iterrows()])
        print("Mask: ", time.clock()-s)

        # placeholder for filling
        plhd = [[i] for i in range(nr_examples)]

        # make matrix with already existing values
        s = time.clock()
        fill_vals = np.matrix([np.pad(np.concatenate(row[1:]), (0, difference_to_full[i]), "constant", constant_values=0) for i, row in X.iterrows()])
        print("Fillings: ", time.clock()-s)


        model_matrix[plhd, ix_mask] = fill_vals
        # create column names: i.e. needs maxTracks pX columns
        colnames = X.columns.values[start:end]
        cols = []

        for col in colnames:
            mult_cols = [col] * self.maxTracks
            cols += mult_cols

        # convert to pandas.DataFrame
        df = pd.DataFrame(model_matrix[:, 1:], columns = cols)

        End = time.clock()
        print("Tracks: ", End-Start)
        # names = standard_features[:]
        # for name in df.columns.values:
        #     names.append(name)
        # with open("../data/featureNames.pickle", "wb") as f:
        #     pickle.dump(names, f)
        # raise("STOP")
        return df


class PtMethod(BaseEstimator, TransformerMixin):

    def __init__(self, method = "sorted", randomState = 42, rowShuffle = True):
        """Initalization of object

        Parameters
        ----------
        method : str
            has to be 'shuffled' or 'sorted'
        randomState : int
            random state for the shuffling process
        rowShuffle : bool
            if true, then every row gets shuffled individually. Takes longer but is also
            cleaner
        """
        if method not in ["sorted", "shuffled"]:
            raise("Invalid track_method. Either 'shuffled' or 'sorted'.")
        self.method = method
        self.randomState = randomState
        self.rowShuffle = rowShuffle

    def fit(self, X):
        return self

    def transform(self, X):
        """Deals with the arbitrariness of the order of tracks variables.

        An usual event in this dataset consists of 10-150 tracks, for each different properties
        are measured (charge, energy,... ). However, these tracks are not arranged in a specific order.
        In order to deal with this arbitrariness one can either sort this events by transverse momentum
        or randomize them by shuffling.
        """
        Start = time.clock()
        if self.method == "sorted":
            def sort_by_pT(df):
                # set NaNs to zero in order for sorting to work correctly, changed back at the end
                X["tracks_PT"] = X["tracks_PT"].fillna(0)

                # Get column names
                cols = np.array(X.columns.values)
                idx = np.where(cols == "tracks_PT")[0][0]

                # Get transverse momentum columns
                pTs = np.where(X.columns.values=="tracks_PT")[0]
                pTs = X.iloc[:,pTs]

                nTrack = len(np.where(cols == "tracks_PT")[0])

                #create a mask for the data matrix with pT sorted indices
                sorted_mask = pTs.values.argsort(axis=1)[:,::-1]
                cols = len(np.unique(cols))-idx
                obs = X.shape[0]

                # placeholder array necessary for numpy to overlay the mask matrix over the data matrix
                plhd = [[i] for i in range(obs)]

                # rearrange every block of features (block of length nTrack for pX, pY,...)
                for i in range(cols):
                    start = idx + i * nTrack
                    end = idx + (i+1) * nTrack
                    X.iloc[:, start:end] = X.iloc[:, start:end].values[plhd, sorted_mask]
                    # print(i+1, " of ", cols, " columns sorted.")

                X["tracks_PT"] = X["tracks_PT"].replace(to_replace = 0, value = 0)

            sort_by_pT(X)

            End = time.clock()
            print("Method: ", End-Start)

            return X
        else:
            def shuffle_momentum(df, shuffle_by_row = self.rowShuffle):
                cols = np.array(df.columns.values)
                idx = np.where(cols == "tracks_PT")[0][0]

                nTrack = len(np.where(cols == "tracks_PT")[0])

                idx_shuffle = np.arange(nTrack)
                cols = len(np.unique(cols)) - idx
                obs = df.shape[0]

                if shuffle_by_row:
                    # Create permutation matrix, creates a matrix with shuffled indices per row
                    np.random.seed = self.randomState
                    shuffle_matrix = np.array([np.random.permutation(nTrack) for i in range(df.shape[0])])

                    # Create placeholder variable for numpy to overlay the mask shuffle_matrix over the data
                    plhd = [[i] for i in range(obs)]

                    # Sort all features after nTrack (pX, pY,...), these are the cols
                    for i in range(cols):
                        start = idx + i * nTrack
                        end = idx + (i+1) * nTrack
                        df.iloc[:, start:end] = df.iloc[:, start:end].values[plhd, shuffle_matrix]
                        # print(i+1, " of ", cols, " columns shuffled.")
                else:
                    # every row is shuffled with the same permutation rule
                    np.random.shuffle(idx_shuffle)
                    i = 0
                    while(i*nTrack+idx < len(cols)):
                        df.iloc[:, (idx+i*nTrack):(idx+(i+1)*nTrack)] = df.iloc[:, (idx+i*nTrack):(idx+(i+1)*nTrack)].iloc[:, idx_shuffle]
                        i+=1
                        # print(i+1, " of ", cols, " columns shuffled.")

            shuffle_momentum(X)
            return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """Transform certain features via log(.+1) map

    Some features (e.g. PT) are spread out very much on the positive axis.
    Often they are more well behaved after a logarithmic transformation
    """

    def __init__(self, attributes, standard=True):
        self.attributes = attributes
        self.standard = standard

    def fit(self, X):
        return self

    def transform(self, X):
        print("Taking logarithm...")
        if self.standard:
            for col in self.attributes:
                X.loc[:, col] = np.log(X[col].values+1)
        else:
            def special_log(row):
                return np.log(row)
            for col in self.attributes:
                X.loc[:, col] = X[col].apply(special_log)

        return X

def logtransform_standard_pipeline(stdF, stdLog, nstdF=None, nstdLog=None, nstdS=None):

    pipeline_standard = Pipeline([
        ("selector", FeatureSelector(stdF)),
        ("logtransform", LogTransformer(stdLog)),
        ("std_scaler", StandardScaler())
        ])

    if nstdF is not None and nstdLog is not None and nstdS is not None:
        pipeline_tracks = Pipeline([
            ("selector", FeatureSelector(nstdF)),
            ("logtransform", LogTransformer(nstdLog, standard=False)),
            ("standardizer", GlobalStandard(nstdS))
            ])

        scale_transformer_pipeline = FeatureUnion(transformer_list=[
            ("pipe_standard", pipeline_standard),
            ("pipe_tracks", pipeline_tracks)
            ])
    else:
        scale_transformer_pipeline = FeatureUnion(transformer_list=[
            ("pipeline_standard", pipeline_standard)
            ])

    return scale_transformer_pipeline


def batch_pipeline(stdF, nstdF, PCA = None):
    pipeline_filler = Pipeline([
        ("selector", FeatureSelector(nstdF)),
        ("filler", TrackFiller()),
        ("ptMethod", PtMethod())
        ])

    pipeline_identity = Pipeline([
        ("selector", FeatureSelector(stdF))
        ])

    if PCA is None:
        batch_pipeline = FeatureUnion([
            ("pipe_identity", pipeline_identity),
            ("pipe_tracks", pipeline_filler)
            ])
    else:
        pipeline_PCA = Pipeline([
            ("PCA", PCA)
            ])
        batch_pipeline = FeatureUnion([
            ("pipe_identity", pipeline_identity),
            ("pipe_tracks", pipeline_filler),
            ("PCA", pipeline_PCA)
            ])

    return batch_pipeline


if __name__ == "__main__":
    sgname = "../data/Val_mu_MC_156388.pickle"
    bgname = "../data/Val_mu_hf_106705.pickle"
    data, labels = get_sg_and_bg(sgname, bgname, labels=True)

    data = data.iloc[:100, :]
    data2 = data.copy()
    # transverse_wo_SPD = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", "muplus_TrEta", "muplus_TrPhi", 'nTrack', 'tracks_PT', 'tracks_IP', 'tracks_IPCHI2', 'tracks_eta', 'tracks_phi', 'tracks_charge', 'tracks_isMuon']
    # 
    # standard_features = transverse_wo_SPD[:7]
    # standard_log = ["muplus_PT", "muminus_PT"]
    # 
    # non_standard_features = transverse_wo_SPD[6:]
    # global_standardized = non_standard_features[1:-2]
    # non_standard_log = ["tracks_PT", "tracks_IPCHI2", "tracks_IP"]
    # 
    # with open("../data/Standardizer_mu_2367829.pickle", "rb") as f:
    #     scale_transformer_pipeline = pickle.load(f)
    # 
    # start = time.clock()
    # 
    # data = scale_transformer_pipeline.transform(data)
    # newFeatures = ["muminus_PT", "muminus_TrEta", "muminus_TrPhi", "muplus_PT", "muplus_TrEta", "muplus_TrPhi", "nTrackN", 'nTrack', 'tracks_PT', 'tracks_IP', 'tracks_IPCHI2', 'tracks_eta', 'tracks_phi', 'tracks_charge', 'tracks_isMuon']
    # standard_features[standard_features.index("nTrack")] = "nTrackN"
    # data = pd.DataFrame(data, columns=newFeatures)
    # print("Time scale/trafo: ", time.clock()-start)
    # 
    # batch_pipeline = batch_pipeline(standard_features, non_standard_features)
    # 
    # data = batch_pipeline.transform(data)

    print(data.head())
    data2 = scale_transformer_pipeline.transform(data2)

    # with h5py.File("../data/Processed_Test_mu_{}.h5".format(data.shape[0]), "w") as hf:
    #     hf.create_dataset("Test", data=data.tolist())
    # 
    # with open("../data/Test_mu_labels_{}.pickle".format(data.shape[0]), "wb") as f:
    #     pickle.dump(labels, f)
    ####
    # PCA
    ####
    """
    n_batches = int(data.shape[0]/2000)
    inc_pca = IncrementalPCA(n_components=354)
    count = 0
    times = []
    for batch in np.array_split(data, n_batches):
        start = time.clock()
        batch = batch_pipeline.transform(batch)
        inc_pca.partial_fit(batch)
        count += 1
        if count % 10 == 0:
            time_per_batch = np.mean(times)
            print("Batch: ", count, " / ", n_batches, " ", time_per_batch, "s")
            seconds = np.round(time_per_batch*(n_batches-count))
            minutes = np.round(seconds/60, 1)
            print("Approximately: {}s - {}min remaining".format(seconds, minutes))
        times.append(time.clock()-start)

    with open("../data/inc_pca354.pickle", "wb") as f:
        pickle.dump(inc_pca, f)

    evar = inc_pca.explained_variance_ratio_
    csum = np.cumsum(inc_pca.explained_variance_ratio_)
    fractions = [0.99, 0.95, 0.90]
    ds = [np.argmax(csum >= f)+1 for f in fractions]
    titlestring = ""

    plt.figure()
    plt.plot(csum, linewidth=2)
    for f, d in zip(fractions, ds):
        titlestring += "{}: - {} -".format(f, d)
        plt.axvline(x=d, c="red", linewidth=1, zorder=0, linestyle="--")
        plt.axhline(y=f, c="red", linewidth=1, zorder=0, linestyle="--")
    titlestring = titlestring[:-1]
    plt.ylabel("Explained variance")
    plt.xlabel("Dimensions")
    plt.title(titlestring)
    plt.grid()
    plt.show()
    """
