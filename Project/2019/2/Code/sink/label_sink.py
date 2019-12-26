from joblib import load
from extract_sink_features import get_sink_features

def predict_sink_label(sink):
    cluster = load('sink_cluster_kmeans.model')
    feature_list = get_sink_features(sink)
    if len(feature_list) == 0:
        print('get sink feature error')
        return 