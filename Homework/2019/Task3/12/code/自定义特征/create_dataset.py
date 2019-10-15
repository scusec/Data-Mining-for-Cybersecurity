import pandas as pd
import csv

def get_features(payload):
    script = payload.lower().count('script')
    java = payload.lower().count('java')
    iframe = payload.lower().count('iframe')
    quto_1 = payload.lower().count('<')
    quto_2 = payload.lower().count('>')
    quto_3 = payload.lower().count('\"')
    quto_4 = payload.lower().count('\'')
    quto_5 = payload.lower().count('%')
    quto_6 = payload.lower().count('(')
    quto_7 = payload.lower().count(')')
    features = [script, java, iframe, quto_1, quto_2, quto_3,
                quto_4, quto_5, quto_6, quto_7]
    return features

if __name__ == '__main__':
    df = pd.read_csv('../data/data.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    df.dropna(axis=0, how='any')

    with open ('../data/dataset.csv', 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(df)):
            features = get_features(df["payload"].loc[i])
            features.append(df["label"].loc[i])
            writer.writerow(features)