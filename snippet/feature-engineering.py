import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans


DATA_DIR = '../input/lish-moa/'
SEED = 42


def data_filter(train, test):
    """cp_type = ctl_vehicleのデータは除外（unknownデータなので）
    """
    train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    test = test[test['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)

    return train, test


def one_hot_encoder(df, cols):
    """sklearnのOneHotEncoderでEncodingを行う
    """
    for col in cols:
        ohe = OneHotEncoder(sparse=False)
        ohe_df = pd.DataFrame(ohe.fit_transform(df[[col]])).add_prefix(col + '_ohe_')
        # 元のDFに結合
        df = pd.concat([df, ohe_df], axis=1)
        # oheしたカラムを除外
        df = df.drop(col, axis=1)

    return df


def get_kmeans_label(df, n_cluster):
    """k-meansで教師なし学習（クラスタ分類）
    """
    km = KMeans(
            n_clusters=n_cluster,
            init='k-means++',
            random_state=SEED
        )
    y_km = km.fit(df)

    return y_km.labels_


def feature_stats(df):
    """基礎統計量の追加
    """
    df.loc[:, 'g-sum'] = df[GENES].sum(axis=1)
    df.loc[:, 'g-mean'] = df[GENES].mean(axis=1)
    df.loc[:, 'g-std'] = df[GENES].std(axis=1)
    df.loc[:, 'g-kurt'] = df[GENES].kurtosis(axis=1)
    df.loc[:, 'g-skew'] = df[GENES].skew(axis=1)

    df.loc[:, 'c-sum'] = df[CELLS].sum(axis=1)
    df.loc[:, 'c-mean'] = df[CELLS].mean(axis=1)
    df.loc[:, 'c-std'] = df[CELLS].std(axis=1)
    df.loc[:, 'c-kurt'] = df[CELLS].kurtosis(axis=1)
    df.loc[:, 'c-skew'] = df[CELLS].skew(axis=1)

    df.loc[:, 'gc-sum'] = df[GENES + CELLS].sum(axis=1)
    df.loc[:, 'gc-mean'] = df[GENES + CELLS].mean(axis=1)
    df.loc[:, 'gc-std'] = df[GENES + CELLS].std(axis=1)
    df.loc[:, 'gc-kurt'] = df[GENES + CELLS].kurtosis(axis=1)
    df.loc[:, 'gc-skew'] = df[GENES + CELLS].skew(axis=1)

    return df


def feature_pca(df, col_list, n_comp, col_type):
    """PCAの特徴量を生成
    """
    pca = (PCA(n_components=n_comp, random_state=SEED).fit_transform(df[col_list]))
    pca_df = pd.DataFrame(pca, columns=[f'{col_type}-pca_{i}' for i in range(n_comp)])
    df = pd.concat([df, pca_df], axis=1)
    return df


def feature_svd(df, col_list, n_comp, col_type):
    """SVDの特徴量を生成
    """
    svd = (TruncatedSVD(n_components=n_comp, random_state=SEED).fit_transform(df[col_list]))
    svd_df = pd.DataFrame(svd, columns=[f'{col_type}-svd_{i}' for i in range(n_comp)])
    df = pd.concat([df, svd_df], axis=1)
    return df


def variance_threshold(df, n=0.4):
    """分散がしきい値以下の特徴量を捨てる
    """
    var_thresh = VarianceThreshold(threshold=n)
    df = pd.DataFrame(var_thresh.fit_transform(df))
    return df


def feature_engineering(train_features, test_features):

    global GENES, CELLS

    # カラムのリストを保持
    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    # filter
    train, test = data_filter(train_features, test_features)

    # 結合
    df = pd.concat([train, test])
    df = df.reset_index(drop=True)

    # k-means cluster
    df.loc[:, 'g-cluster'] = get_kmeans_label(df[GENES], n_cluster=35)
    df.loc[:, 'c-cluster'] = get_kmeans_label(df[CELLS], n_cluster=5)

    # atats feature
    df = feature_stats(df)

    # PCA feature
    df = feature_pca(df, GENES, n_comp=30, col_type='g')
    df = feature_pca(df, CELLS, n_comp=5, col_type='c')

    # SVD feature
    df = feature_svd(df, GENES, n_comp=30, col_type='g')
    df = feature_svd(df, CELLS, n_comp=5, col_type='c')

    # カテゴリDFとnotカテゴリDFに分割（標準化&エンコードのため）
    cat_columns = ['cp_time', 'cp_dose', 'g-cluster', 'c-cluster']
    cat_df = df[['sig_id'] + cat_columns]
    num_df = df.drop(['sig_id'] + cat_columns, axis=1)

    # VarianceThreshold
    num_df = variance_threshold(num_df, n=0.4)

    # notカテゴリDFを正規化
    sscaler = StandardScaler()
    num_df.iloc[:, :] = sscaler.fit_transform(num_df)

    # Robust Scaler
    """
    rscaler = RobustScaler()
    num_df.iloc[:, :] = rscaler.fit_transform(num_df)
    """

    # min max Scaler
    """
    mmscaler = MinMaxScaler()
    num_df.iloc[:, :] = mmscaler.fit_transform(num_df)
    """

    # カテゴリ変数をone-hot-encode
    cat_df = one_hot_encoder(cat_df, cat_columns)

    # カテゴリDFとnotカテゴリDFを結合
    df = pd.concat([cat_df, num_df], axis=1)

    # trainとtestに再分割
    train = df.iloc[:len(train), :]
    test = df.iloc[len(train):, :]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, test


def main():
    """How to use
    """

    # load data
    train_features = pd.read_csv(DATA_DIR + 'train_features.csv')
    train_targets_scored = pd.read_csv(DATA_DIR + 'train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv(DATA_DIR + 'train_targets_nonscored.csv')
    test_features = pd.read_csv(DATA_DIR + 'test_features.csv')

    # 特徴量生成
    train, test = feature_engineering(train_features, test_features)

    # それぞれのカラムのリストを取得
    target_cols = train_targets_scored.drop('sig_id', axis=1).columns.values.tolist()  # 目的変数
    target_cols_non_scored = train_targets_nonscored.drop('sig_id', axis=1).columns.values.tolist()  # pretrain用の目的変数
    feature_cols = [c for c in train.columns if c not in ['sig_id']]  # 学習に使用する説明変数

    # train用のデータセット生成
    train = train.merge(train_targets_scored, on='sig_id')
    target = train[train_targets_scored.columns]

    # pretrain用のデータセット生成
    train_non_scored = train[['sig_id'] + feature_cols].merge(train_targets_nonscored, on='sig_id')
    target_non_scored = train_non_scored[train_targets_nonscored.columns]

    # 確認
    display(train.shape, train.head(), test.shape, test.head(), target.shape, target.head())
    display(train_non_scored.shape, train_non_scored.head(), target_non_scored.shape, target_non_scored.head())
    display(len(target_cols), len(target_cols_non_scored), len(feature_cols))
