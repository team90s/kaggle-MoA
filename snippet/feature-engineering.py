import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.cluster import KMeans


class Config:
    # setting
    is_debug = False
    is_kaggle = False
    is_pretrain = False

    # post prosessing
    is_sub_clipping = False
    is_sub_rounding = False
    is_sub_adjusting_imbalanced_targets = False

    # features
    do_variancethreshold = False
    do_kmeans = False
    do_filter = True
    do_feature_squared = True
    do_feature_stats = True
    do_feature_pca = True
    do_feature_svd = True
    do_feature_fa = True

    # constant
    seed = 42
    n_gene_comp = 70
    n_cell_comp = 10
    n_gene_kmeans_cluster = 30
    n_cell_kmeans_cluster = 5
    n_variance_threshold = 0.7
    p_min = 0.001
    p_max = 0.999
    scaler = 'Rankgauss'  # Standard, Robust, MinMax, Rankgauss, None

    # HyperParameters
    epochs = 80
    seed_avg = [0, 101, 202, 303, 404, 505]
    nfold = 7
    verbose = 0
    lr = 1e-3
    weight_decay = 1e-5
    batch_size = 128


DATA_DIR = '../input/lish-moa/'
config = Config()


def data_filter(train, test):
    """cp_type = ctl_vehicleのデータは除外（unknownデータなので）
    """
    train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    test = test[test['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
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


def kmeans(df, n_cluster, seed=config.seed):
    """k-meansで教師なし学習（クラスタ分類）
    """
    km = KMeans(
            n_clusters=n_cluster,
            init='k-means++',
            random_state=seed
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


def feature_pca(df, col_list, n_comp, col_type='g', seed=config.seed):
    """PCAの特徴量を生成
    """
    pca = (PCA(n_components=n_comp, random_state=seed).fit_transform(df[col_list]))
    pca_df = pd.DataFrame(pca, columns=[f'{col_type}-pca_{i}' for i in range(n_comp)])
    df = pd.concat([df, pca_df], axis=1)
    return df


def feature_svd(df, col_list, n_comp, col_type='g', seed=config.seed):
    """SVDの特徴量を生成
    """
    svd = (TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(df[col_list]))
    svd_df = pd.DataFrame(svd, columns=[f'{col_type}-svd_{i}' for i in range(n_comp)])
    df = pd.concat([df, svd_df], axis=1)
    return df


def feature_fa(df, col_list, n_comp, col_type='g', seed=config.seed):
    """FAの特徴量を生成
    """
    svd = (FactorAnalysis(n_components=n_comp, random_state=seed).fit_transform(df[col_list]))
    svd_df = pd.DataFrame(svd, columns=[f'{col_type}-fa_{i}' for i in range(n_comp)])
    df = pd.concat([df, svd_df], axis=1)
    return df


def feature_squared(df, cols_list):
    """二乗を計算
    """
    for feature in cols_list:
        df.loc[:, f'{feature}_squared'] = df[feature] ** 2
    return df


def variance_threshold(df, n):
    """分散がしきい値以下の特徴量を捨てる
    """
    var_thresh = VarianceThreshold(threshold=n)
    df = pd.DataFrame(var_thresh.fit_transform(df))
    return df


def rankgauss(df, cols, train_len, seed=config.seed):
    """RankGauss
    """
    _train = df.iloc[:train_len, :]
    _test = df.iloc[train_len:, :]

    for col in cols:
        transformer = QuantileTransformer(n_quantiles=100, random_state=seed, output_distribution="normal")
        vec_len = len(_train[col].values)
        vec_len_test = len(_test[col].values)
        raw_vec = _train[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)
        _train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        _test[col] = transformer.transform(_test[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]

    _df = pd.concat([_train, _test])
    _df = _df.reset_index(drop=True)
    return _df


def feature_engineering(train_features, test_features):

    global GENES, CELLS

    # カラムのリストを保持
    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    # カテゴリカラム
    cat_columns = ['cp_time', 'cp_dose']

    train = train_features.copy()
    test = test_features.copy()

    # filter
    if config.do_filter:
        print('do filter')
        train, test = data_filter(train_features, test_features)

    df = pd.concat([train, test])
    df = df.reset_index(drop=True)

    # k-means cluster
    if config.do_kmeans:
        print('do k-means')
        df.loc[:, 'g-cluster'] = kmeans(df[GENES], n_cluster=config.n_gene_kmeans_cluster)
        df.loc[:, 'c-cluster'] = kmeans(df[CELLS], n_cluster=config.n_cell_kmeans_cluster)
        cat_columns = cat_columns + ['g-cluster', 'c-cluster']

    # Stats feature
    if config.do_feature_stats:
        print('do feature_stats')
        df = feature_stats(df)

    # squared
    if config.do_feature_squared:
        print('do feature_squared')
        df = feature_squared(df, CELLS)

    # PCA feature
    if config.do_feature_pca:
        print('do feature_pca')
        df = feature_pca(df, GENES, n_comp=config.n_gene_comp, col_type='g')
        df = feature_pca(df, CELLS, n_comp=config.n_cell_comp, col_type='c')

    # SVD feature
    if config.do_feature_svd:
        print('do feature_svd')
        df = feature_svd(df, GENES, n_comp=config.n_gene_comp, col_type='g')
        df = feature_svd(df, CELLS, n_comp=config.n_cell_comp, col_type='c')

    # FA feature
    if config.do_feature_fa:
        print('do feature_fa')
        df = feature_fa(df, GENES, n_comp=config.n_gene_comp, col_type='g')
        df = feature_fa(df, CELLS, n_comp=config.n_cell_comp, col_type='c')

    # カテゴリのDFとnotカテゴリのDFに分割（標準化&エンコードのため）
    cat_df = df[['sig_id'] + cat_columns]
    num_df = df.drop(['sig_id'] + cat_columns, axis=1)

    # VarianceThreshold
    if config.do_variancethreshold:
        print('do variancethreshold')
        num_df = variance_threshold(num_df, n=config.n_variance_threshold)

    # 正規化
    if config.scaler == 'Rankgauss':
        print('do Rankgauss')
        num_df = rankgauss(num_df, num_df.columns.tolist(), len(train))

    elif config.scaler == 'Standard':
        print('do Standard')
        sscaler = StandardScaler()
        num_df.iloc[:, :] = sscaler.fit_transform(num_df)

    elif config.scaler == 'Robust':
        print('do Robust')
        rscaler = RobustScaler()
        num_df.iloc[:, :] = rscaler.fit_transform(num_df)

    elif config.scaler == 'MinMax':
        print('do MinMax')
        mmscaler = MinMaxScaler()
        num_df.iloc[:, :] = mmscaler.fit_transform(num_df)

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

    # ...
    # ...
    # ...

    # ちなみに、post prosessingはこんな感じ（やるかどうかは要検討）
    if config.is_sub_rounding:
        # https://github.com/team90s/kaggle-MoA/issues/36
        print('rounding...')
        n = 0.0001
        sub.loc[:, target_cols] = sub[target_cols].where(sub[target_cols] > n, 0)

    if config.is_sub_clipping:
        # https://www.kaggle.com/c/lish-moa/discussion/191621
        print('clipping...')
        sub.loc[:, target_cols] = np.clip(sub[target_cols].values, config.p_min, config.p_max)

    if config.is_sub_adjusting_imbalanced_targets:
        # https://www.kaggle.com/c/lish-moa/discussion/191135
        print('adjusting...')
        sub.loc[:, 'atp-sensitive_potassium_channel_antagonist'] = 0.000012
        sub.loc[:, 'erbb2_inhibitor'] = 0.000012
