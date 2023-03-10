#region imports
from AlgorithmImports import *
from Clustering import *
from Metrics import *
from Pair_Checking import *
#endregion
def remove_tickers_with_nan(df, threshold):
    """
    Removes columns with more than threshold null values
    """
    null_values = df.isnull().sum()
    null_values = null_values[null_values > 0]

    to_remove = list(null_values[null_values > threshold].index)
    df = df.drop(columns=to_remove)

    return df

def data_cleaning(df_price):
    df_price = remove_tickers_with_nan(df_price,0)
    df_price = df_price.interpolate(method="linear")
    df_return = df_price.pct_change().iloc[1:]
    return df_price, df_return

'''
df_price: the UNIVERSE, each row is a time tick, each column is a ticker, the entries are prices
min_half_life:  minimium half life value of the spread to consider the pair
                default = 78(number of points in a day)
max_half_life: 
                default = 20000 (~number of points in a year: 78*252)
min_zero_crossings: minimium number of allowed zero crossings
hurst_threshold: mimimium acceptable number for hurst threshold
p_value_threshold:  pvalue threshold for a pair to be cointegrated
'''
def select_pair(df_price, subsample = 25, 
                min_half_life = 78, max_half_life = 20000,
                min_zero_crosings = 12,
                 p_value_threshold = 0.1,
                 hurst_threshold=0.5,
                 num_pairs=4,
                 pair_ranking_metric='p_value',
                 pair_ranking_order=False):
    
    df_price, df_return = data_cleaning(df_price)
    # Apply PCA
    pca = PCA(n_components=5, svd_solver='auto', random_state=0)
    pca.fit(df_return)
    explained_variance = pca.explained_variance_
    X = preprocessing.StandardScaler().fit_transform(pca.components_.T)
    clustered_series_all, clustered_series, counts, clf = apply_OPTICS(X, 
                                        df_return, min_samples=3,cluster_method='xi')
    cluster_size(counts)
    plot_TSNE(X,clf,clustered_series_all)

    for clust in range(len(counts)):
        symbols = list(clustered_series[clustered_series==clust].index)
        means = np.log(df_price[symbols].mean())
        series = np.log(df_price[symbols]).sub(means)
        series.plot(figsize=(10,5))#title='ETFs Time Series for Cluster %d' % (clust+1))
        #plt.ylabel('Normalized log prices', size=12)
        #plt.xlabel('Date', size=12)
        plt.savefig('cluster_{}.png'.format(str(clust+1)), bbox_inches='tight', pad_inches=0.1)


    pairs_unsupervised, unique_tickers = get_candidate_pairs(clustered_series=clustered_series,
                                                                pricing_df=df_price,
                                                                min_half_life=min_half_life,
                                                                max_half_life=max_half_life,
                                                                min_zero_crosings=min_zero_crosings,
                                                                p_value_threshold=p_value_threshold,
                                                                hurst_threshold=hurst_threshold,
                                                                subsample=subsample,
                                                                num_pairs=num_pairs,
                                                                pair_ranking_metric=pair_ranking_metric,
                                                                pair_ranking_order=pair_ranking_order
                                                                )
                            
    return pairs_unsupervised
# Your New Python File
