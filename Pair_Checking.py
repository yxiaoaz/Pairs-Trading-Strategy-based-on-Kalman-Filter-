#region imports
from AlgorithmImports import *
from Metrics import *
#endregion
def check_for_stationarity(X,subsample=25):
    """
    H_0 in adfuller is unit root exists (non-stationary).
    We must observe significant p-value to convince ourselves that the series is stationary.
    :param X: time series
    :param subsample: boolean indicating whether to subsample series
    :return: adf results
    """
    if subsample != 0:
        frequency = round(len(X)/subsample)
        subsampled_X = X[0::frequency]
        result = adfuller(subsampled_X)
    else:
        result = adfuller(X)
    # result contains:
    # 0: t-statistic
    # 1: p-value
    # others: please see https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

    return {'t_statistic': result[0], 'p_value': result[1], 'critical_values': result[4]}

def check_properties(price_series,  p_value_threshold=0.1, min_half_life=78, max_half_life=20000,
                        min_zero_crossings=0, hurst_threshold=0.5, subsample=25,num_pairs=4,
                         pair_ranking_metric='p_value',pair_ranking_order=False):
    """
    Gets two time series as inputs and provides information concerning cointegration stasttics
    Y - b*X : Y is dependent, X is independent
    """

    # for some reason is not giving right results
    # t_statistic, p_value, crit_value = coint(X,Y, method='aeg')

    # perform test manually in both directions
    X = price_series[0]
    Y = price_series[1]
    pairs = [(X, Y), (Y, X)]
    pair_stats = [0] * 2
    criteria_not_verified = 'cointegration'

    # first of all, must verify price series S1 and S2 are I(1)
    stats_Y = check_for_stationarity(np.asarray(Y),subsample=subsample)
    if stats_Y['p_value'] > 0.10:
        stats_X = check_for_stationarity(np.asarray(X),subsample=subsample)
        if stats_X['p_value'] > 0.10:
            # conditions to test cointegration verified

            for i, pair in enumerate(pairs):
                S1 = np.asarray(pair[0])
                S2 = np.asarray(pair[1])
                S1_c = sm.add_constant(S1)

                # Y = bX + c
                # ols: (Y, X)
                results = sm.OLS(S2, S1_c).fit()
                b = results.params[1]

                if b > 0:
                    spread = pair[1] - b * pair[0] # as Pandas Series
                    spread_array = np.asarray(spread) # as array for faster computations

                    stats = check_for_stationarity(spread_array,subsample=subsample)
                    if stats['p_value'] < p_value_threshold:  # verifies required pvalue
                        criteria_not_verified = 'hurst_exponent'

                        hurst_exponent = hurst(spread_array)
                        if hurst_exponent < hurst_threshold:
                            criteria_not_verified = 'half_life'

                            hl = calculate_half_life(spread_array)
                            if (hl >= min_half_life) and (hl < max_half_life):
                                criteria_not_verified = 'mean_cross'

                                zero_cross = zero_crossings(spread_array)
                                if zero_cross >= min_zero_crossings:
                                    criteria_not_verified = 'None'

                                    pair_stats[i] = {'t_statistic': stats['t_statistic'],
                                                        'critical_val': stats['critical_values'],
                                                        'p_value': stats['p_value'],
                                                        'coint_coef': b,
                                                        'zero_cross': zero_cross,
                                                        'half_life': int(round(hl)),
                                                        'hurst_exponent': hurst_exponent,
                                                        'spread': spread,
                                                        'Y_train': pair[1],
                                                        'X_train': pair[0]
                                                        }

    if pair_stats[0] == 0 and pair_stats[1] == 0:
        result = None
        score = 0
        return result, criteria_not_verified,score

    elif pair_stats[0] == 0:  #-> (Y,X), needs to be reversed
        result = 1
    elif pair_stats[1] == 0:
        result = 0
    else: # both combinations are possible
        # select lowest t-statistic as representative test
        if abs(pair_stats[0]['t_statistic']) > abs(pair_stats[1]['t_statistic']):
            result = 0
        else:
            result = 1


    return result, criteria_not_verified, pair_stats[result][pair_ranking_metric]

def find_pairs(data,p_value_threshold=0.1, min_half_life=78, max_half_life=20000,
                   min_zero_crossings=0, hurst_threshold=0.5, subsample=25,
                   num_pairs=4, pair_ranking_metric='p_value',pair_ranking_order=False):
        """
        This function receives a df with the different securities as columns, and aims to find tradable
        pairs within this world. There is a df containing the training data and another one containing test data
        Tradable pairs are those that verify:
            - cointegration
            - minimium half life
            - minimium zero crossings
        :param data_train: df with training prices in columns
        :param data_test: df with testing prices in columns
        :param p_value_threshold:  pvalue threshold for a pair to be cointegrated
        :param min_half_life: minimium half life value of the spread to consider the pair
        :param min_zero_crossings: minimium number of allowed zero crossings
        :param hurst_threshold: mimimium acceptable number for hurst threshold
        :return: pairs that passed test
        """
        n = data.shape[1]
        keys = data.keys()
        pairs_fail_criteria = {'cointegration': 0, 'hurst_exponent': 0, 'half_life': 0, 'mean_cross': 0, 'None': 0}
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                S1 = data[keys[i]]; S2 = data[keys[j]]
                #S1_test = data_test[keys[i]]; S2_test = data_test[keys[j]]
                result, criteria_not_verified, score = check_properties((S1, S2), p_value_threshold, min_half_life, max_half_life,
                                                                      min_zero_crossings, hurst_threshold, subsample,
                                                                      num_pairs=num_pairs, pair_ranking_metric=pair_ranking_metric,pair_ranking_order=pair_ranking_order)
                
                pairs_fail_criteria[criteria_not_verified] += 1
                if result is not None:
                    candidate_pair = None
                    if result==1:
                        candidate_pair=(keys[j], keys[i],score)
                    else:
                        candidate_pair=(keys[i], keys[j],score)
                    if len(pairs)==0:
                        pairs.append(candidate_pair)
                    else:
                        for previous_pair in pairs:
                            if keys[j] in previous_pair or keys[i] in previous_pair:
                                if pair_ranking_order == False:
                                    if candidate_pair[2]<previous_pair[2]:
                                        pairs.remove(previous_pair)
                                        pairs.append(candidate_pair)
                                else:
                                    if candidate_pair[2]>previous_pair[2]:
                                        pairs.remove(previous_pair)
                                        pairs.append(candidate_pair)



        return pairs, pairs_fail_criteria
    
def get_candidate_pairs(clustered_series, pricing_df, min_half_life=78,
                        max_half_life=20000, min_zero_crosings=20, p_value_threshold=0.1, hurst_threshold=0.5,
                        subsample=25,num_pairs=4,pair_ranking_metric='p_value',pair_ranking_order=False):
    """
    This function looks for tradable pairs over the clusters formed previously.
    :param clustered_series: series with cluster label info
    :param pricing_df_train: df with price series from train set
    :param n_clusters: number of clusters
    :param min_half_life: min half life of a time series to be considered as candidate
    :param min_zero_crosings: min number of zero crossings (or mean crossings)
    :param p_value_threshold: p_value to check during cointegration test
    :param hurst_threshold: max hurst exponent value
    :return: list of pairs and its info
    :return: list of unique tickers identified in the candidate pairs universe
    """

    total_pairs, total_pairs_fail_criteria = [], []
    n_clusters = len(clustered_series.value_counts())
    for clust in range(n_clusters):
        sys.stdout.write("\r"+'Cluster {}/{}'.format(clust+1, n_clusters))
        sys.stdout.flush()
        symbols = list(clustered_series[clustered_series == clust].index)
        cluster_pricing_train = pricing_df[symbols]
        pairs, pairs_fail_criteria = find_pairs(cluster_pricing_train,
                                                    p_value_threshold,
                                                    min_half_life,
                                                    max_half_life,
                                                    min_zero_crosings,
                                                    hurst_threshold,
                                                    subsample,
                                                    num_pairs=num_pairs,
                                                    pair_ranking_metric=pair_ranking_metric,
                                                    pair_ranking_order=pair_ranking_order)
        total_pairs.extend(pairs)
        total_pairs_fail_criteria.append(pairs_fail_criteria)

    total_pairs_ranked = sorted(total_pairs, key=lambda tup: tup[2],reverse=pair_ranking_order)[:min(num_pairs,len(total_pairs))]
    total_pairs_ranked = [(p[0],p[1]) for p in total_pairs_ranked]
    print('Found {} pairs'.format(len(total_pairs_ranked)))
    unique_tickers = np.unique([(element[0], element[1]) for element in total_pairs_ranked])
    print('The pairs contain {} unique tickers'.format(len(unique_tickers)))

    # discarded
    review = dict(functools.reduce(operator.add, map(collections.Counter, total_pairs_fail_criteria)))
    print('Pairs Selection failed stage: ', review)

    return total_pairs_ranked, unique_tickers


# Your New Python File
