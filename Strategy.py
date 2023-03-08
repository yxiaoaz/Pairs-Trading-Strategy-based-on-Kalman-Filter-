# region imports
from AlgorithmImports import *
from arch.unitroot.cointegration import engle_granger
from pykalman import KalmanFilter
from scipy.optimize import minimize
from statsmodels.tsa.vector_ar.vecm import VECM
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
#from Select_Pair_main import *
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# endregion

class KalmanPairsTrading(QCAlgorithm):

    '''
    TODO: 
            PairFormation-> List[Tuple()], generate the pairs
            Incorporate weights into CalculateAndTrade
    '''
    def Initialize(self, weight_scheme="committed") -> None:

        #1. Required: Five years of backtest history
        self.SetStartDate(2014, 1, 1)

        #2. Required: Alpha Streams Models:
        self.SetBrokerageModel(BrokerageName.AlphaStreams)

        self.weight_scheme = weight_scheme

        #3. Required: Significant AUM Capacity
        self.SetCash(1000000)
        self.Debug("Initial: total cash ="+str(self.Portfolio.Cash))
        self.AddEquity("SPY",Resolution.Minute)
        #4. Required: Benchmark to SPY
        self.SetBenchmark("SPY")
        
        '''
            Specify which assets' data are of our interest
            assets: the portfolio of pairs
                    list of tuples, [(0a,0b),(0c,0d),(0e,0f)], 
        '''
        self.assets = self.PairFormation(universe=1)
        self.pair_capitals={}
        self.pair_weights={}
        self.pair_trade_states={}
        self.pair_hedge_ratio={}

        self.pair_kf_model={}
        self.pair_curr_mean={}
        self.pair_curr_var={}
        self.pair_upper_threshold={}
        # Add Equity ------------------------------------------------ 
        '''
        for i in range(len(self.assets)):
            self.AddEquity(self.assets[i], Resolution.Minute)
        '''
        for pair in self.assets:
            self.AddEquity(pair[0], Resolution.Minute)
            self.AddEquity(pair[1], Resolution.Minute)
            self.pair_trade_states[pair] = 0 # Set a variable to indicate the trading bias of the portfolio
            if weight_scheme=="committed":
                self.pair_weights[pair] = 1
                self.pair_capitals[pair] = (1/len(self.assets))*self.Portfolio.Cash
            
        # Instantiate our model
        self.Recalibrate()
        
        # Set Scheduled Event Method For Kalman Filter updating.
        self.Schedule.On(self.DateRules.WeekStart(), 
            self.TimeRules.At(0, 0), 
            partial(self.Recalibrate,x='2'))

        # Set Scheduled Event Method For Making Trading Decision.
        self.Schedule.On(self.DateRules.EveryDay(), 
            self.TimeRules.BeforeMarketClose("DBA"), 
            self.CalculateAndTrade)
    
    def PairFormation(self,universe): #-> List[Tuple()]:
        #TODO

        return [('DBA','CGW'),('SHY','SPY')]


    def Recalibrate(self,x='1') -> None:
        for pair in self.assets:
            self.Recalibrate_pairwise(pair)
    
    def Recalibrate_pairwise(self,pair):
        history = self.History(list(pair), 252*2, Resolution.Daily)
        if history.empty: return 
        
        # Select the close column and then call the unstack method
        data = history['close'].unstack(level=0)
        
        # Convert into log-price series to eliminate compounding effect
        log_price = np.log(data)
        
        ### Get Cointegration Vectors
        # Get the cointegration vector
        coint_result = engle_granger(log_price.iloc[:, 0], log_price.iloc[:, 1], trend="c", lags=0)
        coint_vector = coint_result.cointegrating_vector[:2]
        
        # Get the spread
        spread = log_price @ coint_vector
        
        ### Kalman Filter
        '''
        Initialize a Kalman Filter. Using the first 20 data points to optimize its initial state. 
        We assume the market has no regime change so that the transitional matrix and observation matrix is [1].
        '''
        self.pair_kf_model[pair] = KalmanFilter(transition_matrices = [1],
                            observation_matrices = [1],
                            initial_state_mean = spread.iloc[:20].mean(),
                            observation_covariance = spread.iloc[:20].var(),
                            em_vars=['transition_covariance', 'initial_state_covariance'])
        self.pair_kf_model[pair] = self.pair_kf_model[pair].em(spread.iloc[:20], n_iter=5)
        (filtered_state_means, filtered_state_covariances) = self.pair_kf_model[pair].filter(spread.iloc[:20])
        
        # Obtain the current Mean and Covariance Matrix expectations.
        self.pair_curr_mean[pair] = filtered_state_means[-1, :]
        self.pair_curr_var[pair] = filtered_state_covariances[-1, :]
        
        # Initialize a mean series for spread normalization using the Kalman Filter's results.
        mean_series = np.array([None]*(spread.shape[0]-20))
        
        # Roll over the Kalman Filter to obtain the mean series.
        '''
        spread:     [0,1,2,3,4,5,...,19,20,21,22,...252*2-1]
                    -> filtered_state_mean & filtered_state_cov
        
        mean_series:[N,N,N,N,N,N,...,N ,20,21,22,...252*2-1]
        '''

        for i in range(20, spread.shape[0]):
            (self.pair_curr_mean[pair], self.pair_curr_var[pair]) = self.pair_kf_model[pair].filter_update(filtered_state_mean = self.pair_curr_mean[pair],
                                                                    filtered_state_covariance = self.pair_curr_var[pair],
                                                                    observation = spread.iloc[i])
            mean_series[i-20] = float(self.pair_curr_mean[pair])
        
        # Obtain the normalized spread series.   [x-mean(x)]/std(x)
        normalized_spread = (spread.iloc[20:] - mean_series)  
        
        ### Determine Trading Threshold
        self.pair_upper_threshold[pair]= np.sqrt(self.pair_curr_var[pair])

        ### Hedge Ratio (normalized to sum=1)
        self.pair_hedge_ratio[pair] = coint_vector / np.sum(abs(coint_vector))

    
    def CalculateAndTrade(self) -> None:
        for pair in self.assets:
            self.PairwiseCalculate(pair, self.pair_capitals[pair])
        

    '''
    Call this function for each pair.
    pair: a tuple of two Symbol
    '''
    def PairwiseCalculate(self, pair, capital):
        # Get the real-time log close price for all assets and store in a Series
        series = pd.Series()
        for symbol in pair:
            series[symbol] = np.log(self.Securities[symbol].Close)
            
        # Get the spread
        spread = np.sum(series * self.pair_hedge_ratio[pair])
        
        # Update the Kalman Filter with the Series
        (self.pair_curr_mean[pair], self.pair_curr_var[pair]) = self.pair_kf_model[pair].filter_update(filtered_state_mean = self.pair_curr_mean[pair],
                                                                            filtered_state_covariance = self.pair_curr_var[pair],
                                                                            observation = spread)
            
        # Obtain the normalized spread.
        normalized_spread = spread - self.pair_curr_mean[pair]

        # ==============================
        
        # Mean-reversion
        if normalized_spread < -self.pair_upper_threshold[pair]:
            self.Debug("Situation a:")
            self.Debug("The normalized_spread between "+pair[0]+" and "+pair[1]+" is "+normalized_spread)
            self.Debug("The quantity to long for "+pair[0]+" is "+str(self.pair_hedge_ratio[pair][0]*(capital/self.Portfolio.Cash)))
            self.Debug("The quantity to short for "+pair[1]+" is -"+str(self.pair_hedge_ratio[pair][1]*(capital/self.Portfolio.Cash)))
            self.MarketOrder(pair[0], int(self.pair_hedge_ratio[pair][0]*(capital/self.Portfolio.Cash)), True)
            self.MarketOrder(pair[1], int(-self.pair_hedge_ratio[pair][1]*(capital/self.Portfolio.Cash)), True )
                #orders.append(PortfolioTarget(self.asset_list[i], self.trading_weight[i]))
                #self.SetHoldings(orders)
                
            self.pair_trade_states[pair] = 1
                
        elif normalized_spread > self.pair_upper_threshold[pair]:
            #orders = []
            #for i in range(len(self.assets)):
                #orders.append(PortfolioTarget(self.asset_list[i], -1 * self.trading_weight[i]))
                #self.SetHoldings(orders)
            self.Debug("Situation b:")
            self.Debug("The normalized_spread between "+pair[0]+" and "+pair[1]+" is "+normalized_spread)
            self.Debug("The quantity to long for "+pair[1]+" is "+str(self.pair_hedge_ratio[pair][1]*(capital/self.Portfolio.Cash)))
            self.Debug("The quantity to short for "+pair[0]+" is -"+str(self.pair_hedge_ratio[pair][0]*(capital/self.Portfolio.Cash)))
            self.MarketOrder(pair[0], int(-self.pair_hedge_ratio[pair][0]*100), True)
            self.MarketOrder(pair[1], int(self.pair_hedge_ratio[pair][1]*100), True )
            self.pair_trade_states[pair] = -1
                
        # Out of position if spread recovered
        elif self.pair_trade_states[pair] == 1 and normalized_spread > -self.pair_upper_threshold[pair] or self.pair_trade_states[pair] == -1 and normalized_spread < self.pair_upper_threshold[pair]:
            #self.Liquidate()
            self.Debug("Situation b:")
            self.SetHoldings([PortfolioTarget(pair[0], 0), PortfolioTarget(pair[1], 0)])
            self.pair_trade_states[pair] = 0
