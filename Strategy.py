from AlgorithmImports import *
from QuantConnect.DataSource import *
from arch.unitroot.cointegration import engle_granger
from pykalman import KalmanFilter
from scipy.optimize import minimize
from functools import partial
import numpy as np
from math import floor
from matplotlib import pyplot as plt
from Select_Pair_main import *
from pandas.plotting import register_matplotlib_converters
from copy import deepcopy
import time
register_matplotlib_converters()#/home/lean-user/workspace/project
# endregion

class KalmanPairsTrading(QCAlgorithm):

    def Initialize(self) -> None:
        '''
        Changeable Parameters for TESTING
        '''
        self.top_k_per_industry = 5 #select the top k equities from an industry based on metric
        #industry_candidate_metric = market_cap   adjust manually
        self.pair_ranking_metric = 'p_value'# other: 'hurst_exponent', 'zero_cross', 'half_life','critical_val'
        self.pair_ranking_order = False # p_value:False, 
        self.num_pairs = 5 #how many number of pairs to include in trading
        self.buffer_size = 0.4 # portion of position to be left as buffer
        #self.Settings.FreePortfolioValuePercentage = 0.1
        self.min_half_life = 1
        self.max_half_life = floor(252)
        self.formation_period = 252*2
        ''''''
        #
        #Recorders for pairs
        self.recalibrated_atleast_once={}
        self.pair_weights={} #record the absolute weight of each pair
        self.pair_trade_states={} #position of each pair, i.e. long/short/none 
        self.pair_hedge_ratio={}

        self.pair_kf_model={}
        self.pair_curr_mean={}
        self.pair_curr_var={}
        self.pair_upper_threshold={}
        self.z_score_trade_threshold=2 #self.GetParameter("entry-z", 2)# to be multipled by upper_threshold
        self.z_score_exit_threshold=0#self.GetParameter("exit-z",0)
        
        
        
        self.initialize = True # a workaround for getting data...
        self.first_time_form_pair = True
        self.null_value_counts = {}
        self.TradeStartYear = self.GetParameter("TradeStartYear",2007)
        self.SetStartDate(self.TradeStartYear, 1, 1)
        self.SetEndDate(self.TradeStartYear+2 ,12, 31)  
        #2. Required: Alpha Streams Models:
        self.SetBrokerageModel(BrokerageName.AlphaStreams)


        #3. Required: Significant AUM Capacity
        self.SetCash(1000000)
        #self.Debug("Initial: total cash ="+str(self.Portfolio.Cash))
        
        
        #4 Set Benchmark
        self.AddEquity("SPY",Resolution.Minute)
        self.SetBenchmark("SPY")


        # Set Scheduled Event Method For Kalman Filter updating.
        self.Schedule.On(self.DateRules.MonthStart(), 
            self.TimeRules.BeforeMarketClose("SPY"), 
            self.Recalibrate)

         #Set Scheduled Event Method For Making Trading Decision.
        
        lagBeforeClose = self.GetParameter("lagBeforeClose",5)
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
            self.TimeRules.BeforeMarketClose("SPY",lagBeforeClose), 
            self.CalculateAndTrade)

        
        #5 Selection of universe
        self.universe = self.AddUniverse(self.SelectCoarse, self.SelectFine)
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetBrokerageModel(AlphaStreamsBrokerageModel())
        #self.SetSecurityInitializer(MySecurityInitializer(self.BrokerageModel, FuncSecuritySeeder(self.GetLastKnownPrices)))
        
    def Initialize_2(self):
        
        self.candidates = [kvp.Key for kvp in 
                        self.UniverseManager[self.universe.Configuration.Symbol].Members]

        for i in self.candidates:
            self.AddEquity(i,Resolution.Hour)
        '''
            Specify which assets' data are of our interest
            assets: the portfolio of pairs
                    list of tuples, [(0a,0b),(0c,0d),(0e,0f)], 
        '''
        self.assets = self.PairFormation()


        
        # Add Equity ------------------------------------------------ 

        for pair in self.assets:
            self.pair_trade_states[pair] = 0 # Set a variable to indicate the trading bias of the portfolio
            self.recalibrated_atleast_once[pair] = False
            self.AddEquity(pair[0],Resolution.Minute)
            self.AddEquity(pair[1],Resolution.Minute)
            self.pair_weights[pair] = 1
        # Instantiate our model
        self.Recalibrate()

        
    def OnData(self,data)->None:
        if self.initialize:
            self.Initialize_2()
            self.initialize=False
    def SelectCoarse(self, coarse):
        #coarse = sorted(coarse, key=lambda c:c.DollarVolume, reverse=True)
        symbols = [c.Symbol for c in coarse if c.HasFundamentalData and c.Price<1500]
        return symbols
    
    
    
    def SelectFine(self, fine):
        candidates = []
        MORNINGSTAR_INDUSTRY_CODES = {  
        10110010: 'Agricultural Inputs',
        10120010: 'Building Materials',  
        10130010: 'Chemicals',  
        10130020: 'Specialty Chemicals',  
        10140010: 'Lumber & Wood Production',  
        10140020: 'Paper & Paper Products',  
        10150010: 'Aluminum',  
        10150020: 'Copper',  
        10150030: 'Other Industrial Metals & Mining',  
        10150040: 'Gold',  
        10150050: 'Silver',  
        10150060: 'Other Precious Metals & Mining',  
        10160010: 'Coking Coal',  
        10160020: 'Steel',  
        10200010: 'Auto & Truck Dealerships',  
        10200020: 'Auto Manufacturers',  
        10200030: 'Auto Parts',  
        10200040: 'Recreational Vehicles',  
        10220010: 'Furnishings, Fixtures & Appliances',  
        10230010: 'Residential Construction',  
        10240010: 'Textile Manufacturing',  
        10240020: 'Apparel Manufacturing',  
        10240030: 'Footwear & Accessories',  
        10250010: 'Packaging & Containers',  
        10260010: 'Personal Services',  
        10270010: 'Restaurants',  
        10280010: 'Apparel Retail',  
        10280020: 'Department Stores',  
        10280030: 'Home Improvement Retail',  
        10280040: 'Luxury Goods',  
        10280050: 'Internet Retail',  
        10280060: 'Specialty Retail',  
        10290010: 'Gambling',  
        10290020: 'Leisure',  
        10290030: 'Lodging',  
        10290040: 'Resorts & Casinos',  
        10290050: 'Travel Services',  
        10310010: 'Asset Management',  
        10320010: 'Banks—Diversified',  
        10320020: 'Banks—Regional',  
        10320030: 'Mortgage Finance',  
        10330010: 'Capital Markets',  
        10330020: 'Financial Data & Stock Exchanges',  
        10340010: 'Insurance—Life',  
        10340020: 'Insurance—Property & Casualty',  
        10340030: 'Insurance—Reinsurance',  
        10340040: 'Insurance—Specialty',  
        10340050: 'Insurance Brokers',  
        10340060: 'Insurance—Diversified',  
        10350010: 'Shell Companies',  
        10350020: 'Financial Conglomerates',  
        10360010: 'Credit Services',  
        10410010: 'Real Estate—Development',  
        10410020: 'Real Estate Services',  
        10410030: 'Real Estate—Diversified',  
        10420010: 'REIT—Healthcare Facilities',  
        10420020: 'REIT—Hotel & Motel',  
        10420030: 'REIT—Industrial',  
        10420040: 'REIT—Office',  
        10420050: 'REIT—Residential',  
        10420060: 'REIT—Retail',  
        10420070: 'REIT—Mortgage',  
        10420080: 'REIT—Specialty',  
        10420090: 'REIT—Diversified',  
        20510010: 'Beverages—Brewers',  
        20510020: 'Beverages—Wineries & Distilleries',  
        20520010: 'Beverages—Non-Alcoholic',  
        20525010: 'Confectioners',  
        20525020: 'Farm Products',  
        20525030: 'Household & Personal Products',  
        20525040: 'Packaged Foods',  
        20540010: 'Education & Training Services',  
        20550010: 'Discount Stores',  
        20550020: 'Food Distribution',  
        20550030: 'Grocery Stores',  
        20560010: 'Tobacco',  
        20610010: 'Biotechnology',  
        20620010: 'Drug Manufacturers—General',  
        20620020: 'Drug Manufacturers—Specialty & Generic',  
        20630010: 'Healthcare Plans',  
        20645010: 'Medical Care Facilities',  
        20645020: 'Pharmaceutical Retailers',  
        20645030: 'Health Information Services',  
        20650010: 'Medical Devices',  
        20650020: 'Medical Instruments & Supplies',  
        20660010: 'Diagnostics & Research',  
        20670010: 'Medical Distribution',  
        20710010: 'Utilities—Independent Power Producers',  
        20710020: 'Utilities—Renewable',  
        20720010: 'Utilities—Regulated Water',  
        20720020: 'Utilities—Regulated Electric',  
        20720030: 'Utilities—Regulated Gas',  
        20720040: 'Utilities—Diversified',  
        30810010: 'Telecom Services',  
        30820010: 'Advertising Agencies',  
        30820020: 'Publishing',  
        30820030: 'Broadcasting',  
        30820040: 'Entertainment',  
        30830010: 'Internet Content & Information',  
        30830020: 'Electronic Gaming & Multimedia',  
        30910010: 'Oil & Gas Drilling',  
        30910020: 'Oil & Gas E&P',  
        30910030: 'Oil & Gas Integrated',  
        30910040: 'Oil & Gas Midstream',  
        30910050: 'Oil & Gas Refining & Marketing',  
        30910060: 'Oil & Gas Equipment & Services',  
        30920010: 'Thermal Coal',  
        30920020: 'Uranium',  
        31010010: 'Aerospace & Defense',  
        31020010: 'Specialty Business Services',  
        31020020: 'Consulting Services',  
        31020030: 'Rental & Leasing Services',  
        31020040: 'Security & Protection Services',  
        31020050: 'Staffing & Employment Services',  
        31030010: 'Conglomerates',  
        31040010: 'Engineering & Construction',  
        31040020: 'Infrastructure Operations',  
        31040030: 'Building Products & Equipment',  
        31050010: 'Farm & Heavy Construction Machinery',  
        31060010: 'Industrial Distribution',  
        31070010: 'Business Equipment & Supplies',  
        31070020: 'Specialty Industrial Machinery',  
        31070030: 'Metal Fabrication',  
        31070040: 'Pollution & Treatment Controls',  
        31070050: 'Tools & Accessories',  
        31070060: 'Electrical Equipment & Parts',  
        31080010: 'Airports & Air Services',  
        31080020: 'Airlines',  
        31080030: 'Railroads',  
        31080040: 'Marine Shipping',  
        31080050: 'Trucking',  
        31080060: 'Integrated Freight & Logistics',  
        31090010: 'Waste Management',  
        31110010: 'Information Technology Services',  
        31110020: 'Software—Application',  
        31110030: 'Software—Infrastructure',  
        31120010: 'Communication Equipment',  
        31120020: 'Computer Hardware',  
        31120030: 'Consumer Electronics',  
        31120040: 'Electronic Components',  
        31120050: 'Electronics & Computer Distribution',  
        31120060: 'Scientific & Technical Instruments',  
        31130010: 'Semiconductor Equipment & Materials',  
        31130020: 'Semiconductors',  
        31130030: 'Solar'  
        }
        for industry in MORNINGSTAR_INDUSTRY_CODES.keys():
            filtered_fine = [x for x in fine if x.AssetClassification.MorningstarIndustryCode == industry]
            filtered_fine = sorted(filtered_fine,key=lambda f: f.MarketCap, reverse=True)
            candidates.extend([f.Symbol for f in filtered_fine[:min(self.top_k_per_industry,len(filtered_fine))]])
        
        return candidates
        
    def relativeWeight(self,pair):
        total=sum([self.pair_weights[p] for p in self.assets])
        return (self.pair_weights[pair]/total)

    
    def PairFormation(self): #-> List[Tuple()]:
        
        df_price = None
        if self.first_time_form_pair == True:
            df_price = self.History(self.candidates, 
                datetime(self.TradeStartYear-5, 1, 1), 
                datetime(self.TradeStartYear-1, 12, 31),
                Resolution.Daily)
        else:
            df_price = self.History(self.candidates, 
                self.formation_period+1,
                Resolution.Daily)
        
        df_price=df_price['close'].unstack(level=0)

        self.Debug(df_price.shape)
        pairs=select_pair(df_price, subsample = 30, 
                min_half_life = self.min_half_life, max_half_life = self.max_half_life,
                min_zero_crosings = 12,   
                 hurst_threshold=0.5,
                 num_pairs = self.num_pairs,
                 pair_ranking_metric = self.pair_ranking_metric,
                 pair_ranking_order=self.pair_ranking_order)
        if len(pairs)==0:
            self.Initialize()
        self.first_time_form_pair=False
        
        return pairs
        

               

    def Recalibrate(self) -> None:
        if self.initialize==True:
            return
        for pair in self.assets:
            self.Recalibrate_pairwise(pair)

    def Recalibrate_pairwise(self,pair):
        start = time.time()
        history = self.History(list(pair), 252*3, Resolution.Daily)
        end = time.time()
        if history.empty: return 
        
        # Select the close column and then call the unstack method
        data = history['close'].unstack(level=0)
        
        # Convert into log-price series to eliminate compounding effect
        log_price = np.log(data)
        
        ### Get Cointegration Vectors
        # pair[0].log_price = BETA* pair[1].log_price + constant
        if log_price.empty==True or log_price.isnull().values.any() or len(log_price.shape)!=2 or log_price.shape[1]<2:
            return
        coint_result = engle_granger(log_price.iloc[:, 0], log_price.iloc[:, 1], trend="c", lags=0)
        coint_vector = coint_result.cointegrating_vector[:2]
        
        # Get the spread: spread = pair[0].log_price - BETA* pair[1].log_price
        spread = log_price @ coint_vector
       
        ### Kalman Filter
        start = time.time()
        self.pair_kf_model[pair] = KalmanFilter(transition_matrices = [1],
                            observation_matrices = [1],
                            initial_state_mean = spread.iloc[:30].mean(),
                            observation_covariance = spread.iloc[:30].var(),
                            em_vars=['transition_covariance', 'initial_state_covariance'])
        self.pair_kf_model[pair] = self.pair_kf_model[pair].em(spread.iloc[:30], n_iter=5)
        (filtered_state_means, filtered_state_covariances) = self.pair_kf_model[pair].filter(spread.iloc[:30])
        
        # Obtain the current Mean and Covariance Matrix expectations.
        self.pair_curr_mean[pair] = filtered_state_means[-1, :]
        self.pair_curr_var[pair] = filtered_state_covariances[-1, :]
        
        # Initialize a mean series for spread normalization using the Kalman Filter's results.
        mean_series = np.array([None]*(spread.shape[0]-30))
        
        # Roll over the Kalman Filter to obtain the mean series.
        '''
        spread:     [0,1,2,3,4,5,...,19,20,21,22,...252*2-1]
                    -> filtered_state_mean & filtered_state_cov
        
        mean_series:[N,N,N,N,N,N,...,N ,20,21,22,...252*2-1]
        '''
        
        for i in range(30, spread.shape[0]):
            (self.pair_curr_mean[pair], self.pair_curr_var[pair]) = self.pair_kf_model[pair].filter_update(filtered_state_mean = self.pair_curr_mean[pair],
                                                                    filtered_state_covariance = self.pair_curr_var[pair],
                                                                    observation = spread.iloc[i])
            mean_series[i-30] = float(self.pair_curr_mean[pair])
        
        # Obtain the normalized spread series.   [x-mean(x)]/std(x)
        normalized_spread = (spread.iloc[30:] - mean_series)  
        
        ### Determine Trading Threshold
        self.pair_upper_threshold[pair] = np.sqrt(self.pair_curr_var[pair])

        ### Hedge Ratio (normalized to sum=1)
        self.pair_hedge_ratio[pair] = coint_vector / np.sum(abs(coint_vector))
        self.recalibrated_atleast_once[pair] = True
        end=time.time()
      
    def CalculateAndTrade(self) -> None:
        if self.initialize==True:
            return
        start = time.time()
        i=0
        for pair in self.assets:
            if self.recalibrated_atleast_once[pair] == True:
                self.PairwiseCalculateAndTrade(pair)
                i+=1
            

        
        end = time.time()
        #self.Debug("Finished making trading decisions for all pairs"+str(end-start))

    '''
    Call this function for each pair.
    pair: a tuple of two Symbol
    '''
    def PairwiseCalculateAndTrade(self, pair):
        start = time.time()
       
       # Get the real-time log close price for all assets and store in a Series
        series = pd.Series()
        for symbol in pair:
            series[symbol] = np.log(self.Securities[symbol].Price)
      
        # Get the spread
        spread = np.sum(series * self.pair_hedge_ratio[pair])
       
        # Update the Kalman Filter with the Series
        (self.pair_curr_mean[pair], self.pair_curr_var[pair]) = self.pair_kf_model[pair].filter_update(filtered_state_mean = self.pair_curr_mean[pair],
                                                                            filtered_state_covariance = self.pair_curr_var[pair],
                                                                            observation = spread)
        
        # Obtain the normalized spread.
        normalized_spread = spread - self.pair_curr_mean[pair]

        self.pair_upper_threshold[pair] = np.sqrt(self.pair_curr_var[pair])
        
        # Mean-reversion
        if self.pair_trade_states[pair]==0 and normalized_spread < -self.z_score_trade_threshold*self.pair_upper_threshold[pair]:
      
            capital = self.relativeWeight(pair)*self.Portfolio.TotalPortfolioValue
            ratio = abs(self.pair_hedge_ratio[pair][1]/self.pair_hedge_ratio[pair][0])
            if ratio>1:
                if abs(floor(capital/self.Portfolio[pair[1]].Price))>=1 and abs(floor(((1/ratio)*capital)/self.Portfolio[pair[0]].Price))>=1:
                    s= self.Sell(pair[1],floor(capital/self.Portfolio[pair[1]].Price))
                    b=self.Buy(pair[0],floor(((1/ratio)*capital)/self.Portfolio[pair[0]].Price))
                    self.pair_trade_states[pair] = 1
            else:
                if abs(floor(ratio*capital/self.Portfolio[pair[1]].Price))>=1 and abs(floor(capital/self.Portfolio[pair[0]].Price))>=1:
                    s=self.Sell(pair[1],floor(ratio*capital/self.Portfolio[pair[1]].Price))
                    b=self.Buy(pair[0],floor(capital/self.Portfolio[pair[0]].Price))
                    self.pair_trade_states[pair] = 1
        
        elif self.pair_trade_states[pair]==0 and normalized_spread > self.z_score_trade_threshold*self.pair_upper_threshold[pair]:

            capital = self.relativeWeight(pair)*self.Portfolio.TotalPortfolioValue
            ratio = abs(self.pair_hedge_ratio[pair][1]/self.pair_hedge_ratio[pair][0])
            if ratio>1:
                if abs(floor(((1/ratio)*capital)/self.Portfolio[pair[0]].Price))>=1 and abs(floor(capital/self.Portfolio[pair[1]].Price))>=1:
                    s = self.Sell(pair[0],floor(((1/ratio)*capital)/self.Portfolio[pair[0]].Price))
                    b=self.Buy(pair[1],floor(capital/self.Portfolio[pair[1]].Price))
                    self.pair_trade_states[pair] = -1
            else:
                if abs(floor(capital/self.Portfolio[pair[0]].Price))>=1 and abs(floor(ratio*capital/self.Portfolio[pair[1]].Price))>=1:
                    
                    s=self.Sell(pair[0],floor(capital/self.Portfolio[pair[0]].Price))
                    b=self.Buy(pair[1],floor(ratio*capital/self.Portfolio[pair[1]].Price))
                    
                    self.pair_trade_states[pair] = -1
            
            
            
                
        # Out of position if spread recovered
        elif (self.pair_trade_states[pair] == 1 and normalized_spread > -self.z_score_exit_threshold*self.pair_upper_threshold[pair]) or (self.pair_trade_states[pair] == -1 and normalized_spread < self.z_score_exit_threshold*self.pair_upper_threshold[pair]):

            self.Liquidate(pair[0]) 
            self.Liquidate(pair[1])   
            self.pair_trade_states[pair] = 0
        
        
        end=time.time()
        #self.Debug("------ Finsihed calculating Spreads and trading, takes"+str(end-start))
                    
        
        #self.Debug("----------------------------------------------")
