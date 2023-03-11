# region imports
from AlgorithmImports import *
from QuantConnect.DataSource import *
from arch.unitroot.cointegration import engle_granger
from pykalman import KalmanFilter
from scipy.optimize import minimize
from statsmodels.tsa.vector_ar.vecm import VECM
from functools import partial
import numpy as np
from math import floor
from matplotlib import pyplot as plt
from Select_Pair_main import *
from pandas.plotting import register_matplotlib_converters
from copy import deepcopy
register_matplotlib_converters()
# endregion

class KalmanPairsTrading(QCAlgorithm):

    '''
    TODO: 
            PairFormation-> List[Tuple()], generate the pairs
            Incorporate weights into CalculateAndTrade
    '''
    def Initialize(self) -> None:
        '''
        Changeable Parameters for TESTING
        '''
        self.top_k_per_industry = 5 #select the top k equities from an industry based on metric
        #industry_candidate_metric = market_cap   adjust manually
        self.pair_ranking_metric = 'p_value'# other: 'hurst_exponent', 'zero_cross', 'half_life','critical_val'
        self.pair_ranking_order = False # p_value:False, 
        self.num_pairs = 4 #how many number of pairs to include in trading
        self.weight_scheme = "value weighted" #"committed"-> equal weight for all pair
                                         #"equally weighted" -> equal weight among OPEN pairs
                                         #"value weighted" ->  wpt=wpt–1(1+rpt–1) for ALL pairs
        self.buffer_size = 0.4 # portion of position to be left as buffer
        self.Settings.FreePortfolioValuePercentage = 0.1
        self.min_half_life = 1
        self.max_half_life = 252 #floor(252*0.5)
        self.formation_period = 252*2
        self.shuffle_pair_interval_month = 6 # intervals between two shuffling of pairs
        ''''''
        #Recorders for pairs
        self.recalibrated_atleast_once={}
        self.pair_weights={} #record the absolute weight of each pair
        self.pair_trade_states={} #position of each pair, i.e. long/short/none 
        self.pair_hedge_ratio={}

        self.pair_kf_model={}
        self.pair_curr_mean={}
        self.pair_curr_var={}
        self.pair_upper_threshold={}
        self.z_score_threshold=2 # to be multipled by upper_threshold
        
        
        self.initialize = True # a workaround for getting data...
        self.first_time_form_pair = True
        #1. Backtesting starting point
        self.SetStartDate(2021, 1, 1)

        #2. Required: Alpha Streams Models:
        self.SetBrokerageModel(BrokerageName.AlphaStreams)


        #3. Required: Significant AUM Capacity
        self.SetCash(1000000)
        self.Debug("Initial: total cash ="+str(self.Portfolio.Cash))
        
        #4 Set Benchmark
        self.AddEquity("SPY",Resolution.Minute)
        self.SetBenchmark("SPY")


        # Set Scheduled Event Method For Kalman Filter updating.
        self.Schedule.On(self.DateRules.WeekStart(), 
            self.TimeRules.At(0, 0), 
            self.Recalibrate)

        # Set Scheduled Event Method For Making Trading Decision.
        self.Schedule.On(self.DateRules.EveryDay(), 
            self.TimeRules.BeforeMarketClose("SPY"), 
            self.CalculateAndTrade)

        # Set Scheduled Event Method For Rebalancing Weights (only for weight_scheme = "value-weighted")
        self.Schedule.On(self.DateRules.WeekEnd(),
            self.TimeRules.AfterMarketOpen("SPY"),
            self.RebalanceWeights)

        # Set Scheduled Event Method For Reforming Pairs.
        self.month_passed = 0
        self.Schedule.On(self.DateRules.MonthStart(),
            self.TimeRules.AfterMarketOpen("SPY"),
            self.ShufflePairs)
        
        #5 Selection of universe
        #self.universe =self.AddUniverse(self.SelectCoarse)
        self.universe = self.AddUniverse(self.SelectCoarse, self.SelectFine)
        self.to_be_removed = []  # Stack for storing securities to be removed from universe between two reshufflings
        
    def Initialize_2(self):
        
        self.candidates = [kvp.Key for kvp in 
                        self.UniverseManager[self.universe.Configuration.Symbol].Members]
        #self.Debug(self.candidates)
        self.Debug("===================================Refomed Pairs===================================")
        self.Log("[")
        for i in self.candidates:
            self.Log("\'"+i.Value+"\'"+",")
        self.Log("]")
        '''
            Specify which assets' data are of our interest
            assets: the portfolio of pairs
                    list of tuples, [(0a,0b),(0c,0d),(0e,0f)], 
        '''
        self.assets = self.PairFormation()
        
        # Add Equity ------------------------------------------------ 
        '''
        for i in range(len(self.assets)):
            self.AddEquity(self.assets[i], Resolution.Minute)
        '''
        for pair in self.assets:
            #self.AddEquity(pair[0], Resolution.Minute)
            #self.AddEquity(pair[1], Resolution.Minute)
            self.pair_trade_states[pair] = 0 # Set a variable to indicate the trading bias of the portfolio
            self.recalibrated_atleast_once[pair] = False
            self.pair_weights[pair] = 1
        # Instantiate our model
        self.Recalibrate()

        
    def OnData(self,data)->None:
        if self.initialize:
            self.Initialize_2()
            self.initialize=False
    def SelectCoarse(self, coarse):
        #coarse = sorted(coarse, key=lambda c:c.DollarVolume, reverse=True)
        symbols = [c.Symbol for c in coarse if c.HasFundamentalData]
        return symbols
    
    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
        for security in changes.RemovedSecurities:
            self.to_be_removed.append(security.Symbol)

    def ShufflePairs(self):
        self.month_passed+=1

        #check if it's the right time to shuffle pairs
        if self.month_passed == self.shuffle_pair_interval_month:
            old_assets = deepcopy(self.assets) # temporarily store the pairs of last formation period
            self.Initialize_2() # reshuffle self.assets

            #liquidate existing pairs
            for pair in self.assets:
                if pair in old_assets:
                    continue
                else:
                    self.pair_trade_states[pair] = 0 # Set a variable to indicate the trading bias of the portfolio
                    self.recalibrated_atleast_once[pair] = False
                    self.pair_weights[pair] = 1
            
            # for old pairs that are not in position, delete their records
            for old_pair in old_assets:
                
                if old_pair not in self.assets:
                    if self.pair_trade_states[old_pair]==0:
                        self.Debug("Pair reformed, old pairs that are not in position need to be dumped!")
                        self.Debug("Liquidate "+old_pair[0]+", "+old_pair[1])
                        self.Liquidate(old_pair[0])
                        self.Liquidate(old_pair[1])
                        del self.pair_trade_states[old_pair]
                        del self.recalibrated_atleast_once[old_pair]
                        del self.pair_weights[old_pair]
                        del self.pair_hedge_ratio[old_pair]
                        del self.pair_kf_model[old_pair]
                        del self.pair_curr_mean[old_pair]
                        del self.pair_curr_var[old_pair]
                        del self.pair_upper_threshold[old_pair]
                    else:
                        self.assets.append(old_pair)
                        if old_pair[0] in self.to_be_removed:
                            self.Debug("!!!!!!!!!!!!!!Manually add "+old_pair[0]+" to universe !!!!!!!!!!!!!!!!!!")
                            self.AddEquity(old_pair[0],Resolution.Daily)
                        if old_pair[1] in self.to_be_removed:    
                            self.Debug("!!!!!!!!!!!!!!Manually add "+old_pair[1]+" to universe !!!!!!!!!!!!!!!!!!")
                            self.AddEquity(old_pair[1],Resolution.Daily)

    
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
        total=sum([self.pair_weights[p] for p in self.assets])*(1-self.buffer_size)
        return (self.pair_weights[pair]/total)
    
    # for "value-weighted" weight scheme only
    # rebalance the weight of each pair by *(1+Weekly return)
    # Weekly return = Position*Weekly price change of both assets in the pair
    def RebalanceWeights(self):
        if self.weight_scheme=="value weighted":
            self.Debug("--==--==--==--==--==Rebalancing Weight since we are in value weighted mode--==--==--==--==--==")
            for pair in self.assets:
                price_table = self.History([pair[0],pair[1]], 5,Resolution.Daily)['close'].unstack(level=0)
                weighted_return = ((self.Portfolio[pair[0]].Quantity * self.Portfolio[pair[0]].Price) /self.Portfolio.TotalPortfolioValue)*((price_table[pair[0]].iloc[-1]-price_table[pair[0]].iloc[-5])/price_table[pair[0]].iloc[-5])+((self.Portfolio[pair[1]].Quantity * self.Portfolio[pair[1]].Price) /self.Portfolio.TotalPortfolioValue)*((price_table[pair[1]].iloc[-1]-price_table[pair[1]].iloc[-5])/price_table[pair[1]].iloc[-5])
                self.pair_weights[pair]=self.pair_weights[pair]*(1+weighted_return)
    def PairFormation(self): #-> List[Tuple()]:
        #TODO
        #csv = self.Download('https://www.dropbox.com/s/jtq7rgea7gyfdml/intraday_etfs.csv?dl=0')
        '''
        energy_ETF=["VDE", "USO", "XES", "XOP", "UNG", "ICLN", "ERX","UCO", "AMJ", "BNO", "AMLP", "UGAZ", "TAN","ERY", "SCO", "DGAZ"]
        metal_ETF=["GLD", "IAU", "SLV", "GDX", "AGQ", "PPLT", "NUGT", "USLV", "UGLD", "JNUG","DUST", "JDST"]
        tech_ETF=["QQQ", "IGV", "QTEC", "FDN", "FXL", "TECL", "SOXL", "SKYY", "KWEB","TECS", "SOXS"]
        treasury_ETF=["IEF", "SHY", "TLT", "IEI", "TLH", "BIL", "SPTL","TMF", "SCHO", "SCHR", "SPTS", "GOVT","SHV", "TBT", "TBF", "TMV"]
        volatility_ETF=["TVIX", "VIXY", "SPLV", "UVXY", "EEMV", "EFAV", "USMV","SVXY"]
        '''
        '''
        universe = [ 'XLK', 'QQQ', 'BANC', 'BBVA', 'BBD', 'BCH', 'BLX', 'BSBR', 'BSAC', 'SAN',
                    'CIB', 'BXS', 'BAC', 'BOH', 'BMO', 'BK', 'BNS', 'BKU', 'BBT','NBHC', 'OFG',
                    'BFR', 'CM', 'COF', 'C', 'VLY', 'WFC', 'WAL', 'WBK','RBS', 'SHG', 'STT', 'STL', 'SCNB', 'SMFG', 'STI']

        for i in universe:
            self.AddEquity(i,Resolution.Daily)
        '''
        df_price = None
        if self.first_time_form_pair == True:
            df_price = self.History(self.candidates, 
                datetime(2014, 1, 1), 
                datetime(2020, 12, 31), 
                Resolution.Daily)
        else:
            df_price = self.History(self.candidates, 
                self.formation_period+1,
                Resolution.Daily)
            
        df_price=df_price['close'].unstack(level=0)
        self.Debug("Shape of df_price before pair selection: "+str(df_price.shape[0])+" x "+str(df_price.shape[1]))
        self.Debug(df_price.shape)
        pairs=select_pair(df_price, subsample = 30, 
                min_half_life = self.min_half_life, max_half_life = self.max_half_life,
                min_zero_crosings = 12,   
                 hurst_threshold=0.5,
                 num_pairs = self.num_pairs,
                 pair_ranking_metric = self.pair_ranking_metric,
                 pair_ranking_order=self.pair_ranking_order)
        if len(pairs)==0:
            self.Debug("!!!!!!!No Pairs are Found!!!!!!!")
            self.top_k_per_industry+=10
            #self.Initialize()
        self.Debug("-------------Pairs formed! ------------------")
        self.Debug("The length of the pairs collection is:"+str(len(pairs)))
        for i in pairs:
            self.Debug("Pair: "+i[0]+"  "+i[1]+'/n')
        
        
        self.first_time_form_pair=False
        
        return pairs
        
        
        '''return[('DGP', 'SLV', 1), ('DGP', 'UGL', 1), ('DBP', 'DGL', 0), ('DBP', 'USV', 1), ('DGL', 'USV', 1), 
            ('DBA', 'DBB', 0), 
            ('DBA', 'UAG', 0),  ('DBB', 'GRU', 1),
             ('DBB', 'RJA', 1), ('FUD', 'FUE', 0),
              ('FUE', 'UCI', 1), ('GCC', 'GRU', 0), 
             ('GCC', 'RJA', 0), ('GCC', 'RJZ', 0), ('UBG', 'UCI', 1), ('DBE', 'GSG', 1), ('DBE', 'RJN', 1)]
        '''

        #return [('SPY','SHY',1),('DGP', 'SLV', 1)]
               

    def Recalibrate(self) -> None:
        if self.initialize==True:
            return
        for pair in self.assets:
            if self.recalibrated_atleast_once[pair]==False:
                self.Debug("First time recalibration for: "+pair[0]+" "+pair[1])
            self.Recalibrate_pairwise(pair)
    
    def Recalibrate_pairwise(self,pair):
        history = self.History(list(pair), 252*2, Resolution.Daily)
        if history.empty: return 
        
        # Select the close column and then call the unstack method
        data = history['close'].unstack(level=0)
        
        # Convert into log-price series to eliminate compounding effect
        log_price = np.log(data)
        
        ### Get Cointegration Vectors
        # pair[0].log_price = BETA* pair[1].log_price + constant
        coint_result = engle_granger(log_price.iloc[:, 0], log_price.iloc[:, 1], trend="c", lags=0)
        coint_vector = coint_result.cointegrating_vector[:2]
        
        # Get the spread: spread = pair[0].log_price - BETA* pair[1].log_price
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
        self.recalibrated_atleast_once[pair] = True
    
    def CalculateAndTrade(self) -> None:
        if self.initialize==True:
            return
        for pair in self.assets:
            if self.recalibrated_atleast_once[pair] == True:
                self.PairwiseCalculateAndTrade(pair)
        

    '''
    Call this function for each pair.
    pair: a tuple of two Symbol
    '''
    def PairwiseCalculateAndTrade(self, pair):
        # Get the real-time log close price for all assets and store in a Series
        series = pd.Series()
        for symbol in pair:
            series[symbol] = np.log(self.Securities[symbol].Close)
        self.Debug("-----------------Current pair: "+pair[0]+" "+pair[1]+" -----------------------------")
        self.Debug("The normalized coint vectors of the pair are: "+str(self.pair_hedge_ratio[pair][0])+" and "+str(self.pair_hedge_ratio[pair][1]))
        # Get the spread
        spread = np.sum(series * self.pair_hedge_ratio[pair])
        
        # Update the Kalman Filter with the Series
        (self.pair_curr_mean[pair], self.pair_curr_var[pair]) = self.pair_kf_model[pair].filter_update(filtered_state_mean = self.pair_curr_mean[pair],
                                                                            filtered_state_covariance = self.pair_curr_var[pair],
                                                                            observation = spread)
        
        # Obtain the normalized spread.
        normalized_spread = spread - self.pair_curr_mean[pair]
        self.Debug("The normalized spread is: "+str(spread)+" - "+str(self.pair_curr_mean[pair])+" = "+str(normalized_spread))
        # ==============================
        
        # Mean-reversion
        if normalized_spread < -self.z_score_threshold*self.pair_upper_threshold[pair]:
            self.Debug("Situation a:")
            #self.Debug(self.Securities[pair[0]].Close)
            if self.weight_scheme == "equally weighted":
                self.pair_weights[pair]=1
            pair_0_qty = int(self.CalculateOrderQuantity(pair[0], self.relativeWeight(pair)))
            ratio = abs(self.pair_hedge_ratio[pair][1]/self.pair_hedge_ratio[pair][0])
            pair_1_qty = floor(pair_0_qty*ratio)

            self.Debug("The quantity to long for "+pair[0]+" is "+str(pair_0_qty))
            self.Debug("The quantity to short for "+pair[1]+" is "+str(pair_1_qty))
            
            self.Debug("----Do Trade----")
            
            self.Sell(pair[1], pair_1_qty)
            self.Buy(pair[0], pair_0_qty)
                #orders.append(PortfolioTarget(self.asset_list[i], self.trading_weight[i]))
                #self.SetHoldings(orders)
                
            self.pair_trade_states[pair] = 1
                
        elif normalized_spread > self.z_score_threshold*self.pair_upper_threshold[pair]:
            #orders = []
            #for i in range(len(self.assets)):
                #orders.append(PortfolioTarget(self.asset_list[i], -1 * self.trading_weight[i]))
                #self.SetHoldings(orders)
            self.Debug("Situation b:")
            if self.weight_scheme == "equally weighted":
                self.pair_weights[pair]=1
            pair_1_qty = int(self.CalculateOrderQuantity(pair[1], self.relativeWeight(pair)))
            ratio = abs(self.pair_hedge_ratio[pair][0]/self.pair_hedge_ratio[pair][1])
            pair_0_qty = floor(ratio*pair_1_qty)

            self.Debug("The quantity to long for "+pair[1]+" is "+str(pair_1_qty))
            self.Debug("The quantity to short for "+pair[0]+" is "+str(pair_0_qty))
            
            self.Debug("----Do Trade----")
            self.Sell(pair[0], pair_0_qty)
            self.Buy(pair[1], pair_1_qty)
            self.pair_trade_states[pair] = -1
                
        # Out of position if spread recovered
        elif (self.pair_trade_states[pair] == 1 and normalized_spread > -self.pair_upper_threshold[pair]) or (self.pair_trade_states[pair] == -1 and normalized_spread < self.pair_upper_threshold[pair]):
            #self.Liquidate()
            self.Debug("Situation c:")
            self.Liquidate(pair[0]) 
            self.Liquidate(pair[1])   
            self.pair_trade_states[pair] = 0
            if self.weight_scheme == "equally weighted":
                self.pair_weights[pair]=0
        self.Debug("----------------------------------------------")
