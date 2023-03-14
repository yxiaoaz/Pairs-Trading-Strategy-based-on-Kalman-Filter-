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
import time
register_matplotlib_converters()
# endregion
class MySecurityInitializer(BrokerageModelSecurityInitializer):

    def __init__(self, brokerage_model: IBrokerageModel, security_seeder: ISecuritySeeder) -> None:
        super().__init__(brokerage_model, security_seeder)

    def Initialize(self, security: Security) -> None:
        # First, call the superclass definition
        # This method sets the reality models of each security using the default reality models of the brokerage model
        super().Initialize(security)

        # Next, overwrite some of the reality models        
        security.SetBuyingPowerModel(SecurityMarginModel(4))
class KalmanPairsTrading(QCAlgorithm):
    '''
    1. Shorter backtesting time
    2. BeforeMarketClose
    3. Cap on stock price e.g. <1500
    4. Order Type 
     
    '''
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
        self.num_pairs = 5 #how many number of pairs to include in trading
        self.buffer_size = 0.4 # portion of position to be left as buffer
        self.Settings.FreePortfolioValuePercentage = 0.1
        self.min_half_life = 1
        self.max_half_life = floor(252)
        self.formation_period = 252*2
        self.shuffle_pair_interval_month = 6 # intervals between two shuffling of pairs
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
        self.z_score_trade_threshold= self.GetParameter("entry-z", 2)# to be multipled by upper_threshold
        self.z_score_exit_threshold=self.GetParameter("exit-z",0)
        
        
        
        self.initialize = True # a workaround for getting data...
        self.first_time_form_pair = True
        self.null_value_counts = {}
        #1. Backtesting starting point
        self.SetStartDate(2016, 1, 1)
        self.SetEndDate(2019 ,12, 31)  
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
        
        
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
            self.TimeRules.BeforeMarketClose("SPY",5), 
            self.CalculateAndTrade)
        '''
        self.Schedule.On(self.DateRules.EveryDay("SPY"),
                 self.TimeRules.AfterMarketOpen("SPY",120),
                 self.CalculateAndTrade)

        '''
        
        
        # Set Scheduled Event Method For Rebalancing Weights (only for weight_scheme = "value-weighted")
        
        '''
        self.Schedule.On(self.DateRules.WeekEnd(),
            self.TimeRules.AfterMarketOpen("SPY"),
            self.RebalanceWeights)
        '''
        # Set Scheduled Event Method For Reforming Pairs.
        '''
        self.month_passed = 0
        self.Schedule.On(self.DateRules.MonthStart(),
            self.TimeRules.AfterMarketOpen("SPY"),
            self.ShufflePairs)
        '''
        
        
        #5 Selection of universe
        #self.universe =self.AddUniverse(self.SelectCoarse)
        self.Debug("Universe selection starts")
        start = time.time()
        self.universe = self.AddUniverse(self.SelectCoarse, self.SelectFine)
        end = time.time()
        self.Debug("Universe filtering ends, takes "+str(end-start)+" seconds")
        self.UniverseSettings.Resolution = Resolution.Minute
        self.SetBrokerageModel(AlphaStreamsBrokerageModel())
        #self.SetSecurityInitializer(MySecurityInitializer(self.BrokerageModel, FuncSecuritySeeder(self.GetLastKnownPrices)))
        self.to_be_removed = []  # Stack for storing securities to be removed from universe between two reshufflings
        
    def Initialize_2(self):
        
        self.candidates = [kvp.Key for kvp in 
                        self.UniverseManager[self.universe.Configuration.Symbol].Members]
        #self.Debug(self.candidates)
        #self.Debug("===================================Refomed Pairs===================================")
        #self.Log("[")
        for i in self.candidates:
            #self.Log("\'"+i.Value+"\'"+",")
            self.AddEquity(i,Resolution.Hour)
        #self.Log("]")
        '''
            Specify which assets' data are of our interest
            assets: the portfolio of pairs
                    list of tuples, [(0a,0b),(0c,0d),(0e,0f)], 
        '''
        self.Debug("Pair formation starts")
        start = time.time()
        self.assets = self.PairFormation()
        end = time.time()
        self.Debug("Pair formation ends, takes "+str(end-start)+" seconds")

        
        # Add Equity ------------------------------------------------ 
        '''
        for i in range(len(self.assets)):
            self.AddEquity(self.assets[i],  inute)
        '''
        for pair in self.assets:
            #self.AddEquity(pair[0], Resolution.Minute)
            #self.AddEquity(pair[1], Resolution.Minute)
            self.pair_trade_states[pair] = 0 # Set a variable to indicate the trading bias of the portfolio
            self.recalibrated_atleast_once[pair] = False
            self.AddEquity(pair[0],Resolution.Minute)
            self.AddEquity(pair[1],Resolution.Minute)
            self.pair_weights[pair] = 1
            self.null_value_counts[pair] = 0
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
    
    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
        for security in changes.RemovedSecurities:
            self.to_be_removed.append(security.Symbol)
    '''
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
    '''
    
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
        # for "committed": only need to call once
        total=sum([self.pair_weights[p] for p in self.assets])#*(1-self.buffer_size)
        #self.Debug("The relative weight of pair:"+pair[0]+" "+pair[1]+" is "+str(self.pair_weights[pair]/total))
        return (self.pair_weights[pair]/total)
    
    '''
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
    '''
    
    
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
                Resolution.Daily)
        else:
            df_price = self.History(self.candidates, 
                self.formation_period+1,
                Resolution.Daily)
        
        df_price=df_price['close'].unstack(level=0)
        '''
        Debug
        '''
        #self.Debug("The resolution used for pair formation is:")
        indice = df_price.head(20).index.tolist()
        #for i in indice:
            #self.Debug(i)
        ''''''
        #self.Debug("Shape of df_price before pair selection: "+str(df_price.shape[0])+" x "+str(df_price.shape[1]))
        self.Debug(df_price.shape)
        pairs=select_pair(df_price, subsample = 30, 
                min_half_life = self.min_half_life, max_half_life = self.max_half_life,
                min_zero_crosings = 12,   
                 hurst_threshold=0.5,
                 num_pairs = self.num_pairs,
                 pair_ranking_metric = self.pair_ranking_metric,
                 pair_ranking_order=self.pair_ranking_order)
        if len(pairs)==0:
            #self.Debug("!!!!!!!No Pairs are Found!!!!!!!")
            self.top_k_per_industry+=10
            self.Initialize()
        #self.Debug("-------------Pairs formed! ------------------")
        #self.Debug("The length of the pairs collection is:"+str(len(pairs)))
        #for i in pairs:
            #self.Debug("Pair: "+i[0]+"  "+i[1]+'/n')
        
        
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
        start = time.time()
        self.Debug("Recalibration of Kalman starts")
        i=0
        for pair in self.assets:
            
            #if self.recalibrated_atleast_once[pair]==False:
                #self.Debug("First time recalibration for: "+pair[0]+" "+pair[1])
            start1=time.time()
            self.Debug("--Recab for pair "+str(i)+" starts")
            self.Recalibrate_pairwise(pair)
            end1=time.time()
            i+=1
        end = time.time()
        self.Debug("Recalibration for all ends, takes "+str(end-start)+" seconds")

    
    def Recalibrate_pairwise(self,pair):
        self.Debug("----Retrieving history of past 3 years")
        start = time.time()
        history = self.History(list(pair), 252*3, Resolution.Daily)
        end = time.time()
        self.Debug("----Retrieval completes, takes "+str(end-start)+" seconds")
        if history.empty: return 
        
        # Select the close column and then call the unstack method
        data = history['close'].unstack(level=0)
        
        # Convert into log-price series to eliminate compounding effect
        log_price = np.log(data)
        
        ### Get Cointegration Vectors
        # pair[0].log_price = BETA* pair[1].log_price + constant
        if log_price.empty==True or log_price.isnull().values.any() or len(log_price.shape)!=2 or log_price.shape[1]<2:
            #self.Debug("&&&&&&&&&&&&& NAN or table is EMPTY&&&&&&&&&&&&&&&")
            self.null_value_counts[pair]=self.null_value_counts[pair]+1
            '''
            if self.null_value_counts[pair]>=10:
                self.Liquidate(pair[0])
                self.Liquidate(pair[1])
                self.assets.remove(pair)
                del self.pair_trade_states[pair]
                del self.recalibrated_atleast_once[pair]
                del self.pair_weights[pair]
                del self.pair_hedge_ratio[pair]
                del self.pair_kf_model[pair]
                del self.pair_curr_mean[pair]
                del self.pair_curr_var[pair]
                del self.pair_upper_threshold[pair]
                '''
            return
        coint_result = engle_granger(log_price.iloc[:, 0], log_price.iloc[:, 1], trend="c", lags=0)
        coint_vector = coint_result.cointegrating_vector[:2]
        
        # Get the spread: spread = pair[0].log_price - BETA* pair[1].log_price
        spread = log_price @ coint_vector
        #self.Debug("log_price.shape:")
        #self.Debug(log_price)
        #self.Debug("coint_vector.shape:")
        #self.Debug(coint_vector.shape)
        ### Kalman Filter
        '''
        Initialize a Kalman Filter. Using the first 20 data points to optimize its initial state. 
        We assume the market has no regime change so that the transitional matrix and observation matrix is [1].
        '''
        self.Debug("----Initializing a KF model")
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
        self.Debug("----Finished initializing a KF model, takes "+str(end-start)+" seconds")
    def CalculateAndTrade(self) -> None:
        if self.initialize==True:
            return
        self.Debug("Making trading decisions")
        start = time.time()
        i=0
        for pair in self.assets:
            if self.recalibrated_atleast_once[pair] == True:
                self.Debug("--Starts making trading decision for pair "+ str(i))
                start1=time.time()
                self.PairwiseCalculateAndTrade(pair)
                end1=time.time()
                self.Debug("--Finished making trading decision for this pair, takes "+str(end1-start1))
                i+=1
            

        
        end = time.time()
        self.Debug("Finished making trading decisions for all pairs"+str(end-start))

    '''
    Call this function for each pair.
    pair: a tuple of two Symbol
    '''
    def PairwiseCalculateAndTrade(self, pair):
        start = time.time()
        self.Debug("------ Calculating Spreads and trading")
        # Get the real-time log close price for all assets and store in a Series
       # Get the real-time log close price for all assets and store in a Series
        series = pd.Series()
        for symbol in pair:
            series[symbol] = np.log(self.Securities[symbol].Price)
        #self.Debug("xxxxxxx")
        #self.Debug(series.shape)
        #self.Debug("-----------------Current pair: "+pair[0]+" "+pair[1]+" -----------------------------")
        #self.Debug("The normalized coint vectors of the pair are: "+str(self.pair_hedge_ratio[pair][0])+" and "+str(self.pair_hedge_ratio[pair][1]))
        
        # Get the spread
        spread = np.sum(series * self.pair_hedge_ratio[pair])
        #self.Debug(spread)
        #self.Debug(log_price.shape)
        #self.Debug(self.pair_hedge_ratio[pair].shape)
        #self.Debug(np.reshape(self.pair_hedge_ratio[pair],(2,1)).shape)
        
        #self.Debug("-----------------Current pair: "+pair[0]+" "+pair[1]+" -----------------------------")
        #self.Debug("The normalized coint vectors of the pair are: "+str(self.pair_hedge_ratio[pair][0])+" and "+str(self.pair_hedge_ratio[pair][1]))
        # Get the spread
        #self.Debug("serires.shape="+str(series.shape[0])+"x"+str(series.shape[1]))
        #self.Debug("hedge ratio.shape="+str(self.pair_hedge_ratio[pair].shape[0])+"x"+str(self.pair_hedge_ratio[pair].shape[1]))
        
        # Update the Kalman Filter with the Series
        (self.pair_curr_mean[pair], self.pair_curr_var[pair]) = self.pair_kf_model[pair].filter_update(filtered_state_mean = self.pair_curr_mean[pair],
                                                                            filtered_state_covariance = self.pair_curr_var[pair],
                                                                            observation = spread)
        
        # Obtain the normalized spread.
        normalized_spread = spread - self.pair_curr_mean[pair]
        #self.Debug("The normalized spread is: "+str(spread)+" - "+str(self.pair_curr_mean[pair])+" = "+str(normalized_spread))
        # ==============================
        self.pair_upper_threshold[pair] = np.sqrt(self.pair_curr_var[pair])
        #self.Debug("The std of the spread is: "+str(self.pair_upper_threshold[pair]))
        #self.Debug("Deviation from mean spread is (z-score): "+str(normalized_spread/self.pair_upper_threshold[pair]))
        #self.Debug("The current state of this pair is "+str(self.pair_trade_states[pair]))
        # Mean-reversion
        if self.pair_trade_states[pair]==0 and normalized_spread < -self.z_score_trade_threshold*self.pair_upper_threshold[pair]:
           
            #self.Debug("------>State changed to "+str(1))
            #self.Debug("Long ")
            #self.Debug(pair[0])
            #self.Debug("Short")
            #self.Debug(pair[1])
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
            
            
            #pair_0_weight + pair_1_weight = weight
            #abs(pair_0_weight)/abs(pair_1_weight)=ratio
            # 
            '''
            pair_0_weight = (self.relativeWeight(pair)*self.pair_hedge_ratio[pair][1])/(self.pair_hedge_ratio[pair][1]+self.pair_hedge_ratio[pair][0])
            pair_1_weight = self.relativeWeight(pair)-pair_0_weight
            if pair_0_weight<0:
                pair_0_weight, pair_1_weight = pair_1_weight, pair_0_weight
            self.pair_trade_states[pair] = 1
            self.Debug("Long 0 short 1")
            self.Debug("pair_0_weight = ")
            self.Debug(pair_0_weight)
            self.Debug("pair_1_weight = ")
            self.Debug(pair_1_weight)
            self.SetHoldings(pair[0],pair_0_weight)
            self.SetHoldings(pair[1],pair_1_weight)
            '''
            '''
            if self.weight_scheme == "equally weighted":
                self.pair_weights[pair]=1
            #pair_0_qty = int(self.CalculateOrderQuantity(pair[0], self.relativeWeight(pair)))
            pair_0_qty = floor(self.relativeWeight(pair)*self.Portfolio.TotalPortfolioValue - self.Portfolio[pair[0]].HoldingsValue)/self.Portfolio[pair[0]].Price
            ratio = abs(self.pair_hedge_ratio[pair][1]/self.pair_hedge_ratio[pair][0])
            pair_1_qty = floor(pair_0_qty*ratio)
            if abs(pair_0_qty)>=1 and abs(pair_1_qty)>=1:
                self.Debug("Situation a:")
                self.Debug("----Do Trade----")
                self.Debug("The quantity to long for "+pair[0]+" is "+str(pair_0_qty))
                self.Debug("The quantity to short for "+pair[1]+" is "+str(pair_1_qty))
                short_ticket = self.MarketOrder(pair[1], -pair_1_qty)#self.Sell(pair[1], pair_1_qty)
                long_ticket = self.MarketOrder(pair[0], pair_0_qty)#self.Buy(pair[0], pair_0_qty)
                    #orders.append(PortfolioTarget(self.asset_list[i], self.trading_weight[i]))
                    #self.SetHoldings(orders)
                self.Debug("ID of short order:"+str(short_ticket.OrderId))
                self.Debug("ID of long order:"+str(long_ticket.OrderId))
                self.pair_trade_states[pair] = 1
            '''    
        elif self.pair_trade_states[pair]==0 and normalized_spread > self.z_score_trade_threshold*self.pair_upper_threshold[pair]:
            
            #self.Debug("------>State changed to "+str(-1))
            #self.Debug("Long ")
            #self.Debug(pair[1])
            #self.Debug("Short")
            #self.Debug(pair[0])
            capital = self.relativeWeight(pair)*self.Portfolio.TotalPortfolioValue
            #self.Debug("The assigned capital is:")
            #self.Debug(capital)
            ratio = abs(self.pair_hedge_ratio[pair][1]/self.pair_hedge_ratio[pair][0])
            if ratio>1:
                if abs(floor(((1/ratio)*capital)/self.Portfolio[pair[0]].Price))>=1 and abs(floor(capital/self.Portfolio[pair[1]].Price))>=1:
                    #self.Debug("Try to SHORT ")
                    #self.Debug(pair[0])
                    #self.Debug("for "+str((1/ratio)*capital)+"the quantity should be "+str(floor(((1/ratio)*capital)/self.Portfolio[pair[0]].Price)))
                    s = self.Sell(pair[0],floor(((1/ratio)*capital)/self.Portfolio[pair[0]].Price))
                    #self.Debug("Actual quantity filled is "+str(s.QuantityFilled))
                    #self.Debug("Short ordered, the new cash book account is")
                    #self.Debug(self.Portfolio.Cash)

                   
                    #self.Debug("Try to LONG ")
                    #self.Debug(pair[1])
                    #self.Debug("for "+str(capital)+"the quantity should be "+str(floor(capital/self.Portfolio[pair[1]].Price)))
                    b=self.Buy(pair[1],floor(capital/self.Portfolio[pair[1]].Price))
                    #self.Debug("Actual quantity filled is "+str(b.QuantityFilled))
                    #self.Debug("Long ordered, the new cash book account is")
                    #self.Debug(self.Portfolio.Cash)
                    self.pair_trade_states[pair] = -1
            else:
                if abs(floor(capital/self.Portfolio[pair[0]].Price))>=1 and abs(floor(ratio*capital/self.Portfolio[pair[1]].Price))>=1:
                    #self.Debug("Try to SHORT ")
                    #self.Debug(pair[0])
                    #self.Debug("for "+str(capital)+"the quantity should be "+str(floor(capital/self.Portfolio[pair[0]].Price)))
                    s=self.Sell(pair[0],floor(capital/self.Portfolio[pair[0]].Price))
                    #self.Debug("Actual quantity filled is "+str(s.QuantityFilled))
                    #self.Debug("Short ordered, the new cash book account is")
                    #self.Debug(self.Portfolio.Cash)

                    #self.Debug("Try to LONG ")
                    #self.Debug(pair[1])
                    #self.Debug("for "+str(ratio*capital)+"the quantity should be "+str(floor(ratio*capital/self.Portfolio[pair[1]].Price)))
                    b=self.Buy(pair[1],floor(ratio*capital/self.Portfolio[pair[1]].Price))
                    #self.Debug("Actual quantity filled is "+str(b.QuantityFilled))
                    #self.Debug("Long ordered, the new cash book account is")
                    #self.Debug(self.Portfolio.Cash)
                    self.pair_trade_states[pair] = -1
            

            '''
            pair_1_weight = (self.relativeWeight(pair)*self.pair_hedge_ratio[pair][0])/(self.pair_hedge_ratio[pair][0]+self.pair_hedge_ratio[pair][1])
            pair_0_weight = self.relativeWeight(pair)-pair_1_weight
            if pair_1_weight<0:
                pair_0_weight, pair_1_weight = pair_1_weight, pair_0_weight
            self.SetHoldings(pair[0],pair_0_weight)
            self.SetHoldings(pair[1],pair_1_weight)
            self.Debug("Long 1 short 0")
            self.pair_trade_states[pair] = -1
            self.Debug("pair_0_weight = ")
            self.Debug(pair_0_weight)
            self.Debug("pair_1_weight = ")
            self.Debug(pair_1_weight)
            '''
            '''
            #orders = []
            #for i in range(len(self.assets)):
                #orders.append(PortfolioTarget(self.asset_list[i], -1 * self.trading_weight[i]))
                #self.SetHoldings(orders)
            
            if self.weight_scheme == "equally weighted":
                self.pair_weights[pair]=1
            #pair_1_qty = int(self.CalculateOrderQuantity(pair[1], self.relativeWeight(pair)))
            pair_1_qty = floor(self.relativeWeight(pair)*self.Portfolio.TotalPortfolioValue - self.Portfolio[pair[1]].HoldingsValue)/self.Portfolio[pair[1]].Price
            ratio = abs(self.pair_hedge_ratio[pair][0]/self.pair_hedge_ratio[pair][1])
            pair_0_qty = floor(ratio*pair_1_qty)
            if abs(pair_0_qty)>=1 and abs(pair_1_qty)>=1:
                self.Debug("Situation b:")
                self.Debug("----Do Trade----")
                self.Debug("The quantity to long for "+pair[1]+" is "+str(pair_1_qty))
                self.Debug("The quantity to short for "+pair[0]+" is "+str(pair_0_qty))
                short_ticket=self.MarketOrder(pair[0], -pair_0_qty)#self.Sell(pair[0], pair_0_qty)
                long_ticket=self.MarketOrder(pair[1], pair_1_qty)#self.Buy(pair[1], pair_1_qty)
                self.Debug("ID of short order:"+str(short_ticket.OrderId))
                self.Debug("ID of long order:"+str(long_ticket.OrderId))
                self.pair_trade_states[pair] = -1
            '''
            
            
                
        # Out of position if spread recovered
        elif (self.pair_trade_states[pair] == 1 and normalized_spread > -self.z_score_exit_threshold*self.pair_upper_threshold[pair]) or (self.pair_trade_states[pair] == -1 and normalized_spread < self.z_score_exit_threshold*self.pair_upper_threshold[pair]):
            #self.Liquidate()
            #self.Debug("Situation c:")
            #self.Debug("!!!Liquidate state changed to ------>"+str(0))
            self.Liquidate(pair[0]) 
            self.Liquidate(pair[1])   
            self.pair_trade_states[pair] = 0
        
        
        end=time.time()
        self.Debug("------ Finsihed calculating Spreads and trading, takes"+str(end-start))
                    
        
        #self.Debug("----------------------------------------------")

        #"committed"-> equal weight for all pair
                                         #"equally weighted" -> equal weight among OPEN pairs
                                         #"value weighted" ->  wpt=wpt–1(1+rpt–1) for ALL pairs
