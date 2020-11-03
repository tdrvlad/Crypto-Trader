# Crypto-Trader
Bot that uses **Binance client API** to get real-time prices for crypto-currencies and make transactions based on available data.
## `local_market`
Module that manages data comprehension. It can be used for downloading hours of price variations for later tests. It handles 3 use-cases:

 - **Real-time data** (live mode) - when the `get_price_data` method is called, the current price for the exchange of currencies is returned,
 - **Offline data** (test mode) - when previously saved real-time data is locally available, the `get_price_data` returns the exchange price without delays. This is useful for instantly running tests for strategies.
 - **Random-generated data** (test mode) - when there is no previously saved data locally available, the `get_price_data` returns random values in the interval of the `[0;1]` with a default variation etween samples of  `0.01`
### `market_analyser`
This module allows for visualization of the available data.
## `trader`
When the trader runs it gets instant price data chunks (the last *n* samples of price data) by calling the `local_market` methods and places orders depending on the result of the strategy of choice. 
##  `strategies`  
The module will contain possible strategies of trading elaborated by the user. A strategy has to follow the structure of the `BasicStrategy`, having a predefined number of samples (this will dictate the number of samples pulled from the `local_market` data) and a `get_choice` method that computes:
 - an action: 
	 - 1 for buying, -1 for selling or *None*
 - an amount
 
 The strategy of choice is settled with the command:
 `ChosenStrategy = BasicStrategy`
