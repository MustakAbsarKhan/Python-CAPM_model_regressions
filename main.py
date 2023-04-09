#importing libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm

#? loading data from csv file
data = pd.read_csv('ab24dc77d9e26451f5734801f223fc88data.csv')
# data = file data

#? extracting data to variables
BCS_returns = data['BCS'].pct_change() * 100
ULVR_returns = data['ULVR'].pct_change() * 100
BCS_stock_returns = BCS_returns.dropna() #IN %
ULVR_stock_returns = ULVR_returns.dropna() #IN %
Rf = data['Rf']
Rm = data['Rm'].dropna()

#? Calculating Beta For Each Stock
#BCS
BCS_covariance = np.cov(BCS_stock_returns, Rm)[0, 1]
BCS_market_variance = np.var(Rm)
BCS_Beta = BCS_covariance /  BCS_market_variance

#ULVR
ULVR_covariance = np.cov(ULVR_stock_returns, Rm)[0, 1]
ULVR_market_variance = np.var(Rm)
ULVR_Beta = ULVR_covariance /  ULVR_market_variance

#? Running CAPM model for each stock
RiskPremium = (Rm - Rf).mean()
BCS_Er = (Rf + (RiskPremium) * BCS_Beta).mean()
ULVR_Er = (Rf + (RiskPremium) * ULVR_Beta).mean()

print('The Risk Premium is:', "{:.2%}".format(RiskPremium))
print("BCS Er: ", "{:.2%}".format(BCS_Er))
print("ULVR Er: ", "{:.2%}".format(ULVR_Er))

print("--------------------------------------------")
print("--------------------------------------------")


#summery of regressions of two stocks

#? Defining the dependent and independent variables
y_BCS = (data['BCS'] - data['Rf']).dropna()
x_BCS = (data['Rm'] - data['Rf']).dropna()

y_ULVR = (data['ULVR'] - data['Rf']).dropna()
x_ULVR = (data['Rm'] - data['Rf']).dropna()

#? Running the CAPM model and print the summary of regressions
model_BCS = sm.OLS(y_BCS, sm.add_constant(x_BCS))
result_BCS = model_BCS.fit()
print("Summary of Regression for BCS:")
print(result_BCS.summary())

print("-------------------------------------------")

model_ULVR = sm.OLS(y_ULVR, sm.add_constant(x_ULVR))
result_ULVR = model_ULVR.fit()
print("Summary of Regression for ULVR:")
print(result_ULVR.summary())

#! Analysis
#? Based on the regression results, neither BCS nor ULVR have a statistically significant relationship with the market (0). However, ULVR has a higher R-squared value (0.012) than BCS (0.002), which means that ULVR's performance is slightly more correlated with the market compared to BCS. Therefore, in an upward market, ULVR may have a slightly better performance than BCS.
#? Similarly, neither BCS nor ULVR have a statistically significant relationship with the market in a downward direction. However, based on the coefficients, BCS has a lower intercept value (15.7525) than ULVR (1402.2554), which means that BCS may hold its value better than ULVR in a downward market. However, this conclusion should be taken with caution as the relationship is not statistically significant.