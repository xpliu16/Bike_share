## Forecasting hourly demand at a bike sharing service

Bike shares provide a convenient way to utilize an environmentally-friendly form of mobility without needing to own or store a bike. In order for bike shares to be effective, bikes need to be made available at the appropriate place and time. One way trips and the cost of transporting bikes make this logistic task both difficult and critical. A first step is to understand demand for bikes. In perusing the data sets available on the UC Irvine Machine Learning Repository (UCI Machine Learning Repository), I came across a [dataset from a Seoul bike share](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand) that recorded bikes rented and weather factors and holidays at hourly resolution for one year. 

I found this data to be simultaneously quite intuitive while also being poorly behaved for linear regression, providing impetus for me to explore spline regression, generalized linear models, and random forest regressors.

### Initial considerations
The response variable of interest is bikes rented. There is likely no feedback between bikes rented and the weather or holidays, so those can be viewed as purely independent or exogenous. Weather variable may have a time lagged effect (e.g., it may take some time after heavy rains before people head back out). Rental is highly likely to have a daily as well as seasonal effect.

### Potential challenges
1.	**Multicollinearity of features:** “season” is likely related to “temperature”, “date”, and “snow”. After learning more about the weather variables, it seems “solar radiation” is also likely related to “season” and “rain”. “Rain”, “humidity”, and “dew point” are also related to each other. Since we care more about prediction than interpretation (we cannot easily control the weather or holidays), some multicollinearity is okay. However, we need to be careful when tempted to interpret the effect of adding a variable or the value of coefficients due to its correlation and redundancy with other informative variables.

2.	**Nonlinear relationships:** An easy one to see is the upside-down U-shaped effect of temperature – weather that is too cold or too hot are both not conducive to biking. On the dependent variable side, “bikes rented” can only be positive. Given that this is a count-type variable that is arguably a Poisson process, I chose a Poisson generalized linear model (GLM) with a log link function.   

3.	**Multiplicative rather than additive effects:** For instance, if it was raining heavily, “rented bikes” would drop very low, regardless of other variables. A non-functioning day for the bike share also took “rented bikes” to zero, regardless of the value of other variables. This also created heteroskedasticity – when it wasn’t raining there was a large dispersion due to other factors, but when it was raining, there was little variance. The Poisson GLM model is able to accommodate this with the log link converting multiplication to addition.

4.	**Interactions:** Dew point (roughly related to the feeling of “mugginess”) was significantly negatively correlated with bike rentals in the heat of summer, but not when considered across all seasons (in fact, summer dew points tend to be higher than winter, and summer bike usage is also higher; see 1) above). Wind seemed to be a bigger factor in winter. Solar radiation might also potentially be a positive factor in mild weather but become a negative factor when temperatures rise. 

### Data preprocessing
The CSV file did not use UTF-8 encoding and could not be opened up by a default call to pd.read_csv. Using print on the opened file showed file information indicating a European encoding called “cp1252” which could be passed as an argument to read_csv. The European DD/MM date order was dealt with by setting “dayfirst = True” for read_csv.  
The day of the week was not provided and had to be inferred from the dates. 
```
import pandas as pd

with open('data\\SeoulBikeData.csv') as f:
    print(f)
df = pd.read_csv('C:\\Users\\Ping\\Desktop\\Project\\Bike_share\\data\\SeoulBikeData.csv', encoding = 'cp1252', dayfirst=True)

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Day of Week'] = df['Date'].dt.day_name()

df.describe(include='all')

print(df['Hour'][df['Functioning Day']=='No'].count())
print(df['Date'][df['Functioning Day']=='No'].nunique())
print(df['Date'][df['Functioning Day']=='No'].unique())
df['Rented Bike Count'][df['Functioning Day']=='No'].sum()
```
There appeared to be no missing data. However, 295 hours distributed over 13 days were explicitly labeled as non-functioning, during which no bikes were rented. These effective missing data points could be imputed or discarded, with discarding possibly causing problems for the regular period of the time series. 

### Regression by inspection
I examined the daily pattern of bike rentals. As expected, there was a robust daily pattern with high levels of usage during the day but not the night. This pattern was different between work days and non-work days, with the former showing two peaks around rush hour.

<img src="Daily patterns all.png" alt="Daily pattern all">

After normalizing the maximum for each rental day, we see that the daily pattern for Saturdays, Sundays, and holidays are indistinguishable (holidays have the largest SEM bars because it is the smallest sample). 

<img src="Nonwork normalized.png" alt="Daily pattern normalized for non-work days">

So I created an average normalized profile for working days and non-working days:

<img src="Daily pattern work nonwork.png" alt="Daily pattern profiles for workdays and non-workdays">

Assuming a multiplicative effect, I divided all the data by these profiles to de-seasonalize the daily effects. One could also deseason annually, but there is only one year's worth of data, making it less reliable (smoothing would have to be used rather than averaging) and impossible to validate. Also, the annual effect may be largely accounted for by season, temperature, and holiday variables. 

After removing the daily effect, I looked at the relationship between temperature and bikes rented. Both really cold and really hot weather reduced bike usage, creating a nonlinear relationship. After pruning off non-functioning periods, I experimented with cubic splines and B-splines to find parameters allowing a reasonable fit of the temperature relationship.
```
x = df2['Temperature(°C)']
y = df2['de_daily']

transformed_x1 = dmatrix("bs(df2['Temperature(°C)'], df=7)",
                        {'df2.temp':df2['Temperature(°C)']}, return_type='dataframe')

mod = sm.GLM(df2['de_daily'], transformed_x1, family=sm.families.Poisson(link=sm.families.links.log))
res = mod.fit()
y_hat = res.predict()
```
<img src="Bikes rented versus temperature.png" alt="Bike demand versus temperature relationship and spline fit">

A cluster of points had low bike usage unexplained by weather and may relate to other conditions such as high humidity, rain, or very overcast conditions.

Rain was also a major predictor, with few bikes rented when rain was heavy and high variability when there was little or no rain. That variability may or may not be properly accounted for by other predictors – it was still apparent when looking at the amount of rain versus the residual of the temperature fit. The  Poisson GLM should account for some of this behavior where the variance at large mean values is larger than the variance at small mean values. 

<img src="Bikes rented resid versus log rain.png" alt="Bike demand versus rain relationship after temperature fit">

I also explored the other weather predictors with scatterplots and correlations and found some weak relationships, such as a negative relationship with humidity in the summer and overall negative relationship with wind that was stronger in the winter. Surprisingly, visibility was a poor predictor and although very low solar radiation potentially explained some points with atypically low bike usage, it did not have a consistent overall effect.
          
All of these including a dew point-summer season interaction were put together into a final GLM regression on the daily-normalized bike rentals. To make the final prediction, I multiplied by the daily pattern for working or non-working days. 
          
Up to this point, the data had not been treated like a series. Inspired by distributed-lag time regression models, I explored whether adding some history would help. For instance, rain could have a lasting effect as people may not rush out right after rain passes, roads may still be slick and littered with puddles and bike seats wet. In fact, the cross-correlation between bikes rented and rainfall showed a decaying negative relationship that lasted almost 12 hours.

<img src="Rain xcorr.png" alt="Crosscorrelation between bikes rented and rainfall">

I was reluctant to add too many free parameters, but incremental improvements in pseudo-R2 were seen for adding rain history for 4 preceding time steps. Indeed, the coefficients for those terms were negative and decayed with increasing lag. In a more complex model, one might model this as a linearly or exponentially decaying history dependence. Rain also likely has autocorrelations and thus this could conceivably be converted to an ARDL model with lots of exogenous variables and more parameters. As is, the model already performs fairly well on the data. 

<img src="GLM y_actual y_pred.png" alt="Actual versus predicted">
<img src="GLM pred time series.png" alt="Actual versus predicted time series">

### Random Forest Regressor
For comparison, I input the original data to the Scikit-learn random forest regressor. Random forests classifiers train decision trees on random subsets of the data and average the predictions across these trees. This approach can be adapted to continuous regression and can handle nonlinear relationships. 

### Performance evaluation
The random forest regressor, which required pretty low effort and time, performed surprisingly well, achieving a MAE of 59 (with most bikes rented data ranging from 0 to 4000) and R<sup>2</sup> of 0.97 (this metric does not necessarily have the same implications for GLM as for OLS, but is useful for comparing performance e). However, with access to so many free parameters, this could be overfitting. When I split out a 20% testing data set, the performance of the random forest dropped to 0.85. Increasing the number of estimators to the default 100 improved this to 0.87, and further improvements are possible with hyperparameter tuning. The custom regression performed similarly on the training and test data sets at an R<sup>2</sup> of 0.89, took longer, was more subjective, but offered more insight and interpretability.

In the [original paper accompanying this dataset](https://www.tandfonline.com/doi/full/10.1080/22797254.2020.1725789) the authors compared CUBIST, Regularized Random Forest, K-Nearest Neighbors, and Conditional Inference Tree. They found the rule-based CUBIST model could explain about 95% of the variance in the Seoul bike share data set, with temperature and hour of the day being the biggest factors, consistent with our exploratory analysis. It’s quite impressive that there are many machine learning methods able to produce great predictions on difficult data sets. 

### Final thoughts
1.	One of the times with atypically low bike usage relative to prediction was the afternoon of April 15th, 2018. News suggests that Seoul was enshrouded in fine dust and sand blown from China, with a health advisory issued when PM10 exceeded threshold for 2 hours: https://www.upi.com/Top_News/World-News/2018/04/15/Fine-dust-levels-soar-in-South-Korea/5581523776231/
In places like California and the Pacific Northwest, fluctuating pollution levels (e.g., wild fires) should certainly be included in this type of model, as outdoor physical activity would be discouraged at those times. 

2.	Bike rides may fall into clusters with different behavior, such as recreation versus commuting. We may be able to infer this based on repeated daily travel paths, travel on recreational paths, or based on rush hour versus non-rush hour separation. Leisure biking may be more elastic in response to weather predictors.




