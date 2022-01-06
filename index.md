## Forecasting hourly demand at a bike share

Bike shares provide a convenient way to utilize an environmentally-friendly form of mobility without needing to own or store a bike. In order for bike shares to be effective, bikes need to be made available at the appropriate place and time. One way trips and the cost of transporting bikes make this logistic task both difficult and critical. A first step is to understand demand for bikes. In perusing the data sets available on the UC Irvine Machine Learning Repository (UCI Machine Learning Repository), I came across a [dataset from a Seoul bike share](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand) that recorded bikes rented and weather factors and holidays at hourly resolution for one year. 

I found this data to be simultaneously quite intuitive while also being poorly behaved for linear regression, providing impetus for me to explore spline regression, generalized linear models, and random forest regressors.

### Initial considerations
The response variable of interest is bikes rented. There is likely no feedback between bikes rented and the weather or holidays, so those can be viewed as purely independent or exogenous. Weather variable may have a time lagged effect (e.g., it may take some time after heavy rains before people head back out). Rental is highly likely to have a daily as well as seasonal effect.

### Potential challenges
1.	**Multicollinearity of features:** “season” is likely related to “temperature”, “date”, and “snow”. After learning more about the weather variables, it seems “solar radiation” is also likely related to “season” and “rain”. “Rain”, “humidity”, and “dew point” are also related to each other. Since we care more about prediction than interpretation (we cannot easily control the weather or holidays), some multicollinearity is okay. However, we need to be careful when tempted to interpret the effect of adding a variable or the value of coefficients due to its correlation and redundancy with other informative variables.

2.	**Nonlinear relationships:** An easy one to see is the upside-down U-shaped effect of temperature – weather that is too cold or too hot are both not conducive to biking. On the dependent variable side, “bikes rented” can only be positive. Given that this is a count-type variable that is arguably a Poisson process, I chose a Poisson generalized linear model (GLM) with a log link function.   

3.	**Multiplicative rather than additive effects:** For instance, if it was raining heavily, “rented bikes” would drop very low, regardless of other variables. A non-functioning day for the bike share also took “rented bikes” to zero, regardless of the value of other variables. This also created heteroskedasticity – when it wasn’t raining there was a large dispersion due to other factors, but when it was raining, there was little variance. The Poisson GLM model is able to accommodate this with the log link converting multiplication to addition.

4.	**Interactions:** Dew point (roughly related to the feeling of “mugginess”) was significantly negatively correlated with bike rentals in the heat of summer, but not when considered across all seasons (in fact, summer dew points tend to be higher than winter, and summer bike usage is also higher; see 1) above). Wind seemed to be a bigger factor in winter. Solar radiation might also potentially be a positive factor in mild weather but become a negative factor when temperatures rise. 

## Data processing: 
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

## Regression by inspection: 
I examined the daily pattern of bike rentals. As expected, there was a robust daily pattern with high levels of usage during the day but not the night. This pattern was different between work days and non-work days, with the former showing two peaks around rush hour.

![Image](src)


