# Set up the libraries
import pandas as pd
import requests 
from datetime import datetime, timedelta
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# We need to forecast the electricity price for alberta
# Extract Data 
    # Elecricty data: use API key https://api.aeso.ca/web/api/register
        #Through API

    # Parameters for URL
def url_extract_electricity_data(start_date = '02282020', end_date = '02282021'):

    # Date range check "Manually"

    start_date_datetime = datetime.strptime(start_date, '%m%d%Y')
    end_date_datetime = datetime.strptime(end_date, '%m%d%Y')
    time_delta = end_date_datetime - start_date_datetime

    # Date range check "Conditional statement"
    if time_delta.days <= 366:
        print(f"The selected timeframe is { time_delta.days }" )

    elif time_delta.days >= 366:
        print("Date range is over 365 days of data, going to pull multiple timeframes")
        # count = int(time_delta.days / 366)
        current_start = start_date_datetime
        current_end = start_date_datetime + timedelta(days=365)
        
        data_collector = []

        while current_end < end_date_datetime:
            current_time_delta = end_date_datetime - current_end
            print(current_end, current_time_delta.days )

            # Calulate if we need to do a whole year or less
            if current_time_delta.days >= 365:
                print("Using Full Year")
                current_time_delta = timedelta(days=365)
            else:
                print(f"Not Full Year: { current_time_delta.days } ")
                current_time_delta = timedelta(days=current_time_delta.days)
            
            #go and get the data from start date to current end
            data_collector.append(go_take_data_url(current_start, current_end))
            current_start=current_end
            current_end= current_end + current_time_delta
            print(current_start, current_end, current_time_delta)
            
        return pd.concat(data_collector)

        #Through URL

    # Location
    # Weather
# Clean Data
# Plot the data and find trend or correlation
# Feature Engineering : Adding variables needed
# Run the model
    # Backtest the model

def go_take_data_url(start_date, end_date):
    report_type = 'HistoricalPoolPriceReportServlet' # Type of report
    content_type = 'html' # Type of content expected from the AESO server
    
    start_date = start_date.strftime('%m%d%Y')
    end_date = end_date.strftime('%m%d%Y')

    # URL for web scrape
    url = 'http://ets.aeso.ca/ets_web/ip/Market/Reports/{}?beginDate={}&endDate={}&contentType={}'.format(report_type, start_date, end_date, content_type)
    print(url)

    ''' The placeholders {} in the URL string will be replaced with the values of report_type, 
    start_date, end_date, and content_type
    '''

    # Get HTML content using the requests library
    source = requests.get(url).text  # The .text attribute extracts the HTML content as plain text

    # Wrap HTML content in StringIO object
    html_buffer = StringIO(source)

    # Parse the HTML content (stored in source) into a list of DataFrames 
    df_list = pd.read_html(html_buffer)  # pd.read_html() searches for table elements and creates a df for each of them
    print('number of tables =',  len(df_list)) 

    # Create a DataFrame containing the data stored on table #2 (Index = 1) in the HTML
    df = df_list[1]
    
    '''#csv file creation
    file_name = 'aeso_{}_{}_to_{}.csv'.format(report_type, start_date, end_date)
    file_path = 'Data/{}'.format(file_name)
    df.to_csv(file_path)
    '''
    

    # Display the DataFrame
    return df

#population
def population():
    data  = pd.read_csv(r'Data\population.csv')
    print(data)
    
    melted_pop = pd.melt(data,id_vars=['Year'], var_name='Quarter', value_name='Population')
    melted_pop.sort_values(by=['Year','Quarter'],inplace=True)
    melted_pop.reset_index(drop=True, inplace=True)

    
    melted_pop['YearQuarter'] = melted_pop['Year'].astype(str) + melted_pop['Quarter'] 
    # Drop the original 'Year' and 'Quarter' columns if needed
    melted_pop.drop(columns=['Year', 'Quarter'], inplace=True)
    
    melted_pop['YearQuarter'] = pd.to_datetime(melted_pop['YearQuarter'])
    melted_pop.set_index('YearQuarter', inplace=True)
    # Merge using asof merge
    print('Total to Pass', melted_pop)
    
    return melted_pop

    
#feature engineering
def engineering_features(data):
    return (data
        .drop(data.columns[0],axis=1)
            .assign(**{"Date (HE)": data["Date (HE)"].str.replace("*", "", regex=False)})  # Remove asterisks
            .assign(Date=lambda x: pd.to_datetime(x['Date (HE)'].str[:-3], format='%m/%d/%Y'),  # Extract and convert date
                    Hour=lambda x: x['Date (HE)'].str[-2:].astype(int))  # Extract hour as integer
            .drop(columns=["Date (HE)"])  # Drop the original combined Date (HE) column
            .query('Hour.notnull()', engine='python')  # Ensure to keep only rows with valid hours

            .rename(columns={"Price ($)":"price","30Ravg ($)":"avg","AIL Demand (MW)":"ail","Date":"date","Hour":"hour"})
            #.assign(calc_date=lambda x: f"{pd.to_datetime(x['date']).dt.month.astype(str)}-{pd.to_datetime(x['date']).dt.year.astype(str)}")
            .assign(hour_bin=lambda x: pd.cut(x['hour'], bins=6, labels=False))  # Create 6 bins for the hour
            .assign(month=lambda x: pd.to_datetime(x['date']).dt.month)
            .assign(year=lambda x: pd.to_datetime(x['date']).dt.year)
            .assign(# Cyclic transformation of months
                    sin_month=lambda x: np.sin(2 * np.pi * x['month'] / 12),
                    cos_month=lambda x: np.cos(2 * np.pi * x['month'] / 12)
                        )
            .assign(# Cyclic transformation of hours
                    sin_hour=lambda x: np.sin(2 * np.pi * x['hour'] / 24),
                    cos_hour=lambda x: np.cos(2 * np.pi * x['hour'] / 24)
                        )
            .assign(# 3 previous hours as feature
                    prev_price_1=lambda x: x['price'].shift(1),
                    prev_price_2=lambda x: x['price'].shift(2),
                    prev_price_3=lambda x: x['price'].shift(3)
                )
            .dropna() # Ensure no null
            .set_index(["date","hour"])



            )

#plotting the data
def plotting(df):
    # Set the style of seaborn
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Price Time Series
    plt.figure(figsize=(14, 6))
    plt.plot(df.index.get_level_values('date'), df['price'], marker='o', linestyle='-', markersize=2)
    plt.title('Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 2: Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=30, kde=True)
    plt.title('Price Distribution')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.show()

    # Plot 3: Hourly Averages
    hourly_avg = df.groupby('hour')['price'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='hour', y='price', data=hourly_avg)
    plt.title('Average Price by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Average Price ($)')
    plt.show()

    # Plot 4: Seasonality with Sin and Cos of Month
    # Add jitter manually
    jitter_strength = 0.05
    df['sin_month_jittered'] = df['sin_month'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))
    df['cos_month_jittered'] = df['cos_month'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))

    # Plot with jitter
    plt.figure(figsize=(8, 8))
    df_sample=df.sample(500)
    plt.scatter(df_sample['sin_month_jittered'], df_sample['cos_month_jittered'], c=df_sample['price'], cmap='hot_r',vmin=20,vmax=40,alpha=0.8)
    plt.title('Seasonality (Sin & Cos of Month) vs. Price with Jitter')
    plt.xlabel('Sin(Month) with Jitter')
    plt.ylabel('Cos(Month) with Jitter')
    plt.colorbar()
    plt.show()


    # Plot 4: Seasonality with Sin and Cos of Hour
    # Add jitter manually
    jitter_strength = 0.05
    df['sin_hour_jittered'] = df['sin_hour'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))
    df['cos_hour_jittered'] = df['cos_hour'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))

def check_date_size(val):
    if len(val) == 1:
        val = f"{0}+{val}"

def plotting_natalia(df):

    # Plot 1 year progresion
    year_avg = df.groupby('year')['price'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='year', y='price', data=year_avg)
    plt.title('Average Price by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Price ($)')
    plt.show()

    # Plot 1 month progresion
    m_avg = df.groupby('month')['price'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='month', y='price', data=m_avg)
    plt.title('Average Price by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Price ($)')
    plt.show()


    monthly_avg = df.groupby(['month', 'year'])['price'].mean().reset_index()
    monthly_avg.sort_values(by=['year', 'month'], inplace=True)
    monthly_avg['mm-yyyy']=monthly_avg['month'].astype(str)+'-'+monthly_avg['year'].astype(str)
    print(monthly_avg)
    plt.figure(figsize=(20, 6))
    ax = sns.barplot(x='mm-yyyy', y='price', data=monthly_avg)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.title('Average Price by Month')
    plt.xlabel('Date')
    plt.ylabel('Average Price ($)')
    plt.show()

    #cross plot for this
    p_avg = df.groupby(['month', 'year'])['Population'].mean().reset_index()
    p_avg.sort_values(by=['year', 'month'], inplace=True)
    p_avg['mm-yyyy']=monthly_avg['month'].astype(str)+'-'+monthly_avg['year'].astype(str)
    print(p_avg)
    len(p_avg)
    len(monthly_avg)
    plt.figure(figsize=(20, 4))
    
    #print(df.columns)
    #df_single_index = df.copy() 
    #Wdf_single_index.index = df_single_index.index.droplevel(1)
    #print(df.index)
    # Resample to quarterly frequency and aggregate using sum (or other desired aggregation function)
    #quarterly_df = new_df.resample('Q').mean() 
    # Reset index if needed
    plt.scatter(monthly_avg['price'],p_avg['Population']) 
    plt.xlabel('price')
    plt.ylabel('population')
    #print(type(y_test))
    #print(type(predictions))
    plt.legend()
    plt.show()
        
        
#modeling
def modeling(df):
    # Prepare features and target variable
    X = df.drop('price', axis=1)
    y = df['price']

    # Define the model with early stopping rounds in the constructor
    model = xgb.XGBRegressor(
        objective ='reg:squarederror',
        n_estimators=1000,
        early_stopping_rounds=50
    )

    # Setup TimeSeries Cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # Lists to store results of CV testing
    rmse_scores = []
    

    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
   
        # Fit the model
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        print('this is X', X_test)
        # Predict and evaluate
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        rmse_scores.append(rmse)

        # Create a datetime index for plotting
        test_dates = X_test.index.get_level_values('date')
        test_hours = X_test.index.get_level_values('hour')
        datetime_index = pd.to_datetime(test_dates) + pd.to_timedelta(test_hours-1, unit='h')

        # Plotting predictions vs. real values
        plt.figure(figsize=(20, 4))
        plt.plot(datetime_index, y_test, label='Actual Prices', marker='o')
        plt.plot(datetime_index, predictions, label='Predicted Prices', marker='x')
        plt.title(f'Fold {fold} - Predictions vs. Actual Prices')
        plt.xlabel('Date and Hour')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        
        '''#cross plot for this
        plt.figure(figsize=(20, 4))
        plt.scatter(y_test, predictions,s=rmse) 
        plt.title(f'Fold {fold} - Predictions vs. Actual Prices')
        plt.xlabel('Date and Hour')
        plt.ylabel('Price')
        print(f"Test RMSE for fold {fold}: {rmse}")
        #print(type(y_test))
        #print(type(predictions))
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.legend()
        plt.show()
        '''
    # Display average RMSE over all folds
        #print('test=',y_test)
        #print('pred=',predictions)
    print("Average RMSE:", np.mean(rmse_scores))
    
   # index = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')


def linear_reg(df):
    # Prepare features and target variable
    X = df.drop('price', axis=1)
    y = df['price']
    lr = LinearRegression()
    # Setup TimeSeries Cross-validation
    tscv = TimeSeriesSplit(n_splits=50)
    # Lists to store results of CV testing
    rmse_scores = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        lr.fit(X_train, y_train)

        # Predict and evaluate
        predictions = lr.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        rmse_scores.append(rmse)

        # Create a datetime index for plotting
        test_dates = X_test.index.get_level_values('date')
        test_hours = X_test.index.get_level_values('hour')
        datetime_index = pd.to_datetime(test_dates) + pd.to_timedelta(test_hours-1, unit='h')

        '''# Plotting predictions vs. real values
        plt.figure(figsize=(20, 4))
        plt.plot(datetime_index, y_test, label='Actual Prices', marker='o')
        plt.plot(datetime_index, predictions, label='Predicted Prices', marker='x')
        plt.title(f'Fold {fold} - Predictions vs. Actual Prices')
        plt.xlabel('Date and Hour')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        '''
        '''
        #cross plot for this
        plt.figure(figsize=(20, 4))
        plt.scatter(y_test, predictions,s=rmse) 
        plt.title(f'Fold {fold} - Predictions vs. Actual Prices')
        plt.xlabel('Date and Hour')
        plt.ylabel('Price')
        print(f"Test RMSE for fold {fold}: {rmse}")
        #print(type(y_test))
        #print(type(predictions))
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.legend()
        plt.show()
        '''
        
    # Display average RMSE over all folds
        #print('test=',y_test)
        #print('pred=',predictions)
        rmsl=np.mean(rmse_scores)
        print("Average RMSE_lineal:", rmsl)
        return rmsl
#DECIDING BEST MODEL???


#cleaning data



    #df['df']=df['month'].map(str)+'-'+df['year'].map(str)
    #df['df']=pd.to_datetime(df['df'])
    
if __name__ == "__main__":
    print("Running Natalia's Electricy Forecaster")
    #start_date = input('Please enter a start date in MMDDYYYY: ')
    #end_date = input('Please enter a end date in MMDDYYYY: ')
    #elec_df = url_extract_electricity_data(start_date, end_date)
    elec_df = url_extract_electricity_data("01012018", "01012023")
    population = population()
    #merging population with data
    
    print(elec_df)
    engineered_df = engineering_features(elec_df)
    engineered_df = pd.merge_asof(engineered_df, population, left_on='date', right_index=True, direction='forward')
    print(engineered_df)
    #plotting(engineered_df)
    plotting_natalia(engineered_df)
    modeling(engineered_df)
    #linear_reg(engineered_df)