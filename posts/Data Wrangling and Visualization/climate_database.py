import sqlite3
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import calendar

# function for part 2
def query_climate_database(db_file, country, year_begin, year_end, month):
    """
    input:
    db_file: the file name for the database
    country: a string giving the name of a country for which data should be returned.
    year_begin and year_end: two integers giving the earliest and latest years for which should be returned (inclusive).
    month: an integer giving the month of the year for which should be returned.
    
    output (dataframe):
    NAME: The station name.
    LATITUDE: The latitude of the station.
    LONGITUDE: The longitude of the station.
    Country: The name of the country in which the station is located.
    Year: The year in which the reading was taken.
    Month: The month in which the reading was taken.
    Temp: The average temperature at the specified station during the specified year and month.
    """

    # connect to database
    conn = sqlite3.connect(db_file)

    # write sql command using f-strings
    command = f"""
    SELECT S.NAME, S.LATITUDE, S.LONGITUDE, C.Name AS Country, T.Year, T.Month, T.Temp
    FROM temperatures T
    LEFT JOIN stations S
    ON T.id = S.id
    LEFT JOIN countries C
    ON SUBSTR(T.ID, 1, 2) = C."FIPS 10-4"
    WHERE C.Name = "{country}"
    AND T.Year >= {year_begin}
    AND T.Year <= {year_end}
    AND T.Month = {month}
    ORDER BY S.NAME, T.Year, T.Month
    """

    df = pd.read_sql_query(command, conn)
    conn.close()
    return df


# function for part 3
def temperature_coefficient_plot(db_file, country, year_begin, year_end, month, min_obs, **kwargs):
    """
    inputs:
    the first five are the same
    min_obs: the minimum required number of years of data for any given station
    **kwargs: additional keyword arguments passed to px.scatter_mapbox()
    """
    
    # query the database using the function we wrote earlier
    df = query_climate_database(db_file, country, year_begin, year_end, month)

    # filter to have stations which has more than or equal to min_obs
    df = df.groupby(["NAME", "LATITUDE", "LONGITUDE"]).filter(lambda group: len(group) >= min_obs)

    # compute the first coefficient of a linear regression model at a given station
    def coef(data_group):
        x = data_group[["Year"]]
        y = data_group["Temp"]
        LR = LinearRegression()
        LR.fit(x, y)
        return LR.coef_[0]
    
    coefs = df.groupby(["NAME", "LATITUDE", "LONGITUDE"]).apply(coef).reset_index(name = "temp_diff")

    month_name = calendar.month_name[month]

    fig = px.scatter_mapbox(coefs,
                         lat = "LATITUDE",
                         lon = "LONGITUDE",
                         color = "temp_diff",
                         range_color = [-0.15, 0.15],
                         hover_name = "NAME",
                         hover_data = {
                            "LATITUDE": ":.3f",
                            "LONGITUDE": ":.3f",
                            "temp_diff": ":.3f"
                         },
                         labels = {
                             "LATITUDE": "LATITUDE",
                             "LONGITUDE": "LONGITUDE",
                             "temp_diff": "Estimated Yearly Increase (°C)"
                         },
                         title = f"Yearly temperature increase estimates in {month_name}<br>for {country} stations, {year_begin} to {year_end}",
                         **kwargs)
    
    return fig


# function for part 4
def query_station_temperature_data(db_file, country, year_begin, year_end, month):
    """
    input:
    db_file: the file name for the database.
    country: a string giving the name of a country.
    year_begin, year_end: start and end year (inclusive)
    month: specific month.

    output: a dataframe with the following columns
    NAME: name of the station
    LATITUDE, LONGITUDE: location of the station
    STNELEV: elevation of the station
    Country: the name of the country in which the station is located.
    Year: the Year for the reading
    Month: the Month for the reading
    Temp_Range: range of temperature in the station
    Num_Readings: number of readings for the station
    """

    # connect to database
    conn = sqlite3.connect(db_file)


    # write sql command using f-strings
    # only select stations with more than 3 readings in a same month across years
    command = f"""
    WITH yearly_station_counts AS (
        SELECT S.NAME, COUNT(DISTINCT T.Year) as year_count
        FROM temperatures T
        LEFT JOIN stations S ON T.id = S.id
        LEFT JOIN countries C ON SUBSTR(T.ID, 1, 2) = C."FIPS 10-4"
        WHERE C.Name = "{country}"
        AND T.Year >= {year_begin}
        AND T.Year <= {year_end}
        AND T.Month = {month}
        GROUP BY S.NAME
        HAVING year_count >= 3
    )
    SELECT 
        S.NAME, S.LATITUDE, S.LONGITUDE, S.STNELEV,
        C.Name AS Country, 
        T.Month,
        MAX(T.Temp) - MIN(T.Temp) AS Temp_Range
    FROM temperatures T
    LEFT JOIN stations S ON T.id = S.id
    LEFT JOIN countries C ON SUBSTR(T.ID, 1, 2) = C."FIPS 10-4"
    INNER JOIN yearly_station_counts ysc ON S.NAME = ysc.NAME
    WHERE C.Name = "{country}"
    AND T.Year >= {year_begin}
    AND T.Year <= {year_end}
    AND T.Month = {month}
    GROUP BY S.NAME, S.LATITUDE, S.LONGITUDE, S.STNELEV
    """

    df = pd.read_sql_query(command, conn)
    conn.close()
    return df


# part 4 plot function 1
def plot_elevation_temp_variability(db_file, country, year_begin, year_end, month, num_bins=5):

    """
    input:
    db_file: database file path
    country: country name
    year_begin, year_end: time period to analyze
    month: month to analyze (1-12)
    num_bins: number of elevation bins to create
    """
    # Get the data
    df = query_station_temperature_data(db_file, country, year_begin, year_end, month)
    
    # Create elevation bins
    df['Elevation_Bin'] = pd.qcut(df['STNELEV'], 
                                q=num_bins, 
                                labels=[f'{int(x.left)}-{int(x.right)}m' 
                                       for x in pd.qcut(df['STNELEV'], q=num_bins).unique()])
    
    # Create box plot
    fig = px.box(
        df,
        x='Elevation_Bin',
        y='Temp_Range',
        points='all',  # Show all points
        hover_data=['NAME', 'STNELEV'],  # Show station details on hover
        labels={
            'Elevation_Bin': 'Elevation Range',
            'Temp_Range': 'Temperature Range (°C)',
            'NAME': 'Station Name',
            'STNELEV': 'Exact Elevation (m)'
        },
        title=f'Temperature Variability Across Elevation Ranges in {country}<br>{year_begin} to {year_end}, Month {month}'
    )
    
    # Enhance layout
    fig.update_layout(
        height=600,
        width=900,
        title_x=0.5,
        template='plotly_white',
        xaxis_title=f'Elevation Ranges (divided into {num_bins} groups)'
    )
    
    return fig



# part 4 plot function 2
def plot_elevation_temp_trends(db_file, country, decade_ranges, month, max_elevation = 9000):
    """
    input:
    db_file: database file path
    country: country name
    decade_ranges: list of tuples, each containing (start_year, end_year) for a decade
    month: month to analyze (1-12)
    """

    # Initialize list to store data from each decade
    decade_data = []
    
    # Collect data for each decade
    for start_year, end_year in decade_ranges:
        df = query_station_temperature_data(db_file, country, start_year, end_year, month)
        # Filter out stations with unrealistic elevations
        df = df[df['STNELEV'] <= max_elevation]
        df['Decade'] = f'{start_year}s'
        decade_data.append(df)
    
    # Combine all decades' data
    combined_df = pd.concat(decade_data, ignore_index=True)
    
    # Create the title string including the max elevation
    title_string = f'Evolution of Elevation-Temperature Relationship in {country}<br>Month {month} (Stations below {max_elevation}m)'
    
    # Create faceted scatter plot
    fig = px.scatter(
        combined_df,
        x='STNELEV',
        y='Temp_Range',
        facet_col='Decade',
        facet_col_wrap=2,
        trendline='ols',
        hover_data=['NAME', 'LATITUDE', 'LONGITUDE'],
        labels={
            'STNELEV': 'Station Elevation (meters)',
            'Temp_Range': 'Temperature Range (°C)',
            'NAME': 'Station Name',
            'Decade': 'Time Period'
        },
        title=title_string
    )
    
    # Enhance layout and styling
    fig.update_layout(
        height=800,
        width=1000,
        title_x=0.5,
        template='plotly_white',
        showlegend=False
    )
    
    # Update axes titles for better readability
    fig.update_xaxes(title_text='Station Elevation (meters)')
    fig.update_yaxes(title_text='Temperature Range (°C)')
    
    return fig