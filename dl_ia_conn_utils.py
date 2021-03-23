#!/usr/bin/env/python
"""
    This script defines by functions some functionaility frequently used in data science

    @ Jose Angel Velasco (javelascor@indra.es)
    (C) Indra Digital Labs | IA - 2021

"""



def dl_ia_conn_plot_contour(df, title, x_label, y_label):
    """

    :return:
    """

    import plotly.graph_objects as go

    fig = go.Figure(data=
    go.Contour(
        z=df.values,
        x=list(range(df.shape[1])),  # horizontal axis
        y=list(range(df.shape[0])),  # vertical axis
        line_smoothing=0.85,
        contours=dict(
            showlabels=True,  # show labels on contours
            start=0,
            end=18,
            size=1)
    ))
    fig.update_layout(title=title,
                      xaxis_title=x_label,
                      yaxis_title=y_label)
    fig.update_layout(
        yaxis=dict(
            tickvals=np.arange(0, len(df.index)),
            ticktext=df.index
        ),
        xaxis=dict(
            tickvals=np.arange(0, len(df.index)),
            ticktext=df.columns,
            tickangle=90,
            tickfont=dict(size=9)
        )
    )

    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            color="RebeccaPurple",
            size=10))

    fig.show()


def dl_ia_conn_plot_heatmap(df, title, x_label, y_label, x_ticks, y_ticks):
    """ Plot heatmap
    :df param: dataframe to plot
    :title param: string with the title
    :x_label param: string with the label of the x axis
    :y_label param: string with the label of the y axis
    :x_ticks param: list with the ticks of the x axis
    :y_ticks param: list with the ticks of the y axis
    :return:
    """
    import plotly.express as px
    import numpy as np

    fig = px.imshow(df.values)
    fig.update_layout(title=title,
                      yaxis_title=y_label,
                      xaxis_title=x_label)

    fig.update_layout(
        yaxis=dict(
            tickvals=np.arange(0, len(y_ticks)),
            ticktext=y_ticks
        ),
        xaxis=dict(
            tickvals=np.arange(0, len(x_ticks)),
            ticktext=x_ticks
        )
    )
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            color="RebeccaPurple",
            size=11))

    fig.show()


def CONN_create_timestamp(row):
    """ Create a time stamp column from others columns with year, month, day and hour
        Use with apply
        :param row: lambda variable regarding columns of the dataframe
        :return datetime:
    """
    import pandas as pd
    try:
        return pd.Timestamp(int(row['YEAR']), int(row['MONTH']), int(row['DAY']), int(row['HOUR']), int(row['MINUTE']))

    except Exception as exception_msg:
        print('(!) Error in CONN_create_timestamp: ' + str(exception_msg))

def CONN_create_date(row):
    """ convert values of HOUR and MINUTE to datetime
    :param row: lambda variable regarding columns of the dataframe
    :return datetime:
    """
    import pandas as pd
    try:
        return pd.Timestamp(int(row['YEAR']), int(row['MONTH']), int(row['DAY']))
    except Exception as exception_msg:
        print('(!) Error in CONN_create_date: ' + str(exception_msg))

def CONN_create_time(row):
    """ convert values of HOUR and MINUTE to datetime
    :param row: lambda variable regarding columns of the dataframe
    :return datetime:
    """
    import datetime
    try:
        return datetime.time(int(row['HOUR']), int(row['MINUTE']), int(row['SECOND']))
    except Exception as exception_msg:
        print('(!) Error in CONN_create_time: ' + str(exception_msg))

def dl_ia_conn_utils_create_datetime(row):
    """ create datetime with hour and minute
    :param row: lambda variable regarding columns of the dataframe
    :return datetime:
    """
    import datetime
    try:
        return datetime.time(int(row['HOUR']), int(row['MINUTE']))
    except Exception as exception_msg:
        print('(!) Error in dl_ia_conn_utils_create_datetime: ' + str(exception_msg))


def dl_ia_conn_utils_create_date(row):
    """ create date with year, month and day
       :param row: lambda variable regarding columns of the dataframe
       :return datetime:
       """
    import pandas as pd
    try:
        return pd.Timestamp(int(row['YEAR']), int(row['MONTH']), int(row['DAY']))
    except Exception as exception_msg:
        print('(!) Error in dl_ia_conn_utils_create_date: ' + str(exception_msg))

def dl_ia_conn_greemplum():
    """

    :return:
    """



def dl_ia_conn_query_get_data(query, ddbb_settings):
    """
    this function perform the connection to the HORUS  SQL serverdata base and  executes the query provided
    :param query: string with the query
    :param ddbb_settings: List with DB connection settings (driver, server, database, schema,
    user and pass)
    :return error:
    """
    import pyodbc
    import pandas as pd
    error = 0

    try:
        # print('define parameters')

        # for a in db_settings_PRO.keys():
        #    print(a,':', db_settings_PRO[a])

        ### define conection to DDBB
        driver = ddbb_settings['driver']
        server = ddbb_settings['server']
        database = ddbb_settings['database']
        schema = ddbb_settings['schema']
        user = ddbb_settings['user']
        password = ddbb_settings['pass']
        port = ddbb_settings['port']


    except Exception as exception_msg:
        print('(!) Error in dl_ia_conn_query_get_data: ' + str(exception_msg))
        error = 1
        df_input = []
        return error, df_input

    if error == 0:
        try:
            print('Loading data from server:[{}] database:[{}] schema:[{}] port : [{}] '.format(server, database, schema, port))
            ### connect do DDBB and get last 6 hours od data
            # sql_conn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + password + 'Trusted_Connection=yes')
            # sql_conn = pyodbc.connect('DRIVER={SQL Server Native Client RDA 11.0};SERVER=' + server + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + password + 'Trusted_Connection=yes')
            sql_conn_str = 'DRIVER={};SERVER={},{};DATABASE={};UID={};PWD={}'.format(driver, server, port, database, user, password)
            # print(sql_conn_str)
            # sql_conn = pyodbc.connect('DRIVER=' + driver +';SERVER=' + server + ',' + port + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + password)
            sql_conn = pyodbc.connect(sql_conn_str)
            df_input = pd.read_sql(query, sql_conn)
            sql_conn.close()
            return error, df_input

        except Exception as exception_msg:
            print('(!) Error in dl_ia_conn_query_get_data: ' + str(exception_msg))
            error = 2
            df_input = []
            return error, df_input


def dl_ia_conn_initialize_engine(ddbb_settings):
    """ Initialize an SQL ALchemy engine
    :param ddbb_settings: DIctionary with driver user, pass, server, database, schema
    :param engine:
    """
    from sqlalchemy.engine import create_engine

    try:
        engine = create_engine("{}://{}:{}@{}:{}/{}".format(ddbb_settings['driver'],
                                                            ddbb_settings['user'],
                                                            ddbb_settings['pass'],
                                                            ddbb_settings['server'],
                                                            ddbb_settings['port'],
                                                            ddbb_settings['database']))
        print('Engine successfully initialized')
        return engine
    except Exception as exception_msg:
        print('(!) Error in dl_ia_conn_initialize_engine: {}'.format(exception_msg))
        engine = []
        return engine




def check_folder(path_folder):
    """ check that exists a folder, and if not, create it
    :param path_folder: string with the path
    :return error: error code (0:good, 1:bad)
    """
    import os
    error = 0
    try:
        if not os.path.isdir(path_folder):
            CLA_comm('Creating folder: {} '.format(path_folder))
            os.mkdir(path_folder)
    except Exception as exception_msg:
        print('(!) Error in check_folder: ' + str(exception_msg))
        error = 1
        return error



def CONN_config_plotly():
    """ this function configures the plotly visualization

    :return:
    """
    import plotly.io as pio
    import plotly.graph_objects as go
    import plotly.express as px

    pio.renderers.default = "browser"

def CONN_config_matplotlib():
    """ this function configures the matplotlib style

    :return:
    """
    from matplotlib import rc
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)

def CONN_config_pandas():
    """ this function configures the pandas library

    :return:
    """
    import pandas as pd
    pd.set_option('mode.chained_assignment', None)


def read_csv_per_chunks(path):
    """ This function read a large csv file into a dataframe per chunks

    :param path:
    :return df: dataframe
    """
    import pandas as pd
    chunksize_ = 1000
    error = 0

    try:
        TextFileReader = pd.read_csv(path, sep=";", chunksize=chunksize_)
        dfList = []
        for df in TextFileReader:
            dfList.append(df)
        df = pd.concat(dfList, sort=False)

        return error, df

    except Exception as exception_msg:
        print("Error in read_csv_per_chunks {}".format(exception_msg))
        # raise
        error = 1
        df = []
        return error, df

def CONN_comm(msg):
    """ Funtion to show mesages in terminal

    :parm msg: meassge (str)
    :return:
    """
    print('{} {}'.format('-'*20, msg))
    




def plot_timeseries_with_slider(df, var_x, var_y, title):
    """

    :param df:
    :param var_x:
    :param var_y:
    :return:
    """


    import plotly.graph_objects as go
    from CONN_CONN_utils import config_plotly
    config_plotly()

    show = True
    print_= True

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[var_x],
                             y=df[var_y],
                             mode='markers+lines',
                             marker=dict(color='red'),
                             name=var_y))
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title=title,
                      xaxis_title=var_x,
                      yaxis_title=var_y,
                      showlegend=True
                      )

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1d",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3d",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1w",
                         step="day",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\timeseries_with_slider_{}.html".format(var_y))




def CONN_plot_timeseries(df, var_x, var_y):
    """

    :param df:
    :param var_x:
    :param var_y:
    :return:
    """


    import plotly.graph_objects as go
    from CONN_CONN_utils import CONN_config_plotly
    CONN_config_plotly()

    show = True
    print_= True

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[var_x],
                             y=df[var_y],
                             marker=dict(color='red'),
                             mode='markers+lines',
                             name=var_y))
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='Time series',
                      xaxis_title=var_x,
                      yaxis_title=var_y,
                      showlegend=True
                      )

    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\timeseries_{}.html".format(var_y))

def plot_two_timeseries(df, var_x, var_y1, var_y2):
    """

    :param df:
    :param var_x:
    :param var_y1:
    :param var_y2:
    :return:
    """
    import plotly.graph_objects as go
    from CONN_utils import config_plotly, comm
    config_plotly()

    x_label = 'TIME (15 min)'
    y_label = 'FLOW (veh./h)'
    title_ = 'Time series'
    show = True
    print_ = True

    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[var_x],
                                 y=df[var_y1],
                                 marker=dict(color='red'),
                                 mode='markers',
                                 name=var_y1))
        fig.add_trace(go.Scatter(x=df[var_x],
                                 marker=dict(color='blue'),
                                 y=df[var_y2],
                                 mode='markers+lines',
                                 name=var_y2))
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title = title_,
                          xaxis_title = x_label,
                          yaxis_title = y_label,
                          showlegend=True
                          )

        # Add range slider
        fig.update_layout(
            xaxis=go.layout.XAxis(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1d",
                             step="day",
                             stepmode="backward"),
                        dict(count=3,
                             label="3d",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="1w",
                             step="day",
                             stepmode="backward"),
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        if show:
            fig.show()
        if print_:
            fig.write_html("figures\\two_timeseries_{}_{}.html".format(var_y1, var_y2))

    except Exception as exception_msg:
        comm('(!) Error in plot_two_timeseries : {} '.format(exception_msg))



def plot_line(df, var_x, var_y, var_group):
    """
    
    :param df: 
    :param var_x: 
    :param var_y: 
    :param var_group: 
    :return: 
    """
    import plotly.express as px
    from CONN_utils import config_plotly
    config_plotly()

    show = True
    print_ = True
    
    fig = px.line(df,
                  x=var_x,
                  y=var_y,
                  color=var_group,
                  )
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\line_plot_simple_{}_{}.html".format(var_x, var_y))


def CONN_quarter_classify(x):
    """ classify a variabel x into four cuadrants

    :param x: value with values in (0,60)
    :return y: values with values in (1,2,3,4)

    """
    if x <= 15:
        y = 0
    if 30 >= x > 15:
        y = 15
    if 45 >= x > 30:
        y = 30
    if x > 45:
        y = 45
    return y


def model_evaluation(model, X, y, type):
    """ Evaluate regression model

    :param model:
    :param X:
    :param y:
    :param type:
    :return:
    """

    import numpy as np
    from CONN_utils import comm, variability_captured, mape
    from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score, mean_squared_error

    # make prediction
    y_hat = model.predict(X)

    R2 = round(r2_score(y_hat, y),3)
    MAE = round(mean_absolute_error(y_hat, y),3)
    MSE = round(mean_squared_error(y_hat, y),3)
    EV = round(explained_variance_score(y_hat, y),3)
    VC = round(variability_captured(y_hat, y), 3)


    errors = abs(y_hat - y)
    #MAPE = 100 * np.mean(errors / y)
    MAPE = mape(y_hat, y)
    accuracy = 100 - MAPE

    comm('{} - Metrics'.format(type))
    comm('R2 = {:0.2f}'.format(R2))
    comm('EV = {:0.2f}'.format(VC))
    comm('Variability = {:0.2f}'.format(EV))
    comm('MSE = {:0.2f}'.format(MSE))
    comm('MAE = {:0.2f}'.format(MAE))
    comm('MAPE: {:0.4f} %'.format(MAPE))
    comm('Accuracy = {:0.2f} %.'.format(accuracy))

    return R2, MAE

def comm(msg):
    """ function to display in console msg

    :param msg:
    :return:
    """

    print('{} {}'.format('-'*20, msg))


def variability_captured(y, y_hat):
    """ function to calculate the varibility captured or explained variance

    :param y:
    :param y_hat:
    :return:
    """
    import numpy as np
    return round(1-np.var(y_hat-y)/np.var(y_hat), 3)

def mape(y_true, y_pred):
    """ function to calculate the Mean Absolute Percentage Error
        Zero values are treated by vertical translation

    :param y:
    :param y_hat:
    :return:
    """
    import numpy as np
    from CONN_utils import vertical_translation

    y_true = vertical_translation(y_true) # vertical translation +1
    y_pred = vertical_translation(y_pred) # vertical translation +1

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100 ,3)

def subs_zeros_values(y):
    """ subs zero values from from an array by values close to zeros 1e-10
        e.g.: y = np.array([1,4,2,3,7,8,0,0,8,7,0,0,9,8])
    :param y:
    :return:
    """
    import pandas as pd

    df = pd.DataFrame({'y':y})
    df.loc[df['y'] == 0, ['y']] = 1e-9
    return df['y'].values

def vertical_translation(y):
    """ detects in exist a zero value and translate the time series with the minimum value
    :param y:
    :return:
    """
    import numpy as np
    if np.isin(0, y):
        # exists a zero value, find the minimum distinct from zero
        delta = np.min(y[y > 0])
        # vertical translation
        # ym = y + delta
        ym = y + 1
        return ym
    return y




def plot_marginal_dist_plot(df, var_x, var_y):
    """

    :param df:
    :param var_x:
    :param var_y:
    :return:
    """
    from CONN_utils import config_plotly
    config_plotly()
    import plotly.express as px

    show = True
    print_ = True

    fig = px.density_heatmap(df,
                             x = var_x, # title="Click on the legend items!"
                             y = var_y, # add color="species",
                             marginal_x = "box", # histogram, rug
                             marginal_y = "violin")
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\plot_marginal_dist_plot_{}_{}.html".format(var_x, var_y))


def plot_scatter_with_facets(df, var_x, var_y, var_color, var_group):
    """

    :param df:
    :param var_x:
    :param var_y:
    :param var_color:
    :param var_group:
    :return:
    """
    import plotly.express as px
    from CONN_utils import config_plotly
    config_plotly()

    show = True
    print_ = True

    fig = px.scatter(df,
                     x = var_x,
                     y = var_y,
                     color = var_color,
                     facet_col = var_group,
                     marginal_x = "box") # violin, histogram, rug
    fig.update_layout(font = dict(family = "Courier New, monospace",
                                  size = 18,
                                  color = "#7f7f7f"))
    if show:
        fig.show()
    if print_:
        fig.write_html("figures\\plot_scatter_with_facets_{}_{}_{}_{}.html".format(var_x, var_y, var_color, var_group))


def create_lagged_variables(df, variable, number_lags):
    """ create lagged versions of the variable in a dataframe in which each row is an observation

    :param df:
    :param number_lags:
    :return:
    """
    number_lags = 23
    for lag in range(1, number_lags + 1):
        df[variable + '_lag_' + str(lag)] = df[variable].shift(lag)

    # if you want numpy arrays with no null values:
    df.dropna(inplace=True)

    return df




def plot_multi_timeseries():

        start = '2018-05-01 00:00:00'
        end = '2018-05-15 00:00:00'
        df_aux = df[(df['DATE'] > start) & (df['DATE'] < end)]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_aux['DATE'],
                                 y=df_aux['TOTAL_VEHICULOS']*4,
                                 name='Real'))

        fig.add_trace(go.Scatter(x=df_aux['DATE'],
                                 y=df_aux['TOTAL_VEHICULOS_DESCRIPTIVE']*4,
                                 name='Descriptive Model (R2:{:2.2f} / MAE:{:2.2f}%)'.format(r2_desc, MAE_desc)))

        fig.add_trace(go.Scatter(x=df_aux['DATE'],
                                 y=df_aux['TOTAL_VEHICLES_RF_PREDICTION']*4,
                                 name='Random Forest Model (R2:{:2.2f} / MAE:{:2.2f}%)'.format(r2_rf,
                                                                                               MAE_rf)))
        fig.add_trace(go.Scatter(x=df_aux['DATE'],
                                 y=df_aux['TOTAL_VEHICLES_NN_PREDICTION']*4,
                                 mode='lines',
                                 marker_color='rgba(152, 0, 0, .8)',
                                 name='Neural Network Model (R2:{:2.2f} / MAE:{:2.2f}%)'.format(r2_nn,
                                                                                                MAE_nn)))

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Predictive results - Segment: {} -  Dates: {} to {}'.format(segments[seg],
                                                                                                    start[0:10],
                                                                                                    end[0:10]),
                          xaxis_title='Time (15 min. resolution)',
                          yaxis_title='Flow (veh./h)',
                          showlegend=True
                          )

        fig.update_layout(legend=dict(x=0, y=-0.5, bgcolor="white"))

        fig.show()


def plot_multi_timeseries_with_slider():
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['DATE'],
                             y=df['TOTAL_VEHICULOS'],
                             name='Real Data'))

    fig.add_trace(go.Scatter(x=df['DATE'],
                             y=df['TOTAL_VEHICULOS_DESCRIPTIVE'],
                             name='Descriptive Model (R2:{:2.2f} / MAE:{:2.2f}%)'.format(r2_desc, MAE_desc)))

    fig.add_trace(go.Scatter(x=df['DATE'],
                             y=df['TOTAL_VEHICLES_RF_PREDICTION'],
                             name='Random Forest Predictive Model (R2:{:2.2f} / MAE:{:2.2f}%)'.format(r2_rf,
                                                                                                      MAE_rf)))
    # Set x-axis title
    fig.update_xaxes(title_text="Time")

    # Set y-axes titles
    fig.update_yaxes(title_text="Total Vehicles")

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='AUSOL - Road Traffic Flow - Segment: {}'.format(segments[seg]),
                      showlegend=True
                      )

    fig.update_layout(legend=dict(x=0, y=-1.0, bgcolor="white"))

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    fig.show()


def check_null_values(df):
    """

    :param df:
    :return df:
    """
    from CONN_utils import comm
    # check nans
    if df.isna().sum().sum() > 0:
        comm('(!) NAN Values detected')
        print(df.isna().sum())
        df.dropna(inplace=True)
        return df
    elif df.isnull().sum().sum() > 0:
        comm('(!) NULLs Values detected')
        print(df.isnull().sum())
        df.dropna(inplace=True)
        return df
    else:
        comm('Everything ok')
        return df



def CONN_plot_histogram(df, variable, n_bins_, label):
    """ plot a histogram using plotly from a vairable in a dataframe

    :param df: Data frame with the variable
    :param variable: name of the variable (column name)
    :param n_bins_: number of bins of the histogram
    :param label: string with a name for the title
    :return error: error code 0: everything ok, 1: something happened
    """

    import plotly.express as px
    from CONN_utils import CONN_config_plotly, CONN_comm
    import numpy as np
    CONN_config_plotly()

    print_ = True
    show = True
    fontsize_ = 18
    error = 0

    try:
        max_value = int(df[variable].max())
        x_axis = np.arange(0, max_value, int(max_value / 20))

        fig = px.histogram(df, x = variable, nbins = n_bins_, marginal="box")
        fig.update_xaxes(title_text=variable )
        fig.update_layout(font=dict(family="Courier New, monospace", size=fontsize_, color="#7f7f7f"))
        fig.update_layout(title='Histogram - {} - {}'.format(label, variable))
        fig.update_layout(showlegend=True)
        fig.update_traces(opacity=0.9)
        fig.update_layout(bargap=0.2)  # gap between bars of adjacent location coordinates
        fig.update_xaxes(ticktext=x_axis, tickvals=x_axis)

        if print_:
            fig.write_html("figures\\plot_histogram_{}.html".format(variable))
        if show:
            fig.show()
        return error

    except Exception as exception_msg:
        CONN_comm('(!) Error in plot_histogram: ' + str(exception_msg))
        error = 1
        return error




def CONN_time_series_plot(time_index, y1, label1,  title):
    """ Plot one single time series

    :param time_index:
    :param y1:
    :param y2:
    :param label1:
    :param label2:
    :return:
    """

    import plotly.graph_objects as go
    from CONN_utils import CONN_config_plotly
    CONN_config_plotly()

    print_ = True
    show = True
    #title = 'time_series_comparison'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index,
                             y=y1,
                             mode='markers+lines',
                             marker=dict(color='red'),
                             name='{}'.format(label1)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=label1)
    fig.update_xaxes(title_text='Time')
    fig.update_layout(title=title)

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    if print_:
        fig.write_html("figures\\time_series_{}.html".format(label1))
    if show:
        fig.show()



def CONN_time_series_comparison(time_index, y1, y2, label1, label2, title):
    """ Plot two time series with the same time index

    :param time_index:
    :param y1:
    :param y2:
    :param label1:
    :param label2:
    :return:
    """

    import plotly.graph_objects as go
    from CONN_utils import CONN_config_plotly
    CONN_config_plotly()

    print_ = True
    show = True
    #title = 'time_series_comparison'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index,
                             y=y1,
                             mode='markers+lines',
                             marker=dict(color='red'),
                             name='{}'.format(label1)))

    fig.add_trace(go.Scatter(x=time_index,
                             y=y2,
                             mode='markers+lines',
                             marker=dict(color='blue'),
                             name='{}'.format(label2)))

    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text=label1)
    fig.update_xaxes(title_text='Time')
    fig.update_layout(title=title)

    # Add range slider
    fig.update_layout(
        xaxis=go.layout.XAxis(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=3,
                         label="3 day",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1 week",
                         step="day",
                         stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    if print_:
        fig.write_html("figures\\time_series_comparison_{}_{}.html".format(label1, label2))
    if show:
        fig.show()




def CONN_plot_scatterplot_simple(df, var_x, var_y, label):
    """ Produce a simple scatter plot with plotly

    :param df: dataframe that contains the variables
    :param variable_x: variable to plot in x axis
    :param variable_y: variable to plot in y axis
    :param variable_to_color: variable to use as color
    :param variable_to_color: variable to use as size
    :return:
    """
    import plotly.express as px
    from CONN_utils import CONN_config_plotly, CONN_comm
    CONN_config_plotly()

    print_ = True
    show = True
    error = 0

    try:
        fig = px.scatter(df,
                         x = var_x,
                         y = var_y,  # marker = dict(color='blue')
                          )

        fig.update_xaxes(title_text=var_x)
        fig.update_yaxes(title_text=var_y)
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Scatterplot - {}'.format(label))
        fig.update_layout(showlegend=True)

        if print_:
            fig.write_html("figures\\scatterplot_{}_{}_{}.html".format(label, var_x, var_y))
        if show:
            fig.show()

        return error

    except Exception as exception_msg:

        CONN_comm('(!) Error in CONN_plot_scatterplot_simple: ' + str(exception_msg))
        error = 1
        return error

def CONN_plot_scatterplot(df, var_x, var_y, var_color, var_size, label):
    """ Produce a simple scatter plot with plotly

    :param df: dataframe that contains the variables
    :param variable_x: variable to plot in x axis
    :param variable_y: variable to plot in y axis
    :param variable_to_color: variable to use as color
    :param variable_to_color: variable to use as size
    :return:
    """
    import plotly.express as px
    from CONN_utils import CONN_config_plotly, CONN_comm
    CONN_config_plotly()

    print_ = True
    show = True
    error = 0

    try:
        fig = px.scatter(df,
                         x=var_x,
                         y=var_y,  # marker = dict(color='blue')
                         size=var_size,
                         color=var_color)

        fig.update_xaxes(title_text=var_x)
        fig.update_yaxes(title_text=var_y)
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Scatterplot - {}'.format(label))
        fig.update_layout(showlegend=True)

        if print_:
            fig.write_html("figures\\scatterplot_{}_{}_{}.html".format(label, var_x, var_y))
        if show:
            fig.show()

        return error


    except Exception as exception_msg:

        CONN_comm('(!) Error in plot_scatterplot: ' + str(exception_msg))
        error = 1
        return error

def CONN_create_DDBB_connection():
    """ This function build a engine with the connection to the DDBB of the INDRA architecture
    :return engine: ENgine to run queries against DDBB
    """

    from sqlalchemy.engine import create_engine
    from CONN_utils import CONN_comm


    driver = 'postgresql'
    user_ = 'gpadmin'
    pass_ = 'pivotal'
    host = '10.0.2.6'
    port = '5432'
    ddbb = 'gpadmin'

    CONN_comm('Build engine with connection to DDBB:')
    CONN_comm('Driver:{}'.format(driver))
    CONN_comm('User:{}'.format(user_))
    CONN_comm('Pass:{}'.format(pass_))
    CONN_comm('Host:{}'.format(host))
    CONN_comm('Port:{}'.format(port))
    CONN_comm('Database:{}'.format(ddbb))



    engine = create_engine("{}://{}:{}@{}:{}/{}".format(driver, user_, pass_, host, port, ddbb))
    return engine


def CONN_systems_info():
    """ Function that shows the system properties

    """
    import sys
    from platform import python_version
    from CONN_utils import CONN_comm

    #CONN_comm('Python version:{}'.format(python_version()))
    CONN_comm('Python version:{}'.format(sys.version))
    CONN_comm('Path:{}'.format(sys.executable))
    #CONN_comm('Python version info:{}'.format(sys.version_info))

def CONN_anomaly_detection_univariate(df, variable):
    """ Produce anomaly detection with forest isolation with univariate data
    :param df:
    :param variable:

    """

    from sklearn.ensemble import IsolationForest
    import numpy as np
    import pandas as pd
    from CONN_utils import CONN_comm


    error = 0

    try:
        # instantiate model
        isolation_forest = IsolationForest(n_estimators=200)

        # fit model
        isolation_forest.fit(df[variable].values.reshape(-1, 1))
        xx = np.linspace(df[variable].min(), df[variable].max(), len(df)).reshape(-1, 1)
        anomaly_score = isolation_forest.decision_function(xx)

        # make prediction
        outlier = isolation_forest.predict(xx)

        df_out = pd.DataFrame({'X':xx.T.ravel(), 'outlier':outlier.ravel(), 'anomaly_score':anomaly_score.ravel()})

        lower_limit_outlier = df_out[df_out['outlier']==-1]['X'].min()
        upper_limit_outlier = df_out[df_out['outlier']==-1]['X'].max()

        CONN_comm('lower limit outlier:{}'.format(lower_limit_outlier))
        CONN_comm('upper limit outlier:{}'.format(upper_limit_outlier))

        return error, df_out

        # plot
        if 0==1:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(xx, anomaly_score, label='anomaly score')
            plt.fill_between(xx.T[0],
                             np.min(anomaly_score),
                             np.max(anomaly_score),
                             where=outlier == -1,
                             color='r',
                             alpha=.4,
                             label='outlier region')
            plt.legend()
            plt.ylabel('anomaly score')
            plt.xlabel('Sales')
            plt.show();

    except Exception as exception_msg:
        CONN_comm('(!) Error in CONN_anomaly_detection_univariate: ' + str(exception_msg))
        error = 1
        return error
