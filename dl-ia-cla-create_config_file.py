#!/usr/bin/env/python
"""
    This script creates the config file of the project

    @ Jose Angel Velasco (javelascor@indra.es)
    (C) Indra Digital Labs | IA - 2021
"""

from configparser import ConfigParser

#Get the configparser object
config_object = ConfigParser()


# Paths
config_object["paths"] = {
    "data_path":'D:\\data\\clarence',
    'models_path':'D:\\models\\clarence',
    'data_path_ext':'G:\\data\\clarence-data',
    'figures_path':'G:\\figures\\clarence'
}

config_object["data_base"] = {
    "table_desc_model":'h_descriptive_section',
    'table_pred_model':'h_predictive_section'
}


# DB postgresql
config_object["DB_PRE"] = {
    "driver": "postgresql",
    "user": "conf_horus",
    "pass": "CONF_HORUS",
    "server": "10.72.1.16",
    "database": "horus",
     "port": "5432",
    "schema": "hist_horus",
}

# DB SQL server testing
config_object["DB_SER"] = {
    "driver": "{SQL Server}", # ODBC Driver 13 for SQL Server # SQL Server Native Client RDA 11.0
    "user": "sa",
    "pass": "sa$2019",
    "server": "10.72.1.11",
    "database": "HIST_HORUS",
     "port": "5432",
    "schema": "dbo",
}






#Write the above sections to config.ini file
with open('config.ini', 'w') as conf:
    config_object.write(conf)