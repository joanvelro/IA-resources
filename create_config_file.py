from configparser import ConfigParser

#Get the configparser object
config_object = ConfigParser()



config_object["DB_DESCRIPTIVE_SQLSERVER"] = {
    "driver": "ODBC Driver 13 for SQL Server",
    "user": "sa",
    "pass": "sa$2019",
    "server": "10.72.1.11",
    "database": "HIST_HORUS",
    "port": "5432",
    "schema": "dbo",
}

print('ok')


#Write the above sections to config.ini file
with open('config.ini', 'w') as conf:
    config_object.write(conf)