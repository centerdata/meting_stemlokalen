# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:19:19 2023

@author: SeyitHöcük
"""


#%% # Libraries
import pandas as pd
import numpy as np
# system
import os
# geolocation
import geopandas as gpd # gpd.show_versions()
from geopandas.tools import sjoin_nearest
from geopy.geocoders import Nominatim
# others
import gc
from tabulate import tabulate
from collections import Counter
# plot
import matplotlib.pyplot as plt


#%% # Environment
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option("display.max_rows",100)
pd.set_option("display.max_columns",8)
pd.set_option("min_rows",20)
pd.set_option('display.width', 150)
pd.set_option('display.max_colwidth', 25)


#%% # Functions
def find_all_filenames(path_to_dir, suffix=".xlsx", prefix=''):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if (filename.endswith(suffix)) & (filename.startswith(prefix)) ]

def explore_data(df, name='', shuffle=True, head=5):
    print('Filename      :', name, sep=' ', end='\n')
    print('Size of df    :', df.shape)
    print('First 5 cols  :', list(df.columns[:5]), sep=' ')
    print('Amount duplic :', df.duplicated().sum())
    print('Total missing :', df.isna().sum().sum())
    print('')
    print('Columns missings (top '+str(head)+'):')
    print( df.isna().sum()[df.isna().sum()>0].sort_values(ascending=False).head(head) )
    print('')
    if shuffle:
        print('Header information (shuffled):')
        print( tabulate(df.sample(n=10, axis=0, random_state=None).sample(n=5, axis=1, random_state=None).reset_index(drop=True),
                        headers='keys', tablefmt='psql', showindex=False, maxcolwidths=[None,None,None,None]) )
    else:
        print('Header information:')
        print( tabulate(df.iloc[:10,:5], headers='keys', tablefmt='psql', showindex=False,
                        maxcolwidths=[None,None,55,55]) )

def twofilescheck(df1, df2, checktype=1, sortby='', verbose=1):
    df1 = df1.copy().reset_index(drop=True)
    df2 = df2.copy().reset_index(drop=True)
    if sortby:
        df1 = df1.copy().sort_values(by=sortby).reset_index(drop=True)
        df2 = df2.copy().sort_values(by=sortby).reset_index(drop=True)
    if checktype == 0:
        df1 = df1.astype('object')
        df2 = df2.astype('object')
    collist1 = df1.columns
    collist2 = df2.columns
    collists = sorted(list(set( list(collist1) + list(collist2) )))
    identicalcount = 0
    print('\n Checking two DataFrames if they are the same \n')
    for column in collists:
        try:
            sameornot = df1[column].equals(df2[column])
        except:
            if verbose > 1:
                print('Column =', column, '>> is not available in both DataFrames!')
        else:
            if (sameornot == True) & verbose > 0:
                print('Column =', column, '--', sameornot)
            elif (sameornot == False):
                print('Column =', column, '--', sameornot)
            if sameornot == True: identicalcount += 1
    print('\n Are the DataFrames identical?')
    print(df1.equals(df2))
    print('\n How many variables are identical?')
    print(identicalcount, 'out of', len(collists), 'of which df1 =', df1.shape[1],'and df2 =', df2.shape[1])
    print('\n How many different variables?')
    print(set(collist1)^set(collist2))

def weighted_average(df, values, weights):
    d = df[values]
    w = df[weights]
    output = (d * w).sum(min_count=1) / w.sum(min_count=1)
    return output

def weighted_median(df, val, weight):
    df_sorted = df.sort_values(val)
    cumsum = df_sorted[weight].cumsum()
    cutoff = df_sorted[weight].sum(min_count=1) / 2.
    try:
        output = df_sorted[cumsum >= cutoff][val].iloc[0]
    except:
        output = np.nan
    return output


#%% # Paths
mypath = "G:\\Projecten\\Data Science\\8491_BZK_Meting_stemlokalen\\Data\\"
subwms = "WIMS_data\\"
subwmp = "WIMS_data\\WIMS_processed\\"
subksr = "Kiesraad_data\\"
subcbs = "CBS_data\\"
subgeo = "Geolocation\\"
submap = "Mapping\\"

anpath = "G:\\Projecten\\Data Science\\8491_BZK_Meting_stemlokalen\\Analyses\\"
subglv = 'gemeente_level\\'
subwlv = 'wijk_level\\'

fileWAT = "waterschapsverkiezingen 2023_V1.4_15032023.csv"
# filePRV = "Provinciale Statenverkiezingen 2023_V1.4_15032023.csv"
# fileWMS = "waterschapsverkiezingen 2023_V1.4_15032023_Final_deduplicated_modified.xlsx"
# fileWMS = "waterschapsverkiezingen 2023_V1.4_15032023_Final_deduplicated_modified_v2.xlsx"
fileWMS = "Stemlokalen_PS_WS_2023.xlsx"
fileKSR = "Tellingen_PS_2023vs2019 tbv centerdata_20230404.xlsx"
file500 = "cbs_vk500_2020_v2.gpkg"
filePc6 = "CBS-PC6-2020-v1\\CBS_pc6_2020_v1.shp"
fileKWB22 = "kwb-2022.xls" # is niet up to date
fileKWB21 = "kwb-2021.xls"
fileKWB20 = "kwb-2020.xls"
fileKWB19 = "kwb-2020.xlsx"

filemapGWB22 = "pc6hnr20220801_gwb.csv"
filemapGWB21 = "pc6hnr20210801_gwb.csv"
filemapGWB20 = "pc6hnr20200801_gwb.csv"
filemapGWB19 = "pc6hnr20190801_gwb.csv"
filemapGEM22 = "gem2022.csv"
filemapGEM21 = "gem2021.csv"
filemapGEM20 = "gem2020.csv"
filemapGEM19 = "gem2019.csv"
filemapWYK22 = "Wijken2022.csv"
filemapWYK21 = "wijk2021.csv"
filemapWYK20 = "wijk2020.csv"
filemapWYK19 = "wijk2019.csv"


#%% # Initialize
geolocator = Nominatim(user_agent="project_voting") # also Google or Bing possible

myrng = 2               # chosen random number seed
mydpi = 500             # chosen dots-per-inch (dpi) level
do_save_distances = 0   # save distances to a file?
verbose = 1             # how much prints should be made


#%% # Read
# stembureaus en verkiezingen
dfwimsw = pd.read_csv(mypath + subwms + fileWAT) # 2023, waterschappen
# dfwimsp = pd.read_csv(mypath + subwms + filePRV) # 2023, provinces = practically same as waterschappen
dfwimsf = pd.read_excel(mypath + subwmp + fileWMS) # 2023, deduplicated final version of Stemlokalen
dfkiesr = pd.read_excel(mypath + subksr + fileKSR, sheet_name='PS2023') # 2023

# CBS gegevens
gdfbox = gpd.read_file(mypath + subcbs + file500) # 2020, because 2021 (latest) seems not up to date
gdfpc6 = gpd.read_file(mypath + subcbs + filePc6) # 2020, 2021+2022 are promised end of March
dfkwb22 = pd.read_excel(mypath + subcbs + fileKWB22, decimal=',') # 2021, because 2022 is not up to date
dfkwb21 = pd.read_excel(mypath + subcbs + fileKWB21, decimal=',') # 2021, because 2022 is not up to date
dfkwb20 = pd.read_excel(mypath + subcbs + fileKWB20, decimal=',') # 2021, because 2022 is not up to date
dfkwb19 = pd.read_excel(mypath + subcbs + fileKWB19, decimal=',') # 2021, because 2022 is not up to date

# geolocation
# gdfgeo = gpd.read_file(mypath + subgeo + fileGEO) # 2023

# mapping
mapgwb22 = pd.read_csv(mypath + submap + filemapGWB22, delimiter=';')
mapgem22 = pd.read_csv(mypath + submap + filemapGEM22, delimiter=';')
mapwyk22 = pd.read_csv(mypath + submap + filemapWYK22, delimiter=';')
mapgwb21 = pd.read_csv(mypath + submap + filemapGWB21, delimiter=';')
mapgem21 = pd.read_csv(mypath + submap + filemapGEM21, delimiter=';')
mapwyk21 = pd.read_csv(mypath + submap + filemapWYK21, delimiter=';')
mapgwb20 = pd.read_csv(mypath + submap + filemapGWB20, delimiter=';')
mapgem20 = pd.read_csv(mypath + submap + filemapGEM20, delimiter=';', encoding= 'unicode_escape')
mapwyk20 = pd.read_csv(mypath + submap + filemapWYK20, delimiter=';', encoding= 'unicode_escape')
mapgwb19 = pd.read_csv(mypath + submap + filemapGWB19, delimiter=';')
mapgem19 = pd.read_csv(mypath + submap + filemapGEM19, delimiter=';')
mapwyk19 = pd.read_csv(mypath + submap + filemapWYK19, delimiter=';')

# keep originals
do_keep_orig = 0
if do_keep_orig == 1:
    dfwims0 = dfwimsf.copy()
    gdfbox0 = gdfbox.copy()
    gdfpc60 = gdfpc6.copy()
    dfkwb0  = dfkwb22.copy()

# clear memory
gc.collect()


#%% # Explore
do_explore = 0
if do_explore:
    explore_data(dfwimsf)
    explore_data(gdfbox)
    dfwimsf.isna().sum()

do_explore_more = 0
if do_explore_more:

    # find cases with same coordinate (lat,lon) but different address (gem,pc6,str), and vice versa
    dfwimsf_ = dfwimsf.copy()
    dfwimsf_ = dfwimsf_.dropna(subset='Postcode')
    dfwimsf_ = dfwimsf_.apply(lambda x: x.astype(str).str.lower())
    tokeep = False # 'first', 'last', or False

    # same address & same coordinate
    todrop7 = ['Gemeente','Straatnaam','Postcode','Latitude','Longitude']
    dfwimsf_z2 = dfwimsf_.drop_duplicates( subset=todrop7, keep=tokeep ); print(dfwimsf_z2.shape)

    # different address & different coordinate
    todrop4 = ['Latitude','Longitude']
    dfwimsf_x = dfwimsf_.drop_duplicates( subset=todrop4, keep=tokeep )
    dfwimsf_x_indx = dfwimsf_x.index
    todrop2 = ['Gemeente','Straatnaam','Postcode']
    dfwimsf_y = dfwimsf_.drop_duplicates( subset=todrop2, keep=tokeep )
    dfwimsf_y_indx = dfwimsf_y.index
    alldiff_indx = list( set(dfwimsf_x_indx).intersection( set(dfwimsf_y_indx) ) )
    dfwimsf_z1 = dfwimsf_.loc[alldiff_indx]; print(dfwimsf_z1.shape)

    # to check amount
    check_indx = list( (set(dfwimsf_z1.index)^set(dfwimsf_z2.index)) )
    check_indx = list( (set(dfwimsf_z1.index).symmetric_difference( set(dfwimsf_z2.index)) ) )
    print( len(check_indx) )
    dfwimsf_.loc[check_indx, ['Naam stembureau','Gemeente','Straatnaam','Postcode','Latitude','Longitude']]


#%% # Select
cols_wims = ['id','Gemeente','CBS gemeentecode','Naam stembureau','Type stembureau','Gebruiksdoel van het gebouw',
             'Straatnaam','Huisnummer','Huisletter','Postcode','X','Y','Latitude','Longitude',
             'Openingstijd','Sluitingstijd','Toegankelijk voor mensen met een lichamelijke beperking']
dfwimsf = dfwimsf[cols_wims].copy()

cols_wims = ['_id','Gemeente','CBS gemeentecode','Naam stembureau','Type stembureau','Gebruiksdoel van het gebouw',
             'Straatnaam','Huisnummer','Huisletter','Postcode','X','Y','Latitude','Longitude',
             'Openingstijd','Sluitingstijd','Toegankelijk voor mensen met een lichamelijke beperking']
dfwimsw = dfwimsw[cols_wims].copy()

cols_ksr = ['GemeenteCode','GemeenteNaam','Aantal_SB','Cast','TotalCounted']
dfkiesr = dfkiesr[cols_ksr].copy()

dfkwbw = dfkwb21.copy()
# dfkwbw = pd.concat([dfkwbw, dfkwb21.loc[dfkwb21['gwb_code_10'].isin(set(dfkwb21['gwb_code_10'])-set(dfkwbw['gwb_code_10']))]], ignore_index=True)
dfkwbw = pd.concat([dfkwbw, dfkwb20.loc[dfkwb20['gwb_code_10'].isin(set(dfkwb20['gwb_code_10'])-set(dfkwbw['gwb_code_10']))]], ignore_index=True)
dfkwbw = pd.concat([dfkwbw, dfkwb19.loc[dfkwb19['gwb_code_10'].isin(set(dfkwb19['gwb_code_10'])-set(dfkwbw['gwb_code_10']))]], ignore_index=True)
dfkwbw = pd.concat([dfkwbw, dfkwb22.loc[dfkwb22['gwb_code_10'].isin(set(dfkwb22['gwb_code_10'])-set(dfkwbw['gwb_code_10']))]], ignore_index=True)
dfkwbw['gwb_code_8'] = dfkwbw['gwb_code_8'].astype(str)
dfkwbw = dfkwbw.sort_values(by='gwb_code_8').reset_index(drop=True)
dfkwbw['gwb_code_8'] = dfkwbw['gwb_code_8'].astype(int)

cols_kwb = ['gwb_code_8','gm_naam','recs','a_inw','g_wozbag','g_ink_po','g_ink_pi','p_hh_110']
dfkwbw = dfkwbw[cols_kwb]
dfkwbw = dfkwbw[dfkwbw['recs']=='Wijk'].reset_index(drop=True)

# clear memory
del dfkwb22, dfkwb21, dfkwb20, dfkwb19
gc.collect()


#%% # Rename
cols_rename = {'Toegankelijk voor mensen met een lichamelijke beperking':'Toegankelijkheid',
               'CBS gemeentecode':'Gemeentecode'}
dfwimsf.rename(columns=cols_rename, inplace=True)
dfwimsw.rename(columns=cols_rename, inplace=True)

cols_rename = {'GemeenteCode':'Gemeentecode','GemeenteNaam':'Gemeentenaam','Aantal_SB':'aantal_sb',
               'Cast':'kiesgerechtigden?','TotalCounted':'stemmen'}
dfkiesr.rename(columns=cols_rename, inplace=True)

cols_rename = {'Gemcode2022':'Gemeentecode','Gemcode2021':'Gemeentecode','Gemcode2020':'Gemeentecode','Gemcode2019':'Gemeentecode',
               'Gemeente2022':'Gemeentecode','Gemeente2021':'Gemeentecode','Gemeente2020':'Gemeentecode','Gemeente2019':'Gemeentecode',
               'Gemeentenaam2022':'Gemeentenaam','Gemeentenaam2021':'Gemeentenaam','Gemeentenaam2020':'Gemeentenaam','Gemeentenaam2019':'Gemeentenaam',
               'Buurt2020':'Buurt','Buurt2021':'Buurt','Buurt2022':'Buurt','Buurt2019':'Buurt',
               'Wijk2020':'Wijkcode','Wijk2021':'Wijkcode','Wijk2022':'Wijkcode','Wijk2019':'Wijkcode',
               'wijkcode2022':'Wijkcode','wijkcode2021':'Wijkcode','wijkcode2020':'Wijkcode','Wijkcode2019':'Wijkcode',
               'wijknaam2022':'Wijknaam','wijknaam2021':'Wijknaam','wijknaam2020':'Wijknaam','Wijknaam_2019K_NAAM':'Wijknaam'}
mapgem22.rename(columns=cols_rename, inplace=True)
mapwyk22.rename(columns=cols_rename, inplace=True)
mapgwb22.rename(columns=cols_rename, inplace=True)
mapgem21.rename(columns=cols_rename, inplace=True)
mapwyk21.rename(columns=cols_rename, inplace=True)
mapgwb21.rename(columns=cols_rename, inplace=True)
mapgem20.rename(columns=cols_rename, inplace=True)
mapwyk20.rename(columns=cols_rename, inplace=True)
mapgwb20.rename(columns=cols_rename, inplace=True)
mapgem19.rename(columns=cols_rename, inplace=True)
mapwyk19.rename(columns=cols_rename, inplace=True)
mapgwb19.rename(columns=cols_rename, inplace=True)

cols_rename = {'gwb_code_8':'Wijkcode','gm_naam':'Gemeentenaam'}
dfkwbw.rename(columns=cols_rename, inplace=True)


#%% # Merge mapping
mapgwb = mapgwb22.copy()
mapgwb = pd.concat([mapgwb,mapgwb21]).drop_duplicates(subset=['PC6','Huisnummer'], keep='first').reset_index(drop=True)
mapgwb = pd.concat([mapgwb,mapgwb20]).drop_duplicates(subset=['PC6','Huisnummer'], keep='first').reset_index(drop=True)
mapgwb = pd.concat([mapgwb,mapgwb19]).drop_duplicates(subset=['PC6','Huisnummer'], keep='first').reset_index(drop=True)
mapgem = mapgem22.copy()
mapgem = pd.concat([mapgem,mapgem21]).drop_duplicates(subset=['Gemeentecode'], keep='first').reset_index(drop=True)
mapgem = pd.concat([mapgem,mapgem20]).drop_duplicates(subset=['Gemeentecode'], keep='first').reset_index(drop=True)
mapgem = pd.concat([mapgem,mapgem19]).drop_duplicates(subset=['Gemeentecode'], keep='first').reset_index(drop=True)
mapwyk = mapwyk22.copy()
mapwyk = pd.concat([mapwyk,mapwyk21]).drop_duplicates(subset=['Wijkcode'], keep='first').reset_index(drop=True)
mapwyk = pd.concat([mapwyk,mapwyk20]).drop_duplicates(subset=['Wijkcode'], keep='first').reset_index(drop=True)
mapwyk = pd.concat([mapwyk,mapwyk19]).drop_duplicates(subset=['Wijkcode'], keep='first').reset_index(drop=True)

# clear memory
del mapgwb19, mapgwb20, mapgwb21, mapgwb22
del mapgem19, mapgem20, mapgem21, mapgem22
del mapwyk19, mapwyk20, mapwyk21, mapwyk22
gc.collect()


#%% # Append information
mapgwb = pd.merge(mapgwb, mapgem, on='Gemeentecode', how='left')
mapgwb = pd.merge(mapgwb, mapwyk, on='Wijkcode', how='left')


#%% # Clean

# replace -99997's
gdfbox.replace(-99997, np.nan, inplace=True)
gdfpc6.replace(-99997, np.nan, inplace=True)

# replace GM from gemeentecode and leading 0
dfwimsf['Gemeentecode'] = dfwimsf['Gemeentecode'].str.replace('GM','').astype(int)
dfwimsw['Gemeentecode'] = dfwimsw['Gemeentecode'].str.replace('GM','').astype(int)

# replace '.'
dfkwbw.replace('.', 0, inplace=True)
dfkwbw['Wijkcode'] = dfkwbw['Wijkcode'].astype(str)

# postcode mapping is inclusive of house number, which we do not need
todrop = {'Huisnummer','Buurt'}
todrop = list(todrop.intersection(mapgwb.columns))
mapgwb_nonum = (mapgwb.drop(columns=todrop).drop_duplicates()).sort_values(by=['PC6'])
mapgwb_nonum = mapgwb_nonum.drop_duplicates(subset='PC6', keep='first').reset_index(drop=True)
# NOTE >>> 26 (0.005%) postal codes cross municipality border, chosen to keep first
# NOTE >>> 4510 (0.98%) postal codes cross municipality and/or wijk border, chosen to keep first

# type
dfkiesr['aantal_sb'] = dfkiesr['aantal_sb'].astype(int)
tofloat = ['a_inw','g_wozbag','g_ink_po','g_ink_pi','p_hh_110']
dfkwbw[tofloat] = dfkwbw[tofloat].astype(float)


#%% # Convert

# datetime conversion
dfwimsf['Openingstijd'] = pd.to_datetime(dfwimsf['Openingstijd'])
dfwimsf['Sluitingstijd'] = pd.to_datetime(dfwimsf['Sluitingstijd'])
dfwimsf['Openingsduur'] = round( (dfwimsf['Sluitingstijd'] - dfwimsf['Openingstijd'])/np.timedelta64(1,'h'),1 )

# check opening hours
Counter(dfwimsf['Openingsduur']).most_common()

# new format
do_save_new_format = 0
if do_save_new_format:
    filemapGWB_19_22 = 'GWB_mapping_19_to_22.xlsx'
    mapgwb_nonum.to_excel(mypath + submap + filemapGWB_19_22, index=False)


#%% # Features
do_new_features = 0
if do_new_features == 1:
    dfwimsf['Openingsduur_korter'] = (dfwimsf['Openingsduur'] < 13.5)*1
    dfwimsf['Openingsduur_langer'] = (dfwimsf['Openingsduur'] > 13.5)*1
    dfwimsf['Openingsduur_afwijkend'] = (dfwimsf['Openingsduur'] != 13.5)*1


#%% # Coordinate transformations

# create geometry
dfwimsf['geometry'] = gpd.points_from_xy(dfwimsf['X'],dfwimsf['Y'], crs='28992') # RD-coordinates
dfwimsf = gpd.GeoDataFrame(dfwimsf).set_crs(28992)

# create lat lon
gdfbox['geometry_latlon'] = gdfbox['geometry'].to_crs(4326) # Lat-Lon coordinates
gdfbox['geometry_latlon'] = gdfbox['geometry_latlon'].representative_point()

# (Multi)Polygon to representative point
gdfbox['geometry'] = gdfbox['geometry'].representative_point()


#%% # Find or append municipality (gemeente)

# find gemeente
gdfpc6 = pd.merge(gdfpc6, mapgwb_nonum, how='left', on='PC6')
cols_wanted = {'geometry','Gemeentecode','Gemeentenaam','Wijkcode','Wijknaam'}
cols_wanted = list(cols_wanted.intersection(gdfpc6.columns))
gdfbox = sjoin_nearest(gdfbox, gdfpc6[cols_wanted])

# drop duplicates originating from sjoin_nearest (from identical distances?)
gdfbox = gdfbox.drop_duplicates(subset=gdfbox.columns[0], keep='first').reset_index(drop=True)

# drop index_right
todrop = {'index_right'}
gdfbox.drop(columns=todrop, inplace=True)

# sanity check
gdfbox.loc[gdfbox['Gemeentenaam']=='Amsterdam', 'aantal_inwoners'].sum()    # as expected
gdfbox.loc[gdfbox['Gemeentenaam']=='Tilburg', 'aantal_inwoners'].sum()      # as expected
gdfbox.loc[gdfbox['Gemeentenaam']=='Eemsdelta', 'aantal_inwoners'].sum()    # as expected
gdfbox.loc[gdfbox['Gemeentenaam']=='Appingedam', 'aantal_inwoners'].sum()   # as expected
gdfbox.loc[gdfbox['Gemeentenaam']=='Loppersum', 'aantal_inwoners'].sum()    # as expected


#%% # Find nearest
nearest_method = 2

# without municipality border limitation
if nearest_method == 1:
    # sjoin
    joincols = ['geometry','Gemeente','Gemeentecode']
    cols_rename = {'Gemeentecode':'GemeentecodeWIMS'}
    # gdfpc6 = sjoin_nearest(gdfpc6, dfwimsf[joincols].rename(columns=cols_rename), how='left')
    gdfboxn = sjoin_nearest(gdfbox, dfwimsf[joincols].rename(columns=cols_rename), how='left')

    # drop duplicates originating from sjoin_nearest
    gdfboxn = gdfboxn.drop_duplicates(subset=gdfboxn.columns[0], keep='first').reset_index(drop=True)

    # change name for clarity
    cols_rename = {'Gemeente':'Gemeente_nearest_SL','GemeentecodeWIMS':'Gemeentecode_nearest_SL'}
    gdfpc6.rename(columns=cols_rename, inplace=True)
    gdfboxn.rename(columns=cols_rename, inplace=True)

# with municipality border limitation
if nearest_method == 2:
    gdfboxn = gdfbox.copy()
    lijst_gemeentecodes = sorted(list(set( gdfbox['Gemeentecode'].dropna().astype(int) )))
    subsetcols = ['geometry','Gemeente_nearest_SL','Gemeentecode_nearest_SL','index_right']

    for gemeentecode in sorted(lijst_gemeentecodes):
        # condition
        condition1 = gdfbox['Gemeentecode']==gemeentecode
        condition2 = dfwimsf['Gemeentecode']==gemeentecode
        gdfbox_sub = gdfbox[condition1].copy()
        dfwimsf_sub = dfwimsf[condition2].copy()

        # change name for clarity
        cols_rename = {'Gemeente':'Gemeente_nearest_SL','Gemeentecode':'Gemeentecode_nearest_SL'}
        dfwimsf_sub.rename(columns=cols_rename, inplace=True)

        # find nearest gemeente
        gdfbox_sub = sjoin_nearest(gdfbox_sub, dfwimsf_sub[ subsetcols[0:3] ], how='left')

        # drop duplicates originating from sjoin_nearest
        gdfbox_sub = gdfbox_sub.drop_duplicates(subset=gdfbox_sub.columns[0], keep='last')

        # put back
        gdfboxn.loc[condition1, subsetcols[1]] = gdfbox_sub[subsetcols[1]].values
        gdfboxn.loc[condition1, subsetcols[2]] = gdfbox_sub[subsetcols[2]].values
        gdfboxn.loc[condition1, subsetcols[3]] = gdfbox_sub[subsetcols[3]].values

    # fill remaining without border limitation
    conditionnan = gdfboxn['Gemeente_nearest_SL'].isna()
    gdfbox_sub = gdfboxn[conditionnan]

    joincols = ['geometry','Gemeente','Gemeentecode']
    cols_rename = {'Gemeente':'Gemeente_nearest_SL','Gemeentecode':'Gemeentecode_nearest_SL'}
    cols_drop = {'Gemeente_nearest_SL', 'Gemeentecode_nearest_SL','index_right'}
    gdfbox_sub = sjoin_nearest(gdfbox_sub.drop(columns=cols_drop), dfwimsf[joincols].rename(columns=cols_rename), how='left')

    # drop duplicates originating from sjoin_nearest
    gdfbox_sub = gdfbox_sub.drop_duplicates(subset=gdfbox_sub.columns[0], keep='last')

    # put back
    gdfboxn.loc[conditionnan, subsetcols[1]] = gdfbox_sub[subsetcols[1]].values
    gdfboxn.loc[conditionnan, subsetcols[2]] = gdfbox_sub[subsetcols[2]].values
    gdfboxn.loc[conditionnan, subsetcols[3]] = gdfbox_sub[subsetcols[3]].values

# check missing
gdfboxn.isna().sum() # missings can come from mismatch in herindeling gemeente in method 2

# sanity check
gdfboxn.loc[gdfboxn['Gemeente_nearest_SL']=='Amsterdam', 'aantal_inwoners'].sum()   # as expected
gdfboxn.loc[gdfboxn['Gemeente_nearest_SL']=='Tilburg', 'aantal_inwoners'].sum()     # as expected
gdfboxn.loc[gdfboxn['Gemeente_nearest_SL']=='Eemsdelta', 'aantal_inwoners'].sum()   # as expected
gdfboxn.loc[gdfboxn['Gemeente_nearest_SL']=='Appingedam', 'aantal_inwoners'].sum()  # as expected


#%% # Find distances (slowest part)
gdfboxn['distance_nearest_SL'] = np.nan
for index in gdfboxn.index:
    gdfi = gdfboxn.loc[index:index]
    if gdfi['index_right'].values[0] > 0:
        wimsi = dfwimsf.loc[gdfi['index_right']]
        calculateddistance = gdfi.distance( wimsi, align=False )
        gdfboxn.loc[index, 'distance_nearest_SL'] = calculateddistance.values[0]

# check mean and median distance
check1 = weighted_average(gdfboxn, 'distance_nearest_SL', 'aantal_inwoners')
check2 = weighted_median(gdfboxn, 'distance_nearest_SL', 'aantal_inwoners')
print('Mean distance =', check1) # 13 meters difference (408.08/394.57) between yes/no municipality border limitation
print('Median distance =', check2) # 1 meter difference (309.66/308.53) between yes/no municipality border limitation

# clear memory
gc.collect()


#%% # Organize afstanden on gemeente level
cols_interest = ['gemeente','gemeentecode','inwoners','woningwaarde','uitkering','dist_mean','dist_median']
df_afstanden_g = pd.DataFrame(columns=cols_interest)

lijst_gemeentecodes = sorted(list(set( gdfboxn['Gemeentecode'].dropna().astype(int) ))) # from gdfbox, i.e. 2020
lijst_gemeentecodes = sorted(list(set( gdfboxn['Gemeentecode_nearest_SL'].dropna().astype(int) ))) # from wims, i.e. 2023
gemcounter = 0
nmissing_g = 0
for gemeentecode in sorted(lijst_gemeentecodes):
    condition = gdfboxn['Gemeentecode_nearest_SL']==gemeentecode
    if condition.sum() > 0:
        ii = gemcounter
        gdfbox_sub = gdfboxn[condition].copy()
        df_afstanden_g.loc[ii, 'gemeente'] = gdfbox_sub['Gemeente_nearest_SL'].values[0] # or: 'Gemeentenaam'
        df_afstanden_g.loc[ii, 'gemeentecode'] = gdfbox_sub['Gemeentecode_nearest_SL'].values[0] # or: gemeentecode, 'Gemeentecode'
        df_afstanden_g.loc[ii, 'inwoners'] = gdfbox_sub['aantal_inwoners'].sum(min_count=1)
        df_afstanden_g.loc[ii, 'woningwaarde'] = gdfbox_sub['gemiddelde_woz_waarde_woning'].mean()
        df_afstanden_g.loc[ii, 'uitkering'] = gdfbox_sub['aantal_personen_met_uitkering_onder_aowlft'].sum(min_count=1)
        df_afstanden_g.loc[ii, 'dist_mean'] = weighted_average(gdfbox_sub, 'distance_nearest_SL', 'aantal_inwoners')
        df_afstanden_g.loc[ii, 'dist_median'] = weighted_median(gdfbox_sub, 'distance_nearest_SL', 'aantal_inwoners')
        gemcounter += 1
    else:
        nongem = gdfboxn.loc[ gdfboxn['Gemeentecode']==gemeentecode, 'Gemeentenaam' ].values[0]
        print('...Warning, this gemeente has no distances:', gemeentecode, nongem)
        nmissing_g += 1

# save the distances to a file
if do_save_distances == 1:
    savename = 'distances_on_gemeentelevel.xlsx'
    df_afstanden_g.to_excel(anpath + subglv + savename, index=False)

# check
df_afstanden_g.isna().sum()


#%% # Organize afstanden on wijk level
cols_interest = ['Gemeente','Gemeentecode','Wijk','Wijkcode','inwoners','woningwaarde','uitkering','dist_mean','dist_median']
df_afstanden_w = pd.DataFrame(columns=cols_interest)

lijst_wijkcodes = sorted(list(set( gdfboxn['Wijkcode'].dropna() ))) # from gdfbox i.e. 2020
lijst_wijkcodes = sorted(list(set( mapgwb['Wijkcode'].dropna() ))) # from gwb i.e. 2022--2019
wijkcounter = 0
nmissing_wk = 0
for wijkcode in sorted(lijst_wijkcodes):
    condition = gdfboxn['Wijkcode']==wijkcode
    if condition.sum() > 0:
        ii = wijkcounter
        gdfbox_sub = gdfboxn[condition].copy()
        df_afstanden_w.loc[ii, 'Gemeente'] = gdfbox_sub['Gemeentenaam'].values[0] # or: 'Gemeente_nearest_SL'
        df_afstanden_w.loc[ii, 'Gemeentecode'] = gdfbox_sub['Gemeentecode'].values[0] # or: 'Gemeentecode_nearest_SL'
        df_afstanden_w.loc[ii, 'Wijk'] = gdfbox_sub['Wijknaam'].values[0]
        df_afstanden_w.loc[ii, 'Wijkcode'] = wijkcode # or: 'Wijkcode'
        df_afstanden_w.loc[ii, 'inwoners'] = gdfbox_sub['aantal_inwoners'].sum(min_count=1)
        df_afstanden_w.loc[ii, 'woningwaarde'] = gdfbox_sub['gemiddelde_woz_waarde_woning'].mean()
        df_afstanden_w.loc[ii, 'uitkering'] = gdfbox_sub['aantal_personen_met_uitkering_onder_aowlft'].sum(min_count=1)
        df_afstanden_w.loc[ii, 'dist_mean'] = weighted_average(gdfbox_sub, 'distance_nearest_SL', 'aantal_inwoners')
        df_afstanden_w.loc[ii, 'dist_median'] = weighted_median(gdfbox_sub, 'distance_nearest_SL', 'aantal_inwoners')
        wijkcounter += 1
    else:
        nonwyk = mapwyk.loc[mapwyk['Wijkcode']==wijkcode, 'Wijknaam'].values[0]
        if verbose > 1:
            print('...Warning, this wijk has no distances:', wijkcode, nonwyk)
        nmissing_wk += 1

# Add kerncijfers on Wijk level
mergecols = ['Wijkcode','a_inw','g_wozbag','g_ink_po','g_ink_pi','p_hh_110']
df_afstanden_w_ = pd.merge(df_afstanden_w, dfkwbw[mergecols], how='left', on='Wijkcode') # we will not use additional information

# save the distances to a file
if do_save_distances == 1:
    savename = 'distances_on_wijklevel.xlsx'
    df_afstanden_w.to_excel(anpath + subwlv + savename, index=False)

# check
df_afstanden_w.isna().sum()


#%% # Cross-check differences WIMS vs OSV (ondersteunende software verkiezingen)
dfwims_grouped = dfwimsw.groupby(['Gemeentecode','Gemeente']).count().reset_index()
dfwims_grouped = dfwims_grouped[['Gemeentecode','Gemeente','_id']].rename(columns={'_id':'aantal_sb_wims'})
dfkiesr_ccheck = dfkiesr[['Gemeentecode','Gemeentenaam','aantal_sb']].copy()
dfkiesr_ccheck = dfkiesr_ccheck.rename(columns={'Gemeentenaam':'Gemeente'})

# merge
dfccheck = pd.merge(dfwims_grouped, dfkiesr_ccheck, how='left', on=['Gemeentecode'])
dfccheck = dfccheck[['Gemeentecode','Gemeente_x','aantal_sb_wims','aantal_sb']].rename(columns={'Gemeente_x':'Gemeente','aantal_sb':'aantal_sb_ksr'})

# find mismatch
dfccheck = dfccheck.eval('mismatch = aantal_sb_wims - aantal_sb_ksr')

# which ones and how much
dfmismatch = dfccheck[dfccheck['mismatch']!=0].sort_values(by='mismatch')
print('Aantal gemeentes mismatch in #SB =', dfmismatch.shape[0]) # = 86
print('Aantal stembureaus mismatch totaal =', dfccheck.mismatch.abs().sum()) # = 260

# save mismatches
do_save_mismatches = 0
if do_save_mismatches == 1:
    savename = 'mismatching_gemeenten_WIMS_vs_OSV.xlsx'
    dfmismatch.to_excel(anpath + savename, index=False)


#%% # Plots
mycol = 'navy'
myfontsize = 15
tickfontsize = 15
mymarkersz = 7 # default = 6
myfigsize = (10,5) # default = 8,6
plt.rcParams['axes.formatter.min_exponent'] = 5 # default = 0

# histogram
selected_boxes = gdfboxn.loc[gdfboxn['aantal_inwoners']>5,'distance_nearest_SL']

plt.figure(figsize=myfigsize)
plt.hist(selected_boxes, bins=10, range=[0,3000], alpha=0.9, rwidth=0.85, edgecolor='black')
plt.xlabel("Afstand (meter)", labelpad=10, fontsize=myfontsize)
plt.ylabel("Aantal gebieden", labelpad=10, fontsize=myfontsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.tight_layout()

savefile = "plot_distances_histogram_alteast_5_pop.png"
plt.savefig(anpath + savefile, bbox_inches='tight', transparent=False, pad_inches=0.2, dpi=mydpi)
plt.show()

# wijklevel - regular
plt.figure(figsize=myfigsize)
plt.plot(df_afstanden_w['inwoners'], df_afstanden_w['dist_mean'], '.', color=mycol, markersize=mymarkersz)
plt.xlim(-1000,100000)
plt.ylim(-100,6000)
plt.xlabel("Aantal inwoners per wijk", labelpad=10, fontsize=myfontsize)
plt.ylabel("Afstand tot stemlokaal (meter)", labelpad=10, fontsize=myfontsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.grid(True, axis='y', color='#EEEEEE', zorder=0)

savefile = "plot_distances_wijk.png"
plt.savefig(anpath + subwlv + savefile, bbox_inches='tight', transparent=False, pad_inches=0.2, dpi=mydpi)
plt.show()

df_afstanden_w['inwoners'].astype(float).describe()
df_afstanden_w['dist_mean'].astype(float).describe()

# wijklevel - log
plt.figure(figsize=myfigsize)
plt.loglog(df_afstanden_w['inwoners'], df_afstanden_w['dist_mean'], '.', color=mycol, markersize=mymarkersz)
plt.xlim(1e0,1e6)
plt.ylim(1e1,1e4)
plt.xlabel("Aantal inwoners per wijk", labelpad=10, fontsize=myfontsize)
plt.ylabel("Afstand tot stemlokaal (meter)", labelpad=10, fontsize=myfontsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.grid(True, axis='y', color='#EEEEEE', zorder=0)

savefile = "plot_distances_wijk_loglog.png"
plt.savefig(anpath + subwlv + savefile, bbox_inches='tight', transparent=False, pad_inches=0.2, dpi=mydpi)
plt.show()

# gemeentelevel - regular
plt.figure(figsize=myfigsize)
plt.plot(df_afstanden_g['inwoners'], df_afstanden_g['dist_mean'], '.', color=mycol, markersize=mymarkersz)
plt.xlim(-20000,900000)
plt.ylim(150,1200)
plt.xlabel("Aantal inwoners per gemeente", labelpad=10, fontsize=myfontsize)
plt.ylabel("Afstand tot stemlokaal (meter)", labelpad=10, fontsize=myfontsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.grid(True, axis='y', color='#EEEEEE', zorder=0)

savefile = "plot_distances_gemeente.png"
plt.savefig(anpath + subglv + savefile, bbox_inches='tight', transparent=False, pad_inches=0.2, dpi=mydpi)
plt.show()

df_afstanden_w['inwoners'].astype(float).describe()
df_afstanden_w['dist_mean'].astype(float).describe()

# gemeentelevel - log
plt.figure(figsize=myfigsize)
plt.loglog(df_afstanden_g['inwoners'], df_afstanden_g['dist_mean'], '.', color=mycol, markersize=mymarkersz)
plt.xlim(0.5e3,2e6)
plt.ylim(1e2,2e3)
plt.xlabel("Aantal inwoners per gemeente", labelpad=10, fontsize=myfontsize)
plt.ylabel("Afstand tot stemlokaal (meter)", labelpad=10, fontsize=myfontsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)

savefile = "plot_distances_gemeente_loglog.png"
plt.savefig(anpath + subglv + savefile, bbox_inches='tight', transparent=False, pad_inches=0.2, dpi=mydpi)
plt.show()


#%% # Plots 2
dfwimsf_wijk = pd.merge(dfwimsf.rename(columns={'Postcode':'PC6'}), mapgwb_nonum, how='left', on='PC6')
dfwimsf_wijk_gr = dfwimsf_wijk.groupby('Wijkcode').count().reset_index().rename(columns={'id':'count_SL'})
dfwimsf_wijk_gr = dfwimsf_wijk_gr[['Wijkcode','count_SL']]

dfkwbsl = pd.merge(dfkwbw, dfwimsf_wijk_gr, how='left', on='Wijkcode')
dfkwbsl.isna().sum()
dfkwbsl = dfkwbsl.dropna().drop(columns=['recs'])

mycol = 'navy'
myfontsize = 15
tickfontsize = 15
mymarkersz = 7 # default = 6
myfigsize = (10,5) # default = 8,6
plt.rcParams['axes.formatter.min_exponent'] = 5 # default = 0
plt.rcParams['xtick.labelsize'] = tickfontsize

normalized_count = dfkwbsl['count_SL']/dfkwbsl['a_inw']*1000

# inkomen/SL wijklevel - log
plt.figure(figsize=myfigsize)
plt.loglog(dfkwbsl['g_ink_pi'], normalized_count, '.', color=mycol, markersize=mymarkersz)
plt.xlim(1e1,1e2)
plt.ylim(1e-2,1e1)
plt.xlabel('Gemiddeld inkomen per wijk (x 1000 euro)', labelpad=10, fontsize=myfontsize)
plt.ylabel('Aantal stemlokalen per 1000 inwoners', labelpad=10, fontsize=myfontsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.grid(True, axis='y', color='#EEEEEE', zorder=0)

savefile = "plot_inkomen_SL_wijk_loglog.png"
plt.savefig(anpath + subwlv + savefile, bbox_inches='tight', transparent=False, pad_inches=0.2, dpi=mydpi)
plt.show()

# inkomen/afstand wijklevel - log
plt.figure(figsize=myfigsize)
plt.loglog(df_afstanden_w_['g_ink_pi'], df_afstanden_w_['dist_mean'], '.', color=mycol, markersize=mymarkersz)
plt.xlim(1e1,1e2)
plt.ylim(2e1,5e3)
plt.xlabel('Gemiddeld inkomen per wijk (x 1000 euro)', labelpad=10, fontsize=myfontsize)
plt.ylabel('Afstand tot stemlokaal (meter)', labelpad=10, fontsize=myfontsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.grid(True, axis='y', color='#EEEEEE', zorder=0)

savefile = "plot_inkomen_afstand_wijk_loglog.png"
plt.savefig(anpath + subwlv + savefile, bbox_inches='tight', transparent=False, pad_inches=0.2, dpi=mydpi)
plt.show()

# woningwaarde/SL wijklevel - log
plt.figure(figsize=myfigsize)
plt.loglog(dfkwbsl['g_wozbag'], normalized_count, '.', color=mycol, markersize=mymarkersz)
plt.xlim(5e1,2e3)
plt.ylim(1e-2,1e2)
plt.xlabel('Gemiddelde WOZ-waarde per wijk (x 1000 euro)', labelpad=10, fontsize=myfontsize)
plt.ylabel('Aantal stemlokalen per 1000 inwoners', labelpad=10, fontsize=myfontsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.grid(True, axis='y', color='#EEEEEE', zorder=0)

savefile = "plot_woningwaarde_SL_wijk_loglog.png"
plt.savefig(anpath + subwlv + savefile, bbox_inches='tight', transparent=False, pad_inches=0.2, dpi=mydpi)
plt.show()

# inkomen/afstand wijklevel - log
plt.figure(figsize=myfigsize)
plt.loglog(df_afstanden_w_['g_wozbag'], df_afstanden_w_['dist_mean'], '.', color=mycol, markersize=mymarkersz)
plt.xlim(5e1,2e3)
plt.ylim(1e1,1e4)
plt.xlabel('Gemiddelde WOZ-waarde per wijk (x 1000 euro)', labelpad=10, fontsize=myfontsize)
plt.ylabel('Afstand tot stemlokaal (meter)', labelpad=10, fontsize=myfontsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.grid(True, axis='y', color='#EEEEEE', zorder=0)

savefile = "plot_woningwaarde_afstand_wijk_loglog.png"
plt.savefig(anpath + subwlv + savefile, bbox_inches='tight', transparent=False, pad_inches=0.2, dpi=mydpi)
plt.show()


#%% # END



