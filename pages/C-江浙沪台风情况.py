import streamlit as st
import pandas as pd
import functions
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px

st.subheader('江浙沪城市台风灾情图')
gdf = functions.JZH_photo()
html_map = functions.jzh_situation(gdf)
st.components.v1.html(html_map, width=800, height=600)

st.subheader('江浙沪城市台风强度与数量图')
fig1 = functions.draw_JZHtp()
st.plotly_chart(fig1)

st.subheader('江浙沪各城市台风风险指标相关性热力图')
df = pd.read_excel('数据/台风灾害风险评估数据.xlsx')
fig2 = functions.correlation(df)
st.plotly_chart(fig2)

st.subheader('江浙沪各城市台风风险评估图')
sorted_dict = functions.APH()
cities = gpd.read_file('数据/江浙沪边界.shp')
fig3 = functions.jzh_assessment(sorted_dict, cities)
st.plotly_chart(fig3)