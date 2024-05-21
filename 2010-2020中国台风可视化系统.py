import streamlit as st
import matplotlib.pyplot as plt
import functions
import pandas as pd
import geopandas as gpd

# 实现台风数据爬虫（设置了如果爬虫数据已存在，则跳过爬虫）
allinfo = pd.DataFrame()
allinfo = functions.get_tp_data(allinfo)

# 实现中国台风数据筛选（设置了如果筛选后数据已存在，则跳过筛选）    
extent_CN = gpd.read_file('数据/中国省边界.shp')
filtered_CN = functions.filter(allinfo, extent_CN)

# 计算江浙沪地区台风登陆的数量与强度（设置了如果数据已存在，则跳过计算） 
extent_JZH = gpd.read_file('数据/江浙沪边界.shp')
functions.cal_JZH(extent_JZH, filtered_CN)

#计算江浙沪各市的路网密度（设置了如果数据已存在，则跳过计算） 
functions.road_density()

#计算江浙沪各市的平均海拔（设置了如果数据已存在，则跳过计算） 
functions.cal_dem()

plt.rcParams["font.family"] = "SimHei"
st.title("2010-2020中国台风可视化系统")    
form = st.form("my_form")
st.subheader("中国台风灾害介绍")
st.markdown("中国东临西太平洋，具有独特的地理位置和复杂的自然环境，是世界上台风登陆最多、致灾最重的国家之一，也是全世界少数几个受台风影响最为严重的国家，主要表现为大风、暴雨、风暴潮及其引发的次生地质灾害。登陆台风造成的暴雨洪涝是我国灾害损失最强的自然灾害，其中，又以从华南到东北的沿海岸地区受台风灾害影响最为严重。而台风暴雨是台风灾害的主要表现形式，其影响范围深远，性质复杂多变，经常导致洪涝、滑坡和泥石流等次生地质灾害，给人类社会造成巨大冲击。")
st.image("图片/中国台风.jpg")