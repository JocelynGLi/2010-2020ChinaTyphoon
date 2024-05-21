import streamlit as st
import pandas as pd
import functions

st.subheader('台风路径查询')
data = pd.read_excel('数据/中国台风数据.xlsx')
years = pd.unique(data['Year']).tolist()
s_year = st.selectbox('请选择年份', years)
selected_rows = data[data['Year'] == s_year]
selected_rows['Id'] = selected_rows['Id'].astype(str)
id_list = selected_rows['Id'].str[-2:].tolist()
unique_id_list = sorted(set(id_list))
s_id = st.selectbox('请选择台风编号', unique_id_list)
id = str(s_year) + s_id
fig1 = functions.trace_point_shows(data, id)
st.plotly_chart(fig1)

st.subheader('该台风影响的区域')
area, fig2 = functions.tp_extent(id)
st.markdown(f"该台风影响中国的总面积为{area}平方千米")
st.plotly_chart(fig2)