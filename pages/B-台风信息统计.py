import streamlit as st
import pandas as pd
import functions

st.subheader('2010-2020中国台风移动方向雷达图')
fig1 = functions.tp_directions('数据/中国台风数据.xlsx')
st.plotly_chart(fig1)

st.subheader('2010-2020中国台风逐年数量堆叠柱形图')
fig2 = functions.tp_number('Year','年','年份')
st.plotly_chart(fig2)

st.subheader('2010-2020中国台风逐月数量堆叠柱形图')
fig3 = functions.tp_number('Month','月','月份')
st.plotly_chart(fig3)

st.subheader('年度中国台风逐月数量多簇柱形图')
data = pd.read_excel('数据/中国台风数据.xlsx')
years = pd.unique(data['Year']).tolist()
s_year = st.selectbox('请选择年份', years)
fig4 = functions.tp_numberbyyear('Month','月', s_year)
st.plotly_chart(fig4)

st.subheader('历年台风登陆省份柱形图')
fig5 = functions.tp_pr()
st.plotly_chart(fig5)

st.subheader('历年台风登陆强度等级饼图')
fig6 = functions.tp_intensity()
st.plotly_chart(fig6)
