import os
import re
import requests
from lxml import etree
import pandas as pd
from shapely.geometry import Point
from geopandas import GeoDataFrame
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import rasterstats
import folium
from folium import Marker,FeatureGroup
from folium.plugins import MiniMap
import base64

def check_file(path):
    if os.path.exists(path):
        return True
    else:
        return False

#爬取台风数据，返回结果为所有台风数据dataframe
def get_tp_data(allinfo):
    if check_file(path = '数据/台风数据(未筛选).xlsx'):
        allinfo = pd.read_excel('数据/台风数据(未筛选).xlsx')
    else:
        url="http://agora.ex.nii.ac.jp/cgi-bin/dt/search_name2.pl?lang=en&basin=wnp&smp=1&sdp=1&emp=12&edp=31"
        response = requests.get(url)
        response.encoding = 'utf8'
        html = response.text
        d = re.findall(r'<td><a href="/digital-typhoon/summary/wnp/s/(.*?)">',html,re.S) #搜索所有的台风信息
        for p in d: #遍历所有的台风网页
            id = p[:6]  #获取台风的ID
            if 201001<=int(id)<202100: #筛选出2010-2020年的所有台风
                p_url = 'http://agora.ex.nii.ac.jp/digital-typhoon/summary/wnp/l/' + p  #根据台风ID访问具体的台风轨迹信息
                response_1 = requests.get(p_url)
                html_1 = etree.HTML(response_1.text)
                table = html_1 .xpath('//table[@class="TRACKINFO"]')
                table = etree.tostring(table[0], encoding='utf-8').decode()
                df = pd.read_html(table, encoding='utf-8', header=0)[0]
                df.drop(df.columns[[9, 10, 11]], axis=1, inplace=True) #删除无用列
                column=['Year','Month','Day','Hour','Lat','Lon','Pressure(hPa)','Wind(kt)','Class'] #添加列名
                df.columns=column
                df.insert(0, 'Id', id)
                allinfo = pd.concat([allinfo, df])
                print(f'台风{id}爬取完成！')
        allinfo['Time'] = pd.to_datetime(allinfo[['Year', 'Month', 'Day', 'Hour']])
        allinfo['Time'] = allinfo['Time'].dt.strftime('%Y-%m-%d %H:%M')
    allinfo.to_excel('数据/台风数据(未筛选).xlsx', encoding = 'utf-8', index = False)
    return allinfo

#筛选出中国台风数据，返回结果为中国台风数据dataframe
def filter(allinfo, extent):
    if not check_file(path = '数据/中国台风数据.xlsx'):
        ids = pd.unique(allinfo['Id']).tolist()
        filtered_tp = pd.DataFrame()
        intersection = []
        for id in ids:
            single_tp = allinfo[allinfo['Id'] == id]
            points = [Point(lon, lat) for lon, lat in zip(single_tp['Lon'], single_tp['Lat'])] #获取该次台风的各时间台风中心的经纬度数据
            buffered_points = [point.buffer(300/111) for point in points] #设置缓冲区，缓冲区半径为台风7级风圈300km，坐标系为EPSG:4326，所以buffer半径的单位为度，1度约为111km
            geometries = GeoDataFrame(geometry = buffered_points)
            geometries.set_crs('EPSG:4326', inplace=True)
            intersected = gpd.overlay(geometries, extent, how='intersection') #获取台风影响范围和中国的重合部分
            if not intersected.empty: #若台风7级风圈与研究区有重合范围，则保存该台风数据
                filtered_tp = pd.concat([filtered_tp, single_tp])
                intersection.append(intersected)
        filtered_tp.to_excel('数据/中国台风数据.xlsx', index = False)
    else:
        filtered_tp = pd.read_excel('数据/中国台风数据.xlsx')
    return filtered_tp

#绘制台风路径图，返回结果为图
def trace_point_shows(df, id):
    token = '' # fill in with your token
    df = df[df['Id'] == int(id)]
    df.columns = ['台风编号', '年', '月', '日', '时刻', '纬度', '经度', '气压(hPa)', '风速(kt)', '等级', '时间']
    fig = px.scatter_mapbox(df,
                            hover_data=['时间','气压(hPa)','等级'],
                            lon='经度',
                            lat='纬度',
                            color='风速(kt)',
                            hover_name='台风编号',
                            size_max=14,
                            color_continuous_scale="Sunset")
    
    for index, row in df.iterrows():
        # 添加圆形形状
        lon_c = row['经度']
        lat_c = row['纬度']
        R = 300 / 111.32  # 半径转换成度数，假设每度经纬度距离大约为111.32km
        theta = np.linspace(0, 2*np.pi, 100)
        circle_lon = lon_c + R * np.cos(theta)
        circle_lat = lat_c + R * np.sin(theta)
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=circle_lon,
            lat=circle_lat,
            marker=dict(size=0),
            hoverinfo='none',
            line=dict(width=2, color="grey"),
            opacity=0.3,
            showlegend=False  # 不在图例中显示
        ))
    
    fig.update_layout(mapbox={'accesstoken': token, 'center': {'lon': df['经度'].iloc[0], 'lat': df['纬度'].iloc[0]}, 'zoom': 2, 'style': 'light'}, margin={'l': 1, 'r': 1, 't': 1, 'b': 1})
    return fig

#绘制台风影响范围图，返回结果为影响面积area和图fig
def tp_extent(id):
    data = pd.read_excel('数据/中国台风数据.xlsx')
    pr = gpd.read_file("数据/中国省边界.shp", crs='EPSG:4326')
    df = data[data['Id'] == int(id)]
    points = [Point(lon, lat) for lon, lat in zip(df['Lon'], df['Lat'])]
    buffered_points = [point.buffer(300/111) for point in points]
    gdf = gpd.GeoDataFrame(geometry=buffered_points, crs='EPSG:4326')
    intersected = gpd.overlay(gdf, pr, how='intersection')
    crs = 'EPSG:32650' 
    extent = intersected.to_crs(crs)
    area = round(extent.geometry.area.sum()/1000000,2)
    intersected_json = intersected.__geo_interface__
    fig = px.choropleth_mapbox(
        intersected, 
        geojson=intersected_json, 
        locations=intersected.index, 
        color_discrete_sequence=['#62559f'],
        mapbox_style="carto-positron",
        zoom=2.5, 
        center={"lat": 35, "lon": 110},
        labels={'constant': '影响区域'},
        hover_data={'pr_中文': True}
    )
    fig.update_traces(marker_line_color='rgba(0,0,0,0)')
    return area,fig

#生成江浙沪城市台风灾情geojson数据,返回geojson
def JZH_photo():
    if not check_file(path = '数据/poi.geojson'):
        cities = gpd.read_file('数据/江浙沪边界.shp')
        all_cities=cities['市'].tolist()
        #使用百度地图API获取各个城市的中心经纬度
        typhoon_city = pd.DataFrame({})
        service ="http://api.map.baidu.com/geocoding/v3/?"
        output = "json"
        AK= '' # fill in with your AK
        typhoon_city['name']=all_cities
        typhoon_city['photo']=None
        typhoon_city['lon']=None
        typhoon_city['lat']=None
        for a in all_cities:
            parameters = f"address={a}&output={output}&ak={AK}"
            url = service + parameters
            response = requests.get(url)
            text=response.text
            dic=json.loads(text)
            status = dic["status"]
            if status==0:
                lng = dic["result"]["location"]["lng"]
                lat = dic["result"]["location"]["lat"]
                typhoon_city.loc[typhoon_city['name'] == a, 'photo'] = '图片/江浙沪城市台风灾害灾情概况/{}.jpg'.format(a[:-1])
                typhoon_city.loc[typhoon_city['name'] == a, 'lon'] = lng
                typhoon_city.loc[typhoon_city['name'] == a, 'lat'] = lat
        typhoon_city.to_excel('数据/城市台风图.xlsx', index = False)
        name=typhoon_city['name']
        photo=typhoon_city['photo']
        lat=typhoon_city['lat']
        lon=typhoon_city['lon']
        features=[]
        for i in range(len(typhoon_city)):
            feature={'type':'Feature','properties':{"NAME":name[i],'photo':photo[i]},'geometry':{'type':'Point','coordinates':[lon[i],lat[i]]}}
            features.append(feature)
        geojson_data={'type':'FeatureCollection','features':features}
        with open('数据/poi.geojson','w') as f:
            json.dump(geojson_data,f)
        gdf=gpd.read_file('数据/poi.geojson')
    else:
        gdf=gpd.read_file('数据/poi.geojson')
    return gdf

#计算江浙沪地区台风登陆的数量与强度
def cal_JZH(extent_JZH, filtered_CN):
    if not check_file(path = '数据/江浙沪城市台风.xlsx'):
        all_cities=extent_JZH['市'].tolist()
        cities_affected=[] #用于存储每次台风影响的城市
        city_intensity = {} #用于统计每个城市受到的历次台风等级之和，该等级来源于爬虫数据
        counts = {} #用于统计每个城市10年间过境的台风个数
        for city in all_cities:
            city_intensity[city] = 0 #city_intensity的keys为'江浙沪边界.shp'中的各个城市的名字，values初始值为0
            counts[city] = 0
        ids = pd.unique(filtered_CN['Id']).tolist() 
        for id in ids:
            single_tp = filtered_CN[filtered_CN['Id'] == id]
            points = [Point(lon, lat) for lon, lat in zip(single_tp['Lon'], single_tp['Lat'])] #获取该次台风的各时间台风中心的经纬度数据
            buffered_points = [point.buffer(300/111) for point in points] #设置缓冲区，缓冲区半径为台风7级风圈300km，坐标系为EPSG:4326，所以buffer半径的单位为度，1度约为111km
            geometries = GeoDataFrame(geometry = buffered_points)
            geometries.set_crs('EPSG:4326', inplace=True)
            intersected = gpd.overlay(geometries, extent_JZH, how='intersection') #获取台风影响范围和江浙沪地区的重合部分
            if not intersected.empty: #若台风7级风圈与研究区有重合范围，则保存该台风数据
                classes=single_tp['Class'].tolist()
                intensity=sum(classes)/len(classes) #计算该台风的平均等级
                affected_cities = list(set(intersected['市'].tolist())) #获取受到该台风影响的城市，删除其中的重复值
                cities_affected.append(affected_cities)
                for c in affected_cities:
                    city_intensity[c] += intensity #在字典中加上此次台风的平均等级
        for m in cities_affected:
            for n in m:
                if n in counts:
                    counts[n] += 1
                else:
                    counts[n] = 1
        counts = dict(sorted(counts.items(), key=lambda x: x[0]))
        ave_intensity = {} #用于统计每个城市10年间受到的平均台风等级
        for i in all_cities:
            ave_intensity[i]=round(city_intensity[i]/counts[i],2)
        ave_intensity = dict(sorted(ave_intensity.items(), key=lambda x: x[0]))
        df1 = pd.DataFrame(list(ave_intensity.items()), columns=['城市', '台风强度'])
        df2 = pd.DataFrame(list(counts.items()), columns=['城市', '台风数量'])
        result_df = pd.merge(df1, df2, on='城市')
        result_df.to_excel('数据/江浙沪城市台风.xlsx', encoding = 'utf-8', index = False)
    return True

#绘制江浙沪各城市台风强度和数量图，返回结果为图fig
def draw_JZHtp():
    df = pd.read_excel('数据/江浙沪城市台风.xlsx')
    cities_list = df['城市']
    intensity_values = df['台风强度']
    counts_values = df['台风数量']
    bar_trace = go.Bar(
        x=cities_list,
        y=intensity_values,
        name='强度',
        marker=dict(color='#ffe17c')
    )
    line_trace = go.Scatter(
        x=cities_list,
        y=counts_values,
        mode='lines+markers',
        name='频数',
        yaxis='y2',
        line=dict(color='#bb68be')
    )
    layout = go.Layout( 
        xaxis=dict(title='城市',  tickangle=45, dtick=1),
        yaxis=dict(title='历年台风平均强度'),
        yaxis2=dict(title='历年台风总数量（个）', overlaying='y', side='right'),
        legend=dict(x=1.1, y=0.5) 
    )
    fig = go.Figure(data=[bar_trace, line_trace], layout=layout)
    fig.update_layout(width=900)
    return fig

#计算江浙沪各市的路网密度
def road_density():
    df=pd.read_excel('数据/台风灾害风险评估数据.xlsx',header=0)
    if '路网密度' not in df.columns:
        df['路网密度'] = float('nan')
        cities = gpd.read_file('数据/JZH_cities.shp')  
        road_CN=gpd.read_file('数据/china_roads.shp',encoding="utf8")
        density=[]
        for i in cities['市'].tolist():
            city_name=cities[cities["市"]==i]
            polygon=city_name.iloc[0]["geometry"]
            road=road_CN.intersection(polygon)
            a=round(polygon.area/1000000,4)
            l=round(sum(road.length)/1000,3)
            den=round(l/a,4)
            density.append(den)
        cities['road_density']=density
        for index, row in cities.iterrows():
            city_name = row['市']
            road_density=row['road_density']
            row_index = df[df.iloc[:, 0] == city_name].index
            df.iloc[row_index, df.columns.get_loc('路网密度')] = road_density
        df.to_excel('数据/台风灾害风险评估数据.xlsx', index=False)
    return True

#计算江浙沪各市的平均海拔
def cal_dem():
    df = pd.read_excel('数据/台风灾害风险评估数据.xlsx', header=0)
    if '平均海拔' not in df.columns:
        df['平均海拔'] = float('nan')
        cities = gpd.read_file('数据/江浙沪边界.shp')
        raster_file = '数据/DEM_JZH.tif'
        for i in range(len(cities)):
            selected_feature = cities.iloc[i]
            selected_vector_data = gpd.GeoDataFrame(geometry=[selected_feature.geometry])
            stats = rasterstats.zonal_stats(selected_vector_data, raster_file, stats="mean", nodata=-9999)
            row_index = df.index[df.iloc[:, 0] == selected_feature['市']][0]                
            df.loc[row_index, '平均海拔'] = stats[0]['mean']
        df.to_excel('数据/台风灾害风险评估数据.xlsx', index=False)
    return True

#使用层次分析法分析各城市的台风风险分数，返回结果为字典
def APH():
    weights = {
        "人口密度": 0.0438,
        "第一产业占GDP比重": 0.0233,
        "GDP": 0.0233,
        "医生人数": 0.0116,
        "医院床位数": 0.0233,
        "水利管理及相近行业从业人员数": 0.0116,
        "客运总量": 0.0151,
        "建成区绿化覆盖率": 0.1190,
        "平均海拔": 0.1784,
        "路网密度": 0.0116,
        "台风强度": 0.3593,
        "台风数量": 0.1797
    }
    df1 = pd.read_excel('数据/台风灾害风险评估数据.xlsx',header=0)
    df2 = pd.read_excel('数据/江浙沪城市台风.xlsx',header=0)
    df = pd.merge(df1, df2, on='城市')
    normalized = pd.DataFrame()
    normalized[df.columns[0]] = df.iloc[:, 0]
    for column_name in df.columns[1:]:
        print(column_name)
        indicator = df[column_name]
        value_max = max(indicator)
        value_min = min(indicator)
        if column_name in ['医生人数', '医院床位数', '水利管理及相近行业从业人员数', '建成区绿化覆盖率', '平均海拔']:# 负指标
            normalized_column = (value_max - indicator) / (value_max - value_min)
        else:# 正指标
            normalized_column = (indicator - value_min) / (value_max - value_min)
        normalized[column_name] = normalized_column
    total_score = np.dot(normalized.iloc[:, 1:13], list(weights.values()))
    normalized['总分'] = total_score
    keys = normalized.iloc[:, 0]
    values = normalized.iloc[:, -1]
    result_dict = dict(zip(keys, values))
    normalized.to_excel('数据/normalized.xlsx', index=False, encoding='utf-8-sig')
    sorted_dict = dict(sorted(result_dict.items(), key=lambda x: x[0][0]))
    return sorted_dict

#绘制历年台风移动方向雷达图，返回结果为图fig
def tp_directions(filepath):
    df = pd.read_excel(filepath)
    df_grouped = df.groupby('Id')
    angles = {}
    for name, group in df_grouped:
        start_lat, start_lon = group.iloc[0]['Lat'], group.iloc[0]['Lon']
        end_lat, end_lon = group.iloc[-1]['Lat'], group.iloc[-1]['Lon']
        angle = np.arctan2(end_lat - start_lat, end_lon - start_lon) * 180 / np.pi
        if angle < 0:
            angle += 360
        angles[name] = angle
    quadrants_count = {'NE': 0, 'SE': 0, 'SW': 0, 'NW': 0}
    for angle in angles.values():
        if 0 <= angle < 90:
            quadrants_count['NE'] += 1
        elif 90 <= angle < 180:
            quadrants_count['SE'] += 1
        elif 180 <= angle < 270:
            quadrants_count['SW'] += 1
        else:
            quadrants_count['NW'] += 1
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(quadrants_count.values()),
        theta=['NE', 'SE', 'SW', 'NW'],
        fill='toself',
        line=dict(color='#bb68be'),
        name='台风数量'
    ))
    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                direction="clockwise",
                rotation=45
            ),
            radialaxis=dict(
                visible=True,
                range=[0, max(quadrants_count.values())]
            ),
            bgcolor='#ffe17c'
        ),
        showlegend=True,
    )
    return fig

#绘制历年台风数量图，返回结果为图fig
def tp_number(field1, field2,xlabel):
    df1 = pd.read_excel('数据/中国台风数据.xlsx')
    df1_unique = df1.drop_duplicates(subset=['Id'], keep='first')
    df1_year = df1_unique[field1].value_counts()
    sorted_df1_year = df1_year.sort_index()
    df2 = pd.read_excel('数据/台风登陆数据.xlsx')
    df2_year = df2[field2].value_counts()
    sorted_df2_year = df2_year.sort_index()
    if field1 == 'Month':
        all_months = pd.Series(range(1, 13))
        sorted_df2_year = sorted_df2_year.reindex(all_months, fill_value=0)
    diff = sorted_df1_year - sorted_df2_year
    trace1 = go.Bar(x=sorted_df2_year.index, y=diff.values, name='影响中国但未在中国登陆的台风', marker=dict(color='#ffe17c'))
    trace2 = go.Bar(x=sorted_df2_year.index, y=sorted_df2_year.values, name='在中国登陆的台风', marker=dict(color='#bb68be'))
    trace = [trace1, trace2]
    layout = go.Layout(
        xaxis=dict(
            title=xlabel,
            tickmode='linear',
            dtick=1
        ),
        yaxis=dict(title='数量'),
        barmode='stack',
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )
    fig = go.Figure(data=trace, layout=layout)
    return fig

#绘制某年台风数量逐月图，返回结果为图fig
def tp_numberbyyear(field1, field2, year):
    df1 = pd.read_excel('数据/中国台风数据.xlsx')
    df1_unique = df1.drop_duplicates(subset=['Id'], keep='first')
    df1_year = df1_unique[df1_unique['Year'] == year][field1].value_counts().sort_index()
    df1_year = df1_year.reindex(range(1, 13), fill_value=0)
    df2 = pd.read_excel('数据/台风登陆数据.xlsx')
    df2_year = df2[df2['年'] == year][field2].value_counts().sort_index()
    df2_year = df2_year.reindex(range(1, 13), fill_value=0)
    for i in range(len(df1_year.values)):
        if df1_year.values[i] < df2_year.values[i]:
            df1_year.values[i] = df2_year.values[i]
    trace1 = go.Bar(x=df1_year.index, y=df1_year.values, name='影响中国的台风', marker=dict(color='#ffe17c'))
    trace2 = go.Bar(x=df2_year.index, y=df2_year.values, name='在中国登陆的台风', marker=dict(color='#bb68be'))
    trace = [trace1, trace2]
    layout = go.Layout(
        xaxis=dict(
            title='月份',
            tickmode='linear',
            dtick=1
        ),
        yaxis=dict(title='数量'),
        barmode='group',
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )
    fig = go.Figure(data=trace, layout=layout)
    return fig

#绘制历年台风登陆省份柱形图，返回结果为图fig
def tp_pr():
    df = pd.read_excel('数据/台风登陆数据.xlsx')
    df_stat = df['省份'].value_counts()
    sorted = df_stat.sort_index()
    x_values = sorted.index
    y_values = sorted.values
    fig = go.Figure([go.Bar(x=x_values, y=y_values,marker=dict(color='#ffe17c'))])
    fig.update_layout(
        xaxis=dict(
            title='省份',
            tickmode='array',
            tickvals=x_values,
            ticktext=x_values,
            tickangle=45
        ),  
        yaxis=dict(title='数量')
    )
    
    return fig

#绘制历年台风登陆强度等级饼图，返回结果为图fig
def tp_intensity():
    df = pd.read_excel('数据/台风登陆数据.xlsx')
    df['登陆时强度等级'] = df['登陆时强度等级'].map({
        'TD': '热带低压',
        'TS': '热带风暴',
        'STS': '强热带风暴',
        'TY': '台风',
        'STY': '强台风',
        'SuperTY': '超强台风'
    })
    df_stat = df['登陆时强度等级'].value_counts()
    sorted_counts = df_stat.sort_index()
    labels = sorted_counts.index
    values = sorted_counts.values
    custom_colors = ['#62559f','#965c9d','#ba6893','#dd8587','#eda784','#f1e3a1']
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        textinfo='label+percent',
        insidetextorientation='radial',
        textposition='outside',
        marker_colors=custom_colors
    ))
    fig.update_layout(
        showlegend=False
    )
    return fig

#绘制江浙沪城市台风灾情图，返回结果为html_map
def jzh_situation(gdf):
    tiles = 'http://webst01.is.autonavi.com/appmaptile?style=7&x={x}&y={y}&z={z}'
    m=folium.Map(location=[31,120],tiles=tiles,name="高德地图",attr="高德地图", zoom_start=6,width=800,height=500)
    layer=FeatureGroup(name='POI',control=True)
    for i in range(len(gdf)):
        record=gdf.iloc[i]
        lng=float(record["geometry"].x)
        lat=float(record["geometry"].y)
        name=record['NAME']
        with open(record['photo'], "rb") as p:
            img_data = p.read()
            img_base64 = base64.b64encode(img_data).decode("utf-8")
        html=f"""<h4>{name}</h4>
                <img src="data:image/jpeg;base64,{img_base64}">"""
        popup=folium.Popup(html,max_width=300)
        Marker(location=(lat,lng),popup=popup).add_to(layer)
    layer.add_to(m)
    folium.LayerControl().add_to(m)
    minimap=MiniMap(toggle_display=True)
    m.add_child(minimap)
    html_map = m._repr_html_()   
    return html_map

#绘制江浙沪各城市台风风险评估图，返回结果为图fig
def jzh_assessment(sorted_dict, cities):
    cities_sorted = cities.sort_values(by='市')
    cities_sorted['score'] = [sorted_dict[city] for city in cities_sorted['市']]
    fig = px.choropleth_mapbox(cities_sorted, 
                                geojson=cities_sorted.geometry, 
                                locations=cities_sorted.index, 
                                color='score',
                                color_continuous_scale="Sunset",
                                mapbox_style="carto-positron",
                                zoom=5, 
                                center={"lat": 31, "lon": 120},
                                opacity=0.7,
                                labels={'score': '评分'},
                                hover_data={'市': True, 'score': True}
                                )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

#绘制江浙沪各城市台风风险指标相关性热力图，返回结果为图fig
def correlation(df):
    df = df.drop(columns=[df.columns[0]])    
    fig = go.Figure(data=go.Heatmap(
            z=df.corr(),
            x=df.columns,
            y=df.columns[::-1],
            colorscale='Sunset'
        ))
    fig.update_layout(
        width=700,
        height=650
    )
    fig.update_xaxes(tickangle=45)
    return fig
