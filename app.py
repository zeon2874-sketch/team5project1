import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import koreanize_matplotlib
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(page_title="서울 신혼부부 최적 주거지 추천 (고도화)", layout="wide")

# 자치구별 중심 좌표 (지도 시각화용)
GU_COORDS = {
    '강남구': [37.4959, 127.0664], '강동구': [37.5492, 127.1464], '강북구': [37.6469, 127.0147],
    '강서구': [37.5658, 126.8226], '관악구': [37.4653, 126.9438], '광진구': [37.5481, 127.0857],
    '구로구': [37.4954, 126.8581], '금천구': [37.4600, 126.9008], '노원구': [37.6552, 127.0771],
    '도봉구': [37.6658, 127.0317], '동대문구': [37.5838, 127.0507], '동작구': [37.4965, 126.9443],
    '마포구': [37.5509, 126.9086], '서대문구': [37.5820, 126.9356], '서초구': [37.4769, 127.0378],
    '성동구': [37.5506, 127.0409], '성북구': [37.5891, 127.0182], '송파구': [37.5048, 127.1144],
    '양천구': [37.5270, 126.8543], '영등포구': [37.5206, 126.9139], '용산구': [37.5311, 126.9811],
    '은평구': [37.6176, 126.9227], '종로구': [37.5859, 126.9848], '중구': [37.5579, 126.9941],
    '중랑구': [37.5953, 127.0936]
}

@st.cache_data
def load_data():
    """데이터 로드 및 기본 전처리 (로컬 및 배포 환경 대응)"""
    # 파일 경로 후보군 (GitHub 업로드 구조 고려)
    paths = [
        'outputs/analysis_base_table.csv',
        'team5/outputs/analysis_base_table.csv',
        '../outputs/analysis_base_table.csv'
    ]
    
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                st.error(f"파일을 읽는 중 오류가 발생했습니다 ({path}): {e}")
    
    st.error("데이터 파일(analysis_base_table.csv)을 찾을 수 없습니다. 경로를 확인해 주세요.")
    return None

def calculate_scores(df, weights):
    """가중치 기반 점수 산출"""
    temp_df = df.copy()
    
    # 정규화 대상 컬럼
    cols_to_norm = {
        '평균전세가': True, '평균월세': True, '범죄건수': True,
        '거래량': False, '공원수': False, '마트수': False,
        '정비사업수': False, '병원수': False, '만족도': False
    }
    
    scaler = MinMaxScaler()
    for col, reverse in cols_to_norm.items():
        if col in temp_df.columns:
            norm_val = scaler.fit_transform(temp_df[[col]])
            temp_df[f'{col}_점수'] = (1 - norm_val) * 100 if reverse else norm_val * 100
    
    # 요소별 그룹화 점수
    temp_df['가격_요소'] = (temp_df['평균전세가_점수'] + temp_df['평균월세_점수']) / 2
    temp_df['인프라_요소'] = (temp_df['공원수_점수'] + temp_df['마트수_점수'] + temp_df['병원수_점수']) / 3
    temp_df['안전_요소'] = (temp_df['범죄건수_점수'] + temp_df['만족도_점수']) / 2
    temp_df['개발_요소'] = temp_df['정비사업수_점수']
    temp_df['거래량_요소'] = temp_df['거래량_점수']
    
    # 최종 종합 점수
    temp_df['종합점수'] = (
        temp_df['가격_요소'] * (weights['가격'] / 100) +
        temp_df['인프라_요소'] * (weights['인프라'] / 100) +
        temp_df['안전_요소'] * (weights['안전'] / 100) +
        temp_df['개발_요소'] * (weights['개발'] / 100) +
        temp_df['거래량_요소'] * (weights['거래량'] / 100)
    )
    
    return temp_df.sort_values(by='종합점수', ascending=False)

def main():
    st.title("🏠 서울 신혼부부 최적 주거지 추천 (고도화 버전)")
    st.markdown("---")
    
    df = load_data()
    if df is None: return

    # --- 사이드바: 글로벌 필터 및 가중치 ---
    st.sidebar.header("🔍 상세 조건 설정")
    
    st.sidebar.subheader("💰 예산 상한선 (만원)")
    jeonse_limit = st.sidebar.slider("평균 전세가", 0, int(df['평균전세가'].max()), 45000, 1000)
    rent_limit = st.sidebar.slider("평균 월세", 0, int(df['평균월세'].max()), 100, 5)
    
    st.sidebar.subheader("⚖️ 중요도 가중치 (%)")
    w_price = st.sidebar.slider("가격 (낮음 선호)", 0, 100, 30)
    w_infra = st.sidebar.slider("인프라 (공원/마트/병원)", 0, 100, 20)
    w_safety = st.sidebar.slider("안전 (치안/만족도)", 0, 100, 20)
    w_dev = st.sidebar.slider("개발 가치 (정비사업)", 0, 100, 15)
    w_vol = st.sidebar.slider("거래 활성도 (거래량)", 0, 100, 15)
    
    total_w = w_price + w_infra + w_safety + w_dev + w_vol
    norm_f = 100 / total_w if total_w != 0 else 1
    weights = {'가격': w_price * norm_f, '인프라': w_infra * norm_f, '안전': w_safety * norm_f, '개발': w_dev * norm_f, '거래량': w_vol * norm_f}

    # 데이터 필터링 및 점수 계산
    filtered_df = df[(df['평균전세가'] <= jeonse_limit) & (df['평균월세'] <= rent_limit)]
    if filtered_df.empty:
        st.error("조건에 맞는 자치구가 없습니다. 예산을 늘려보세요.")
        return
    
    scored_df = calculate_scores(filtered_df, weights)
    
    # --- 탭 구성 ---
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 추천 요약", "📊 구별 상세 비교", "🏙️ 인프라 심층 분석", "🛡️ 안전/개발 가치"])

    # --- Tab 1: 추천 요약 ---
    with tab1:
        st.subheader(f"🏆 신혼부부 추천 Top 5")
        top_5 = scored_df.head(5).reset_index(drop=True)
        cols = st.columns(5)
        for i, row in top_5.iterrows():
            with cols[i]:
                st.metric(label=f"{i+1}위 {row['자치구']}", value=f"{row['종합점수']:.1f}점")
                st.write(f"💰 전세: {row['평균전세가']:,.0f} / 월세: {row['평균월세']:,.1f}")

        col_map, col_chart = st.columns([6, 4])
        with col_map:
            st.markdown("#### 📍 자치구별 점수 지도")
            m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
            for idx, row in scored_df.iterrows():
                gu = row['자치구']
                if gu in GU_COORDS:
                    color = 'red' if idx in top_5.index else 'blue'
                    folium.CircleMarker(location=GU_COORDS[gu], radius=row['종합점수']/5, 
                                        popup=f"{gu}: {row['종합점수']:.1f}점", 
                                        color=color, fill=True, fill_opacity=0.5).add_to(m)
            folium_static(m)
        
        with col_chart:
            st.markdown("#### 📊 종합 점수 순위 (Top 15)")
            fig = px.bar(scored_df.head(15), x='종합점수', y='자치구', orientation='h', color='종합점수',
                         color_continuous_scale='Viridis', text_auto='.1f')
            st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: 구별 상세 비교 ---
    with tab2:
        st.subheader("🔍 자치구 맞춤형 비교 분석")
        selected_gus = st.multiselect("비교할 자치구를 선택하세요", scored_df['자치구'].tolist(), default=top_5['자치구'].tolist()[:3])
        
        if selected_gus:
            comp_df = scored_df[scored_df['자치구'].isin(selected_gus)]
            
            # 레이더 차트 (요소별 비교)
            radar_cols = ['가격_요소', '인프라_요소', '안전_요소', '개발_요소', '거래량_요소']
            fig_radar = go.Figure()
            for gu in selected_gus:
                gu_data = comp_df[comp_df['자치구'] == gu]
                fig_radar.add_trace(go.Scatterpolar(
                    r=gu_data[radar_cols].values[0],
                    theta=['가격', '인프라', '안전', '개발', '거래량'],
                    fill='toself', name=gu
                ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="요소별 점수 레이더 비교")
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # 주요 지표 멀티 바 차트
            metrics = st.selectbox("비교할 지표 선택", ['평균전세가', '평균월세', '거래량', '범죄건수', '만족도'])
            fig_bar = px.bar(comp_df, x='자치구', y=metrics, color='자치구', title=f"구별 {metrics} 수치 비교")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.dataframe(comp_df[['자치구', '종합점수'] + metrics.split()].style.highlight_max(axis=0))

    # --- Tab 3: 인프라 심층 분석 ---
    with tab3:
        st.subheader("🏙️ 편의 시설 및 인프라 분포")
        infra_cols = ['공원수', '마트수', '병원수']
        
        col_box, col_scatter = st.columns(2)
        with col_box:
            # 인프라 지표별 박스플롯
            melted_infra = df.melt(value_vars=infra_cols, var_name='항목', value_name='수치')
            fig_box = px.box(melted_infra, x='항목', y='수치', points="all", title="서울시 전체 인프라 분포 및 이상치")
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col_scatter:
            # 버블 차트 (전세가 vs 인프라 총합)
            df['인프라_총합'] = df[infra_cols].sum(axis=1)
            fig_bubble = px.scatter(df, x='평균전세가', y='인프라_총합', size='병원수', color='자치구',
                                    hover_name='자치구', title="가격 대비 인프라 가성비 분석",
                                    labels={'인프라_총합': '인프라 수치 합계 (공원+마트+병원)'})
            st.plotly_chart(fig_bubble, use_container_width=True)

    # --- Tab 4: 안전/개발 가치 ---
    with tab4:
        st.subheader("🛡️ 주거 안전 및 미래 개발 잠람성")
        
        c1, c2 = st.columns(2)
        with c1:
            # 범죄 vs 경찰 만족도 상관관계 산점도
            fig_safety = px.scatter(df, x='범죄건수', y='만족도', text='자치구', size='범죄건수', 
                                    color='만족도', title="치안 만족도 분석",
                                    labels={'만족도': '경찰 서비스 만족도'})
            st.plotly_chart(fig_safety, use_container_width=True)
        
        with c2:
            # 정비사업 현황 바 차트
            fig_dev = px.bar(df.sort_values('정비사업수', ascending=False), x='자치구', y='정비사업수',
                             color='정비사업수', title="자치구별 정비사업(재개발/재건축) 추진 현황",
                             color_continuous_scale='Reds')
            st.plotly_chart(fig_dev, use_container_width=True)

        # 상관관계 히트맵
        st.markdown("#### 🔗 전체 지표 간 상관관계 분석")
        corr = df[['평균전세가', '평균월세', '거래량', '공원수', '마트수', '병원수', '범죄건수', '만족도', '정비사업수']].corr()
        fig_heat = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', 
                             title="데이터 지표 간 상관계수 (Heatmap)")
        st.plotly_chart(fig_heat, use_container_width=True)

if __name__ == "__main__":
    main()
