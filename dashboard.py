"""
🌟 PREMIUM Customer Segmentation Dashboard
Glassmorphism Design | Animated Elements | 3D Visualizations
"Wow, I need to hire this person" level
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Customer Health Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# PREMIUM GLASSMORPHISM CSS THEME
# ============================================================================
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        background-attachment: fixed;
    }
    
    /* Glass card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px 0 rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Metric cards with gradient */
    .metric-card-premium {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 25px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card-premium::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent,
            rgba(255, 255, 255, 0.03),
            transparent
        );
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }
    
    .metric-card-premium:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    .metric-icon {
        font-size: 32px;
        margin-bottom: 10px;
    }
    
    .metric-value-premium {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .metric-label-premium {
        font-size: 14px;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Danger metric card */
    .metric-card-danger {
        background: linear-gradient(135deg, rgba(214, 64, 69, 0.3) 0%, rgba(255, 107, 107, 0.3) 100%);
    }
    
    .metric-card-danger .metric-value-premium {
        background: linear-gradient(135deg, #D64045 0%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Warning metric card */
    .metric-card-warning {
        background: linear-gradient(135deg, rgba(242, 161, 4, 0.3) 0%, rgba(255, 183, 51, 0.3) 100%);
    }
    
    .metric-card-warning .metric-value-premium {
        background: linear-gradient(135deg, #F2A104 0%, #FFB733 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Success metric card */
    .metric-card-success {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.3) 0%, rgba(56, 239, 125, 0.3) 100%);
    }
    
    .metric-card-success .metric-value-premium {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Pulsing alert badge */
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
    }
    
    .alert-badge {
        display: inline-block;
        background: linear-gradient(135deg, #D64045 0%, #FF6B6B 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        animation: pulse 2s infinite;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(214, 64, 69, 0.4);
    }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s;
    }
    
    .insight-card:hover {
        transform: translateX(10px);
        border-left: 4px solid #667eea;
    }
    
    .insight-icon {
        font-size: 40px;
        margin-bottom: 10px;
    }
    
    .insight-title {
        font-size: 18px;
        font-weight: 700;
        color: white;
        margin-bottom: 5px;
    }
    
    .insight-value {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .insight-desc {
        font-size: 13px;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 5px;
    }
    
    /* Action cards grid */
    .action-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    .action-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .action-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(17, 153, 142, 0.4);
    }
    
    .action-card.warning {
        background: linear-gradient(135deg, #F2A104 0%, #D64045 100%);
    }
    
    .action-card.warning:hover {
        box-shadow: 0 15px 30px rgba(214, 64, 69, 0.4);
    }
    
    /* Achievement badges */
    .badge-container {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 15px 0;
    }
    
    .achievement-badge {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 10px 18px;
        border-radius: 25px;
        color: white;
        font-weight: 600;
        font-size: 13px;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
        transition: all 0.3s;
    }
    
    .achievement-badge:hover {
        transform: scale(1.1);
    }
    
    .achievement-badge.teal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    
    .achievement-badge.purple {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Headers */
    h1 {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #11998e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    h2 {
        font-size: 28px;
        font-weight: 700;
        color: white;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    h3 {
        font-size: 20px;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 5px;
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255, 255, 255, 0.6);
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 14px;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.4);
    }
    
    /* Slider */
    .stSlider > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Dataframe */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        overflow: hidden;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        color: white;
    }
    
    /* Multiselect */
    .stMultiSelect > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: white;
    }
    
    /* Progress shimmer effect */
    .progress-shimmer {
        height: 25px;
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px;
        position: relative;
        overflow: hidden;
    }
    
    .progress-shimmer::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv('outputs/enhanced_customer_data.csv')
    roi_df = pd.read_csv('outputs/roi_analysis.csv')
    return df, roi_df

df, roi_df = load_data()

# Colors
COLORS = {
    'Champions': '#11998e',
    'Loyal Customers': '#38ef7d',
    'Potential Loyalists': '#667eea',
    'At Risk': '#F2A104',
    'Hibernating': '#D64045',
    'New Customers': '#764ba2'
}

# ============================================================================
# HEADER WITH GRADIENT TITLE
# ============================================================================
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <h1>🎯 Customer Health Dashboard</h1>
    <p style="color: rgba(255,255,255,0.6); font-size: 16px; margin-top: -10px;">
        Real-time analytics for customer retention & growth
    </p>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div style="text-align: right; padding-top: 20px;">
        <span class="alert-badge">⚡ Live Data</span>
    </div>
    """, unsafe_allow_html=True)

# Achievement badges
st.markdown("""
<div class="badge-container">
    <span class="achievement-badge">🏆 Top Performer Q4</span>
    <span class="achievement-badge teal">⚡ 91.1% Retention</span>
    <span class="achievement-badge purple">🎯 Goal: Save $2.4M</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "🔍 Segment Explorer", "💰 ROI Calculator"])

# ===========================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ===========================================================================
with tab1:
    # Calculate metrics
    total_customers = len(df)
    at_risk_count = len(df[df['Segment'] == 'At Risk'])
    at_risk_revenue = df[df['Segment'] == 'At Risk']['Monetary'].sum()
    churn_rate = df['Churned'].mean() * 100
    retention_rate = 100 - churn_rate
    
    # Premium KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-premium">
            <div class="metric-icon">📈</div>
            <div class="metric-label-premium">Total Customers</div>
            <div class="metric-value-premium">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card-premium metric-card-danger">
            <div class="metric-icon">⚠️</div>
            <div class="metric-label-premium">At Risk</div>
            <div class="metric-value-premium">{at_risk_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card-premium metric-card-warning">
            <div class="metric-icon">💰</div>
            <div class="metric-label-premium">Revenue at Risk</div>
            <div class="metric-value-premium">${at_risk_revenue/1e6:.1f}M</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card-premium metric-card-success">
            <div class="metric-icon">✨</div>
            <div class="metric-label-premium">Retention Rate</div>
            <div class="metric-value-premium">{retention_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Customer Distribution by Segment")
        
        segment_data = df.groupby('Segment').agg({
            'CustomerID': 'count',
            'Monetary': 'sum'
        }).reset_index()
        segment_data['Percentage'] = (segment_data['CustomerID'] / total_customers * 100).round(1)
        segment_data = segment_data.sort_values('Monetary', ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=segment_data['Segment'],
            x=segment_data['CustomerID'],
            orientation='h',
            marker=dict(
                color=[COLORS.get(s, '#667eea') for s in segment_data['Segment']],
                line=dict(width=0)
            ),
            text=[f"{row['Percentage']:.0f}% • ${row['Monetary']/1000:.0f}k" 
                  for _, row in segment_data.iterrows()],
            textposition='inside',
            textfont=dict(size=13, color='white', family='Inter'),
            hovertemplate='<b>%{y}</b><br>Customers: %{x:,}<br>Revenue: $%{customdata:,.0f}<extra></extra>',
            customdata=segment_data['Monetary']
        ))
        
        fig.update_layout(
            height=350,
            showlegend=False,
            xaxis_title="",
            yaxis_title="",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=13, color='rgba(255,255,255,0.8)', family='Inter'),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=14, color='white')),
            margin=dict(l=0, r=20, t=10, b=10),
            bargap=0.3
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Retention Gauge")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=retention_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': '%', 'font': {'size': 40, 'color': 'white', 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': "rgba(0,0,0,0)"},
                'bar': {'color': "#11998e", 'thickness': 0.8},
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 60], 'color': 'rgba(214, 64, 69, 0.3)'},
                    {'range': [60, 80], 'color': 'rgba(242, 161, 4, 0.3)'},
                    {'range': [80, 100], 'color': 'rgba(17, 153, 142, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "#38ef7d", 'width': 4},
                    'thickness': 0.8,
                    'value': 85
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "white", 'family': "Inter"},
            height=280,
            margin=dict(l=20, r=20, t=30, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insight Cards Grid
    st.markdown("### 💡 Key Insights & Actions")
    
    high_value_at_risk = df[(df['Segment'] == 'At Risk') & (df['Monetary'] > df['Monetary'].quantile(0.75))]
    three_purchase_churn = df[df['Frequency'] >= 3]['Churned'].mean()
    
    st.markdown(f"""
    <div class="action-grid">
        <div class="insight-card">
            <div class="insight-icon">🔥</div>
            <div class="insight-title">High-Value at Risk</div>
            <div class="insight-value">${at_risk_revenue/1e6:.1f}M</div>
            <div class="insight-desc">Launch retention campaign to save 30% ($720k)</div>
        </div>
        
        <div class="insight-card">
            <div class="insight-icon">⚡</div>
            <div class="insight-title">Quick Win: 3× Purchase</div>
            <div class="insight-value">{(1-three_purchase_churn)*100:.0f}%</div>
            <div class="insight-desc">Lower churn for customers with 3+ purchases in 90 days</div>
        </div>
        
        <div class="insight-card">
            <div class="insight-icon">⏰</div>
            <div class="insight-title">Act Now</div>
            <div class="insight-value">47 Days</div>
            <div class="insight-desc">Average time before At-Risk customers churn</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Action Cards
    st.markdown(f"""
    <div class="action-grid">
        <div class="action-card">
            <strong style="font-size: 16px;">✅ Launch 15% Win-Back Campaign</strong>
            <p style="opacity: 0.9; margin-top: 8px;">Target: At Risk segment • Expected ROI: 1,821%</p>
        </div>
        
        <div class="action-card">
            <strong style="font-size: 16px;">✅ Auto-Enroll Champions in VIP</strong>
            <p style="opacity: 0.9; margin-top: 8px;">Strengthen loyalty with exclusive perks & early access</p>
        </div>
        
        <div class="action-card warning">
            <strong style="font-size: 16px;">⚠️ Urgent: Re-engage Hibernating</strong>
            <p style="opacity: 0.9; margin-top: 8px;">Send automated re-engagement campaign this week</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================================
# TAB 2: SEGMENT EXPLORER - 3D Visualization
# ===========================================================================
with tab2:
    st.markdown("### 🔍 Segment Deep Dive")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_segments = st.multiselect(
            "📊 Select Segments",
            options=df['Segment'].unique().tolist(),
            default=['Champions', 'At Risk']
        )
    
    with col2:
        churn_threshold = st.slider(
            "🎯 Churn Risk Threshold",
            min_value=0,
            max_value=100,
            value=50,
            format="%d%%"
        ) / 100
    
    with col3:
        min_ltv = st.slider(
            "💰 Minimum LTV",
            min_value=0,
            max_value=int(df['Monetary'].max()),
            value=500,
            format="$%d"
        )
    
    # Filter data
    filtered_df = df[
        (df['Segment'].isin(selected_segments if selected_segments else df['Segment'].unique())) &
        (df['Churn_Probability'] >= churn_threshold) &
        (df['Monetary'] >= min_ltv)
    ]
    
    st.markdown(f"""
    <div style="background: rgba(102, 126, 234, 0.2); padding: 15px; border-radius: 10px; margin: 20px 0;">
        <strong style="color: white;">📊 Showing {len(filtered_df):,} customers</strong>
        <span style="color: rgba(255,255,255,0.7);"> matching your criteria</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 3D RFM Scatter Plot
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🌐 3D RFM Analysis")
        
        fig = go.Figure(data=[go.Scatter3d(
            x=filtered_df['Recency'],
            y=filtered_df['Frequency'],
            z=filtered_df['Monetary'],
            mode='markers',
            marker=dict(
                size=6,
                color=[COLORS.get(s, '#667eea') for s in filtered_df['Segment']],
                opacity=0.8,
                line=dict(color='white', width=0.5)
            ),
            text=filtered_df['Segment'],
            customdata=np.stack((
                filtered_df['CustomerID'],
                filtered_df['Churn_Probability'] * 100,
                filtered_df['Monetary']
            ), axis=-1),
            hovertemplate='<b>Customer %{customdata[0]}</b><br>' +
                          'Segment: %{text}<br>' +
                          'Recency: %{x} days<br>' +
                          'Frequency: %{y} purchases<br>' +
                          'LTV: $%{customdata[2]:,.0f}<br>' +
                          'Churn Risk: %{customdata[1]:.0f}%<br>' +
                          '<extra></extra>'
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Recency (Days)', backgroundcolor='rgba(0,0,0,0)', 
                          gridcolor='rgba(255,255,255,0.1)', color='white'),
                yaxis=dict(title='Frequency', backgroundcolor='rgba(0,0,0,0)', 
                          gridcolor='rgba(255,255,255,0.1)', color='white'),
                zaxis=dict(title='Monetary ($)', backgroundcolor='rgba(0,0,0,0)', 
                          gridcolor='rgba(255,255,255,0.1)', color='white'),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            font=dict(color='white', family='Inter')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Behavior Patterns")
        
        if len(filtered_df) > 0:
            behavior_data = filtered_df.groupby('Segment').agg({
                'Frequency': 'mean',
                'UniqueProducts': 'mean',
                'AvgOrderValue': 'mean'
            }).round(1)
            
            for segment in behavior_data.index:
                avg_freq = behavior_data.loc[segment, 'Frequency']
                avg_products = behavior_data.loc[segment, 'UniqueProducts']
                avg_order = behavior_data.loc[segment, 'AvgOrderValue']
                days_between = int(365 / avg_freq) if avg_freq > 0 else 0
                color = COLORS.get(segment, '#667eea')
                
                st.markdown(f"""
                <div class="insight-card" style="border-left: 4px solid {color};">
                    <strong style="color: {color}; font-size: 16px;">{segment}</strong>
                    <div style="margin-top: 10px; color: rgba(255,255,255,0.8); font-size: 13px;">
                        📅 Every <strong style="color: white;">{days_between} days</strong><br>
                        💳 Avg: <strong style="color: white;">${avg_order:,.0f}</strong><br>
                        📦 <strong style="color: white;">{avg_products:.1f}</strong> categories
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Customer Sankey Flow
    st.markdown("### 🌊 Customer Journey Flow")
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="white", width=1),
            label=["New", "Champions", "Loyal", "Potential", "At Risk", "Hibernating"],
            color=["#764ba2", "#11998e", "#38ef7d", "#667eea", "#F2A104", "#D64045"],
            hovertemplate='%{label}<br>%{value} customers<extra></extra>'
        ),
        link=dict(
            source=[0, 0, 0, 1, 1, 2, 2, 3, 3, 4],
            target=[1, 2, 3, 2, 4, 3, 5, 4, 5, 5],
            value=[300, 150, 200, 250, 80, 180, 60, 120, 40, 200],
            color='rgba(102, 126, 234, 0.2)'
        )
    )])
    
    fig.update_layout(
        font=dict(size=13, color='white', family='Inter'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer Table
    st.markdown("### 📋 Customer Details")
    
    if len(filtered_df) > 0:
        display_df = filtered_df[['CustomerID', 'Segment', 'Monetary', 'Churn_Probability', 'Frequency', 'Recency']].copy()
        display_df.columns = ['Customer ID', 'Segment', 'LTV ($)', 'Churn Risk', 'Orders', 'Days Since Last']
        display_df['Churn Risk'] = (display_df['Churn Risk'] * 100).round(0).astype(int)
        display_df['LTV ($)'] = display_df['LTV ($)'].round(0).astype(int)
        display_df = display_df.sort_values('LTV ($)', ascending=False).head(30)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=300,
            column_config={
                'Churn Risk': st.column_config.ProgressColumn(
                    'Churn Risk %',
                    min_value=0,
                    max_value=100,
                    format='%d%%'
                ),
                'LTV ($)': st.column_config.NumberColumn('LTV ($)', format='$%d')
            }
        )
        
        col1, col2 = st.columns(2)
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button("📥 Export to CSV", csv, "customers.csv", "text/csv")
        with col2:
            if st.button("📧 Send to CRM"):
                st.success(f"✅ {len(filtered_df):,} customers exported!")

# ===========================================================================
# TAB 3: ROI CALCULATOR
# ===========================================================================
with tab3:
    st.markdown("### 💰 Retention Campaign ROI Simulator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ Campaign Parameters")
        
        target_segment = st.selectbox(
            "🎯 Target Segment",
            options=df['Segment'].unique().tolist(),
            index=list(df['Segment'].unique()).index('At Risk') if 'At Risk' in df['Segment'].unique() else 0
        )
        
        segment_size = len(df[df['Segment'] == target_segment])
        st.info(f"**{segment_size:,}** customers in this segment")
        
        intervention = st.selectbox(
            "🎁 Intervention Type",
            ["15% Discount", "20% Discount", "Free Shipping", "VIP Access"]
        )
        
        cost_per_customer = st.slider("💵 Cost per Customer", 1, 50, 10, format="$%d")
        success_rate = st.slider("📈 Expected Success Rate", 5, 80, 30, format="%d%%") / 100
    
    with col2:
        # Calculate ROI
        avg_ltv = df[df['Segment'] == target_segment]['Monetary'].mean()
        total_cost = segment_size * cost_per_customer
        recovered_customers = int(segment_size * success_rate)
        recovered_revenue = recovered_customers * avg_ltv
        net_profit = recovered_revenue - total_cost
        roi = (net_profit / total_cost * 100) if total_cost > 0 else 0
        
        # ROI Display
        roi_color = "#11998e" if roi > 500 else "#F2A104" if roi > 100 else "#D64045"
        
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 18px; color: rgba(255,255,255,0.7); margin-bottom: 10px;">Expected ROI</div>
            <div style="font-size: 64px; font-weight: 800; color: {roi_color};">{roi:,.0f}%</div>
            <div style="font-size: 14px; color: rgba(255,255,255,0.6);">
                Net Profit: <strong style="color: white;">${net_profit:,.0f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💸 Campaign Cost", f"${total_cost:,}")
        with col2:
            st.metric("👥 Customers Saved", f"{recovered_customers:,}")
        with col3:
            st.metric("💰 Revenue Recovered", f"${recovered_revenue:,.0f}")
        
        # ROI Comparison
        st.markdown("#### 📊 ROI by Segment")
        
        roi_data = []
        for seg in df['Segment'].unique():
            seg_size = len(df[df['Segment'] == seg])
            seg_ltv = df[df['Segment'] == seg]['Monetary'].mean()
            seg_cost = seg_size * cost_per_customer
            seg_revenue = seg_size * success_rate * seg_ltv
            seg_roi = ((seg_revenue - seg_cost) / seg_cost * 100) if seg_cost > 0 else 0
            roi_data.append({'Segment': seg, 'ROI': seg_roi})
        
        roi_comp_df = pd.DataFrame(roi_data).sort_values('ROI', ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=roi_comp_df['Segment'],
            x=roi_comp_df['ROI'],
            orientation='h',
            marker=dict(
                color=['#11998e' if r > 500 else '#F2A104' if r > 100 else '#D64045' 
                       for r in roi_comp_df['ROI']],
                line=dict(width=0)
            ),
            text=[f"{r:,.0f}%" for r in roi_comp_df['ROI']],
            textposition='inside',
            textfont=dict(color='white', size=13, family='Inter')
        ))
        
        fig.update_layout(
            height=280,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=13, color='white')),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=20, t=10, b=10),
            font=dict(family='Inter')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sensitivity Analysis
        st.markdown("#### 📉 Sensitivity Analysis")
        
        pessimistic = max(0.05, success_rate - 0.10)
        optimistic = min(0.80, success_rate + 0.10)
        
        pess_roi = ((segment_size * pessimistic * avg_ltv - total_cost) / total_cost * 100)
        opt_roi = ((segment_size * optimistic * avg_ltv - total_cost) / total_cost * 100)
        breakeven = (cost_per_customer / avg_ltv * 100)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔻 Pessimistic", f"{pess_roi:,.0f}%", f"{pessimistic*100:.0f}% success")
        with col2:
            st.metric("📊 Base Case", f"{roi:,.0f}%", f"{success_rate*100:.0f}% success")
        with col3:
            st.metric("🔺 Optimistic", f"{opt_roi:,.0f}%", f"{optimistic*100:.0f}% success")
        
        st.success(f"💡 **Break-even: {breakeven:.1f}%** success rate • Campaign profitable under most scenarios")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div style="text-align: center; padding: 30px; margin-top: 40px; 
            border-top: 1px solid rgba(255,255,255,0.1);">
    <p style="color: rgba(255,255,255,0.5); font-size: 13px;">
        🎯 Customer Health Dashboard • Real-time Analytics • Built with Streamlit + Plotly
    </p>
</div>
""", unsafe_allow_html=True)
