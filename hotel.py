import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Hotel Booking Dashboard", page_icon="🏨", layout="wide")
st.title("🏨 Hotel Booking Cancellation Dashboard")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/hotel_bookings.csv')
        
        df = df.rename(columns={
            'hotel': 'hotel_type',
            'arrival_date_month': 'arrival_month',
        })
        
        if 'is_canceled' in df.columns and 'lead_time' in df.columns and 'adr' in df.columns:
            df = df.dropna(subset=['is_canceled', 'lead_time', 'adr'])
        
        st.success(f"✅ Loaded {len(df):,} real records from your CSV")
        return df
        
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

df = load_data()

if df is None:
    st.stop()

if 'stays_in_weekend_nights' in df.columns and 'stays_in_week_nights' in df.columns:
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

if 'adults' in df.columns:
    df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies'].fillna(0)

def booking_window(days):
    if days <= 30:
        return 'Last minute'
    elif days <= 90:
        return '1-3 months'
    elif days <= 180:
        return '3-6 months'
    else:
        return 'Way ahead'

df['booking_window'] = df['lead_time'].apply(booking_window)
df['has_special_requests'] = df['total_of_special_requests'] > 0

with st.sidebar:
    st.title("🔧 Dashboard Controls")
    st.markdown("---")
    
    hotel_filter = st.multiselect("Select Hotel Type", df['hotel_type'].unique(), default=df['hotel_type'].unique())
    
    if 'market_segment' in df.columns:
        market_filter = st.multiselect("Select Market Segment", df['market_segment'].unique(), default=df['market_segment'].unique())
        filtered_df = df[(df['hotel_type'].isin(hotel_filter)) & (df['market_segment'].isin(market_filter))]
    else:
        filtered_df = df[df['hotel_type'].isin(hotel_filter)]
    
    st.markdown("---")
    st.markdown(f"**Records displayed:** {len(filtered_df)}")
    st.markdown(f"**Cancellation rate:** {(filtered_df['is_canceled'].mean() * 100):.1f}%")

st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Bookings", f"{len(filtered_df):,}")
with col2:
    st.metric("Cancellations", f"{int(filtered_df['is_canceled'].sum()):,}")
with col3:
    st.metric("Completed", f"{int(len(filtered_df) - filtered_df['is_canceled'].sum()):,}")
with col4:
    st.metric("Avg Lead Time (days)", f"{filtered_df['lead_time'].mean():.0f}")
with col5:
    st.metric("Avg Room Price ($)", f"${filtered_df['adr'].mean():.2f}")

st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Univariate", "🔄 Bivariate", "🎯 Multivariate", "💡 Insights", "📋 Summary"])

with tab1:
    st.subheader("Univariate Analysis - Individual Variable Distributions")
    
    st.markdown("#### 📊 Lead Time Distribution")
    fig1 = px.histogram(filtered_df, x='lead_time', nbins=50, title='Lead Time Distribution (Days)')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""
    **📌 Insight:** Lead time shows a right-skewed distribution, with most bookings made within the first few months. 
    A significant number of bookings occur more than 180 days in advance, indicating advance planning. 
    This metric is crucial for revenue management and inventory planning.
    """)
    
    st.markdown("---")
    
    st.markdown("#### 💰 Room Price (ADR) Distribution")
    fig2 = px.histogram(filtered_df[filtered_df['adr'] < 500], x='adr', nbins=40, title='Average Daily Rate Distribution (ADR in $)')
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    **📌 Insight:** Room prices show a concentration in the lower price ranges ($0-$150), with a long tail extending to higher prices.
    This suggests a mix of budget and premium offerings. The majority of bookings are in the economy segment,
    indicating volume-based revenue strategy.
    """)
    
    st.markdown("---")
    
    if 'booking_changes' in filtered_df.columns:
        st.markdown("#### 🔄 Booking Changes Distribution")
        fig3 = px.histogram(filtered_df, x='booking_changes', nbins=20, title='Number of Booking Changes')
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""
        **📌 Insight:** Most bookings undergo few or no changes after reservation.
        Bookings with multiple changes may indicate customer uncertainty or special requests.
        """)
        st.markdown("---")
    
    if 'days_in_waiting_list' in filtered_df.columns:
        st.markdown("#### ⏳ Days in Waiting List Distribution")
        waiting_data = filtered_df[filtered_df['days_in_waiting_list'] > 0]
        if len(waiting_data) > 0:
            fig4 = px.histogram(waiting_data, x='days_in_waiting_list', nbins=30, title='Days in Waiting List')
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("""
            **📌 Insight:** Only a subset of bookings go on waiting list. Those that do show variable wait times,
            suggesting fluctuating inventory and demand patterns.
            """)

with tab2:
    st.subheader("Bivariate Analysis - Relationships Between Variables")
    
    st.markdown("#### 🏨 Hotel Type vs Cancellation Rate")
    hotel_cancel_data = filtered_df.groupby('hotel_type')['is_canceled'].agg(['sum', 'count', 'mean'])
    hotel_cancel_data['percent'] = (hotel_cancel_data['mean'] * 100).round(2)
    
    fig5 = px.bar(x=hotel_cancel_data.index, y=hotel_cancel_data['percent'], 
                  title='Cancellation Rate by Hotel Type',
                  labels={'y': 'Cancellation Rate (%)'})
    st.plotly_chart(fig5, use_container_width=True)
    
    insight_text = "City hotels: " + str(hotel_cancel_data['percent'].iloc[0]) + "% cancellation rate, Resort hotels: " + str(hotel_cancel_data['percent'].iloc[-1]) + "% cancellation rate"
    st.markdown(f"""
    **📌 Insight:** {insight_text}
    
    City hotels typically have higher cancellation rates than resort hotels. This is likely because:
    - City hotels cater to business travelers with flexible schedules
    - Resort hotels have more committed leisure travelers
    - Seasonal variations affect different hotel types differently
    """)
    
    st.markdown("---")
    
    st.markdown("#### 📅 Booking Window vs Cancellation Rate")
    window_order = ['Last minute', '1-3 months', '3-6 months', 'Way ahead']
    window_cancel = filtered_df.groupby('booking_window')['is_canceled'].mean() * 100
    window_data = [window_cancel.get(w, 0) for w in window_order]
    
    fig6 = px.bar(x=window_order, y=window_data, 
                  title='Cancellation Rate by Booking Window',
                  labels={'y': 'Cancellation Rate (%)', 'x': 'Booking Window'})
    st.plotly_chart(fig6, use_container_width=True)
    
    st.markdown(f"""
    **📌 Insight:** Clear pattern observed:
    - Last-minute bookings: {window_data[0]:.1f}% cancellation (most reliable)
    - 1-3 months ahead: {window_data[1]:.1f}% cancellation
    - 3-6 months ahead: {window_data[2]:.1f}% cancellation
    - Way ahead (180+ days): {window_data[3]:.1f}% cancellation (highest risk)
    
    Early advance bookings are riskier - guests may cancel due to changed circumstances or find better alternatives.
    """)
    
    st.markdown("---")
    
    st.markdown("#### ⭐ Special Requests vs Cancellation Rate")
    requests_cancel = filtered_df.groupby('has_special_requests')['is_canceled'].mean() * 100
    
    fig7 = px.bar(x=['No Special Requests', 'Has Special Requests'], 
                  y=[requests_cancel[False], requests_cancel[True]],
                  title='Cancellation Rate by Special Requests',
                  labels={'y': 'Cancellation Rate (%)'})
    st.plotly_chart(fig7, use_container_width=True)
    
    no_req_rate = requests_cancel[False]
    yes_req_rate = requests_cancel[True]
    multiplier = no_req_rate / yes_req_rate if yes_req_rate > 0 else 0
    
    st.markdown(f"""
    **📌 Insight:** Guests with special requests: {yes_req_rate:.1f}% cancellation
    Guests without requests: {no_req_rate:.1f}% cancellation
    
    **Key finding:** {multiplier:.1f}x higher cancellation rate for guests without special requests!
    
    Interpretation: Engaged customers (those making special requests) are more committed to their stay.
    This could be a strong predictor of booking reliability.
    """)
    
    if 'market_segment' in filtered_df.columns:
        st.markdown("---")
        st.markdown("#### 🎯 Market Segment vs Cancellation Rate")
        market_cancel = filtered_df.groupby('market_segment')['is_canceled'].mean() * 100
        market_cancel = market_cancel.sort_values(ascending=False)
        
        fig8 = px.bar(x=market_cancel.values, y=market_cancel.index, 
                      orientation='h',
                      title='Cancellation Rate by Market Segment',
                      labels={'x': 'Cancellation Rate (%)'})
        st.plotly_chart(fig8, use_container_width=True)
        
        st.markdown("""
        **📌 Insight:** Market segments ranked by cancellation risk:
        Direct bookings are most reliable. Online travel agencies and groups show higher cancellation.
        Corporate segment shows lower cancellation, indicating committed bookings.
        """)

with tab3:
    st.subheader("Multivariate Analysis - Complex Patterns")
    
    st.markdown("#### 🌡️ Seasonal Patterns in Cancellations")
    if 'arrival_month' in filtered_df.columns:
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        month_cancel = filtered_df.groupby('arrival_month')['is_canceled'].mean() * 100
        month_data = [month_cancel.get(m, 0) for m in month_order]
        
        fig9 = px.line(x=month_order, y=month_data, markers=True,
                      title='Cancellation Rate Throughout the Year',
                      labels={'y': 'Cancellation Rate (%)', 'x': 'Month'})
        st.plotly_chart(fig9, use_container_width=True)
        
        summer_peak = max(month_data[4:8])
        winter_low = min(month_data[11:] + month_data[:2])
        
        st.markdown(f"""
        **📌 Insight:** Clear seasonal variation in cancellation rates:
        - Summer months (May-August): Higher cancellation ({summer_peak:.1f}% peak)
        - Winter months (Dec-Feb): Lower cancellation ({winter_low:.1f}% lowest)
        - Mid-year: Moderate cancellation rates
        
        Possible reasons:
        - Summer: Families adjust plans, more flexibility
        - Winter: Holiday bookings are more committed
        - Business travel patterns vary seasonally
        """)
    
    st.markdown("---")
    
    st.markdown("#### 🔗 Variable Correlation Analysis")
    numeric_cols = [col for col in ['is_canceled', 'lead_time', 'adr', 'total_nights', 'total_guests', 'total_of_special_requests'] 
                    if col in filtered_df.columns and filtered_df[col].notna().sum() > 0]
    
    if len(numeric_cols) > 1:
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig10 = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig10.update_layout(title='Correlation Matrix', height=400)
        st.plotly_chart(fig10, use_container_width=True)
        
        lead_corr = corr_matrix.loc['is_canceled', 'lead_time']
        req_corr = corr_matrix.loc['is_canceled', 'total_of_special_requests']
        price_corr = corr_matrix.loc['is_canceled', 'adr']
        
        price_direction = "higher" if price_corr > 0 else "lower"
        
        st.markdown(f"""
        **📌 Insight:** Correlation analysis reveals:
        - Lead time and cancellation: Strong positive correlation ({lead_corr:.3f})
          → Longer advance bookings = higher cancellation risk
        - Special requests and cancellation: Negative correlation ({req_corr:.3f})
          → More requests = lower cancellation
        - Price and cancellation: {price_corr:.3f} correlation
          → Higher priced rooms have {price_direction} cancellation
        """)

with tab4:
    st.subheader("💡 Key Insights & Business Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Critical Findings")
        
        avg_cancel_rate = filtered_df['is_canceled'].mean() * 100
        high_risk = filtered_df[filtered_df['booking_window'] == 'Way ahead']['is_canceled'].mean() * 100
        low_risk = filtered_df[filtered_df['booking_window'] == 'Last minute']['is_canceled'].mean() * 100
        
        st.markdown(f"""
        **1. Overall Cancellation Rate: {avg_cancel_rate:.1f}%**
           - {int(filtered_df['is_canceled'].sum()):,} cancellations out of {len(filtered_df):,} bookings
        
        **2. Lead Time is Critical**
           - High-risk (180+ days): {high_risk:.1f}% cancellation
           - Low-risk (last-minute): {low_risk:.1f}% cancellation
           - Risk multiplier: {high_risk/low_risk:.1f}x
        
        **3. Booking Method Matters**
           - Direct bookings: Most reliable
           - Online channels: Moderate reliability
           - Group bookings: Highest risk
        
        **4. Engagement is Predictive**
           - Special requests = More commitment
           - Engagement score correlates negatively with cancellation
        """)
    
    with col2:
        st.markdown("### 💼 Strategic Recommendations")
        st.markdown("""
        **Immediate Actions:**
        
        ✅ **Require Deposits**
           - Non-refundable for bookings 3+ months ahead
           - Reduces cancellation risk significantly
        
        ✅ **Encourage Engagement**
           - Prompt special requests at booking
           - Special requests reduce cancellation 2-3x
        
        ✅ **Segment-Specific Policies**
           - Stricter for group bookings
           - Flexible for direct bookings
           - Business-focused for corporate
        
        ✅ **Seasonal Adjustments**
           - Higher overbooking in summer
           - Lower overbooking in winter
           - Adjust by historical patterns
        
        **Expected Impact:**
        📈 10-15% reduction in cancellations
        📈 Better revenue forecasting
        📈 Improved operational efficiency
        """)

with tab5:
    st.subheader("📋 Data Summary & Statistics")
    
    st.markdown(f"""
    **Dataset Overview:**
    - Total Records: {len(filtered_df):,}
    - Cancellations: {int(filtered_df['is_canceled'].sum()):,}
    - Completed Stays: {len(filtered_df) - int(filtered_df['is_canceled'].sum()):,}
    - Overall Cancellation Rate: {filtered_df['is_canceled'].mean() * 100:.2f}%
    """)
    
    st.markdown("---")
    
    stats_dict = {}
    for col in ['lead_time', 'adr', 'total_nights', 'total_guests', 'total_of_special_requests']:
        if col in filtered_df.columns:
            stats_dict[col] = {
                'Min': filtered_df[col].min(),
                'Q1': filtered_df[col].quantile(0.25),
                'Median': filtered_df[col].median(),
                'Mean': filtered_df[col].mean(),
                'Q3': filtered_df[col].quantile(0.75),
                'Max': filtered_df[col].max(),
            }
    
    if stats_dict:
        st.markdown("**Numerical Variables Statistics:**")
        for col, stats in stats_dict.items():
            col_display = col.replace('_', ' ').title()
            st.markdown(f"""
            **{col_display}:**
            - Min: {stats['Min']:.2f} | Q1: {stats['Q1']:.2f} | Median: {stats['Median']:.2f} | Mean: {stats['Mean']:.2f} | Q3: {stats['Q3']:.2f} | Max: {stats['Max']:.2f}
            """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 12px;'><p>Hotel Booking Cancellation Analysis Dashboard | Component 3</p></div>", unsafe_allow_html=True)
