import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Hotel Booking Dashboard", page_icon="🏨", layout="wide")
st.title("🏨 Hotel Booking Cancellation Dashboard")

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        # Load the dataset
        df = pd.read_csv('data/hotel_bookings.csv')
        
        # Rename columns for clarity
        df = df.rename(columns={
            'hotel': 'hotel_type',
            'arrival_date_month': 'arrival_month',
        })
        
        # Basic cleaning: drop rows where key columns are missing
        if 'is_canceled' in df.columns and 'lead_time' in df.columns and 'adr' in df.columns:
            df = df.dropna(subset=['is_canceled', 'lead_time', 'adr'])
        
        st.success(f"✅ Loaded {len(df):,} real records from your CSV")
        return df
        
    except FileNotFoundError:
        st.error("Error: The file 'data/hotel_bookings.csv' was not found.")
        st.info("Please make sure the CSV file is in a folder named 'data' in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

df = load_data()

# Stop the app if data loading failed
if df is None:
    st.stop()

# --- Feature Engineering ---
# Create 'total_nights'
if 'stays_in_weekend_nights' in df.columns and 'stays_in_week_nights' in df.columns:
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

# Create 'total_guests'
if 'adults' in df.columns:
    df['total_guests'] = df['adults'].fillna(0) + df['children'].fillna(0) + df['babies'].fillna(0)
    # Filter out bookings with 0 guests
    df = df[df['total_guests'] > 0]

# Create 'booking_window' categories
def booking_window(days):
    if days <= 30:
        return 'Last minute (0-30 days)'
    elif days <= 90:
        return '1-3 months (31-90 days)'
    elif days <= 180:
        return '3-6 months (91-180 days)'
    else:
        return 'Way ahead (181+ days)'

df['booking_window'] = df['lead_time'].apply(booking_window)

# Create 'has_special_requests'
if 'total_of_special_requests' in df.columns:
    df['has_special_requests'] = df['total_of_special_requests'] > 0

# --- Sidebar Filters ---
with st.sidebar:
    st.title("🔧 Dashboard Controls")
    st.markdown("---")
    
    # Hotel Type Filter
    hotel_types = df['hotel_type'].unique()
    hotel_filter = st.multiselect("Select Hotel Type", hotel_types, default=hotel_types)
    
    # Market Segment Filter
    if 'market_segment' in df.columns:
        market_segments = df['market_segment'].unique()
        market_filter = st.multiselect("Select Market Segment", market_segments, default=market_segments)
        
        # Apply filters
        filtered_df = df[
            (df['hotel_type'].isin(hotel_filter)) & 
            (df['market_segment'].isin(market_filter))
        ]
    else:
        filtered_df = df[df['hotel_type'].isin(hotel_filter)]
    
    # Display summary metrics in the sidebar
    st.markdown("---")
    st.markdown(f"**Records displayed:** {len(filtered_df):,}")
    if not filtered_df.empty:
        st.markdown(f"**Cancellation rate:** {(filtered_df['is_canceled'].mean() * 100):.1f}%")

# Stop if filters result in no data
if filtered_df.empty:
    st.warning("No data matches the current filters. Please adjust your selection.")
    st.stop()

# --- Main Page KPIs ---
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Bookings", f"{len(filtered_df):,}")
with col2:
    st.metric("Cancellations", f"{int(filtered_df['is_canceled'].sum()):,}")
with col3:
    completed_bookings = len(filtered_df) - filtered_df['is_canceled'].sum()
    st.metric("Completed Stays", f"{int(completed_bookings):,}")
with col4:
    st.metric("Avg Lead Time (days)", f"{filtered_df['lead_time'].mean():.0f}")
with col5:
    # Filter out $0 ADR for a more meaningful average
    st.metric("Avg Room Price ($)", f"${filtered_df[filtered_df['adr'] > 0]['adr'].mean():.2f}")

st.markdown("---")

# --- Tabs for Analysis ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Univariate", "🔄 Bivariate", "🎯 Multivariate", "💡 Recommendations", "📋 Summary"])

with tab1:
    st.subheader("Univariate Analysis - Individual Variable Distributions")
    
    st.markdown("#### 📊 Lead Time Distribution")
    fig1 = px.histogram(filtered_df, x='lead_time', nbins=50, 
                        title='Lead Time Distribution (Days)',
                        color_discrete_sequence=['#1f77b4']) # Vibrant blue
    fig1.update_layout(bargap=0.1)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""
    **📌 Insight:** The lead time is heavily right-skewed. This means most bookings are made "Last minute" (within 30 days),
    but there's also a very long tail of people who book many months, sometimes almost a year, in advance.
    This split suggests two different customer behaviors: the spontaneous traveler and the long-term planner.
    """)
    
    st.markdown("---")
    
    st.markdown("#### 💰 Room Price (ADR) Distribution")
    # Filter out extreme outliers for a clearer plot
    adr_data = filtered_df[(filtered_df['adr'] > 0) & (filtered_df['adr'] < 500)]
    fig2 = px.histogram(adr_data, x='adr', nbins=40, 
                        title='Average Daily Rate Distribution (ADR in $)',
                        color_discrete_sequence=['#2ca02c']) # Green
    fig2.update_layout(bargap=0.1)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    **📌 Insight:** Most bookings are concentrated in the $50-$150 price range, which is likely the hotel's
    core business. There's a smaller bump at higher prices, representing premium rooms or peak season pricing.
    The data suggests a volume-based strategy, focusing on filling rooms at these competitive rates.
    """)
    
    st.markdown("---")
    
    if 'booking_changes' in filtered_df.columns:
        st.markdown("#### 🔄 Booking Changes Distribution")
        fig3 = px.histogram(filtered_df, x='booking_changes', 
                            title='Number of Booking Changes',
                            color_discrete_sequence=['#ff7f0e']) # Orange
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""
        **📌 Insight:** The vast majority of bookings (the first bar) have **zero** changes.
        This implies that once a booking is made, it's usually final. A small number of guests
        make one or two changes, but it's not common.
        """)
        st.markdown("---")
    
    if 'days_in_waiting_list' in filtered_df.columns:
        st.markdown("#### ⏳ Days in Waiting List Distribution")
        waiting_data = filtered_df[filtered_df['days_in_waiting_list'] > 0]
        if not waiting_data.empty:
            fig4 = px.histogram(waiting_data, x='days_in_waiting_list', nbins=30, 
                                title='Days in Waiting List (For those who waited)',
                                color_discrete_sequence=['#d62728']) # Red
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("""
            **📌 Insight:** Very few bookings ever end up on a waiting list. For the small fraction
            that do, most are cleared relatively quickly, though some can wait for a long time.
            This suggests the hotel's inventory management is either very effective or the demand
            rarely exceeds supply to the point of needing a long waitlist.
            """)
        else:
            st.info("No bookings in the filtered data spent time on the waiting list.")

with tab2:
    st.subheader("Bivariate Analysis - Relationships Between Variables")
    
    st.markdown("#### 🏨 Hotel Type vs Cancellation Rate")
    hotel_cancel_data = filtered_df.groupby('hotel_type')['is_canceled'].mean().reset_index()
    hotel_cancel_data['percent'] = (hotel_cancel_data['is_canceled'] * 100)
    
    fig5 = px.bar(hotel_cancel_data, x='hotel_type', y='percent', 
                  title='Cancellation Rate by Hotel Type',
                  labels={'percent': 'Cancellation Rate (%)', 'hotel_type': 'Hotel Type'},
                  color='hotel_type',
                  color_discrete_map={'City Hotel': '#1f77b4', 'Resort Hotel': '#2ca02c'},
                  text=hotel_cancel_data['percent'].apply(lambda x: f'{x:.1f}%'),
                  hover_data={'hotel_type': True, 'percent': ':.1f'}) # 'hotel_type' set to True for visibility
    fig5.update_traces(textposition='outside')
    st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown(f"""
    **📌 Insight:** There's a noticeable difference here. **City Hotels** tend to have a
    significantly higher cancellation rate than **Resort Hotels**.
    
    This might be because:
    - City hotel bookings are often for business, which can be volatile.
    - Resort hotel bookings are typically for leisure/vacations, which are planned more firmly.
    """)
    
    st.markdown("---")
    
    st.markdown("#### 📅 Booking Window vs Cancellation Rate")
    
    window_order = ['Last minute (0-30 days)', '1-3 months (31-90 days)', '3-6 months (91-180 days)', 'Way ahead (181+ days)']
    # Convert the Series to a DataFrame
    window_cancel_df = (filtered_df.groupby('booking_window')['is_canceled'].mean() * 100).reset_index()
    # Rename columns for clarity
    window_cancel_df.columns = ['Booking Window', 'Cancellation Rate (%)']
    
    fig6 = px.bar(window_cancel_df, 
                  x='Booking Window', 
                  y='Cancellation Rate (%)', 
                  title='Cancellation Rate by Booking Window',
                  labels={'Cancellation Rate (%)': 'Cancellation Rate (%)', 'Booking Window': 'Booking Window'},
                  color='Booking Window',
                  color_discrete_sequence=px.colors.sequential.YlOrRd[2::2],
                  text='Cancellation Rate (%)',
                  hover_data={'Booking Window': True, 'Cancellation Rate (%)': ':.1f'}) 
    
    fig6.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    # Ensure the x-axis keeps the logical order
    fig6.update_layout(xaxis={'categoryorder':'array', 'categoryarray': window_order})
    st.plotly_chart(fig6, use_container_width=True)
    
    st.markdown(f"""
    **📌 Insight:** This shows a very clear trend: **the further in advance a guest books, the more likely they are to cancel.**
    
    - Bookings made 'Way ahead' (6+ months) have the highest risk.
    - 'Last minute' bookings are the safest and almost never get canceled.
    
    This makes intuitive sense—plans change over a long period, and people might find better deals or alternative options as their trip approaches.
    """)
    
    st.markdown("---")
    
    if 'has_special_requests' in filtered_df.columns:
        st.markdown("#### ⭐ Special Requests vs Cancellation Rate")
        requests_cancel = (filtered_df.groupby('has_special_requests')['is_canceled'].mean() * 100).reset_index()
        requests_cancel['has_special_requests_str'] = requests_cancel['has_special_requests'].map({True: 'Has Special Requests', False: 'No Special Requests'})
        
        fig7 = px.bar(requests_cancel, x='has_special_requests_str', y='is_canceled', 
                      title='Cancellation Rate by Special Requests',
                      labels={'is_canceled': 'Cancellation Rate (%)', 'has_special_requests_str': 'Made Special Requests?'},
                      color='has_special_requests_str',
                      color_discrete_map={'No Special Requests': '#d62728', 'Has Special Requests': '#2ca02c'},
                      text=requests_cancel['is_canceled'].apply(lambda x: f'{x:.1f}%'),
                      hover_data={'has_special_requests_str': True, 'is_canceled': ':.1f'})
        fig7.update_traces(textposition='outside')
        fig7.update_xaxes(title_text="Made Special Requests?") # Cleaner axis title
        st.plotly_chart(fig7, use_container_width=True)
        
        no_req_rate = requests_cancel[requests_cancel['has_special_requests'] == False]['is_canceled'].values[0]
        yes_req_rate = requests_cancel[requests_cancel['has_special_requests'] == True]['is_canceled'].values[0]
        
        st.markdown(f"""
        **📌 Insight:** This is a fascinating finding. Guests who **make special requests** are
        **significantly less likely to cancel** (a {yes_req_rate:.1f}% rate) compared to those
        who make no requests (a {no_req_rate:.1f}% rate).
        
        This suggests that making a request (e.g., "king bed," "high floor") is a sign of
        **customer engagement** and commitment to the stay.
        """)
    
    if 'market_segment' in filtered_df.columns:
        st.markdown("---")
        st.markdown("#### 🎯 Market Segment vs Cancellation Rate")
        
        # Convert the Series to a DataFrame
        market_cancel_df = (filtered_df.groupby('market_segment')['is_canceled'].mean() * 100).reset_index()
        # Rename columns for clarity
        market_cancel_df.columns = ['Market Segment', 'Cancellation Rate (%)']
        
        fig8 = px.bar(market_cancel_df, 
                      y='Market Segment', 
                      x='Cancellation Rate (%)', 
                      orientation='h',
                      title='Cancellation Rate by Market Segment',
                      labels={'Cancellation Rate (%)': 'Cancellation Rate (%)', 'Market Segment': 'Market Segment'},
                      color='Market Segment',
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      text='Cancellation Rate (%)',
                      hover_data={'Market Segment': True, 'Cancellation Rate (%)': ':.1f'})
        
        fig8.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        # This layout option automatically sorts the bars from lowest to highest
        fig8.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig8, use_container_width=True)
        
        st.markdown("""
        **📌 Insight:** The market segment (how the booking was made) matters a lot.
        - **Groups** and **Online Travel Agents (OTA)** show the highest cancellation rates. This might be due to flexible booking policies or "shopping around."
        - **Direct** bookings and **Corporate** accounts appear to be much more reliable, with lower cancellation rates.
        """)

with tab3:
    st.subheader("Multivariate Analysis - Complex Patterns")
    
    st.markdown("#### 🌡️ Seasonal Patterns in Cancellations")
    if 'arrival_month' in filtered_df.columns:
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        month_cancel = (filtered_df.groupby('arrival_month')['is_canceled'].mean() * 100).reindex(month_order)
        
        fig9 = px.line(month_cancel, x=month_cancel.index, y=month_cancel.values, 
                       title='Cancellation Rate Throughout the Year',
                       labels={'y': 'Cancellation Rate (%)', 'x': 'Month'},
                       markers=True)
        fig9.update_traces(line_color='#8c564b', marker=dict(size=8)) # Brown line
        fig9.update_layout(hovermode="x unified")
        st.plotly_chart(fig9, use_container_width=True)
        
        st.markdown(f"""
        **📌 Insight:** We can see a clear seasonal pattern.
        - Cancellation rates start low at the beginning of the year.
        - They **rise steadily** through spring and **peak in the summer** (June-August).
        - After summer, they drop off again, hitting their lowest point in the late autumn/early winter.
        
        This suggests that summer travel plans are more flexible or subject to change,
        while winter holiday or business bookings are more firm.
        """)
    
    st.markdown("---")
    
    st.markdown("#### 💸 Lead Time vs. Price (ADR) and Cancellation")
    
    # Sample the data to avoid overplotting and improve performance
    sample_df = filtered_df.sample(n=min(5000, len(filtered_df)), random_state=42)
    
    fig11 = px.scatter(sample_df, 
                       x='lead_time', 
                       y='adr', 
                       color='is_canceled',
                       title='Lead Time vs. ADR, Colored by Cancellation',
                       labels={'lead_time': 'Lead Time (Days)', 'adr': 'Average Daily Rate ($)', 'is_canceled': 'Canceled?'},
                       color_discrete_map={0: '#2ca02c', 1: '#d62728'},
                       hover_data=['total_nights', 'market_segment', 'hotel_type'])
    
    fig11.update_traces(marker=dict(size=5, opacity=0.7))
    st.plotly_chart(fig11, use_container_width=True)
    st.markdown(f"""
    *(This plot is based on a random sample of {min(5000, len(filtered_df)):,} bookings to keep the dashboard fast.)*
    
    **📌 Insight:** This scatter plot helps us see a few patterns at once:
    - **Cancellations (red dots)** appear to be more common at **longer lead times**, which confirms our earlier finding.
    - There isn't a simple, clear link between price (ADR) and cancellation. You see both red (canceled) and green (kept) dots across all price ranges.
    - However, there seems to be a cluster of **high-lead-time, low-to-mid-ADR** bookings that get canceled. This could be people booking a "safe" option far in advance, only to cancel it later.
    """)
    
    st.markdown("---")
    
    st.markdown("#### 🔗 Variable Correlation Analysis")
    numeric_cols = [col for col in ['is_canceled', 'lead_time', 'adr', 'total_nights', 'total_guests', 'total_of_special_requests'] 
                    if col in filtered_df.columns]
    
    if len(numeric_cols) > 1:
        corr_matrix = filtered_df[numeric_cols].corr()
        
        # Create the heatmap
        fig10 = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r', # Reversed Red-Blue scale
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate="%{text:.2f}", # Format text to 2 decimal places
            hoverongaps=False
        ))
        fig10.update_layout(title='Correlation Matrix of Key Variables', height=500)
        st.plotly_chart(fig10, use_container_width=True)
        
        # Pull out key correlations with 'is_canceled'
        if 'is_canceled' in corr_matrix.columns and 'lead_time' in corr_matrix.index:
            lead_corr = corr_matrix.loc['is_canceled', 'lead_time']
        else:
            lead_corr = 0.0
            
        if 'is_canceled' in corr_matrix.columns and 'total_of_special_requests' in corr_matrix.index:
            req_corr = corr_matrix.loc['is_canceled', 'total_of_special_requests']
        else:
            req_corr = 0.0
        
        st.markdown(f"""
        **📌 Insight:** This matrix shows how variables move together. Focus on the `is_canceled` row:
        - **Lead Time:** Shows a positive correlation ({lead_corr:.2f}). This confirms our finding: longer lead time = higher cancellation likelihood.
        - **Total of Special Requests:** Shows a negative correlation ({req_corr:.2f}). This also confirms our finding: more requests = lower cancellation likelihood.
        - Other correlations are quite weak, suggesting they aren't strong individual predictors.
        """)

with tab4:
    st.subheader("💡 Key Conclusions & Potential Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Critical Conclusions")
        
        avg_cancel_rate = filtered_df['is_canceled'].mean() * 100
        
        st.markdown(f"""
        Based on the data, we've learned a few key things about what makes a booking "risky":
        
        **1. Overall Cancellation Rate: {avg_cancel_rate:.1f}%**
           - This is our baseline. Any group with a rate much higher than this is a problem area.
        
        **2. Lead Time is a Key Factor**
           - Bookings made **180+ days in advance** seem to be the *most likely* to be canceled.
           - Bookings made **within 30 days** are the *safest*.
        
        **3. How They Book Matters**
           - Bookings from **Online Travel Agents (OTAs)** and **Groups** appear to be canceled more often.
           - **Direct** bookings seem to be more reliable.
        
        **4. Engagement is a Good Sign**
           - Guests who make **special requests** are *less* likely to cancel. This suggests they are more committed to their stay.
        
        **5. Seasonality is Real**
           - The **summer months** see the highest wave of cancellations, while the **end of the year** is the most stable.
        """)
    
    with col2:
        st.markdown("### 💼 Potential Strategies to Consider")
        st.markdown("""
        Given these conclusions, here are a few strategies the hotel *might* consider to
        help manage cancellations:
        
        **Actions to Explore:**
        
        ✅ **Tiered Deposit Policies**
           - For bookings made 6+ months in advance, perhaps require a small, non-refundable deposit.
           - This *could* discourage "placeholder" bookings that are likely to be canceled.
        
        ✅ **Encourage Engagement at Booking**
           - During the booking process, prompt guests to make requests ("Need a quiet room?", "Celebrating an anniversary?").
           - Since engaged guests cancel less, this *may* help build commitment.
        
        ✅ **Segment-Specific Follow-ups**
           - For high-risk segments like "Groups" or "OTA," send a confirmation email a month
             before arrival. This *might* help identify potential cancellations earlier.
        
        ✅ **Smart Overbooking**
           - Knowing that summer months have higher cancellations, the hotel *could*
             adjust its overbooking strategy slightly during this period to maximize occupancy.
        
        **Potential Impact:**
        By testing these ideas, the hotel *may* be able to:
        📈 Reduce the overall cancellation rate.
        📈 Improve revenue forecasting by having a better sense of confirmed stays.
        📈 Optimize room availability and staff planning.
        """)

with tab5:
    st.subheader("📋 Data Summary & Statistics")
    
    st.markdown(f"""
    **Dataset Overview (Based on Filters):**
    - Total Records: {len(filtered_df):,}
    - Cancellations: {int(filtered_df['is_canceled'].sum()):,}
    - Completed Stays: {len(filtered_df) - int(filtered_df['is_canceled'].sum()):,}
    - **Overall Cancellation Rate: {filtered_df['is_canceled'].mean() * 100:.2f}%**
    """)
    
    st.markdown("---")
    
    # Show descriptive statistics for key numerical columns
    cols_to_describe = [
        'lead_time', 'adr', 'total_nights', 'total_guests', 'total_of_special_requests'
    ]
    # Filter list to only include columns that actually exist in the dataframe
    existing_cols = [col for col in cols_to_describe if col in filtered_df.columns]
    
    if existing_cols:
        stats_df = filtered_df[existing_cols].describe().transpose()
        stats_df = stats_df[['min', '25%', '50%', 'mean', '75%', 'max']]
        stats_df.columns = ['Min', 'Q1 (25%)', 'Median (50%)', 'Mean', 'Q3 (75%)', 'Max']
        stats_df = stats_df.applymap(lambda x: f"{x:,.2f}") # Format all numbers
        
        st.markdown("**Numerical Variables Statistics:**")
        st.dataframe(stats_df, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 12px;'><p>Hotel Booking Cancellation Analysis Dashboard</p></div>", unsafe_allow_html=True)