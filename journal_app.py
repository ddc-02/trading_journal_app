import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import calendar 
from datetime import datetime, timedelta
from data_manager import load_data_from_google_sheet, load_summary_metrics

# --- Set Streamlit Page Config ---
st.set_page_config(
    page_title="ddc Trading Journal",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed sidebar for a cleaner look
)

# Use hex codes for consistency
POSITIVE_COLOR = '#3F9114' # Darker Green for positive PnL
NEGATIVE_COLOR = '#cc2936' # Red for negative PnL

# --- Load Data and Metrics ---
df_trades = load_data_from_google_sheet()
metrics = load_summary_metrics()

if df_trades.empty or not metrics:
    st.error("Could not load data. Please check configuration.")
    st.stop()

# --- Dark Theme Plotly Template ---
DARK_THEME_TEMPLATE = 'plotly_dark'

# --- New Data Processing Function for Flat Equity Curve ---
def get_time_series_pnl(df_trades):
    """
    Calculates cumulative PnL for every day and uses forward-fill to show 
    a flat line on non-trading days.
    """
    if df_trades.empty:
        return pd.DataFrame({'Date': [], 'Cumulative PnL': []})
        
    # 1. Calculate PnL per day (using Date Closed as index)
    daily_pnl = df_trades.groupby(df_trades['Date Closed'].dt.date)['Net PnL (₹)'].sum()
    
    # 2. Calculate cumulative PnL for trading days
    daily_pnl_cum = daily_pnl.cumsum()
    
    # Convert index to datetime objects for date range operation
    daily_pnl_cum.index = pd.to_datetime(daily_pnl_cum.index)

    # 3. Create a date range for ALL days
    start_date = daily_pnl_cum.index.min()
    end_date = daily_pnl_cum.index.max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # 4. Re-index to the full range and Forward-Fill (ffill) the PnL
    cumulative_series = daily_pnl_cum.reindex(all_dates)
    
    # Fill the very first value if it was NaT (should be 0 if the first day had no trades)
    cumulative_series = cumulative_series.fillna(method='ffill')
    cumulative_series = cumulative_series.fillna(0) # In case ffill fails or the whole series is empty

    # 5. Convert back to DataFrame for Plotly
    df_pnl_timeseries = cumulative_series.reset_index()
    df_pnl_timeseries.columns = ['Date', 'Cumulative PnL']
    
    return df_pnl_timeseries


def render_calculator_popover():
    """Renders the position/PnL calculator using st.popover."""
    
    with st.popover("🧮 Calculator (PnL)", use_container_width=True):
        st.markdown("### Simple PnL Calculator")
        
        # --- Input Section (Based on your Excel layout) ---
        
        # Column layout for input (Sold, Bought, Lots/Credit-Debit)
        col_sold, col_bought, col_lots = st.columns(3)

        with col_sold:
            st.markdown("#### Sold (Avg)")
            # Use 'Avg 1' as the primary sold price input
            sold_price = st.number_input("Price Sold", min_value=0.0, value=0.0, format="%.2f", key="calc_sold")
        
        with col_bought:
            st.markdown("#### Bought (Avg)")
            # Use 'Avg 1' as the primary bought price input
            bought_price = st.number_input("Price Bought", min_value=0.0, value=0.0, format="%.2f", key="calc_bought")
            
        with col_lots:
            st.markdown("#### Quantity")
            # Lots/Quantity input
            lots = st.number_input("Lots/Qty", min_value=1, value=1, step=1, key="calc_lots")

        st.markdown("---")
        
        # --- Calculation Logic ---
        # Assuming Net PnL is (Sold Price * Lots) - (Bought Price * Lots)
        # Your Excel image shows: 4230 (Sold) and -1500 (Bought) leading to 2730.
        # This implies: (56.4 * 100) = 5640? or (56.4 * 75) = 4230. 
        # Let's assume the input is price/share and the 'lots' is the total quantity/units.
        
        # Based on your image (Avg 1: 56.4 * 1, Avg 2: 20 * 1), and final PnL 2730.
        # If your trade is an exit (Sold) and entry (Bought) of a single instrument:
        # Net PnL = (Exit Price - Entry Price) * Quantity
        
        # Let's align with the final row of your Excel: 4230 - 1500 = 2730
        # We will expose the gross amount for Sold and Bought based on the Lot/Qty input.

        # Assuming 'lots' is the total *units* being traded, not 'lots' of a contract size.
        # If 'lots' = 1, we assume it means 1 unit. If you trade 75 units, input 75.
        
        # Calculate Gross Sold Value and Gross Bought Value
        gross_sold = sold_price * lots *65
        gross_bought = bought_price * lots*65
        
        # Final Net PnL
        net_pnl = gross_sold - gross_bought

        # --- Output Section ---
        
        col_gross_sold, col_gross_bought, col_net = st.columns(3)
        
        with col_gross_sold:
            st.metric("Gross Sold Value (₹)", f"₹{gross_sold:,.2f}")
            
        with col_gross_bought:
            # We display the bought value as a debit (negative) to match your Excel logic
            st.metric("Gross Bought Debit (₹)", f"₹{-gross_bought:,.2f}")

        with col_net:
            st.metric(
                "Net Credit/Debit (₹)", 
                f"₹{net_pnl:,.2f}", 
                delta_color="off" # Keep it white/main color
            )
            
        # Optional: Add a simple interpretation
        if net_pnl > 0:
            st.success("Result: Profit/Credit")
        elif net_pnl < 0:
            st.error("Result: Loss/Debit")
        else:
            st.info("Result: Breakeven")

# Note: If your 'Lots' input means a multiplier (e.g., 1 lot = 100 shares), 
# you should adjust the calculation: gross_sold = sold_price * lots * 100


# --- Functions for Visualization (Updated plot_cumulative_pnl) ---

def plot_cumulative_pnl(df):
    """Generates the cumulative PnL area chart (Equity Curve) with flat lines for non-trading days."""
    
    # --- KEY CHANGE: Get the full time series data ---
    df_pnl_plot = get_time_series_pnl(df)
    
    if df_pnl_plot.empty:
        # Handle case where no data is present after filtering
        return go.Figure().update_layout(title="Equity Curve (No Trades)", template=DARK_THEME_TEMPLATE)

    # Use go.Scatter with 'fill' for an area chart
    fig = go.Figure()
    
    # Convert hex to RGB for semi-transparent fill
    r, g, b = int(POSITIVE_COLOR[1:3], 16), int(POSITIVE_COLOR[3:5], 16), int(POSITIVE_COLOR[5:7], 16)
    # Using 0.35 opacity
    fill_color_rgba = f'rgba({r}, {g}, {b}, 0.35)' 
    
    fig.add_trace(go.Scatter(
        x=df_pnl_plot['Date'], # Use the full date range
        y=df_pnl_plot['Cumulative PnL'],
        fill='tozeroy',  # Fill area down to y=0
        line=dict(
            color=POSITIVE_COLOR, 
            width=3,            
            shape='spline',      # Curved line segments on PnL changes
            smoothing=0.6        # Tones down the curvature
        ),
        fillcolor=fill_color_rgba,
        mode='lines',
        name='Equity Curve'
    ))
    
    fig.update_layout(
        title='Equity Curve (Cumulative PnL)', 
        xaxis_title="", 
        yaxis_title="PnL (₹)",
        template=DARK_THEME_TEMPLATE,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.add_hline(y=0, line_dash="dash", line_color= 'white', line_width=1)
    return fig

# plot_monthly_pnl remains the same

# File: journal_app.py (Updated plot_monthly_pnl function)

def plot_monthly_pnl(df):
    """Generates the total PnL for each month (PnL by Month), labeling the x-axis by month
    and placing the PnL amount on/above the bar."""
    
    # 1. Calculate Monthly PnL
    df['Month_Sort'] = df['Date Closed'].dt.to_period('M') 
    df['Month_Display'] = df['Date Closed'].dt.strftime('%b %Y') 
    
    monthly_pnl = df.groupby('Month_Sort').agg(
        {'Net PnL (₹)': 'sum', 'Month_Display': 'first'}
    ).reset_index()
    
    monthly_pnl.columns = ['Month_Sort', 'Total PnL', 'Month_Display']
    
    # Sort the DataFrame by the Month_Sort period object
    monthly_pnl = monthly_pnl.sort_values(by='Month_Sort')
    
    # 2. Determine color and format text
    monthly_pnl['Color'] = monthly_pnl['Total PnL'].apply(lambda x: POSITIVE_COLOR if x >= 0 else NEGATIVE_COLOR)
    
    # Format the PnL amount for display (e.g., '₹5,600.00')
    monthly_pnl['Text_Label'] = monthly_pnl['Total PnL'].apply(lambda x: f"₹{x:,.2f}")
    
    # Determine text color for visibility (use white/light gray on the dark theme)
    TEXT_COLOR = 'white' 

    # 3. Create Plotly Bar Chart
    fig = go.Figure(data=[
        go.Bar(
            x=monthly_pnl['Month_Display'], 
            y=monthly_pnl['Total PnL'],
            marker_color=monthly_pnl['Color'],
            
            # --- KEY CHANGE FOR DATA LABELS ---
            text=monthly_pnl['Text_Label'],
            textposition='outside', # Place text outside/above the bar
            textfont=dict(color=TEXT_COLOR, size=12, weight='bold'),
            # ----------------------------------
        )
    ])

    fig.update_layout(
        title='Monthly Net PnL (₹)',
        xaxis_title="", 
        xaxis={'type': 'category'}, 
        yaxis_title="Total PnL (₹)",
        template=DARK_THEME_TEMPLATE,
        margin=dict(l=20, r=20, t=40, b=20),
        # Increase the top margin if necessary to prevent text from being clipped
        # We also want to leave space for negative labels that are 'outside'
        yaxis=dict(rangemode='tozero', showgrid=True),
    )
    # Adding a zero line for reference
    # fig.add_hline(y=0, line_dash="dash", line_color=NEGATIVE_COLOR, line_width=1)
    
    return fig

# --- PnL Data Preparation (Remains the same for the calendar/lookup) ---
daily_pnl = df_trades.groupby(df_trades['Date Closed'].dt.date)['Net PnL (₹)'].sum().reset_index()
daily_pnl.columns = ['Date', 'PnL']
daily_pnl['Date'] = pd.to_datetime(daily_pnl['Date'])
daily_pnl['PnL'] = daily_pnl['PnL'].round(2)

# Create a PnL lookup dictionary for fast access: {datetime.date: PnL_Amount}
pnl_lookup = daily_pnl.set_index(daily_pnl['Date'].dt.date)['PnL'].to_dict()


# --- Core Calendar Rendering Function (Renders one month's HTML) ---
def render_calendar_month_html(target_date, pnl_lookup, day_names):
# ... (render_calendar_month_html remains the same) ...
    """Generates the HTML string for a single month's calendar grid."""
    
    cal = calendar.Calendar(firstweekday=calendar.MONDAY) # Start week on Monday
    month_days = cal.monthdatescalendar(target_date.year, target_date.month)
    
    html_content = "<div class='calendar-grid-single'>"
    
    # 1. Day Headers (Mon, Tue, ...)
    html_content += "".join([f"<div class='day-header'>{name}</div>" for name in day_names])
    
    # 2. Day Cells
    for week in month_days:
        for day in week:
            cell_class = "calendar-cell"
            day_content = '' 
            pnl_display = "" 
            
            # Check if day is in the current month
            if day.month != target_date.month:
                cell_class += " dimmed" # Dim outside days
            else:
                day_content = day.day
                pnl_amount = pnl_lookup.get(day, None)
                
                if pnl_amount is not None:
                    # Format PnL and set box style
                    pnl_str = f"₹{pnl_amount:,.2f}"
                    pnl_class = "pnl-positive" if pnl_amount >= 0 else "pnl-negative"
                    pnl_display = f"<div class='pnl-box {pnl_class}'>{pnl_str}</div>"
                
            # Build the cell HTML - Concatenate onto a single line
            html_content += (
                f'<div class="{cell_class}">'
                f'<div class="day-number">{day_content}</div>'
                f'{pnl_display}'
                f'</div>'
            )

    html_content += "</div>"
    return html_content.replace('\n', '').strip()

# File: journal_app.py (Updated render_scorecard_popover function)

def calculate_score(indication, weight):
    """Translates indication to a value and calculates the score."""
    if indication == 'Upside':
        value = 1
    elif indication == 'Downside':
        value = -1
    else: # Neutral
        value = 0
    return value * weight


def render_scorecard_popover():
    """Renders the interactive trend analyzer/scorecard in a compact, no-scroll format."""
    
    # Use a wide popover for the table-like layout
    with st.popover("📈 Trend Analyzer", use_container_width=True):
        st.markdown("### Technical Analysis Scorecard")
        
        # Define the variables and their default weights based on your image
        variables = {
            "50 EMA": 1.75,
            "20 EMA": 1.5,
            "200 EMA": 0.5,
            "MACD": 0.75,
            "RSI": 1.2,
            "Price Action": 0.8,
            "Technicals": 1.5,
            "SuperTrend 1": 1.2,
            "SuperTrend 2": 1.2,
            "SuperTrend 3": 1.2,
            "Personal Sentiment": 0.5,
        }
        
        # --- UI Table Setup ---
        
        # Headers: Removed the 'Weight' column header
        col_var, col_ind, col_score = st.columns([3, 2, 1])
        with col_var: st.markdown("##### Variable")
        with col_ind: st.markdown("##### Indication")
        with col_score: st.markdown("##### Score")

        # Reduce the margin/padding for a tighter fit
        st.markdown(
            """
            <style>
            /* Targets the space around select boxes for compactness */
            .stSelectbox {
                margin-top: -10px;
                margin-bottom: -10px;
            }
            /* Targets the space around markdown text for compactness */
            h5 {
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---") # Visual separator for headers
        
        total_weight = 0.0
        total_score = 0.0

        # Create interactive rows
        for i, (variable_name, default_weight) in enumerate(variables.items()):
            # Use the new 3-column structure for the row
            col_v, col_i, col_s = st.columns([3, 2, 1])

            with col_v:
                # Use st.text to remove extra padding
                st.text(variable_name) 

            with col_i:
                # Select box for Indication
                indication = st.selectbox(
                    "Indication",
                    ['Neutral', 'Upside', 'Downside'],
                    index=0, 
                    label_visibility='collapsed',
                    key=f"ind_{i}"
                )
            
            # Note: The Weight is now implicitly used in the calculation but NOT displayed
            score = calculate_score(indication, default_weight)
            
            with col_s:
                # Display score 
                st.text(f"{score:.2f}")

            # Aggregate totals
            total_weight += default_weight
            total_score += score
            
        st.markdown("---")
        
        # --- Final Results Section ---
        
        # Adjusted columns for final results (Removed the weight column output)
        col_spacer1, col_total_score = st.columns([4, 2])
        
        # Removed the Total Weight metric
            
        with col_total_score:
            st.metric("Final Score", f"{total_score:.2f}")
        
        # Determine the conclusion (Logic remains the same)
        if total_score > 0.5:
            conclusion = "Strong Upside"
            color = POSITIVE_COLOR
        elif total_score < -0.5:
            conclusion = "Strong Downside"
            color = NEGATIVE_COLOR
        else:
            conclusion = "Neutral"
            color = "#E5E5E5" 

        st.markdown(
            f"""
            <div style='text-align: right; margin-top: 10px;'>
                Conclusion: <span style='color:{color}; font-weight:bold;'>{conclusion}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )

# --- Main Calendar View Logic (Displays two months) ---
def render_calendar_view(pnl_lookup, daily_pnl): 
# ... (render_calendar_view remains the same) ...
    """Handles navigation, CSS, and displays two calendar months side-by-side."""
    
    # --- Month Navigation and State Management ---
    if 'current_date' not in st.session_state:
        if not daily_pnl.empty:
            st.session_state.current_date = daily_pnl['Date'].max().replace(day=1)
        else:
            st.session_state.current_date = datetime.now().replace(day=1)

    current_date = st.session_state.current_date
    previous_month_date = (current_date - timedelta(days=1)).replace(day=1)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # 1. Navigation Header
    col_nav_1, col_nav_2, col_nav_3 = st.columns([1, 2, 1])

    with col_nav_1:
        if st.button('◀️ Previous Month', use_container_width=True):
            st.session_state.current_date = previous_month_date
            st.rerun()

    with col_nav_2:
        st.markdown(f"<h3 style='text-align: center; color: white;'>{previous_month_date.strftime('%B %Y')} / {current_date.strftime('%B %Y')}</h3>", unsafe_allow_html=True)

    with col_nav_3:
        if st.button('Next Month ▶️', use_container_width=True):
            next_month = current_date.replace(day=28) + timedelta(days=4) 
            st.session_state.current_date = next_month.replace(day=1)
            st.rerun()
            
    # 2. Custom Calendar CSS Styles (Updated class name for grid)
    st.markdown(
        f"""
        <style>
        .calendar-grid-container {{
            display: flex;
            gap: 20px; /* Space between the two months */
        }}
        .calendar-grid-wrapper {{
            flex: 1;
        }}
        .calendar-title {{
            text-align: center; 
            font-size: 1.2em; 
            margin-bottom: 10px;
            font-weight: bold;
        }}
        .calendar-grid-single {{ /* Updated class name from .calendar-grid */
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            border: 1px solid #444;
            border-radius: 5px;
            overflow: hidden;
            width: 100%;
        }}
        /* Keep all other .day-header, .calendar-cell, etc. styles the same */
        .day-header, .calendar-cell {{
            padding: 10px 5px;
            border: 1px solid #333;
            text-align: center;
            min-height: 80px; 
        }}
        .day-header {{
            background-color: #2c2c2c;
            font-weight: bold;
            color: #ccc;
            min-height: 20px;
        }}
        .calendar-cell {{
            background-color: #1e1e1e;
            position: relative;
            color: #aaa;
        }}
        .day-number {{
            position: absolute;
            top: 5px;
            left: 5px;
            font-size: 0.8em;
            color: #777;
        }}
        .pnl-box {{
            margin-top: 15px;
            padding: 5px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 1em;
        }}
        .pnl-positive {{
            background-color: {POSITIVE_COLOR}33; 
            color: {POSITIVE_COLOR};
        }}
        .pnl-negative {{
            background-color: {NEGATIVE_COLOR}33; 
            color: {NEGATIVE_COLOR};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # 3. Render Two Months Side-by-Side
    col_prev, col_curr = st.columns(2)

    with col_prev:
        st.markdown(f'<div class="calendar-title">{previous_month_date.strftime("%B %Y")}</div>', unsafe_allow_html=True)
        html_prev = render_calendar_month_html(previous_month_date, pnl_lookup, day_names)
        st.markdown(f'<div class="calendar-grid-wrapper">{html_prev}</div>', unsafe_allow_html=True)

    with col_curr:
        st.markdown(f'<div class="calendar-title">{current_date.strftime("%B %Y")}</div>', unsafe_allow_html=True)
        html_curr = render_calendar_month_html(current_date, pnl_lookup, day_names)
        st.markdown(f'<div class="calendar-grid-wrapper">{html_curr}</div>', unsafe_allow_html=True)


# --- UI LAYOUT ---
st.markdown("## Dhruv's Analytics Dashboard")
col_title, col_calc, col_analyzer = st.columns([6, 1, 1])



with col_calc:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True) 
    render_calculator_popover() # Existing PnL calculator

with col_analyzer:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True) 
    # NEW: Call the function to render the Scorecard popover
    render_scorecard_popover()


# 1. Key Metrics Section (KPIs)
with st.container(border=True):
    st.markdown("### 🎯 Performance Overview")
    
    # --- CSS to Force Metric Value to White ---
    st.markdown(
        """
        <style>
        div[data-testid='stMetricValue'] {
            color: white !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # CHANGE: Create 6 columns instead of 5
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7) 

    # Function to apply color based on value for a 'swaggier' look
    def format_metric(label, value, is_pnl=False):
        if value == "N/A":
            st.metric(label=label, value=value)
            return 
        
        # Display the metric without any custom color logic
        st.metric(
            label=label, 
            value=value,
            # Ensure delta coloring is off
            delta_color="off" 
        )
    
    # Calculate and format the W:L Ratio string
    # Assuming "Wins" and "Losses" keys exist in your metrics dictionary (as per image)
    win_count = metrics.get("Wins", "N/A")
    loss_count = metrics.get("Losses", "N/A")
    
    if win_count != "N/A" and loss_count != "N/A":
        wl_ratio_str = f"{win_count} : {loss_count}"
    else:
        wl_ratio_str = "N/A"
        
    # Display the metrics across 6 columns
    with col1: 
        # NEW METRIC: Win : Loss Count (e.g., "17W : 11L")
        format_metric("Win : Loss Count", wl_ratio_str) 
        
    with col2: 
        # Original Metric: Win %
        format_metric("Win %", metrics.get("Win %", "N/A"))
        
    with col3: 
        # Original Metric: Avg Return (₹)
        format_metric("Avg Return (₹)", metrics.get("Avg Return (₹)", "N/A"), is_pnl=True)
        
    with col4: 
        # Original Metric: Profit Factor
        format_metric("Profit Factor", metrics.get("Profit Factor", "N/A"))
        
    with col5: 
        # Original Metric: Cumulative PnL (₹)
        format_metric("Cumulative PnL (₹)", metrics.get("Cumulative PnL (₹)", "N/A"), is_pnl=True)
        
    with col6: 
        # Original Metric: Current Drawdown
        format_metric("Current Drawdown", metrics.get("Current Drawdown", "N/A"))

    with col7: 
        # Original Metric: Median Drawdown
        format_metric("Median Drawdown", metrics.get("Median Drawdown", "N/A"))


st.markdown("---")

# 2. Charts Section (Two columns for major charts)
col_chart_1, col_chart_2 = st.columns(2)

with col_chart_1:
    with st.container(border=True):
        st.plotly_chart(plot_cumulative_pnl(df_trades), use_container_width=True)

with col_chart_2:
    with st.container(border=True):
        st.plotly_chart(plot_monthly_pnl(df_trades), use_container_width=True)

st.markdown("---")

# 3. Calendar View 
st.markdown("### 📅 Monthly Trade Calendar Comparison")

# The logic is now contained within the function above
render_calendar_view(pnl_lookup, daily_pnl)


col_cal, col_refresh = st.columns([4, 1])

with col_refresh:
    st.subheader("Controls")
    # Refresh Button (moved to its own column for prominence)
    if st.button('🔄 Refresh Data', use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Spacer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("Data auto-updates every day.")


st.markdown("---")

# 4. Raw Trade Log (Full Width)
st.subheader("📚 Detailed Trade Log")

st.dataframe(df_trades, use_container_width=True)
