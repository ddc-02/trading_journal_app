import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import streamlit as st
import json # Not strictly needed if using ServiceAccountCredentials, but good to keep
from datetime import datetime

# --- Configuration ---
# IMPORTANT: Replace these placeholders with your actual values
SERVICE_ACCOUNT_FILE = 'service_account.json'
SPREADSHEET_NAME = 'Options' # e.g., 'My 2024 Trading Log'
TRADES_WORKSHEET_NAME = 'Tracker' # The name of the sheet with the raw trade data
METRICS_WORKSHEET_NAME = 'Tracker' # The name of the sheet with the summary metrics

# --- Authentication Helper ---
def get_google_sheet_client():
    """Initializes and returns the gspread client using Streamlit secrets."""
    try:
        # Check if secrets are available
        if not st.secrets:
            st.error("GCP service account secrets not found in .streamlit/secrets.toml.")
            return None
        
        # Use st.secrets to authenticate gspread
        # The key in st.secrets must match the header in secrets.toml ([gcp_service_account])
        client = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        return client
        
    except Exception as e:
        # Note: In production, avoid exposing e in st.error for security.
        st.error(f"Failed to authenticate with Google Sheets. Check your secrets.toml configuration. Error: {e}")
        return None

@st.cache_data(ttl=43200)
def load_data_from_google_sheet():
    """Connects to Google Sheets, loads trade data, and cleans it."""
    try:
        client = get_google_sheet_client()
        if not client:
            return pd.DataFrame()
        
        sheet = client.open(SPREADSHEET_NAME)
        trades_ws = sheet.worksheet(TRADES_WORKSHEET_NAME)
        
        # We fetch all data to determine headers and the log start point
        all_data = trades_ws.get_all_values()
        
        # Determine the start row for the trade log (Row 11 in Excel, index 10 in Python)
        TRADE_LOG_START_ROW_INDEX = 10
        
        # --- CRITICAL FIX 1: Clean and Prepare Headers ---
        raw_headers = all_data[TRADE_LOG_START_ROW_INDEX]
        
        cleaned_headers = []
        empty_col_count = 0
        
        for h in raw_headers:
            h_stripped = h.strip()
            if h_stripped == '':
                cleaned_headers.append(f'EMPTY_COL_{empty_col_count}')
                empty_col_count += 1
            else:
                cleaned_headers.append(h_stripped)
        
        trade_data = all_data[TRADE_LOG_START_ROW_INDEX + 1:]
        
        df = pd.DataFrame(trade_data, columns=cleaned_headers)
        
        # --- CRITICAL FIX 2: Drop the identified EMPTY_COL columns ---
        cols_to_drop = [col for col in df.columns if col.startswith('EMPTY_COL_')]
        df.drop(columns=cols_to_drop, inplace=True)
        
        # --- Data Cleaning and Type Conversion ---
        
        # Convert PnL, RRR columns to numeric
        pnl_cols = ['Net PnL (₹)', 'Gross PnL (₹)', 'Charges'] 
        for col in pnl_cols:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col].replace({'': 0, '-': 0}), errors='coerce')
            
        # Convert date columns to datetime
        date_cols = ['Date Opened', 'Date Closed']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
            
        # Drop rows where essential data (like PnL) is missing
        df.dropna(subset=['Net PnL (₹)'], inplace=True)
        
        # Create Month column for analysis
        df['Month'] = df['Date Closed'].dt.to_period('M')
        
        return df
        
    except Exception as e:
        st.error(f"Error in data_manager.py: Failed to load or process Google Sheet data. Details: {e}")
        return pd.DataFrame()

# --- NEW FUNCTION FOR TRADE LOGGING ---
def append_trade_to_sheet(new_trade_data: dict):
    """Appends a new trade dictionary to the Google Sheet."""
    try:
        client = get_google_sheet_client()
        if not client:
            return False, "Failed to authenticate with Google Sheets client."
        
        sheet = client.open(SPREADSHEET_NAME)
        worksheet = sheet.worksheet(TRADES_WORKSHEET_NAME) # Use the trades worksheet

        # Get the row where data starts to determine the next trade number (Row 11 is index 10)
        TRADE_LOG_START_ROW_INDEX = 10
        
        # Get all values from the first column from the start of the trade log down
        # This is a robust way to determine the next row index
        trade_index_col = worksheet.col_values(1)
        next_trade_num = len(trade_index_col) - TRADE_LOG_START_ROW_INDEX 

        # The row data must be a LIST of values in the correct column order of your sheet.
        # ASSUMING COLUMN ORDER (YOU MUST VERIFY THIS ORDER IN YOUR SHEET)
        row_values = [
            next_trade_num,                                 # 1. # (Trade Number)
            new_trade_data.get('Date Opened', ''),          # 2. Date Opened
            new_trade_data.get('Date Closed', ''),          # 3. Date Closed
            new_trade_data.get('Symbol', ''),               # 4. Symbol
            new_trade_data.get('Short Strike', ''),         # 5. Short Strike
            new_trade_data.get('Long Strike', ''),          # 6. Long Strike
            new_trade_data.get('Entry Premium (₹)', 0),    # 7. Entry Premium (Defaulted to 0)
            new_trade_data.get('Total Width (pts)', 0),     # 8. Total Width (Defaulted to 0)
            new_trade_data.get('Max Loss (₹)', 0),          # 9. Max Loss (Defaulted to 0)
            new_trade_data.get('Exit Premium (₹)', 0),      # 10. Exit Premium (Defaulted to 0)
            new_trade_data.get('Gross PnL (₹)', 0),         # 11. Gross PnL (Defaulted to 0)
            new_trade_data.get('Charges', 0),               # 12. Charges
            new_trade_data.get('Net PnL (₹)', 0),           # 13. Net PnL (₹)
            new_trade_data.get('RRR', ''),                  # 14. RRR
            new_trade_data.get('Remarks', '')               # 15. Remarks
        ]

        # Append the new row to the worksheet
        worksheet.append_row(row_values)
        
        # Invalidate the cache so the app reloads with the new data
        st.cache_data.clear()
        
        return True, "Trade successfully logged and dashboard will update."
    
    except Exception as e:
        return False, f"Error logging trade to sheet: {e}"


def safe_get_cell_value(worksheet, range_name, default_value="N/A"):
    """Safely retrieves a single cell value from the worksheet."""
    try:
        # get() returns a list of lists (rows), e.g., [['Value']]
        values = worksheet.get(range_name)
        
        # Check if values is not empty and the first row is not empty
        if values and values[0] and values[0][0].strip():
            return values[0][0]
        else:
            return default_value
    except Exception:
        # Catch any other access error
        return default_value

# --- LOAD SUMMARY METRICS (Minor update to Win:Loss key) ---

@st.cache_data(ttl=43200)
def load_summary_metrics():
    """Loads the summary metrics from the top section of your sheet."""
    try:
        client = get_google_sheet_client()
        if not client:
            return {}

        sheet = client.open(SPREADSHEET_NAME)
        metrics_ws = sheet.worksheet(METRICS_WORKSHEET_NAME)
        
        # 1. Safely load the main metrics row (A2:H2)
        metrics_row = metrics_ws.get('A2:H2')
        # Check if the row is present, otherwise use a safe default list
        metrics = metrics_row[0] if metrics_row and metrics_row[0] else ["N/A"] * 8
        
        # 2. Safely load individual drawdown metrics
        # If C6 is empty, safe_get_cell_value will return "N/A"
        current_drawdown = safe_get_cell_value(metrics_ws, 'A6')
        median_drawdown = safe_get_cell_value(metrics_ws, 'B6')
        
        return {
            # Use safe list access for A2:H2 (indices 0 to 7)
            "Wins": metrics[0],
            "Losses": metrics[1],
            "Win %": metrics[2],
            "Avg Return (₹)": metrics[3],
            "Profit Factor": metrics[7],
            "Cumulative PnL (₹)": metrics[6],
            # Use safely retrieved single values
            "Current Drawdown": current_drawdown,
            "Median Drawdown": median_drawdown
        }
        
    except Exception as e:
        # Keep the original error message for debugging purposes
        st.error(f"Error loading metrics: {e}")
        return {}