# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from pmdarima import auto_arima
    FORECASTING_AVAILABLE = True
except ImportError as e:
    FORECASTING_AVAILABLE = False
    st.error(f"Forecasting packages not available: {e}")

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SalesForecaster:
    def __init__(self):
        self.models = {}
        
    def prepare_data(self, historical_data, item_id=None, customer_name=None):
        """Prepare time series data from raw sales data"""
        try:
            df = historical_data.copy()
            
            # Convert date column
            df['Posting Date'] = pd.to_datetime(df['Posting Date'])
            df['Month'] = df['Posting Date'].dt.to_period('M').dt.to_timestamp()
            
            # Apply accounting convention
            df['Sales_Quantity'] = np.where(df['Invoiced Quantity'] < 0, 
                                          abs(df['Invoiced Quantity']), 0)
            df['Sales_Revenue'] = np.where(df['Sales Amount'] > 0, 
                                         df['Sales Amount'], 0)
            
            # Aggregate by month, item, and customer
            monthly_data = df.groupby(['Month', 'Item No', 'Name']).agg({
                'Sales_Revenue': 'sum',
                'Sales_Quantity': 'sum',
                'Posting Date': 'count'
            }).reset_index()
            
            monthly_data = monthly_data.rename(columns={'Posting Date': 'Transaction_Count'})
            
            # Apply filters if specified
            if item_id:
                monthly_data = monthly_data[monthly_data['Item No'] == item_id]
            if customer_name:
                monthly_data = monthly_data[monthly_data['Name'] == customer_name]
                
            return monthly_data.sort_values('Month')
            
        except Exception as e:
            raise Exception(f"Data preparation error: {str(e)}")
    
    def fit_model(self, ts_vector, model_type='ets', horizon=6):
        """Generic model fitting function"""
        try:
            if len(ts_vector) < 12:
                return None
                
            if model_type == 'ets':
                model = ExponentialSmoothing(
                    ts_vector,
                    seasonal_periods=min(12, len(ts_vector)//2),
                    trend='add',
                    seasonal='add' if len(ts_vector) >= 24 else None
                ).fit()
                forecast = model.forecast(horizon)
                residuals = model.resid
                aic = getattr(model, 'aic', None)
                
            elif model_type == 'arima':
                model = auto_arima(
                    ts_vector,
                    seasonal=True,
                    m=min(12, len(ts_vector)//2),
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore'
                )
                forecast = model.predict(n_periods=horizon)
                residuals = model.resid()
                aic = model.aic()
                
            else:
                return None
                
            return {
                'model': model,
                'forecast': forecast.values if hasattr(forecast, 'values') else forecast,
                'residuals': residuals,
                'aic': aic
            }
        except Exception as e:
            print(f"{model_type.upper()} model error: {e}")
            return None
    
    def forecast(self, historical_data, item_id, customer_name, horizon=6):
        """Main forecasting function for both revenue and quantity"""
        try:
            # Prepare data
            ts_data = self.prepare_data(historical_data, item_id, customer_name)
            
            if len(ts_data) < 12:
                return {
                    'error': f'Insufficient data: only {len(ts_data)} months available (need at least 12)'
                }
            
            # Forecast revenue
            revenue_forecast = self.fit_model(ts_data['Sales_Revenue'].values, 'ets', horizon)
            quantity_forecast = self.fit_model(ts_data['Sales_Quantity'].values, 'ets', horizon)
            
            # Generate future dates
            last_date = pd.to_datetime(ts_data['Month'].iloc[-1])
            future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(horizon)]
            
            return {
                'success': True,
                'revenue_forecast': revenue_forecast['forecast'] if revenue_forecast else None,
                'quantity_forecast': quantity_forecast['forecast'] if quantity_forecast else None,
                'future_dates': future_dates,
                'historical_data': ts_data,
                'last_historical_revenue': ts_data['Sales_Revenue'].iloc[-1],
                'last_historical_quantity': ts_data['Sales_Quantity'].iloc[-1],
                'forecast_horizon': horizon
            }
            
        except Exception as e:
            return {
                'error': f'Forecasting error: {str(e)}'
            }

def create_matplotlib_plot(historical_data, revenue_forecast, quantity_forecast, future_dates, title):
    """Create matplotlib plot for forecasts"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Revenue plot
    ax1.plot(historical_data['Month'], historical_data['Sales_Revenue'], 
             label='Historical Revenue', color='blue', linewidth=2, marker='o')
    ax1.plot(future_dates, revenue_forecast, 
             label='Revenue Forecast', color='red', linewidth=2, linestyle='--', marker='s')
    ax1.set_title(f'{title} - Revenue', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Revenue ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Quantity plot
    ax2.plot(historical_data['Month'], historical_data['Sales_Quantity'], 
             label='Historical Quantity', color='green', linewidth=2, marker='o')
    ax2.plot(future_dates, quantity_forecast, 
             label='Quantity Forecast', color='orange', linewidth=2, linestyle='--', marker='s')
    ax2.set_title(f'{title} - Quantity', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Quantity', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    st.title("📈 Sales Forecasting Dashboard")
    
    # Initialize forecaster
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = SalesForecaster()
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    
    if not FORECASTING_AVAILABLE:
        st.error("""
        ❌ Required forecasting packages are not available. 
        Please ensure your requirements.txt includes:
        - statsmodels
        - pmdarima
        - scikit-learn
        """)
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Data Upload", "Single Item Forecast", "Batch Forecast", "Download Center"]
    )
    
    if app_mode == "Data Upload":
        show_data_upload()
    elif app_mode == "Single Item Forecast":
        show_single_forecast()
    elif app_mode == "Batch Forecast":
        show_batch_forecast()
    elif app_mode == "Download Center":
        show_download_center()

def show_data_upload():
    st.header("📁 Data Upload & Preview")
    
    uploaded_file = st.file_uploader(
        "Upload your sales data (Excel or CSV)",
        type=['xlsx', 'xls', 'csv'],
        help="Required columns: Posting Date, Item No, Name, Invoiced Quantity, Sales Amount"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.sales_data = df
            st.success("✅ Data uploaded successfully!")
            
            # Data preview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Unique Items", df['Item No'].nunique())
            with col3:
                st.metric("Unique Customers", df['Name'].nunique())
            with col4:
                total_sales = df[df['Invoiced Quantity'] < 0]['Invoiced Quantity'].sum()
                st.metric("Total Sales Quantity", f"{-total_sales:,.0f}")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Show basic data summary
            with st.expander("Show Data Summary"):
                st.write(df.describe())
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("👆 Please upload your sales data to get started")

def show_single_forecast():
    st.header("🔍 Single Item Forecast")
    
    if 'sales_data' not in st.session_state:
        st.warning("⚠️ Please upload data first in the 'Data Upload' section")
        return
    
    df = st.session_state.sales_data
    
    col1, col2, col3 = st.columns(3)
    with col1:
        items = df['Item No'].unique()
        selected_item = st.selectbox("Select Item", items)
    with col2:
        customers = df[df['Item No'] == selected_item]['Name'].unique()
        selected_customer = st.selectbox("Select Customer", customers)
    with col3:
        forecast_horizon = st.slider("Forecast Horizon (months)", 1, 12, 6)
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                result = st.session_state.forecaster.forecast(
                    df, selected_item, selected_customer, forecast_horizon
                )
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    display_forecast_results(result, selected_item, selected_customer)
                    
            except Exception as e:
                st.error(f"Forecasting failed: {str(e)}")

def display_forecast_results(result, item, customer):
    st.success("✅ Forecast generated successfully!")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Last Month Revenue", f"${result['last_historical_revenue']:,.2f}")
    with col2:
        st.metric("Last Month Quantity", f"{result['last_historical_quantity']:,.0f}")
    with col3:
        avg_rev = np.mean(result['revenue_forecast'])
        st.metric("Avg Revenue Forecast", f"${avg_rev:,.2f}")
    with col4:
        avg_qty = np.mean(result['quantity_forecast'])
        st.metric("Avg Quantity Forecast", f"{avg_qty:,.0f}")
    
    # Matplotlib visualization
    st.subheader("📊 Forecast Visualization")
    fig = create_matplotlib_plot(
        result['historical_data'],
        result['revenue_forecast'],
        result['quantity_forecast'],
        result['future_dates'],
        f"{item} - {customer}"
    )
    st.pyplot(fig)
    
    # Forecast tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Revenue Forecast")
        revenue_df = pd.DataFrame({
            'Month': [d.strftime('%Y-%m') for d in result['future_dates']],
            'Forecasted_Revenue': result['revenue_forecast']
        })
        st.dataframe(revenue_df.style.format({
            'Forecasted_Revenue': '${:,.2f}'
        }), use_container_width=True)
    
    with col2:
        st.subheader("📦 Quantity Forecast")
        quantity_df = pd.DataFrame({
            'Month': [d.strftime('%Y-%m') for d in result['future_dates']],
            'Forecasted_Quantity': result['quantity_forecast']
        })
        st.dataframe(quantity_df.style.format({
            'Forecasted_Quantity': '{:,.0f}'
        }), use_container_width=True)
    
    # Download section
    st.subheader("💾 Download Forecast")
    download_data = []
    for i, (date, rev, qty) in enumerate(zip(result['future_dates'], 
                                           result['revenue_forecast'], 
                                           result['quantity_forecast'])):
        download_data.append({
            'Item_No': item,
            'Customer': customer,
            'Month': date.strftime('%Y-%m'),
            'Revenue_Forecast': rev,
            'Quantity_Forecast': qty
        })
    
    download_df = pd.DataFrame(download_data)
    csv = download_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="forecast_{item}_{customer}.csv">📥 Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def show_batch_forecast():
    st.header("📊 Batch Forecast")
    
    if 'sales_data' not in st.session_state:
        st.warning("⚠️ Please upload data first in the 'Data Upload' section")
        return
    
    df = st.session_state.sales_data
    
    st.info("This will generate forecasts for all item-customer combinations with sufficient data (≥12 months)")
    
    if st.button("Run Batch Forecast", type="primary"):
        with st.spinner("Running batch forecast... This may take a while"):
            try:
                # Simple batch processing demonstration
                ts_data = st.session_state.forecaster.prepare_data(df)
                combinations = ts_data.groupby(['Item No', 'Name']).agg({
                    'Sales_Revenue': ['count', 'sum']
                }).reset_index()
                combinations.columns = ['Item No', 'Name', 'Months_Data', 'Total_Revenue']
                viable_combinations = combinations[combinations['Months_Data'] >= 12]
                
                st.session_state.batch_results = {
                    'viable_combinations': viable_combinations,
                    'total_count': len(viable_combinations)
                }
                
                st.success(f"✅ Found {len(viable_combinations)} viable item-customer combinations")
                st.dataframe(viable_combinations.head(20), use_container_width=True)
                
            except Exception as e:
                st.error(f"Batch analysis failed: {str(e)}")

def show_download_center():
    st.header("💾 Download Center")
    
    if 'sales_data' not in st.session_state:
        st.warning("⚠️ Please upload data first")
        return
    
    st.info("Download processed data and forecast templates")
    
    df = st.session_state.sales_data
    
    # Process data for download
    processed_data = st.session_state.forecaster.prepare_data(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Processed Monthly Data")
        st.write("Aggregated monthly sales data by item and customer")
        csv_processed = processed_data.to_csv(index=False)
        b64 = base64.b64encode(csv_processed.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="processed_sales_data.csv">📥 Download Processed Data</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        st.dataframe(processed_data.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📋 Forecast Template")
        st.write("Template for manual forecasting input")
        # Create a simple template
        unique_items = df['Item No'].unique()
        template_data = []
        for item in unique_items[:10]:  # Limit for demo
            template_data.append({
                'Item_No': item,
                'Customer': 'All',
                'Base_Quantity': 100,
                'Growth_Rate': 0.05,
                'Seasonality_Factor': 1.0
            })
        
        template_df = pd.DataFrame(template_data)
        csv_template = template_df.to_csv(index=False)
        b64 = base64.b64encode(csv_template.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="forecast_template.csv">📥 Download Template</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
