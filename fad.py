# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting App",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📈 Sales Forecasting Dashboard")
    st.markdown("""
    This app provides sales forecasting capabilities using simple moving averages 
    and trend analysis. Upload your sales data as a CSV file to get started.
    """)
    
    # Initialize session state
    if 'sales_data' not in st.session_state:
        st.session_state.sales_data = None
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose Mode",
        ["Data Upload", "Simple Forecast", "Data Analysis", "Export Data"]
    )
    
    if app_mode == "Data Upload":
        show_data_upload()
    elif app_mode == "Simple Forecast":
        show_simple_forecast()
    elif app_mode == "Data Analysis":
        show_data_analysis()
    elif app_mode == "Export Data":
        show_export_data()

def show_data_upload():
    st.header("📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your sales data (CSV format)",
        type=['csv'],
        help="Required columns: Posting Date, Item No, Name, Invoiced Quantity, Sales Amount"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Basic validation
            required_cols = ['Posting Date', 'Item No', 'Name', 'Invoiced Quantity', 'Sales Amount']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("""
                Your CSV should contain these columns:
                - Posting Date
                - Item No  
                - Name
                - Invoiced Quantity
                - Sales Amount
                """)
                return
            
            st.session_state.sales_data = df
            st.success("✅ Data uploaded successfully!")
            
            # Show basic stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Unique Items", df['Item No'].nunique())
            with col3:
                st.metric("Unique Customers", df['Name'].nunique())
            with col4:
                # Calculate total sales quantity (negative values are sales)
                sales_qty = df[df['Invoiced Quantity'] < 0]['Invoiced Quantity'].sum()
                st.metric("Total Sales Quantity", f"{-sales_qty:,.0f}")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data summary
            with st.expander("Show Data Summary"):
                st.write("**Basic Statistics:**")
                st.write(df[['Invoiced Quantity', 'Sales Amount']].describe())
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("👆 Please upload your sales data as a CSV file to get started")

def show_simple_forecast():
    st.header("🔮 Simple Sales Forecast")
    
    if st.session_state.sales_data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' section")
        return
    
    df = st.session_state.sales_data
    
    # Process the data
    processed_data = process_sales_data(df)
    
    if processed_data is None:
        st.error("❌ Error processing data. Please check your data format.")
        return
    
    # Forecasting options
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_method = st.selectbox(
            "Forecast Method",
            ["Moving Average (3 months)", "Moving Average (6 months)", "Linear Trend", "Last Value"]
        )
    
    with col2:
        forecast_months = st.slider("Months to Forecast", 1, 12, 3)
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                # Generate forecasts for top items
                top_items = get_top_items(processed_data, top_n=5)
                forecasts = {}
                
                for item in top_items:
                    item_data = processed_data[processed_data['Item No'] == item]
                    monthly_sales = item_data.groupby('Month')['Sales_Revenue'].sum().sort_index()
                    
                    if len(monthly_sales) >= 3:
                        forecast = generate_simple_forecast(
                            monthly_sales.values, 
                            forecast_method, 
                            forecast_months
                        )
                        forecasts[item] = {
                            'historical': monthly_sales.values,
                            'forecast': forecast,
                            'last_date': monthly_sales.index[-1],
                            'months_data': len(monthly_sales)
                        }
                
                if forecasts:
                    display_forecasts(forecasts, forecast_method)
                else:
                    st.warning("Not enough data for forecasting. Need at least 3 months of data.")
                
            except Exception as e:
                st.error(f"Forecasting error: {str(e)}")

def process_sales_data(df):
    """Process raw sales data into monthly aggregates"""
    try:
        df_clean = df.copy()
        df_clean['Posting Date'] = pd.to_datetime(df_clean['Posting Date'], errors='coerce')
        
        # Remove rows with invalid dates
        df_clean = df_clean.dropna(subset=['Posting Date'])
        
        df_clean['Month'] = df_clean['Posting Date'].dt.to_period('M').dt.to_timestamp()
        
        # Apply accounting convention: negative quantities are sales
        df_clean['Sales_Quantity'] = np.where(
            df_clean['Invoiced Quantity'] < 0, 
            abs(df_clean['Invoiced Quantity']), 
            0
        )
        df_clean['Sales_Revenue'] = np.where(
            df_clean['Sales Amount'] > 0, 
            df_clean['Sales Amount'], 
            0
        )
        
        return df_clean
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return None

def get_top_items(processed_data, top_n=5):
    """Get top selling items by revenue"""
    try:
        item_sales = processed_data.groupby('Item No')['Sales_Revenue'].sum()
        return item_sales.nlargest(top_n).index.tolist()
    except:
        return []

def generate_simple_forecast(historical_data, method, months):
    """Generate simple forecast using basic methods"""
    try:
        if method == "Moving Average (3 months)":
            # 3-month moving average
            last_avg = np.mean(historical_data[-3:])
            return [last_avg] * months
        
        elif method == "Moving Average (6 months)":
            # 6-month moving average
            if len(historical_data) >= 6:
                last_avg = np.mean(historical_data[-6:])
            else:
                last_avg = np.mean(historical_data)
            return [last_avg] * months
        
        elif method == "Linear Trend":
            # Simple linear trend
            if len(historical_data) >= 3:
                x = np.arange(len(historical_data))
                trend = np.polyfit(x, historical_data, 1)
                future_x = np.arange(len(historical_data), len(historical_data) + months)
                forecast = np.polyval(trend, future_x)
                # Ensure no negative forecasts
                return np.maximum(forecast, 0)
            else:
                last_value = historical_data[-1]
                return [last_value] * months
        
        else:  # Last Value
            last_value = historical_data[-1]
            return [last_value] * months
    except:
        # Fallback: use last value
        last_value = historical_data[-1] if len(historical_data) > 0 else 0
        return [last_value] * months

def display_forecasts(forecasts, method):
    """Display forecast results"""
    st.success(f"✅ Forecast generated using {method}")
    
    for item, data in forecasts.items():
        with st.expander(f"📊 Forecast for {item} ({data['months_data']} months data)", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Last Month Revenue", f"${data['historical'][-1]:,.2f}")
            with col2:
                avg_forecast = np.mean(data['forecast'])
                st.metric("Avg Forecast", f"${avg_forecast:,.2f}")
            with col3:
                change_pct = ((avg_forecast - data['historical'][-1]) / data['historical'][-1]) * 100
                st.metric("Change", f"{change_pct:+.1f}%")
            
            # Trend analysis
            st.write("**Trend Analysis:**")
            hist_avg = np.mean(data['historical'][-3:])
            forecast_avg = np.mean(data['forecast'])
            
            if forecast_avg > hist_avg * 1.15:
                st.success("📈 Strong growth expected (+15%+)")
            elif forecast_avg > hist_avg * 1.05:
                st.info("↗️ Moderate growth expected (+5-15%)")
            elif forecast_avg > hist_avg * 0.95:
                st.info("➡️ Stable performance expected (±5%)")
            elif forecast_avg > hist_avg * 0.85:
                st.warning("↘️ Moderate decline expected (-5-15%)")
            else:
                st.error("📉 Significant decline expected (-15%+)")
            
            # Forecast table
            forecast_dates = [
                (pd.to_datetime(data['last_date']) + pd.DateOffset(months=i+1)).strftime('%Y-%m')
                for i in range(len(data['forecast']))
            ]
            
            forecast_df = pd.DataFrame({
                'Month': forecast_dates,
                'Forecasted_Revenue': data['forecast'],
                'Confidence_Range': f"±{np.std(data['historical'][-6:])/np.mean(data['historical'][-6:])*100:.1f}%" if len(data['historical']) >= 6 else "±N/A"
            })
            
            st.dataframe(forecast_df.style.format({
                'Forecasted_Revenue': '${:,.2f}'
            }), use_container_width=True)
            
            # Download button for this forecast
            download_df = pd.DataFrame({
                'Month': forecast_dates,
                'Item_No': item,
                'Forecasted_Revenue': data['forecast'],
                'Method': method
            })
            
            csv = download_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="forecast_{item}.csv">📥 Download {item} Forecast</a>'
            st.markdown(href, unsafe_allow_html=True)

def show_data_analysis():
    st.header("📊 Data Analysis")
    
    if st.session_state.sales_data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' section")
        return
    
    df = st.session_state.sales_data
    processed_data = process_sales_data(df)
    
    if processed_data is None:
        st.error("❌ Error processing data")
        return
    
    # Basic analysis
    st.subheader("Sales Summary")
    
    # Top items
    top_items = processed_data.groupby('Item No').agg({
        'Sales_Revenue': 'sum',
        'Sales_Quantity': 'sum',
        'Posting Date': 'count'
    }).nlargest(10, 'Sales_Revenue')
    
    top_items.columns = ['Total_Revenue', 'Total_Quantity', 'Transaction_Count']
    
    st.write("**Top 10 Items by Revenue:**")
    st.dataframe(top_items.style.format({
        'Total_Revenue': '${:,.2f}',
        'Total_Quantity': '{:,.0f}'
    }), use_container_width=True)
    
    # Top customers
    top_customers = processed_data.groupby('Name').agg({
        'Sales_Revenue': 'sum',
        'Sales_Quantity': 'sum',
        'Posting Date': 'count'
    }).nlargest(10, 'Sales_Revenue')
    
    top_customers.columns = ['Total_Revenue', 'Total_Quantity', 'Transaction_Count']
    
    st.write("**Top 10 Customers by Revenue:**")
    st.dataframe(top_customers.style.format({
        'Total_Revenue': '${:,.2f}',
        'Total_Quantity': '{:,.0f}'
    }), use_container_width=True)
    
    # Monthly trends
    monthly_trends = processed_data.groupby('Month').agg({
        'Sales_Revenue': 'sum',
        'Sales_Quantity': 'sum',
        'Posting Date': 'count'
    }).reset_index()
    
    monthly_trends.columns = ['Month', 'Total_Revenue', 'Total_Quantity', 'Transaction_Count']
    
    st.write("**Monthly Sales Trends:**")
    st.dataframe(monthly_trends.style.format({
        'Total_Revenue': '${:,.2f}',
        'Total_Quantity': '{:,.0f}'
    }), use_container_width=True)

def show_export_data():
    st.header("💾 Export Data")
    
    if st.session_state.sales_data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' section")
        return
    
    df = st.session_state.sales_data
    processed_data = process_sales_data(df)
    
    if processed_data is None:
        st.error("❌ Error processing data")
        return
    
    st.info("Export processed data for further analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Processed Monthly Data")
        st.write("Aggregated monthly sales by item and customer")
        
        monthly_data = processed_data.groupby(['Month', 'Item No', 'Name']).agg({
            'Sales_Revenue': 'sum',
            'Sales_Quantity': 'sum'
        }).reset_index()
        
        csv_data = monthly_data.to_csv(index=False)
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="monthly_sales_data.csv">📥 Download Monthly Data (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        st.dataframe(monthly_data.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📋 Sales Summary")
        st.write("Item and customer level summaries")
        
        item_summary = processed_data.groupby('Item No').agg({
            'Sales_Revenue': ['sum', 'mean', 'count']
        }).round(2)
        item_summary.columns = ['Total_Revenue', 'Avg_Revenue', 'Transaction_Count']
        item_summary = item_summary.reset_index()
        
        csv_summary = item_summary.to_csv(index=False)
        b64 = base64.b64encode(csv_summary.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="sales_summary.csv">📥 Download Sales Summary (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        st.dataframe(item_summary.head(10), use_container_width=True)
    
    # Raw data export
    st.subheader("📄 Raw Data Export")
    st.write("Download the original uploaded data")
    
    csv_raw = df.to_csv(index=False)
    b64 = base64.b64encode(csv_raw.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="original_sales_data.csv">📥 Download Original Data (CSV)</a>'
    st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
