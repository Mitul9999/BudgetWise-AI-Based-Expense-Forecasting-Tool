import streamlit as st
from data_utils import load_transactions, save_transaction, ensure_files, load_users
from auth import register_user, authenticate
from categorizer import categorize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
import os

ensure_files() 
st.set_page_config(page_title="BudgetWise", layout="wide")

if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'df' not in st.session_state:
    st.session_state['df'] = load_transactions()
if 'current_file' not in st.session_state:
    st.session_state['current_file'] = "transactions_large.csv"


st.title("💰 BudgetWise — Expense Analyzer")


logged_in = st.session_state['username'] is not None

if logged_in:
    menu = st.sidebar.selectbox("Menu", ["Dashboard", "Logout"])
else:
    menu = st.sidebar.selectbox("Menu", ["Login", "Register"])


# AUTHENTICATION->

if menu == "Register" and not logged_in:
    st.header("Create Account")
    uname = st.text_input("Username")
    email = st.text_input("Email")
    pwd = st.text_input("Password", type="password")
    if st.button("Register"):
        ok, msg = register_user(uname, email, pwd)
        if ok:
            st.success(msg + " — please login now.")
        else:
            st.error(msg)
    st.stop()

elif menu == "Login" and not logged_in:
    st.header("Login")
    user_input = st.text_input("Username or Email")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        ok, msg_or_user = authenticate(user_input, pwd)
        if ok:
            st.session_state['username'] = msg_or_user
            st.rerun() 
        else:
            st.error(msg_or_user)
    st.stop()

elif menu == "Logout" and logged_in:
    del st.session_state['username']
    st.session_state['df'] = load_transactions()
    st.info("Logged out. Use the sidebar to login/register.")
    st.rerun()


if not logged_in:
    st.info("Please login or register from the sidebar to use the application.")
    st.stop()

# DASHBOARD->
current_user = st.session_state['username']
df = st.session_state['df']
st.markdown(f"**Logged in as:** {current_user}")


st.sidebar.subheader("Upload Custom Data (Optional)")
uploaded_file = st.sidebar.file_uploader(
    f"Upload CSV ({st.session_state['current_file']} format required)", 
    type="csv"
)

if uploaded_file is not None:
    try:
        new_df = pd.read_csv(uploaded_file)
        
        required_cols = ['Date', 'Description', 'Category', 'Amount', 'Type']
    
        new_df.columns = [str(col).strip() for col in new_df.columns]
       
        if 'User' not in new_df.columns:
            new_df['User'] = current_user
        
        if not all(col in new_df.columns for col in required_cols):
             st.error(f"Uploaded CSV must contain all required columns: {', '.join(required_cols)}")
             st.stop()
        
        #Feature Enginneering Methods->
        new_df['Date'] = pd.to_datetime(new_df['Date'], errors='coerce')
        new_df['Amount'] = pd.to_numeric(new_df['Amount'], errors='coerce')
        new_df['Description'] = new_df['Description'].astype(str)
        new_df['User'] = current_user
        
    
        new_df['Category'] = new_df.apply(
            lambda row: categorize(row['Description']) if pd.notna(row['Description']) else "Other", axis=1
        )
      
        new_df.dropna(subset=['Date', 'Amount'], inplace=True)
        
        
        full_df_disk = load_transactions()
        # Filter out the current user's data from the disk
        all_other_users_df = full_df_disk[full_df_disk['User'] != current_user]
        
        # Concatenate both df together->
        st.session_state['df'] = pd.concat([all_other_users_df, new_df], ignore_index=True)
        
        st.success(f"Custom data loaded and processed! Total {len(new_df)} valid records processed for {current_user}.")
        
       
        st.session_state['df'].to_csv(st.session_state['current_file'], index=False)

        st.rerun()

    except Exception as e:
        st.error(f"Error processing uploaded data: {e}. Please check file format.")
        

st.header("Add a Transaction")
with st.form("tx_form", clear_on_submit=True):
    date = st.date_input("Date", value=datetime.today())
    description = st.text_input("Description (e.g., 'Uber ride', 'Walmart groceries')")
    amount = st.number_input("Amount (₹)", min_value=0.0, step=1.0)
    
   
    cat_manual = st.text_input("Category (optional)", value="", placeholder="Leave blank for auto-detect") 
    
    tx_type = st.selectbox("Type", ["Expense", "Income"])
    submitted = st.form_submit_button("Add Transaction")
    if submitted:
        if amount <= 0.0:
            st.error("Amount must be greater than zero.")
            st.stop()
            
       
        if cat_manual.strip() != "":
            cat = cat_manual.strip()
        else:
            cat = categorize(description)

        record = {
            "Date": pd.to_datetime(date).strftime("%Y-%m-%d"),
            "Description": description,
            "Category": cat,
            "Amount": float(amount),
            "Type": tx_type,
            "User": current_user
        }
        save_transaction(record)
        st.session_state['df'] = load_transactions()
        st.success("Transaction saved!")
        st.rerun()


user_df = df[df['User'] == current_user].copy()

# Date conversion->
user_df['Date'] = pd.to_datetime(user_df['Date'], errors='coerce')
user_df.dropna(subset=['Date'], inplace=True) 

st.subheader("Filter Transactions")

col_filters = st.columns(3)

if user_df.empty:
    st.info("Start by adding a transaction above!")
    st.stop()


valid_dates_df = user_df[user_df['Date'].notna()]

if valid_dates_df.empty:
    safe_date = datetime.today().date()
    min_date_data = safe_date
    max_date_data = safe_date
else:
    min_date_data = valid_dates_df['Date'].min().date()
    max_date_data = valid_dates_df['Date'].max().date()

# Date Filtering->
date_range = col_filters[0].date_input(
    "Date Range",
    [min_date_data, max_date_data],
    min_value=min_date_data,
    max_value=max_date_data 
)

if len(date_range) == 2:

    start_date = pd.to_datetime(date_range[0]).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(date_range[1]).strftime('%Y-%m-%d')
    

    filtered_df = user_df[
        (user_df['Date'].dt.strftime('%Y-%m-%d') >= start_date) & 
        (user_df['Date'].dt.strftime('%Y-%m-%d') <= end_date)
    ].copy()
else:
    filtered_df = user_df.copy()

# Category Filtering->
filtered_df['Category'] = filtered_df['Category'].astype(str)
all_categories = ['All Categories'] + filtered_df['Category'].unique().tolist()
selected_category = col_filters[1].selectbox("Category", all_categories)
if selected_category != 'All Categories':
    filtered_df = filtered_df[filtered_df['Category'] == selected_category]

# Type Filtering->
filtered_df['Type'] = filtered_df['Type'].astype(str)
all_types = ['All Types'] + filtered_df['Type'].unique().tolist()
selected_type = col_filters[2].selectbox("Type", all_types)
if selected_type != 'All Types':
    filtered_df = filtered_df[filtered_df['Type'] == selected_type]


st.subheader("Your Transactions")
if filtered_df.empty:
    st.write("No transactions found matching the selected filters.")
else:
  
    filtered_df['Date_Display'] = filtered_df['Date'].dt.strftime('%Y-%m-%d')
    display_cols = ["Date_Display", "Description", "Category", "Amount", "Type"]
    st.dataframe(
        filtered_df.sort_values("Date", ascending=False)
                   .reset_index(drop=True)[display_cols]
                   .rename(columns={'Date_Display': 'Date'})
    )

    # Export or Download the Filtered file->
    @st.cache_data
    def convert_df_to_csv(df):
        export_df = df.copy()
        if 'User' in export_df.columns:
            export_df = export_df.drop(columns=['User'])
        if 'Date' in export_df.columns:
            export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')
        return export_df[['Date', 'Description', 'Category', 'Amount', 'Type']].to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(filtered_df)
    st.download_button(
        label="Export Filtered Data as CSV",
        data=csv,
        file_name=f"{current_user}_transactions_filtered.csv",
        mime="text/csv",
    )



st.subheader("Summary Metrics & Charts")

if not filtered_df.empty:
    total = filtered_df.loc[filtered_df['Type']=='Expense','Amount'].sum()
    total_income = filtered_df.loc[filtered_df['Type']=='Income','Amount'].sum()
    avg_tx = filtered_df['Amount'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spent (₹)", f"{total:.2f}")
    col2.metric("Total Income (₹)", f"{total_income:.2f}")
    col3.metric("Average Tx (₹)", f"{avg_tx:.2f}")

    expense_df = filtered_df[filtered_df['Type']=='Expense'].copy()
    if not expense_df.empty:
        
        # CategoryWise Grouping->
        expense_df['Category'] = expense_df['Category'].astype(str)
        by_cat = expense_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        
        col_chart_1, col_chart_2 = st.columns(2)

        with col_chart_1:
            st.markdown("##### Category-wise Spending")
            fig, ax = plt.subplots(figsize=(6,4))
            by_cat.plot(kind='bar', ax=ax, edgecolor='black', color=sns.color_palette("pastel")[0])
            ax.set_ylabel("Amount (₹)")
            ax.set_xlabel("Category")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

        with col_chart_2:
            st.markdown("##### Category Share (Expense)")
            fig2, ax2 = plt.subplots(figsize=(5,4))
            by_cat.plot(kind='pie', autopct="%1.1f%%", ax=ax2, startangle=90, counterclock=False)
            ax2.set_ylabel("")
            st.pyplot(fig2)

        daily = expense_df.groupby('Date')['Amount'].sum().sort_index()
        if not daily.empty:
            st.markdown("##### Daily Spending Trend")
            fig3, ax3 = plt.subplots(figsize=(10,4))
            ax3.plot(daily.index, daily.values, marker='o', linestyle='-', color='red')
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Total (₹)")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig3)

    
        st.subheader("📈 Monthly Expense Forecasting (AI)")

        # Prepare data for forecasting
        expense_df['Month'] = expense_df['Date'].dt.to_period('M')
        monthly_totals = expense_df.groupby('Month')['Amount'].sum().reset_index()
        
        if len(monthly_totals) >= 2:
            
            monthly_totals['Month_Num'] = np.arange(len(monthly_totals)) + 1
            
            X = monthly_totals[['Month_Num']]
            y = monthly_totals['Amount']
            
            model = LinearRegression()
            model.fit(X, y)
            
            next_month_num = len(monthly_totals) + 1
            future_X = np.array([[next_month_num]])
            forecast_amount = model.predict(future_X)[0]
            
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
           
            last_period = monthly_totals['Month'].iloc[-1]
            next_month_date = (last_period + 1).start_time 
            next_month_str = next_month_date.strftime("%B %Y")

            col_forecast_1, col_forecast_2, col_forecast_3 = st.columns(3)
            
            col_forecast_1.metric(
                label=f"Forecasted Expense ({next_month_str})",
                value=f"₹ {max(0, forecast_amount):.2f}",
                delta_color="inverse"
            )
            
            col_forecast_2.metric(
                label="Trend Strength (R²)",
                value=f"{r2:.2f}",
                help="R-squared measures how well the linear model fits the data (1.0 is a perfect fit)."
            )
            
            col_forecast_3.metric(
                label="Monthly Trend (Slope)",
                value=f"₹ {model.coef_[0]:.2f}",
                delta=f"{model.coef_[0]:.2f}",
                delta_color="inverse",
                help="The average change in spending (increase/decrease) per month."
            )

            fig4, ax4 = plt.subplots(figsize=(10,4))
            
            ax4.plot(monthly_totals['Month_Num'], monthly_totals['Amount'], marker='o', label='Actual Spending', color='skyblue')
            
            trend_X = np.array([[1], [next_month_num]])
            trend_y = model.predict(trend_X)
            ax4.plot(trend_X, trend_y, linestyle='--', color='gray', label='Linear Trend')
            
            ax4.scatter(next_month_num, forecast_amount, color='red', marker='X', s=100, label='Forecast')
            
            ax4.set_xlabel("Historical Month Count (1=Oldest)")
            ax4.set_ylabel("Total Expense (₹)")
            ax4.set_title("Monthly Spending Trend & Forecast")
            ax4.legend()
            st.pyplot(fig4)

        else:
            st.info("Need transactions spanning at least two different months in the selected date range to generate a forecast.")
else:
    st.write("Add expense transactions and filter to view metrics and charts.")
