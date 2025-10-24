import pandas as pd
import os
from datetime import datetime

TRANSACTION_FILE = "transactions_large.csv"  
USER_FILE = "users.csv"

def ensure_files():
    if not os.path.exists(TRANSACTION_FILE):
        df = pd.DataFrame(columns=["Date","Description","Category","Amount","Type","User"])
        df.to_csv(TRANSACTION_FILE, index=False)
    
    if not os.path.exists(USER_FILE):
        df = pd.DataFrame(columns=["username","email","password_hash","joined_on"])
        df.to_csv(USER_FILE, index=False)

def load_transactions():
    ensure_files()
    
    try:
        df = pd.read_csv(TRANSACTION_FILE)
        load_transactions.TRANSACTION_FILE = TRANSACTION_FILE
        return df
    except pd.errors.EmptyDataError:
        load_transactions.TRANSACTION_FILE = TRANSACTION_FILE
        return pd.DataFrame(columns=["Date","Description","Category","Amount","Type","User"])

def save_transaction(record: dict):
    ensure_files()
    df = load_transactions()
    new_df = pd.DataFrame([record])
    new_df = new_df.reindex(columns=df.columns, fill_value=None)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(TRANSACTION_FILE, index=False)

def load_users():
    ensure_files()
    
    try:
        return pd.read_csv(USER_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["username","email","password_hash","joined_on"])

def add_user(user_dict: dict):
    ensure_files()
    users = load_users()
    new_user_df = pd.DataFrame([user_dict])
    new_user_df = new_user_df.reindex(columns=users.columns, fill_value=None)
    users = pd.concat([users, new_user_df], ignore_index=True)
    users.to_csv(USER_FILE, index=False)

def username_exists(username):
    users = load_users()
    return username in users['username'].values

def email_exists(email):
    users = load_users()
    return email in users['email'].values
