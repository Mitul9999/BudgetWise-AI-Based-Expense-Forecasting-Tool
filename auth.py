from data_utils import load_users, add_user, username_exists, email_exists
from datetime import datetime
import bcrypt

def hash_password(password: str) -> str:
   
    pw = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pw, salt)
    return hashed.decode('utf-8')

def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def register_user(username: str, email: str, password: str):
    if username_exists(username):
        return False, "Username already exists"
    if email_exists(email):
        return False, "Email already exists"
    password_hash = hash_password(password)
    user_dict = {
        "username": username,
        "email": email,
        "password_hash": password_hash,
        "joined_on": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }
    add_user(user_dict)
    return True, "User registered successfully"

def authenticate(username_or_email: str, password: str):
    users = load_users()
    found = users[(users['username'] == username_or_email) | (users['email'] == username_or_email)]
    if found.empty:
        return False, "User not found"
    row = found.iloc[0]
    if check_password(password, row['password_hash']):
        return True, row['username'] 
    return False, "Invalid password"
