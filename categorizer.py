import re

KEYWORD_MAP = {
    "Food": ["restaurant","cafe","zomato","dominos","pizza","burger","mcdonald","dine"],
    "Groceries": ["grocery","supermarket","dmart","bigbasket","groceries","big bazaar","ration"],
    "Transport": ["uber","ola","taxi","bus","metro","train","petrol","fuel"],
    "Rent": ["rent","rental","landlord","emi","loan","mortgage"],
    "Shopping": ["amazon","flipkart","clothes","shopping","shirt","shoes","clothing","store"],
    "Bills": ["electricity","water","internet","phone","bill","broadband","utility","telecom"],
    "Entertainment": ["netflix","prime","movie","cinema","spotify","concert","gaming","ticket"],
    "Healthcare": ["doctor","hospital","clinic","pharmacy","medicine","medicare"],
    "Education": ["course","udemy","coursera","tuition","school","college","book"],
}

def categorize(description):
    desc = str(description).lower().strip()
    
    if not desc:
        return "Other"
    
    # Direct keyword matching
    for cat, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in desc:
                return cat
    
    # Fallback rules
    if re.search(r'\b(market|bazaar|super)\b', desc):
        return "Groceries"
    if re.search(r'\b(cab|auto|rail|air)\b', desc):
        return "Transport"
    
    return "Other" 
