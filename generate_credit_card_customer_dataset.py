import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 8000
OUTPUT_PATH = "outputs/bank_credit_card_customers_8000.csv"

np.random.seed(RANDOM_SEED)

segments = {
    "Heavy_Spenders": {
        "weight": 0.20,
        "monthly_spend": (2500, 6500),
        "yearly_spend": (30000, 78000),
        "transactions": (35, 90),
        "avg_trans": (80, 220),
        "offers": (12, 40),
        "balance": (1800, 9000),
        "cash_advance": (0, 350),
        "credit_limit": (12000, 40000),
        "days_since_last": (3, 18),
    },
    "Frequent_Users": {
        "weight": 0.25,
        "monthly_spend": (1400, 3500),
        "yearly_spend": (18000, 42000),
        "transactions": (22, 60),
        "avg_trans": (60, 180),
        "offers": (10, 30),
        "balance": (900, 4200),
        "cash_advance": (0, 220),
        "credit_limit": (8000, 26000),
        "days_since_last": (4, 20),
    },
    "Moderate_Users": {
        "weight": 0.30,
        "monthly_spend": (700, 1800),
        "yearly_spend": (9000, 22000),
        "transactions": (12, 35),
        "avg_trans": (40, 120),
        "offers": (6, 20),
        "balance": (450, 2200),
        "cash_advance": (0, 150),
        "credit_limit": (5000, 16000),
        "days_since_last": (8, 35),
    },
    "Low_Activity_Users": {
        "weight": 0.15,
        "monthly_spend": (200, 900),
        "yearly_spend": (2500, 11000),
        "transactions": (4, 18),
        "avg_trans": (20, 80),
        "offers": (2, 12),
        "balance": (150, 1000),
        "cash_advance": (0, 80),
        "credit_limit": (2000, 9000),
        "days_since_last": (20, 90),
    },
    "Dormant_Customers": {
        "weight": 0.07,
        "monthly_spend": (50, 350),
        "yearly_spend": (600, 4500),
        "transactions": (1, 8),
        "avg_trans": (10, 65),
        "offers": (0, 4),
        "balance": (20, 600),
        "cash_advance": (0, 40),
        "credit_limit": (1000, 6000),
        "days_since_last": (45, 180),
    },
    "Cash_Advance_Users": {
        "weight": 0.03,
        "monthly_spend": (300, 1400),
        "yearly_spend": (3500, 17000),
        "transactions": (8, 25),
        "avg_trans": (30, 95),
        "offers": (1, 10),
        "balance": (1000, 5000),
        "cash_advance": (250, 2200),
        "credit_limit": (3500, 18000),
        "days_since_last": (6, 25),
    },
}

segment_names = list(segments.keys())
segment_probs = [segments[s]["weight"] for s in segment_names]

income_bins = ["Low", "Medium", "High", "Very High"]
occupation = [
    "Salaried Employee",
    "Business Owner",
    "Professional",
    "Self-Employed",
    "Student",
    "Retired",
    "Government Employee",
    "Freelancer",
]
regions = [
    "North",
    "South",
    "East",
    "West",
    "Central",
    "Metro",
]
card_types = ["Silver", "Gold", "Platinum", "Titanium"]
payment_methods = ["Auto Debit", "UPI", "Bank Transfer", "Credit Card Payment", "Wallet"]

def random_between(low, high):
    return float(np.random.uniform(low, high))

rows = []
for i in range(1, N_CUSTOMERS + 1):
    segment = np.random.choice(segment_names, p=segment_probs)
    seg_params = segments[segment]

    age = int(np.random.randint(21, 72))
    gender = np.random.choice(["Male", "Female", "Other"])
    income_category = np.random.choice(income_bins)
    occupation_name = np.random.choice(occupation)
    region = np.random.choice(regions)
    card_type = np.random.choice(card_types)
    card_age_months = int(np.random.randint(6, 120))

    monthly_income = random_between(2000, 30000)
    credit_limit = random_between(*seg_params["credit_limit"])
    account_balance = random_between(*seg_params["balance"])
    monthly_spend = random_between(*seg_params["monthly_spend"])
    yearly_spend = random_between(*seg_params["yearly_spend"])
    total_transactions = int(np.random.randint(*seg_params["transactions"]))
    avg_trans = random_between(*seg_params["avg_trans"])
    offers_redeemed = int(np.random.randint(*seg_params["offers"]))
    offer_utilization_rate = round(np.random.uniform(0.08, 0.75), 3)
    payment_method = np.random.choice(payment_methods)
    monthly_cash_advance = random_between(*seg_params["cash_advance"])
    yearly_cash_advance = monthly_cash_advance * 12 * np.random.uniform(0.85, 1.15)
    revolving_balance = max(0.0, account_balance * np.random.uniform(0.1, 0.95))
    days_since_last_transaction = int(np.random.randint(*seg_params["days_since_last"]))
    customer_segment = segment
    usage_level = (
        "Very High"
        if segment in ["Heavy_Spenders", "Frequent_Users"]
        else "High"
        if segment == "Moderate_Users"
        else "Low"
        if segment in ["Low_Activity_Users", "Dormant_Customers"]
        else "Medium"
    )

    row = {
        "Customer_ID": f"CUST_{i:05d}",
        "Age": age,
        "Gender": gender,
        "Income_Category": income_category,
        "Occupation": occupation_name,
        "Region": region,
        "Card_Type": card_type,
        "Card_Age_Months": card_age_months,
        "Credit_Limit": round(credit_limit, 2),
        "Account_Balance": round(account_balance, 2),
        "Monthly_Income": round(monthly_income, 2),
        "Monthly_Spend": round(monthly_spend, 2),
        "Yearly_Spend": round(yearly_spend, 2),
        "Total_Purchases": round(monthly_spend * 12, 2),
        "Total_Transactions": total_transactions,
        "Avg_Transactions_Per_Month": round(avg_trans, 2),
        "Offers_Redeemed": offers_redeemed,
        "Offer_Utilization_Rate": offer_utilization_rate,
        "Payment_Method": payment_method,
        "Monthly_Cash_Advance": round(monthly_cash_advance, 2),
        "Yearly_Cash_Advance": round(yearly_cash_advance, 2),
        "Revolving_Balance": round(revolving_balance, 2),
        "Days_Since_Last_Transaction": days_since_last_transaction,
        "Customer_Segment": customer_segment,
        "Usage_Level": usage_level,
    }

    rows.append(row)


df = pd.DataFrame(rows)

# Ensure a clean output directory
from pathlib import Path
Path("outputs").mkdir(exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)
print(f"Created {len(df)} customer records at {OUTPUT_PATH}")
print(df.head().to_string(index=False))
