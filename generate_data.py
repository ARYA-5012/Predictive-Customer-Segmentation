"""
Data Generator for Customer Segmentation Project
Creates realistic e-commerce transaction data similar to UCI Online Retail dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# Configuration
NUM_CUSTOMERS = 4500
NUM_TRANSACTIONS = 50000
START_DATE = datetime(2010, 1, 1)
END_DATE = datetime(2011, 12, 31)

# Generate customer base with different behavioral profiles
customer_profiles = {
    'champion': {'count': int(NUM_CUSTOMERS * 0.10), 'freq_range': (20, 50), 'monetary_mult': 3.0},
    'loyal': {'count': int(NUM_CUSTOMERS * 0.15), 'freq_range': (10, 25), 'monetary_mult': 2.0},
    'potential': {'count': int(NUM_CUSTOMERS * 0.20), 'freq_range': (5, 15), 'monetary_mult': 1.5},
    'at_risk': {'count': int(NUM_CUSTOMERS * 0.15), 'freq_range': (8, 20), 'monetary_mult': 1.8},
    'hibernating': {'count': int(NUM_CUSTOMERS * 0.20), 'freq_range': (1, 5), 'monetary_mult': 0.8},
    'new': {'count': int(NUM_CUSTOMERS * 0.20), 'freq_range': (1, 3), 'monetary_mult': 1.0},
}

# Products catalog
products = [
    ('85123A', 'WHITE HANGING HEART T-LIGHT HOLDER', 2.55),
    ('71053', 'WHITE METAL LANTERN', 3.39),
    ('84406B', 'CREAM CUPID HEARTS COAT HANGER', 2.75),
    ('84029G', 'KNITTED UNION FLAG HOT WATER BOTTLE', 3.39),
    ('84029E', 'RED WOOLLY HOTTIE WHITE HEART', 3.39),
    ('22752', 'SET 7 BABUSHKA NESTING BOXES', 7.65),
    ('21730', 'GLASS STAR FROSTED T-LIGHT HOLDER', 4.25),
    ('22633', 'HAND WARMER UNION JACK', 1.85),
    ('22632', 'HAND WARMER RED POLKA DOT', 1.85),
    ('84879', 'ASSORTED COLOUR BIRD ORNAMENT', 1.69),
    ('22745', 'POPPY\'S PLAYHOUSE BEDROOM', 2.10),
    ('22748', 'POPPY\'S PLAYHOUSE KITCHEN', 2.10),
    ('22749', 'FELTCRAFT PRINCESS CHARLOTTE DOLL', 3.75),
    ('22310', 'IVORY KNITTED MUG COSY', 1.65),
    ('84625A', 'BLUE NEW BAROQUE CANDLESTICK', 5.45),
    ('23084', 'RABBIT NIGHT LIGHT', 1.95),
    ('23298', 'SPOTLIGHT ANTIQUE SILVER', 6.95),
    ('23300', 'SPOTTY BUNTING', 4.95),
    ('22386', 'JUMBO BAG PINK POLKADOT', 1.95),
    ('21672', 'WHITE SPOT RED CERAMIC DRAWER KNOB', 1.25),
]

countries = ['United Kingdom'] * 80 + ['Germany', 'France', 'EIRE', 'Spain', 'Netherlands', 
              'Belgium', 'Switzerland', 'Portugal', 'Australia', 'Norway']

def generate_transactions():
    transactions = []
    customer_id = 12346
    invoice_no = 536365
    
    for profile_name, profile in customer_profiles.items():
        for _ in range(profile['count']):
            customer_id += 1
            num_orders = random.randint(*profile['freq_range'])
            
            # Determine customer's last activity based on profile
            if profile_name == 'champion':
                last_active_days = random.randint(1, 30)
            elif profile_name == 'loyal':
                last_active_days = random.randint(10, 60)
            elif profile_name == 'at_risk':
                last_active_days = random.randint(60, 150)
            elif profile_name == 'hibernating':
                last_active_days = random.randint(150, 365)
            elif profile_name == 'new':
                last_active_days = random.randint(1, 45)
            else:
                last_active_days = random.randint(30, 120)
            
            customer_end = END_DATE - timedelta(days=last_active_days)
            customer_start = max(START_DATE, customer_end - timedelta(days=random.randint(60, 500)))
            
            order_dates = sorted([customer_start + timedelta(
                days=random.randint(0, (customer_end - customer_start).days)
            ) for _ in range(num_orders)])
            
            country = random.choice(countries)
            
            for order_date in order_dates:
                invoice_no += 1
                items_count = random.randint(1, 8)
                
                for _ in range(items_count):
                    product = random.choice(products)
                    quantity = random.randint(1, 24)
                    unit_price = product[2] * profile['monetary_mult'] * random.uniform(0.8, 1.2)
                    
                    transactions.append({
                        'Invoice': str(invoice_no),
                        'StockCode': product[0],
                        'Description': product[1],
                        'Quantity': quantity,
                        'InvoiceDate': order_date,
                        'Price': round(unit_price, 2),
                        'Customer ID': customer_id,
                        'Country': country
                    })
    
    # Add some cancelled orders (about 2%)
    df = pd.DataFrame(transactions)
    cancelled_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
    
    for idx in cancelled_indices:
        cancel_invoice = 'C' + df.loc[idx, 'Invoice']
        cancel_row = df.loc[idx].copy()
        cancel_row['Invoice'] = cancel_invoice
        cancel_row['Quantity'] = -cancel_row['Quantity']
        df = pd.concat([df, pd.DataFrame([cancel_row])], ignore_index=True)
    
    # Add some missing customer IDs (about 5%)
    missing_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
    df.loc[missing_indices, 'Customer ID'] = np.nan
    
    return df.sample(frac=1).reset_index(drop=True)

if __name__ == "__main__":
    print("Generating realistic e-commerce transaction data...")
    df = generate_transactions()
    df.to_csv('data/online_retail_data.csv', index=False)
    print(f"Generated {len(df)} transactions for {df['Customer ID'].nunique()} customers")
    print(f"Saved to data/online_retail_data.csv")
    print(f"\nData Summary:")
    print(df.describe())
