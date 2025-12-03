"""
Create Sample Transaction Data for Fraud Detection
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
"""

import csv
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)

# Merchant categories
merchant_categories = ['retail', 'online', 'gas', 'restaurant', 'grocery', 'entertainment', 'hotel', 'travel']
card_types = ['credit', 'debit', 'prepaid']

# Generate transaction data
transactions = []
start_date = datetime(2024, 1, 1, 8, 0, 0)
transaction_id = 1

# Generate 5000 transactions
for i in range(5000):
    # Base transaction time
    hours_offset = i * 2  # Transactions every 2 hours
    transaction_time = start_date + timedelta(hours=hours_offset)
    
    # User ID (1-500 users)
    user_id = random.randint(1, 500)
    
    # Merchant ID (1-200 merchants)
    merchant_id = random.randint(1, 200)
    
    # Merchant category
    merchant_category = random.choice(merchant_categories)
    
    # Card type
    card_type = random.choice(card_types)
    
    # Determine if fraud (5% fraud rate)
    is_fraud = 1 if random.random() < 0.05 else 0
    
    # Amount based on fraud status and category
    if is_fraud == 1:
        # Fraud transactions tend to be higher amounts, especially online
        if merchant_category == 'online':
            amount = round(random.uniform(1000, 5000), 2)
        else:
            amount = round(random.uniform(500, 2000), 2)
        # Some fraud at unusual times
        if random.random() < 0.3:
            transaction_time = transaction_time.replace(hour=random.choice([22, 23, 0, 1, 2, 3]))
    else:
        # Normal transactions
        if merchant_category == 'online':
            amount = round(random.uniform(50, 500), 2)
        elif merchant_category == 'restaurant':
            amount = round(random.uniform(30, 200), 2)
        elif merchant_category == 'gas':
            amount = round(random.uniform(20, 100), 2)
        elif merchant_category == 'grocery':
            amount = round(random.uniform(40, 300), 2)
        elif merchant_category == 'retail':
            amount = round(random.uniform(25, 400), 2)
        elif merchant_category == 'entertainment':
            amount = round(random.uniform(50, 250), 2)
        elif merchant_category == 'hotel':
            amount = round(random.uniform(100, 800), 2)
        else:  # travel
            amount = round(random.uniform(200, 1000), 2)
    
    transactions.append({
        'transaction_id': transaction_id,
        'amount': amount,
        'user_id': user_id,
        'timestamp': transaction_time.strftime('%Y-%m-%d %H:%M:%S'),
        'merchant_category': merchant_category,
        'merchant_id': merchant_id,
        'card_type': card_type,
        'is_fraud': is_fraud
    })
    
    transaction_id += 1

# Write to CSV
with open('data/raw/transactions.csv', 'w', newline='') as csvfile:
    fieldnames = ['transaction_id', 'amount', 'user_id', 'timestamp', 'merchant_category', 'merchant_id', 'card_type', 'is_fraud']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for transaction in transactions:
        writer.writerow(transaction)

print(f"Generated {len(transactions)} transactions")
print(f"Fraud transactions: {sum(t['is_fraud'] for t in transactions)} ({sum(t['is_fraud'] for t in transactions)/len(transactions)*100:.2f}%)")
print(f"Data saved to: data/raw/transactions.csv")

