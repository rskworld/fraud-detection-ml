# Transaction Data Description

<!--
Project: Fraud Detection System using ML
Developer: Molla Samser (Founder) - https://rskworld.in
Designer & Tester: Rima Khatun
Email: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Copyright © 2025 RSK World. All rights reserved.
-->

## Dataset Overview

This directory contains sample transaction data for training and testing the fraud detection system.

## File: transactions.csv

### Description
A synthetic transaction dataset containing 5,000 transactions with realistic fraud patterns.

### Columns

1. **transaction_id** (int): Unique identifier for each transaction
2. **amount** (float): Transaction amount in currency units
3. **user_id** (int): Unique identifier for the user (1-500)
4. **timestamp** (datetime): Date and time of the transaction (YYYY-MM-DD HH:MM:SS)
5. **merchant_category** (string): Category of merchant
   - retail
   - online
   - gas
   - restaurant
   - grocery
   - entertainment
   - hotel
   - travel
6. **merchant_id** (int): Unique identifier for the merchant (1-200)
7. **card_type** (string): Type of card used
   - credit
   - debit
   - prepaid
8. **is_fraud** (int): Fraud label (0 = legitimate, 1 = fraud)

### Statistics

- **Total Transactions**: 5,000
- **Fraud Rate**: ~5% (241 fraud transactions)
- **Time Period**: January 2024 onwards
- **Users**: 500 unique users
- **Merchants**: 200 unique merchants

### Fraud Patterns

The dataset includes realistic fraud patterns:
- Higher transaction amounts for fraudulent transactions
- Online transactions more likely to be fraudulent
- Some fraud occurs at unusual hours (late night/early morning)
- Fraud transactions often have amounts between $500-$5000

### Usage

```python
import pandas as pd

# Load the data
df = pd.read_csv('data/raw/transactions.csv')

# Basic statistics
print(f"Total transactions: {len(df)}")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
```

### Generating More Data

To generate additional sample data:

```bash
python create_sample_data.py
```

Or modify the script to generate different amounts:
- Change `range(5000)` to generate more/fewer transactions
- Adjust fraud rate by changing `random.random() < 0.05`
- Modify amount ranges for different patterns

## Notes

- This is synthetic data for demonstration purposes
- Real-world data should be used for production models
- Data includes realistic patterns but is not based on actual transactions
- All personal information is synthetic

## Contact

**Founder**: Molla Samser  
**Designer & Tester**: Rima Khatun  
**Website**: https://rskworld.in  
**Email**: help@rskworld.in, support@rskworld.in

© 2025 RSK World. All rights reserved.

