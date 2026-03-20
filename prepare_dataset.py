"""
Prepare instruction dataset for HSI stock prediction fine-tuning
"""
import pandas as pd
import json
import os
from datetime import datetime

def prepare_training_data():
    # Load collected data
    data_path = "data/raw/hsi_prices.csv"
    if not os.path.exists(data_path):
        print("Please run data collector first: python src/data/collector.py")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    
    # Create instruction dataset
    dataset = []
    
    # Group by stock
    for stock in df['Company'].unique():
        stock_data = df[df['Company'] == stock]
        
        # Get recent data points
        for i in range(min(10, len(stock_data) - 1)):
            current = stock_data.iloc[-i-1] if i < len(stock_data) else stock_data.iloc[-1]
            prev = stock_data.iloc[-i-2] if i+1 < len(stock_data) else stock_data.iloc[-2]
            
            # Calculate price change
            price_change = ((current['Close'] - prev['Close']) / prev['Close']) * 100
            
            # Create instruction
            instruction = {
                "instruction": f"Analyze {stock} stock price movement",
                "input": f"Date: {current['Date']}, "
                        f"Price: ${current['Close']:.2f}, "
                        f"Volume: {current['Volume']:.0f}, "
                        f"24h change: {price_change:.2f}%",
                "output": f"Based on technical analysis, {stock} shows "
                         f"{'bullish' if price_change > 0 else 'bearish'} momentum "
                         f"with a {abs(price_change):.1f}% change. "
                         f"Current price is ${current['Close']:.2f} with volume "
                         f"{current['Volume']:.0f}."
            }
            dataset.append(instruction)
    
    # Save dataset
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/training_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✅ Created dataset with {len(dataset)} samples")
    print(f"Saved to: data/processed/training_dataset.json")
    
    # Show sample
    if dataset:
        print("\nSample instruction:")
        print(json.dumps(dataset[0], indent=2))

if __name__ == "__main__":
    prepare_training_data()