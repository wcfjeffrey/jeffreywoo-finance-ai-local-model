"""
HSI Stock Data Collector
Fetches historical data for Hang Seng Index constituents
"""
import yfinance as yf
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HSIDataCollector:
    def __init__(self):
        self.stocks = {
            '0700.HK': 'Tencent',
            '9988.HK': 'Alibaba',
            '0005.HK': 'HSBC',
            '0011.HK': 'Hang Seng Bank',
            '0388.HK': 'HKEX',
        }
    
    def collect_prices(self, period='1y'):
        all_data = {}
        for symbol, name in self.stocks.items():
            logger.info(f"Collecting {name} ({symbol})")
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period)
                hist['Company'] = name
                all_data[symbol] = hist
                logger.info(f"  OK {len(hist)} records")
            except Exception as e:
                logger.error(f"  Failed: {e}")
        return all_data
    
    def save_data(self, data, output_path='data/raw/hsi_prices.csv'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if data:
            combined = pd.concat(data.values(), keys=data.keys())
            combined.to_csv(output_path)
            logger.info(f"Saved to {output_path}")
            return combined
        return None

if __name__ == '__main__':
    print("Starting HSI Data Collection...")
    collector = HSIDataCollector()
    data = collector.collect_prices()
    if data:
        collector.save_data(data)
        print(f"Complete! Collected {len(data)} stocks")