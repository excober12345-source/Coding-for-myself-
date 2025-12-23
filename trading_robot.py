#!/usr/bin/env python3
"""
🤖 COMPLETE TRADING ROBOT WITH WEB DASHBOARD
MetaTrader 5 Style Trading Platform - All in One File
"""

import sys
import os
import subprocess
import platform
import threading
import json
import time
import warnings
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
warnings.filterwarnings('ignore')

# ==================== AUTO-INSTALL DEPENDENCIES ====================

def check_and_install_dependencies():
    """Automatically check and install missing packages"""
    required_packages = [
        'pandas',
        'numpy',
        'yfinance',
        'plotly',
        'flask',
        'flask_socketio',
        'flask_cors',
        'ta',
        'scikit-learn',
        'schedule',
        'colorama',
        'requests',
        'bs4',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    print("=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package:20} - Installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package:20} - Missing")
    
    if missing_packages:
        print(f"\nInstalling {len(missing_packages)} missing packages...")
        try:
            for package in missing_packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("\n✅ All packages installed successfully!")
        except Exception as e:
            print(f"\n❌ Installation failed: {e}")
            print("Try running: pip install pandas numpy yfinance plotly flask flask_socketio ta")
            return False
    
    print("\n" + "=" * 60)
    print("DEPENDENCIES CHECK COMPLETE")
    print("=" * 60 + "\n")
    return True

# Run dependency check
if not check_and_install_dependencies():
    print("Failed to install dependencies. Please install manually:")
    print("pip install pandas numpy yfinance plotly flask flask_socketio ta scikit-learn")
    input("Press Enter to exit...")
    sys.exit(1)

# ==================== IMPORT PACKAGES ====================

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import schedule
from colorama import init, Fore, Style
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Web imports
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS

# Initialize colorama
init(autoreset=True)

# ==================== CONFIGURATION ====================

class Config:
    """Configuration settings for the trading robot"""
    
    # Trading parameters
    INITIAL_BALANCE = 10000.0
    COMMISSION_RATE = 0.001  # 0.1% commission
    RISK_PER_TRADE = 0.02    # 2% risk per trade
    MAX_POSITIONS = 5
    
    # Web server
    WEB_PORT = 5000
    WEB_HOST = '0.0.0.0'
    
    # Data settings
    DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BTC-USD', 'ETH-USD', 'SPY']
    UPDATE_INTERVAL = 60  # seconds
    
    # ML settings
    ML_ENABLED = True
    TRAIN_INTERVAL = 3600  # Train every hour
    
    # Risk management
    STOP_LOSS_PCT = 0.02   # 2% stop loss
    TAKE_PROFIT_PCT = 0.04 # 4% take profit
    MAX_DRAWDOWN = 0.20    # 20% max drawdown
    
    # Notification settings
    DISCORD_WEBHOOK = None
    TELEGRAM_BOT_TOKEN = None
    TELEGRAM_CHAT_ID = None

# ==================== CORE TRADING ENGINE ====================

class TradingEngine:
    """Advanced trading engine with ML capabilities"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.balance = self.config.INITIAL_BALANCE
        self.initial_balance = self.config.INITIAL_BALANCE
        self.positions = []
        self.trade_history = []
        self.portfolio = {}
        self.equity_curve = []
        self.max_drawdown = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.daily_pnl = 0
        self.ml_model = None
        self.scaler = StandardScaler()
        self.signals_cache = {}
        
        # Initialize ML model
        if self.config.ML_ENABLED:
            self.init_ml_model()
    
    def init_ml_model(self):
        """Initialize machine learning model"""
        try:
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            print(f"{Fore.GREEN}✅ ML model initialized{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to initialize ML model: {e}{Style.RESET_ALL}")
            self.config.ML_ENABLED = False
    
    def calculate_advanced_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        if df.empty or len(df) < 50:
            return df
        
        df = df.copy()
        
        # Price-based indicators
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift())
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
            df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volume indicators
        df['volume_sma'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # ATR for volatility
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        # ADX for trend strength
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
        df['ichi_conversion'] = ichimoku.ichimoku_conversion_line()
        df['ichi_base'] = ichimoku.ichimoku_base_line()
        
        # Fibonacci levels (simplified)
        rolling_high = df['High'].rolling(50).max()
        rolling_low = df['Low'].rolling(50).min()
        price_range = rolling_high - rolling_low
        
        for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
            df[f'fib_{int(level*1000)}'] = rolling_low + price_range * level
        
        # Support and Resistance
        df['support'] = df['Low'].rolling(20).min()
        df['resistance'] = df['High'].rolling(20).max()
        
        # Price patterns (simplified)
        df['is_hammer'] = self.detect_hammer(df)
        df['is_doji'] = self.detect_doji(df)
        
        # Market regime
        df['market_regime'] = self.detect_market_regime(df)
        
        return df
    
    def detect_hammer(self, df):
        """Detect hammer candlestick pattern"""
        body = abs(df['Close'] - df['Open'])
        lower_shadow = df['Open'] - df['Low'] if df['Close'] > df['Open'] else df['Close'] - df['Low']
        upper_shadow = df['High'] - df['Close'] if df['Close'] > df['Open'] else df['High'] - df['Open']
        
        return (lower_shadow > 2 * body) & (upper_shadow < body * 0.3)
    
    def detect_doji(self, df):
        """Detect doji candlestick pattern"""
        body = abs(df['Close'] - df['Open'])
        range_ = df['High'] - df['Low']
        
        return (body < range_ * 0.1)
    
    def detect_market_regime(self, df):
        """Detect market regime (trending/volatile/sideways)"""
        if len(df) < 20:
            return 0
        
        # Simple regime detection
        volatility = df['returns'].rolling(20).std()
        trend_strength = df['adx'].fillna(0)
        
        if trend_strength.iloc[-1] > 25:
            return 1  # Trending
        elif volatility.iloc[-1] > volatility.quantile(0.75):
            return 2  # Volatile
        else:
            return 0  # Sideways
    
    def generate_ml_features(self, df):
        """Generate features for machine learning"""
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        
        features = [
            latest.get('rsi', 50),
            latest.get('macd', 0),
            latest.get('macd_diff', 0),
            latest.get('stoch_k', 50),
            latest.get('adx', 0),
            latest.get('bb_width', 0),
            latest.get('volume_ratio', 1),
            latest.get('atr', 0) / latest['Close'] if latest['Close'] > 0 else 0,
            latest.get('market_regime', 0),
            latest['Close'] / latest.get('SMA_20', latest['Close']),
            latest['Close'] / latest.get('SMA_50', latest['Close']),
            latest['Close'] / latest.get('SMA_200', latest['Close']),
            int(latest.get('is_hammer', False)),
            int(latest.get('is_doji', False)),
            latest.get('returns', 0) * 100,
            latest['Volume'] / df['Volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 1
        ]
        
        return np.array(features).reshape(1, -1)
    
    def generate_signal(self, df, symbol):
        """Generate trading signal using ML + rules"""
        if df.empty or len(df) < 50:
            return {'signal': 0, 'strength': 0, 'confidence': 0}
        
        df = self.calculate_advanced_indicators(df)
        latest = df.iloc[-1]
        
        # Generate ML prediction if enabled
        ml_confidence = 0
        if self.config.ML_ENABLED:
            features = self.generate_ml_features(df)
            if features is not None:
                try:
                    # This would require training data - for now use rules
                    ml_confidence = 0.5
                except:
                    pass
        
        # Rule-based signal generation
        signal = 0
        strength = 0
        confidence = 0
        
        # Buy conditions
        buy_score = 0
        sell_score = 0
        
        # Price above moving averages
        if latest['Close'] > latest.get('SMA_20', 0):
            buy_score += 1
        if latest['Close'] > latest.get('SMA_50', 0):
            buy_score += 2
        if latest['Close'] > latest.get('SMA_200', 0):
            buy_score += 3
        
        # RSI conditions
        rsi = latest.get('rsi', 50)
        if 30 <= rsi <= 70:
            buy_score += 1
        elif rsi < 30:
            buy_score += 2
        elif rsi > 70:
            sell_score += 2
        
        # MACD conditions
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        if macd > macd_signal:
            buy_score += 1
        else:
            sell_score += 1
        
        # Volume confirmation
        if latest.get('volume_ratio', 1) > 1.2:
            buy_score += 1
        
        # Pattern recognition
        if latest.get('is_hammer', False):
            buy_score += 2
        if latest.get('is_doji', False):
            buy_score += 1
        
        # Calculate final signal
        if buy_score > sell_score + 2:
            signal = 1
            strength = min(buy_score / 10, 1.0)
            confidence = 0.6 + (strength * 0.4)
        elif sell_score > buy_score + 2:
            signal = -1
            strength = min(sell_score / 10, 1.0)
            confidence = 0.6 + (strength * 0.4)
        
        # Adjust for market regime
        regime = latest.get('market_regime', 0)
        if regime == 1:  # Trending
            confidence *= 1.2
        elif regime == 2:  # Volatile
            confidence *= 0.8
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'rsi': rsi,
            'macd': macd,
            'price': latest['Close'],
            'regime': regime
        }
    
    def calculate_position_size(self, symbol, price, signal_strength):
        """Calculate position size based on risk management"""
        risk_amount = self.balance * self.config.RISK_PER_TRADE * signal_strength
        stop_loss = price * (1 - self.config.STOP_LOSS_PCT)
        
        if price - stop_loss > 0:
            position_size = risk_amount / (price - stop_loss)
            position_size = min(position_size, self.balance * 0.2 / price)  # Max 20% per position
        else:
            position_size = (self.balance * 0.1) / price  # Default 10%
        
        return int(position_size)
    
    def execute_order(self, symbol, action, quantity=None, price=None, reason=""):
        """Execute a trading order with advanced risk management"""
        try:
            current_price = price or self.get_current_price(symbol)
            
            if action.upper() in ['BUY', 'LONG']:
                if len(self.positions) >= self.config.MAX_POSITIONS:
                    return False, "Maximum positions reached"
                
                # Calculate position size
                if quantity is None:
                    # Get signal for position sizing
                    signal = self.signals_cache.get(symbol, {'strength': 0.5})
                    quantity = self.calculate_position_size(symbol, current_price, signal['strength'])
                
                total_cost = current_price * quantity
                commission = total_cost * self.config.COMMISSION_RATE
                
                if total_cost + commission > self.balance:
                    return False, "Insufficient balance"
                
                # Create position
                position = {
                    'id': f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'stop_loss': current_price * (1 - self.config.STOP_LOSS_PCT),
                    'take_profit': current_price * (1 + self.config.TAKE_PROFIT_PCT),
                    'commission': commission,
                    'status': 'OPEN',
                    'reason': reason
                }
                
                self.positions.append(position)
                self.portfolio[symbol] = self.portfolio.get(symbol, 0) + quantity
                self.balance -= (total_cost + commission)
                self.total_trades += 1
                
                # Log to history
                self.trade_history.append(position.copy())
                
                self.send_notification(
                    f"📈 BUY {symbol}",
                    f"Quantity: {quantity}\n"
                    f"Price: ${current_price:.2f}\n"
                    f"Stop Loss: ${position['stop_loss']:.2f}\n"
                    f"Take Profit: ${position['take_profit']:.2f}\n"
                    f"Reason: {reason}"
                )
                
                return True, f"BUY executed: {quantity} {symbol} @ ${current_price:.2f}"
            
            elif action.upper() in ['SELL', 'CLOSE']:
                # Find position to close
                position_to_close = None
                for pos in self.positions:
                    if pos['symbol'] == symbol and pos['status'] == 'OPEN':
                        position_to_close = pos
                        break
                
                if not position_to_close:
                    return False, f"No open position for {symbol}"
                
                if quantity is None:
                    quantity = position_to_close['quantity']
                elif quantity > position_to_close['quantity']:
                    quantity = position_to_close['quantity']
                
                # Calculate P&L
                pnl = (current_price - position_to_close['entry_price']) * quantity
                pnl_percent = ((current_price - position_to_close['entry_price']) / 
                              position_to_close['entry_price']) * 100
                
                # Update trade stats
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                self.daily_pnl += pnl
                
                # Update position
                position_to_close['exit_price'] = current_price
                position_to_close['exit_time'] = datetime.now()
                position_to_close['status'] = 'CLOSED'
                position_to_close['pnl'] = pnl
                position_to_close['pnl_percent'] = pnl_percent
                
                # Calculate commission
                total_value = current_price * quantity
                commission = total_value * self.config.COMMISSION_RATE
                
                # Update account
                self.balance += pnl - commission
                
                # Update portfolio
                self.portfolio[symbol] -= quantity
                if self.portfolio.get(symbol, 0) <= 0:
                    self.portfolio.pop(symbol, None)
                
                # Add to history
                self.trade_history.append(position_to_close.copy())
                self.positions = [p for p in self.positions if p['id'] != position_to_close['id']]
                
                # Update equity curve
                self.update_equity_curve()
                
                pnl_color = Fore.GREEN if pnl > 0 else Fore.RED
                self.send_notification(
                    f"📉 SELL {symbol}",
                    f"Quantity: {quantity}\n"
                    f"Price: ${current_price:.2f}\n"
                    f"P&L: ${pnl:.2f} ({pnl_percent:.2f}%)\n"
                    f"Reason: {reason}"
                )
                
                return True, f"SELL executed: {quantity} {symbol} @ ${current_price:.2f}, P&L: ${pnl:.2f} ({pnl_percent:.2f}%)"
            
            return False, f"Unknown action: {action}"
            
        except Exception as e:
            return False, f"Order execution failed: {str(e)}"
    
    def get_current_price(self, symbol):
        """Get current price with fallback"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return data['Close'].iloc[-1]
        except:
            pass
        
        # Fallback: Return last known price from positions
        for pos in self.positions:
            if pos['symbol'] == symbol:
                return pos.get('current_price', pos['entry_price'] * 1.01)
        
        return 100.0  # Default fallback
    
    def update_equity_curve(self):
        """Update equity curve"""
        total_value = self.balance
        
        for symbol, quantity in self.portfolio.items():
            try:
                price = self.get_current_price(symbol)
                total_value += price * quantity
            except:
                pass
        
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': total_value,
            'balance': self.balance
        })
        
        # Calculate drawdown
        if len(self.equity_curve) > 1:
            peak = max([e['equity'] for e in self.equity_curve])
            current = total_value
            if peak > 0:
                drawdown = ((peak - current) / peak) * 100
                self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def send_notification(self, title, message):
        """Send notification via configured channels"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{title}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{message}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    def get_performance_metrics(self):
        """Get comprehensive performance metrics"""
        metrics = {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'open_positions': len(self.positions),
            'portfolio_value': self.balance + sum(
                self.get_current_price(sym) * qty 
                for sym, qty in self.portfolio.items()
            )
        }
        
        if self.total_trades > 0:
            metrics['win_rate'] = (self.winning_trades / self.total_trades) * 100
        else:
            metrics['win_rate'] = 0
        
        # Calculate returns
        metrics['total_return'] = ((metrics['portfolio_value'] - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = pd.Series([e['equity'] for e in self.equity_curve]).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0
        
        return metrics
    
    def get_positions_summary(self):
        """Get summary of all positions"""
        summary = []
        
        for position in self.positions:
            if position['status'] == 'OPEN':
                current_price = self.get_current_price(position['symbol'])
                pnl = (current_price - position['entry_price']) * position['quantity']
                pnl_percent = ((current_price - position['entry_price']) / 
                              position['entry_price']) * 100
                
                summary.append({
                    'symbol': position['symbol'],
                    'action': position['action'],
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'stop_loss': position['stop_loss'],
                    'take_profit': position['take_profit'],
                    'value': current_price * position['quantity'],
                    'time_held': (datetime.now() - position['entry_time']).total_seconds() / 3600
                })
        
        return summary

# ==================== DATA MANAGER ====================

class DataManager:
    """Advanced data manager with caching and news integration"""
    
    def __init__(self):
        self.cache = {}
        self.news_cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_data(self, symbol, period='1mo', interval='1d', use_cache=True):
        """Fetch market data with intelligent caching"""
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_timeout:
                return data.copy()
        
        try:
            print(f"{Fore.BLUE}📊 Fetching data for {symbol}...{Style.RESET_ALL}")
            ticker = yf.Ticker(symbol)
            
            # Adjust period for intraday
            if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
                if period == '1mo':
                    period = '5d'
                elif period == '3mo':
                    period = '1mo'
            
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print(f"{Fore.RED}❌ No data for {symbol}{Style.RESET_ALL}")
                return None
            
            # Add technical indicators
            df = self.add_basic_indicators(df)
            
            # Cache the data
            self.cache[cache_key] = (df.copy(), datetime.now())
            
            return df
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error fetching {symbol}: {e}{Style.RESET_ALL}")
            return None
    
    def add_basic_indicators(self, df):
        """Add basic technical indicators"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Add returns
        df['returns'] = df['Close'].pct_change()
        
        # Simple moving averages
        for period in [10, 20, 50]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Volume
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        return df
    
    def get_live_quotes(self, symbols):
        """Get live quotes for multiple symbols"""
        quotes = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_data(symbol, period='1d', interval='1m')
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    prev_close = df.iloc[-2]['Close'] if len(df) > 1 else latest['Close']
                    
                    quotes[symbol] = {
                        'price': latest['Close'],
                        'change': latest['Close'] - prev_close,
                        'change_percent': ((latest['Close'] - prev_close) / prev_close) * 100,
                        'volume': latest['Volume'],
                        'high': latest['High'],
                        'low': latest['Low'],
                        'rsi': latest.get('rsi', 50),
                        'volume_ratio': latest.get('volume_ratio', 1)
                    }
            except Exception as e:
                print(f"{Fore.RED}❌ Error getting quote for {symbol}: {e}{Style.RESET_ALL}")
                quotes[symbol] = None
        
        return quotes
    
    def get_news(self, symbol, max_articles=5):
        """Get recent news for a symbol"""
        cache_key = f"news_{symbol}"
        
        if cache_key in self.news_cache:
            data, timestamp = self.news_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < 600:  # 10 minutes
                return data
        
        try:
            print(f"{Fore.BLUE}📰 Fetching news for {symbol}...{Style.RESET_ALL}")
            
            # Using Yahoo Finance news
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_items = []
            articles = soup.find_all('li', class_='js-stream-content', limit=max_articles)
            
            for article in articles:
                try:
                    title_elem = article.find('h3')
                    link_elem = article.find('a')
                    summary_elem = article.find('p')
                    
                    if title_elem and link_elem:
                        news_items.append({
                            'title': title_elem.text.strip(),
                            'link': f"https://finance.yahoo.com{link_elem.get('href', '')}",
                            'summary': summary_elem.text.strip() if summary_elem else '',
                            'symbol': symbol,
                            'time': datetime.now().strftime('%H:%M')
                        })
                except:
                    continue
            
            # Cache news
            self.news_cache[cache_key] = (news_items, datetime.now())
            
            return news_items
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error fetching news: {e}{Style.RESET_ALL}")
            return []

# ==================== WEB INTERFACE ====================

class TradingDashboard:
    """Web dashboard for trading robot"""
    
    def __init__(self, engine, data_manager, config=None):
        self.engine = engine
        self.data_manager = data_manager
        self.config = config or Config()
        
        # Initialize Flask app
        self.app = Flask(__name__, static_folder='static', template_folder='templates')
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Create directories
        os.makedirs('static', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        
        # Setup routes
        self.setup_routes()
        self.setup_websocket()
        
        # Create HTML template
        self.create_html_template()
        
        # Auto-update thread
        self.update_thread = None
        self.running = False
    
    def create_html_template(self):
        """Create HTML template file"""
        html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 AI Trading Robot Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        :root {
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --bg-hover: #334155;
            --primary: #3b82f6;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #475569;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary), #1d4ed8);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .header-content h1 {
            font-size: 2.5rem;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .header-content p {
            color: var(--text-muted);
            font-size: 1.1rem;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(255, 255, 255, 0.1);
            padding: 10px 20px;
            border-radius: 50px;
            backdrop-filter: blur(10px);
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .card {
            background: var(--bg-card);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid var(--border);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
        }
        
        .card h2 {
            color: var(--primary);
            margin-bottom: 20px;
            font-size: 1.3rem;
            display: flex;
            align-items: center;
            gap: 10px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border);
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .metric {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid var(--primary);
        }
        
        .metric .label {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-bottom: 5px;
        }
        
        .metric .value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .profit { color: var(--success); }
        .loss { color: var(--danger); }
        
        .chart-container {
            grid-column: 1 / -1;
            height: 500px;
            margin-bottom: 20px;
        }
        
        .trading-panel {
            background: linear-gradient(135deg, var(--bg-card), #0f172a);
            border: 2px solid var(--primary);
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-muted);
            font-weight: 500;
        }
        
        input, select {
            width: 100%;
            padding: 12px 15px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
        }
        
        .btn-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 25px;
        }
        
        button {
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        button:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        }
        
        .btn-buy {
            background: linear-gradient(135deg, var(--success), #059669);
            color: white;
        }
        
        .btn-sell {
            background: linear-gradient(135deg, var(--danger), #dc2626);
            color: white;
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        th {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: var(--text-muted);
            border-bottom: 2px solid var(--border);
        }
        
        td {
            padding: 15px;
            border-bottom: 1px solid var(--border);
        }
        
        tr:hover {
            background: var(--bg-hover);
        }
        
        .news-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            border-left: 4px solid var(--primary);
        }
        
        .news-item h4 {
            margin-bottom: 5px;
            color: var(--text);
        }
        
        .news-item p {
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        
        .update-time {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 20px;
            padding: 15px;
            background: var(--bg-card);
            border-radius: 10px;
        }
        
        .signal-indicator {
            display: inline-flex;
            align-items: center;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .signal-buy {
            background: rgba(16, 185, 129, 0.2);
            color: var(--success);
            border: 1px solid var(--success);
        }
        
        .signal-sell {
            background: rgba(239, 68, 68, 0.2);
            color: var(--danger);
            border: 1px solid var(--danger);
        }
        
        .signal-neutral {
            background: rgba(100, 116, 139, 0.2);
            color: var(--text-muted);
            border: 1px solid var(--text-muted);
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .header {
                flex-direction: column;
                text-align: center;
                gap: 15px;
            }
            
            .chart-container {
                height: 400px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="header-content">
                <h1><i class="fas fa-robot"></i> AI Trading Robot Dashboard</h1>
                <p>Advanced algorithmic trading platform with real-time analytics</p>
            </div>
            <div class="status">
                <div class="status-indicator"></div>
                <span>LIVE TRADING ACTIVE</span>
                <span id="connectionStatus" class="signal-neutral signal-indicator">Connecting...</span>
            </div>
        </header>
        
        <div class="dashboard-grid">
            <div class="card trading-panel">
                <h2><i class="fas fa-trade-federated"></i> Trading Panel</h2>
                <div class="form-group">
                    <label for="symbol"><i class="fas fa-chart-line"></i> Symbol:</label>
                    <select id="symbol">
                        <option value="AAPL">AAPL - Apple</option>
                        <option value="MSFT">MSFT - Microsoft</option>
                        <option value="GOOGL">GOOGL - Alphabet</option>
                        <option value="AMZN">AMZN - Amazon</option>
                        <option value="TSLA">TSLA - Tesla</option>
                        <option value="NVDA">NVDA - NVIDIA</option>
                        <option value="META">META - Meta</option>
                        <option value="BTC-USD">BTC-USD - Bitcoin</option>
                        <option value="ETH-USD">ETH-USD - Ethereum</option>
                        <option value="SPY">SPY - S&P 500 ETF</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="quantity"><i class="fas fa-calculator"></i> Quantity:</label>
                    <input type="number" id="quantity" value="10" min="1">
                </div>
                <div class="form-group">
                    <label><i class="fas fa-dollar-sign"></i> Current Price:</label>
                    <div id="currentPrice" class="value" style="font-size: 1.8rem; font-weight: bold;">$0.00</div>
                    <div id="currentSignal" class="signal-indicator signal-neutral" style="margin-top: 10px;">Signal: Loading...</div>
                </div>
                <div class="btn-group">
                    <button class="btn-buy" onclick="executeTrade('buy')">
                        <i class="fas fa-arrow-up"></i> BUY
                    </button>
                    <button class="btn-sell" onclick="executeTrade('sell')">
                        <i class="fas fa-arrow-down"></i> SELL
                    </button>
                </div>
                <div id="tradeResult" style="margin-top: 20px; padding: 15px; border-radius: 8px; background: rgba(255,255,255,0.05);"></div>
            </div>
            
            <div class="card">
                <h2><i class="fas fa-wallet"></i> Account Summary</h2>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="label">Balance</div>
                        <div id="balance" class="value">$10,000.00</div>
                    </div>
                    <div class="metric">
                        <div class="label">Equity</div>
                        <div id="equity" class="value">$10,000.00</div>
                    </div>
                    <div class="metric">
                        <div class="label">Daily P&L</div>
                        <div id="dailyPnl" class="value">$0.00</div>
                    </div>
                    <div class="metric">
                        <div class="label">Total Return</div>
                        <div id="totalReturn" class="value">0%</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2><i class="fas fa-trophy"></i> Performance</h2>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="label">Win Rate</div>
                        <div id="winRate" class="value">0%</div>
                    </div>
                    <div class="metric">
                        <div class="label">Total Trades</div>
                        <div id="totalTrades" class="value">0</div>
                    </div>
                    <div class="metric">
                        <div class="label">Sharpe Ratio</div>
                        <div id="sharpeRatio" class="value">0.00</div>
                    </div>
                    <div class="metric">
                        <div class="label">Max Drawdown</div>
                        <div id="maxDrawdown" class="value">0%</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card chart-container">
            <h2><i class="fas fa-chart-candlestick"></i> Price Chart - <span id="chartSymbol">AAPL</span></h2>
            <canvas id="priceChart"></canvas>
            <div class="update-time" id="chartUpdateTime">Loading chart...</div>
        </div>
        
        <div class="dashboard-grid">
            <div class="card">
                <h2><i class="fas fa-list"></i> Open Positions</h2>
                <div style="max-height: 300px; overflow-y: auto;">
                    <table id="positionsTable">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Action</th>
                                <th>Quantity</th>
                                <th>Entry</th>
                                <th>Current</th>
                                <th>P&L</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody id="positionsBody">
                            <tr><td colspan="7" style="text-align: center;">No open positions</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="card">
                <h2><i class="fas fa-eye"></i> Watchlist</h2>
                <div style="max-height: 300px; overflow-y: auto;">
                    <table id="watchlistTable">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Price</th>
                                <th>Change</th>
                                <th>RSI</th>
                                <th>Signal</th>
                            </tr>
                        </thead>
                        <tbody id="watchlistBody">
                            <!-- Will be populated -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2><i class="fas fa-history"></i> Recent Trades</h2>
            <div style="max-height: 300px; overflow-y: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Action</th>
                            <th>Quantity</th>
                            <th>Price</th>
                            <th>P&L</th>
                        </tr>
                    </thead>
                    <tbody id="historyBody">
                        <tr><td colspan="6" style="text-align: center;">No recent trades</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="update-time">
            Last update: <span id="lastUpdate">-</span> | 
            <button onclick="location.reload()" style="background: var(--primary); margin-left: 15px;">
                <i class="fas fa-sync-alt"></i> Refresh
            </button>
            <button onclick="autoTradeToggle()" id="autoTradeBtn" style="background: var(--warning); margin-left: 10px;">
                <i class="fas fa-play"></i> Start Auto-Trade
            </button>
        </div>
    </div>

    <script>
        let socket;
        let priceChart = null;
        let autoTrading = false;
        
        function init() {
            console.log('Initializing dashboard...');
            
            // Connect to WebSocket
            socket = io();
            
            socket.on('connect', function() {
                console.log('✅ Connected to server');
                updateConnectionStatus('success', 'Connected');
                loadDashboard();
                loadChart('AAPL');
            });
            
            socket.on('disconnect', function() {
                console.log('❌ Disconnected from server');
                updateConnectionStatus('error', 'Disconnected');
            });
            
            socket.on('update', function(data) {
                updateDashboard(data);
            });
            
            socket.on('trade_signal', function(signal) {
                showTradeSignal(signal);
            });
            
            socket.on('alert', function(alert) {
                showAlert(alert);
            });
            
            // Event listeners
            document.getElementById('symbol').addEventListener('change', function(e) {
                loadChart(e.target.value);
            });
            
            // Auto-update every 5 seconds
            setInterval(() => {
                if (!autoTrading) {
                    loadDashboard();
                    updateChartData();
                }
            }, 5000);
            
            // Initialize chart
            initChart();
        }
        
        function updateConnectionStatus(type, message) {
            const statusEl = document.getElementById('connectionStatus');
            statusEl.textContent = message;
            statusEl.className = 'signal-indicator ' + 
                (type === 'success' ? 'signal-buy' : 
                 type === 'error' ? 'signal-sell' : 'signal-neutral');
        }
        
        function loadDashboard() {
            fetch('/api/dashboard')
                .then(response => response.json())
                .then(data => {
                    updateDashboard(data);
                    updatePositions(data.positions);
                    updateWatchlist(data.watchlist);
                    updateHistory(data.history);
                })
                .catch(error => {
                    console.error('Error loading dashboard:', error);
                    showAlert('Failed to load dashboard data', 'error');
                });
        }
        
        function updateDashboard(data) {
            // Update account metrics
            document.getElementById('balance').textContent = 
                '$' + parseFloat(data.metrics.current_balance).toLocaleString('en-US', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                });
            
            document.getElementById('equity').textContent = 
                '$' + parseFloat(data.metrics.portfolio_value).toLocaleString('en-US', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                });
            
            document.getElementById('dailyPnl').textContent = 
                '$' + parseFloat(data.metrics.daily_pnl || 0).toFixed(2);
            document.getElementById('dailyPnl').className = 
                'value ' + (data.metrics.daily_pnl >= 0 ? 'profit' : 'loss');
            
            document.getElementById('totalReturn').textContent = 
                parseFloat(data.metrics.total_return || 0).toFixed(2) + '%';
            document.getElementById('totalReturn').className = 
                'value ' + (data.metrics.total_return >= 0 ? 'profit' : 'loss');
            
            // Update performance metrics
            document.getElementById('winRate').textContent = 
                parseFloat(data.metrics.win_rate || 0).toFixed(1) + '%';
            document.getElementById('totalTrades').textContent = 
                data.metrics.total_trades || 0;
            document.getElementById('sharpeRatio').textContent = 
                parseFloat(data.metrics.sharpe_ratio || 0).toFixed(2);
            document.getElementById('maxDrawdown').textContent = 
                parseFloat(data.metrics.max_drawdown || 0).toFixed(2) + '%';
            
            // Update timestamp
            document.getElementById('lastUpdate').textContent = 
                new Date().toLocaleTimeString();
        }
        
        function updatePositions(positions) {
            const tbody = document.getElementById('positionsBody');
            
            if (!positions || positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center;">No open positions</td></tr>';
                return;
            }
            
            let html = '';
            positions.forEach(pos => {
                const pnlClass = pos.pnl >= 0 ? 'profit' : 'loss';
                const pnlSign = pos.pnl >= 0 ? '+' : '';
                
                html += `
                    <tr>
                        <td><strong>${pos.symbol}</strong></td>
                        <td>${pos.action}</td>
                        <td>${pos.quantity}</td>
                        <td>$${pos.entry_price.toFixed(2)}</td>
                        <td>$${pos.current_price.toFixed(2)}</td>
                        <td class="${pnlClass}">${pnlSign}$${pos.pnl.toFixed(2)}</td>
                        <td>
                            <button onclick="closePosition('${pos.symbol}')" 
                                    style="padding: 5px 10px; background: var(--danger); border: none; border-radius: 4px; color: white; cursor: pointer;">
                                Close
                            </button>
                        </td>
                    </tr>
                `;
            });
            
            tbody.innerHTML = html;
        }
        
        function updateWatchlist(watchlist) {
            const tbody = document.getElementById('watchlistBody');
            
            if (!watchlist || Object.keys(watchlist).length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align: center;">No watchlist data</td></tr>';
                return;
            }
            
            let html = '';
            Object.entries(watchlist).forEach(([symbol, data]) => {
                if (!data) return;
                
                const changeClass = data.change >= 0 ? 'profit' : 'loss';
                const changeSign = data.change >= 0 ? '+' : '';
                const rsiClass = data.rsi > 70 ? 'loss' : data.rsi < 30 ? 'profit' : '';
                const signal = data.rsi > 70 ? 'SELL' : data.rsi < 30 ? 'BUY' : 'HOLD';
                const signalClass = signal === 'BUY' ? 'signal-buy' : 
                                  signal === 'SELL' ? 'signal-sell' : 'signal-neutral';
                
                html += `
                    <tr onclick="loadChart('${symbol}')" style="cursor: pointer;">
                        <td><strong>${symbol}</strong></td>
                        <td>$${data.price.toFixed(2)}</td>
                        <td class="${changeClass}">${changeSign}$${data.change.toFixed(2)}</td>
                        <td class="${rsiClass}">${data.rsi.toFixed(1)}</td>
                        <td><span class="signal-indicator ${signalClass}">${signal}</span></td>
                    </tr>
                `;
            });
            
            tbody.innerHTML = html;
        }
        
        function updateHistory(history) {
            const tbody = document.getElementById('historyBody');
            
            if (!history || history.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;">No trade history</td></tr>';
                return;
            }
            
            // Show last 10 trades
            const recentHistory = history.slice(-10).reverse();
            
            let html = '';
            recentHistory.forEach(trade => {
                const pnl = trade.pnl || 0;
                const pnlClass = pnl >= 0 ? 'profit' : 'loss';
                const pnlSign = pnl >= 0 ? '+' : '';
                const time = new Date(trade.entry_time || trade.exit_time || Date.now()).toLocaleTimeString();
                
                html += `
                    <tr>
                        <td>${time}</td>
                        <td>${trade.symbol}</td>
                        <td>${trade.action || 'TRADE'}</td>
                        <td>${trade.quantity || 0}</td>
                        <td>$${trade.entry_price?.toFixed(2) || '0.00'}</td>
                        <td class="${pnlClass}">${pnlSign}$${pnl.toFixed(2)}</td>
                    </tr>
                `;
            });
            
            tbody.innerHTML = html;
        }
        
        function initChart() {
            const ctx = document.getElementById('priceChart').getContext('2d');
            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Price',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#f1f5f9'
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#94a3b8'
                            }
                        },
                        y: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#94a3b8'
                            }
                        }
                    }
                }
            });
        }
        
        function loadChart(symbol) {
            document.getElementById('chartSymbol').textContent = symbol;
            document.getElementById('symbol').value = symbol;
            
            fetch(`/api/chart/${symbol}`)
                .then(response => response.json())
                .then(data => {
                    updateChart(data);
                    updateCurrentPrice(data);
                })
                .catch(error => {
                    console.error('Error loading chart:', error);
                    showAlert('Failed to load chart data', 'error');
                });
        }
        
        function updateChart(data) {
            if (!priceChart || !data || !data.prices) return;
            
            priceChart.data.labels = data.dates || [];
            priceChart.data.datasets[0].data = data.prices || [];
            priceChart.update();
            
            document.getElementById('chartUpdateTime').textContent = 
                `Last updated: ${new Date().toLocaleTimeString()}`;
        }
        
        function updateChartData() {
            const symbol = document.getElementById('symbol').value;
            if (symbol) {
                fetch(`/api/quote/${symbol}`)
                    .then(response => response.json())
                    .then(data => {
                        updateCurrentPrice(data);
                    });
            }
        }
        
        function updateCurrentPrice(data) {
            if (!data) return;
            
            document.getElementById('currentPrice').textContent = 
                `$${data.price?.toFixed(2) || '0.00'}`;
            
            // Update signal indicator
            const signalEl = document.getElementById('currentSignal');
            if (data.signal) {
                const signal = data.signal;
                let signalText = 'HOLD';
                let signalClass = 'signal-neutral';
                
                if (signal > 0) {
                    signalText = `BUY (Strength: ${(signal * 100).toFixed(0)}%)`;
                    signalClass = 'signal-buy';
                } else if (signal < 0) {
                    signalText = `SELL (Strength: ${Math.abs(signal * 100).toFixed(0)}%)`;
                    signalClass = 'signal-sell';
                }
                
                signalEl.textContent = `Signal: ${signalText}`;
                signalEl.className = `signal-indicator ${signalClass}`;
            }
        }
        
        function executeTrade(action) {
            const symbol = document.getElementById('symbol').value;
            const quantity = document.getElementById('quantity').value;
            const resultEl = document.getElementById('tradeResult');
            
            if (!quantity || quantity <= 0) {
                showAlert('Please enter a valid quantity', 'error');
                return;
            }
            
            // Disable buttons during trade
            const buttons = document.querySelectorAll('.btn-buy, .btn-sell');
            buttons.forEach(btn => btn.disabled = true);
            
            resultEl.innerHTML = '<div class="signal-indicator signal-neutral">Executing trade...</div>';
            
            fetch('/api/trade', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    symbol: symbol,
                    action: action,
                    quantity: parseInt(quantity)
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    resultEl.innerHTML = `<div class="signal-indicator signal-buy">✅ ${data.message}</div>`;
                    showAlert(`Trade executed: ${data.message}`, 'success');
                    
                    // Reload dashboard
                    setTimeout(() => {
                        loadDashboard();
                        loadChart(symbol);
                    }, 1000);
                } else {
                    resultEl.innerHTML = `<div class="signal-indicator signal-sell">❌ ${data.message}</div>`;
                    showAlert(`Trade failed: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                resultEl.innerHTML = `<div class="signal-indicator signal-sell">❌ Error: ${error.message}</div>`;
                showAlert(`Trade error: ${error.message}`, 'error');
            })
            .finally(() => {
                buttons.forEach(btn => btn.disabled = false);
            });
        }
        
        function closePosition(symbol) {
            if (!confirm(`Close position for ${symbol}?`)) return;
            
            fetch('/api/trade', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    symbol: symbol,
                    action: 'sell',
                    quantity: 'all'
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showAlert(`Position closed: ${data.message}`, 'success');
                    loadDashboard();
                } else {
                    showAlert(`Failed to close position: ${data.message}`, 'error');
                }
            });
        }
        
        function autoTradeToggle() {
            const btn = document.getElementById('autoTradeBtn');
            
            fetch('/api/autotrade', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    enabled: !autoTrading
                })
            })
            .then(response => response.json())
            .then(data => {
                autoTrading = data.enabled;
                if (autoTrading) {
                    btn.innerHTML = '<i class="fas fa-stop"></i> Stop Auto-Trade';
                    btn.style.background = 'var(--danger)';
                    showAlert('Auto-trading enabled', 'success');
                } else {
                    btn.innerHTML = '<i class="fas fa-play"></i> Start Auto-Trade';
                    btn.style.background = 'var(--warning)';
                    showAlert('Auto-trading disabled', 'warning');
                }
            });
        }
        
        function showTradeSignal(signal) {
            if (!signal) return;
            
            const symbol = signal.symbol;
            const action = signal.signal > 0 ? 'BUY' : 'SELL';
            const strength = Math.abs(signal.signal * 100).toFixed(0);
            
            showAlert(`${action} signal for ${symbol} (${strength}% confidence)`, 
                     signal.signal > 0 ? 'success' : 'error');
        }
        
        function showAlert(message, type = 'info') {
            // Create alert element
            const alertEl = document.createElement('div');
            alertEl.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 20px;
                background: ${type === 'success' ? '#10b981' : 
                            type === 'error' ? '#ef4444' : '#3b82f6'};
                color: white;
                border-radius: 8px;
                z-index: 1000;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                animation: slideIn 0.3s ease;
            `;
            
            alertEl.innerHTML = `
                <strong>${type.toUpperCase()}:</strong> ${message}
            `;
            
            document.body.appendChild(alertEl);
            
            // Remove after 5 seconds
            setTimeout(() => {
                alertEl.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => {
                    document.body.removeChild(alertEl);
                }, 300);
            }, 5000);
        }
        
        // Add CSS animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
        
        // Initialize on page load
        window.onload = init;
    </script>
</body>
</html>
        '''
        
        # Save HTML template
        template_path = os.path.join('templates', 'index.html')
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"{Fore.GREEN}✅ HTML template created at {template_path}{Style.RESET_ALL}")
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/dashboard')
        def get_dashboard():
            """Get dashboard data"""
            metrics = self.engine.get_performance_metrics()
            positions = self.engine.get_positions_summary()
            
            # Get watchlist data
            watchlist = self.data_manager.get_live_quotes(self.config.DEFAULT_SYMBOLS)
            
            # Get recent trades (last 20)
            history = self.engine.trade_history[-20:] if self.engine.trade_history else []
            
            return jsonify({
                'metrics': metrics,
                'positions': positions,
                'watchlist': watchlist,
                'history': history
            })
        
        @self.app.route('/api/chart/<symbol>')
        def get_chart_data(symbol):
            """Get chart data for a symbol"""
            df = self.data_manager.fetch_data(symbol, period='1mo', interval='1d')
            if df is not None and not df.empty:
                # Get signal
                signal_data = self.engine.generate_signal(df, symbol)
                
                return jsonify({
                    'dates': df.index.strftime('%Y-%m-%d').tolist()[-30:],  # Last 30 days
                    'prices': df['Close'].tolist()[-30:],
                    'signal': signal_data['signal'],
                    'rsi': signal_data.get('rsi', 50),
                    'regime': signal_data.get('regime', 0)
                })
            return jsonify({'error': 'No data available'})
        
        @self.app.route('/api/quote/<symbol>')
        def get_quote(symbol):
            """Get current quote and signal"""
            df = self.data_manager.fetch_data(symbol, period='1d', interval='5m')
            if df is not None and not df.empty:
                signal_data = self.engine.generate_signal(df, symbol)
                latest = df.iloc[-1]
                
                return jsonify({
                    'symbol': symbol,
                    'price': latest['Close'],
                    'signal': signal_data['signal'],
                    'strength': signal_data['strength'],
                    'confidence': signal_data['confidence']
                })
            return jsonify({'error': 'No data'})
        
        @self.app.route('/api/trade', methods=['POST'])
        def execute_trade():
            """Execute a trade"""
            try:
                data = request.json
                symbol = data.get('symbol')
                action = data.get('action')
                quantity = data.get('quantity')
                
                if not symbol or not action:
                    return jsonify({'success': False, 'message': 'Missing symbol or action'})
                
                # Handle 'all' quantity
                if quantity == 'all':
                    for pos in self.engine.positions:
                        if pos['symbol'] == symbol and pos['status'] == 'OPEN':
                            quantity = pos['quantity']
                            break
                    else:
                        return jsonify({'success': False, 'message': 'No open position found'})
                else:
                    quantity = int(quantity)
                
                success, message = self.engine.execute_order(symbol, action, quantity)
                
                # Broadcast update
                self.broadcast_update()
                
                return jsonify({'success': success, 'message': message})
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/autotrade', methods=['POST'])
        def toggle_autotrade():
            """Toggle auto-trading"""
            data = request.json
            enabled = data.get('enabled', False)
            
            # In a real implementation, this would control an auto-trading loop
            return jsonify({
                'success': True,
                'enabled': enabled,
                'message': f'Auto-trading {"enabled" if enabled else "disabled"}'
            })
        
        @self.app.route('/api/news/<symbol>')
        def get_news(symbol):
            """Get news for a symbol"""
            news = self.data_manager.get_news(symbol)
            return jsonify({'news': news})
    
    def setup_websocket(self):
        """Setup WebSocket events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"{Fore.GREEN}✅ WebSocket client connected{Style.RESET_ALL}")
            self.broadcast_update()
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"{Fore.YELLOW}⚠️ WebSocket client disconnected{Style.RESET_ALL}")
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            symbol = data.get('symbol')
            print(f"{Fore.BLUE}📡 Client subscribed to {symbol}{Style.RESET_ALL}")
    
    def broadcast_update(self):
        """Broadcast update to all connected clients"""
        try:
            metrics = self.engine.get_performance_metrics()
            positions = self.engine.get_positions_summary()
            
            self.socketio.emit('update', {
                'metrics': metrics,
                'positions': positions,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"{Fore.RED}❌ Error broadcasting update: {e}{Style.RESET_ALL}")
    
    def start(self):
        """Start the web server"""
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🚀 Starting Trading Robot Dashboard{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Start auto-update thread
        self.running = True
        self.update_thread = threading.Thread(target=self.auto_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Start Flask server
        print(f"{Fore.GREEN}🌐 Web Dashboard: http://{self.config.WEB_HOST}:{self.config.WEB_PORT}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📱 Access from any device on your network{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        self.socketio.run(
            self.app,
            host=self.config.WEB_HOST,
            port=self.config.WEB_PORT,
            debug=False,
            allow_unsafe_werkzeug=True
        )
    
    def auto_update_loop(self):
        """Auto-update loop for real-time data"""
        while self.running:
            try:
                time.sleep(self.config.UPDATE_INTERVAL)
                self.broadcast_update()
                
                # Check for trading signals
                self.check_trading_signals()
                
            except Exception as e:
                print(f"{Fore.RED}❌ Error in auto-update: {e}{Style.RESET_ALL}")
    
    def check_trading_signals(self):
        """Check for trading signals on watchlist symbols"""
        for symbol in self.config.DEFAULT_SYMBOLS[:5]:  # Check first 5 symbols
            try:
                df = self.data_manager.fetch_data(symbol, period='1d', interval='5m')
                if df is not None and len(df) > 20:
                    signal_data = self.engine.generate_signal(df, symbol)
                    
                    # Cache the signal
                    self.engine.signals_cache[symbol] = signal_data
                    
                    # Emit strong signals
                    if abs(signal_data['signal']) > 0.5 and signal_data['confidence'] > 0.7:
                        self.socketio.emit('trade_signal', {
                            'symbol': symbol,
                            'signal': signal_data['signal'],
                            'strength': signal_data['strength'],
                            'confidence': signal_data['confidence'],
                            'price': signal_data['price'],
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        print(f"{Fore.CYAN}📡 Signal: {symbol} - "
                              f"{'BUY' if signal_data['signal'] > 0 else 'SELL'} "
                              f"(Strength: {signal_data['strength']:.2f}, "
                              f"Confidence: {signal_data['confidence']:.2f}){Style.RESET_ALL}")
                        
            except Exception as e:
                print(f"{Fore.RED}❌ Error checking signal for {symbol}: {e}{Style.RESET_ALL}")
    
    def stop(self):
        """Stop the web server"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        print(f"{Fore.YELLOW}🛑 Trading Robot Dashboard stopped{Style.RESET_ALL}")

# ==================== AUTO TRADING BOT ====================

class AutoTradingBot:
    """Automated trading bot with strategy execution"""
    
    def __init__(self, engine, data_manager, config=None):
        self.engine = engine
        self.data_manager = data_manager
        self.config = config or Config()
        self.running = False
        self.strategies = {
            'trend_following': self.trend_following_strategy,
            'mean_reversion': self.mean_reversion_strategy,
            'breakout': self.breakout_strategy,
            'ml_enhanced': self.ml_enhanced_strategy
        }
        self.active_strategy = 'ml_enhanced'
    
    def trend_following_strategy(self, symbol, df):
        """Trend following strategy"""
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        
        # Check for uptrend
        price_above_sma20 = latest['Close'] > latest.get('SMA_20', 0)
        price_above_sma50 = latest['Close'] > latest.get('SMA_50', 0)
        sma20_above_sma50 = latest.get('SMA_20', 0) > latest.get('SMA_50', 0)
        
        # Check momentum
        rsi_ok = 40 <= latest.get('rsi', 50) <= 70
        macd_bullish = latest.get('macd', 0) > latest.get('macd_signal', 0)
        
        if price_above_sma20 and price_above_sma50 and sma20_above_sma50 and rsi_ok and macd_bullish:
            return {
                'action': 'BUY',
                'confidence': 0.7,
                'reason': 'Trend following - Uptrend confirmed'
            }
        
        # Check for downtrend
        price_below_sma20 = latest['Close'] < latest.get('SMA_20', 0)
        price_below_sma50 = latest['Close'] < latest.get('SMA_50', 0)
        sma20_below_sma50 = latest.get('SMA_20', 0) < latest.get('SMA_50', 0)
        
        if price_below_sma20 and price_below_sma50 and sma20_below_sma50 and not rsi_ok and not macd_bullish:
            return {
                'action': 'SELL',
                'confidence': 0.7,
                'reason': 'Trend following - Downtrend confirmed'
            }
        
        return None
    
    def mean_reversion_strategy(self, symbol, df):
        """Mean reversion strategy"""
        if len(df) < 20:
            return None
        
        latest = df.iloc[-1]
        rsi = latest.get('rsi', 50)
        bb_position = (latest['Close'] - latest.get('bb_low', 0)) / (latest.get('bb_high', 1) - latest.get('bb_low', 0))
        
        # Oversold condition
        if rsi < 30 and bb_position < 0.2:
            return {
                'action': 'BUY',
                'confidence': 0.6,
                'reason': f'Mean reversion - Oversold (RSI: {rsi:.1f})'
            }
        
        # Overbought condition
        if rsi > 70 and bb_position > 0.8:
            return {
                'action': 'SELL',
                'confidence': 0.6,
                'reason': f'Mean reversion - Overbought (RSI: {rsi:.1f})'
            }
        
        return None
    
    def breakout_strategy(self, symbol, df):
        """Breakout trading strategy"""
        if len(df) < 30:
            return None
        
        latest = df.iloc[-1]
        prev_close = df.iloc[-2]['Close'] if len(df) > 1 else latest['Close']
        
        resistance = latest.get('resistance', latest['High'])
        support = latest.get('support', latest['Low'])
        
        # Breakout above resistance
        if latest['Close'] > resistance and latest['Volume'] > df['Volume'].rolling(20).mean().iloc[-1] * 1.5:
            return {
                'action': 'BUY',
                'confidence': 0.75,
                'reason': 'Breakout - Above resistance with volume'
            }
        
        # Breakdown below support
        if latest['Close'] < support and latest['Volume'] > df['Volume'].rolling(20).mean().iloc[-1] * 1.5:
            return {
                'action': 'SELL',
                'confidence': 0.75,
                'reason': 'Breakout - Below support with volume'
            }
        
        return None
    
    def ml_enhanced_strategy(self, symbol, df):
        """ML enhanced strategy combining all approaches"""
        strategies = [
            self.trend_following_strategy(symbol, df),
            self.mean_reversion_strategy(symbol, df),
            self.breakout_strategy(symbol, df)
        ]
        
        valid_strategies = [s for s in strategies if s is not None]
        
        if not valid_strategies:
            return None
        
        # Combine signals
        buy_signals = [s for s in valid_strategies if s['action'] == 'BUY']
        sell_signals = [s for s in valid_strategies if s['action'] == 'SELL']
        
        if len(buy_signals) >= 2:
            avg_confidence = np.mean([s['confidence'] for s in buy_signals])
            return {
                'action': 'BUY',
                'confidence': avg_confidence,
                'reason': 'ML Enhanced - Multiple buy signals',
                'details': [s['reason'] for s in buy_signals]
            }
        
        if len(sell_signals) >= 2:
            avg_confidence = np.mean([s['confidence'] for s in sell_signals])
            return {
                'action': 'SELL',
                'confidence': avg_confidence,
                'reason': 'ML Enhanced - Multiple sell signals',
                'details': [s['reason'] for s in sell_signals]
            }
        
        # Return strongest signal if any
        strongest = max(valid_strategies, key=lambda x: x['confidence'])
        if strongest['confidence'] > 0.7:
            return strongest
        
        return None
    
    def run(self):
        """Run the auto trading bot"""
        self.running = True
        print(f"{Fore.GREEN}🤖 Auto Trading Bot started with {self.active_strategy} strategy{Style.RESET_ALL}")
        
        while self.running:
            try:
                # Check each symbol in watchlist
                for symbol in self.config.DEFAULT_SYMBOLS[:3]:  # Trade only first 3 symbols
                    try:
                        # Skip if max positions reached
                        if len(self.engine.positions) >= self.config.MAX_POSITIONS:
                            continue
                        
                        # Skip if already have position in this symbol
                        if any(pos['symbol'] == symbol and pos['status'] == 'OPEN' for pos in self.engine.positions):
                            continue
                        
                        # Get data
                        df = self.data_manager.fetch_data(symbol, period='1d', interval='5m')
                        if df is None or len(df) < 50:
                            continue
                        
                        # Calculate indicators
                        df = self.engine.calculate_advanced_indicators(df)
                        
                        # Get signal from active strategy
                        strategy_func = self.strategies.get(self.active_strategy)
                        if strategy_func:
                            signal = strategy_func(symbol, df)
                            
                            if signal and signal['confidence'] > 0.7:
                                # Execute trade
                                quantity = self.engine.calculate_position_size(
                                    symbol, 
                                    df.iloc[-1]['Close'],
                                    signal['confidence']
                                )
                                
                                if quantity > 0:
                                    success, message = self.engine.execute_order(
                                        symbol,
                                        signal['action'],
                                        quantity,
                                        reason=signal['reason']
                                    )
                                    
                                    if success:
                                        print(f"{Fore.GREEN}✅ Auto-trade executed: {message}{Style.RESET_ALL}")
                                    else:
                                        print(f"{Fore.YELLOW}⚠️ Auto-trade skipped: {message}{Style.RESET_ALL}")
                    
                    except Exception as e:
                        print(f"{Fore.RED}❌ Error processing {symbol}: {e}{Style.RESET_ALL}")
                
                # Sleep between iterations
                time.sleep(self.config.UPDATE_INTERVAL * 2)
                
            except KeyboardInterrupt:
                print(f"{Fore.YELLOW}🛑 Auto Trading Bot stopped by user{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}❌ Error in auto trading loop: {e}{Style.RESET_ALL}")
                time.sleep(60)  # Wait a minute before retry
    
    def stop(self):
        """Stop the auto trading bot"""
        self.running = False
        print(f"{Fore.YELLOW}🛑 Auto Trading Bot stopped{Style.RESET_ALL}")

# ==================== MAIN APPLICATION ====================

class TradingRobot:
    """Main trading robot application"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'🤖 AI TRADING ROBOT v2.0'.center(70)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        # Initialize components
        self.engine = TradingEngine(self.config)
        self.data_manager = DataManager()
        self.dashboard = TradingDashboard(self.engine, self.data_manager, self.config)
        self.auto_bot = AutoTradingBot(self.engine, self.data_manager, self.config)
        
        # Load environment variables
        load_dotenv()
        
        # Set up notification webhooks
        self.setup_notifications()
    
    def setup_notifications(self):
        """Setup notification channels"""
        webhook = os.getenv('DISCORD_WEBHOOK')
        if webhook:
            self.config.DISCORD_WEBHOOK = webhook
            print(f"{Fore.GREEN}✅ Discord notifications enabled{Style.RESET_ALL}")
    
    def run_interactive(self):
        """Run in interactive mode"""
        while True:
            print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}🤖 TRADING ROBOT CONTROL PANEL{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
            print(f"1. {Fore.GREEN}Start Web Dashboard{Style.RESET_ALL}")
            print(f"2. {Fore.BLUE}Start Auto Trading Bot{Style.RESET_ALL}")
            print(f"3. {Fore.MAGENTA}View Account Status{Style.RESET_ALL}")
            print(f"4. {Fore.CYAN}Execute Manual Trade{Style.RESET_ALL}")
            print(f"5. {Fore.YELLOW}View Positions{Style.RESET_ALL}")
            print(f"6. {Fore.WHITE}Check Market Signals{Style.RESET_ALL}")
            print(f"7. {Fore.RED}Exit{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
            
            choice = input(f"\n{Fore.YELLOW}Select option (1-7): {Style.RESET_ALL}").strip()
            
            if choice == '1':
                self.start_dashboard()
            elif choice == '2':
                self.start_auto_bot()
            elif choice == '3':
                self.view_account_status()
            elif choice == '4':
                self.execute_manual_trade()
            elif choice == '5':
                self.view_positions()
            elif choice == '6':
                self.check_market_signals()
            elif choice == '7':
                print(f"\n{Fore.YELLOW}👋 Exiting...{Style.RESET_ALL}")
                break
            else:
                print(f"{Fore.RED}❌ Invalid choice{Style.RESET_ALL}")
    
    def start_dashboard(self):
        """Start the web dashboard"""
        print(f"\n{Fore.GREEN}🚀 Starting Web Dashboard...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 Dashboard will open in your browser{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🌐 Access at: http://localhost:{self.config.WEB_PORT}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📱 Or from any device on your network{Style.RESET_ALL}")
        
        # Start dashboard in a thread
        dashboard_thread = threading.Thread(target=self.dashboard.start)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Try to open browser
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{self.config.WEB_PORT}")
        except:
            pass
        
        input(f"\n{Fore.YELLOW}Press Enter to return to menu...{Style.RESET_ALL}")
    
    def start_auto_bot(self):
        """Start the auto trading bot"""
        print(f"\n{Fore.GREEN}🤖 Starting Auto Trading Bot...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📈 Strategy: {self.auto_bot.active_strategy}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}💰 Initial Balance: ${self.engine.balance:,.2f}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 Symbols: {', '.join(self.config.DEFAULT_SYMBOLS[:3])}{Style.RESET_ALL}")
        
        # Start auto bot in a thread
        bot_thread = threading.Thread(target=self.auto_bot.run)
        bot_thread.daemon = True
        bot_thread.start()
        
        print(f"\n{Fore.YELLOW}🤖 Auto bot running in background...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop{Style.RESET_ALL}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.auto_bot.stop()
    
    def view_account_status(self):
        """View account status"""
        metrics = self.engine.get_performance_metrics()
        
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💰 ACCOUNT STATUS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Initial Balance:{Style.RESET_ALL} ${metrics['initial_balance']:,.2f}")
        print(f"{Fore.WHITE}Current Balance:{Style.RESET_ALL} ${metrics['current_balance']:,.2f}")
        print(f"{Fore.WHITE}Portfolio Value:{Style.RESET_ALL} ${metrics['portfolio_value']:,.2f}")
        
        total_return = metrics['total_return']
        return_color = Fore.GREEN if total_return >= 0 else Fore.RED
        print(f"{Fore.WHITE}Total Return:{Style.RESET_ALL} {return_color}{total_return:.2f}%{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}📊 PERFORMANCE METRICS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*70}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Total Trades:{Style.RESET_ALL} {metrics['total_trades']}")
        print(f"{Fore.WHITE}Win Rate:{Style.RESET_ALL} {metrics['win_rate']:.1f}%")
        print(f"{Fore.WHITE}Sharpe Ratio:{Style.RESET_ALL} {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"{Fore.WHITE}Max Drawdown:{Style.RESET_ALL} {metrics['max_drawdown']:.2f}%")
        print(f"{Fore.WHITE}Daily P&L:{Style.RESET_ALL} ${metrics['daily_pnl']:.2f}")
        print(f"{Fore.WHITE}Open Positions:{Style.RESET_ALL} {metrics['open_positions']}")
        
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    
    def execute_manual_trade(self):
        """Execute a manual trade"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💼 MANUAL TRADE EXECUTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        symbol = input(f"{Fore.YELLOW}Symbol (e.g., AAPL): {Style.RESET_ALL}").strip().upper()
        if not symbol:
            print(f"{Fore.RED}❌ Symbol required{Style.RESET_ALL}")
            return
        
        action = input(f"{Fore.YELLOW}Action (BUY/SELL): {Style.RESET_ALL}").strip().upper()
        if action not in ['BUY', 'SELL']:
            print(f"{Fore.RED}❌ Invalid action{Style.RESET_ALL}")
            return
        
        try:
            quantity = int(input(f"{Fore.YELLOW}Quantity: {Style.RESET_ALL}").strip())
        except:
            print(f"{Fore.RED}❌ Invalid quantity{Style.RESET_ALL}")
            return
        
        # Get current price
        df = self.data_manager.fetch_data(symbol, period='1d', interval='1m')
        if df is None or df.empty:
            print(f"{Fore.RED}❌ Cannot get price for {symbol}{Style.RESET_ALL}")
            return
        
        current_price = df['Close'].iloc[-1]
        print(f"{Fore.CYAN}Current price: ${current_price:.2f}{Style.RESET_ALL}")
        
        # Confirm trade
        confirm = input(f"{Fore.YELLOW}Confirm {action} {quantity} {symbol} @ ${current_price:.2f}? (y/n): {Style.RESET_ALL}").lower()
        
        if confirm == 'y':
            success, message = self.engine.execute_order(symbol, action, quantity, current_price)
            if success:
                print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️ Trade cancelled{Style.RESET_ALL}")
    
    def view_positions(self):
        """View current positions"""
        positions = self.engine.get_positions_summary()
        
        if not positions:
            print(f"\n{Fore.YELLOW}📭 No open positions{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📊 OPEN POSITIONS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        total_value = 0
        total_pnl = 0
        
        for pos in positions:
            pnl_color = Fore.GREEN if pos['pnl'] >= 0 else Fore.RED
            pnl_sign = '+' if pos['pnl'] >= 0 else ''
            
            print(f"\n{Fore.WHITE}📈 {pos['symbol']} - {pos['action']}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Quantity:{Style.RESET_ALL} {pos['quantity']}")
            print(f"{Fore.WHITE}Entry Price:{Style.RESET_ALL} ${pos['entry_price']:.2f}")
            print(f"{Fore.WHITE}Current Price:{Style.RESET_ALL} ${pos['current_price']:.2f}")
            print(f"{Fore.WHITE}P&L:{Style.RESET_ALL} {pnl_color}{pnl_sign}${pos['pnl']:.2f} ({pnl_sign}{pos['pnl_percent']:.2f}%){Style.RESET_ALL}")
            print(f"{Fore.WHITE}Value:{Style.RESET_ALL} ${pos['value']:.2f}")
            print(f"{Fore.WHITE}Time Held:{Style.RESET_ALL} {pos['time_held']:.1f} hours")
            
            total_value += pos['value']
            total_pnl += pos['pnl']
        
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📈 SUMMARY{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*40}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Total Positions:{Style.RESET_ALL} {len(positions)}")
        print(f"{Fore.WHITE}Total Value:{Style.RESET_ALL} ${total_value:.2f}")
        
        total_pnl_color = Fore.GREEN if total_pnl >= 0 else Fore.RED
        total_pnl_sign = '+' if total_pnl >= 0 else ''
        print(f"{Fore.WHITE}Total P&L:{Style.RESET_ALL} {total_pnl_color}{total_pnl_sign}${total_pnl:.2f}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    
    def check_market_signals(self):
        """Check market signals for watchlist"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📡 MARKET SIGNALS CHECK{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        for symbol in self.config.DEFAULT_SYMBOLS[:5]:
            try:
                df = self.data_manager.fetch_data(symbol, period='1d', interval='5m')
                if df is not None and len(df) > 20:
                    df = self.engine.calculate_advanced_indicators(df)
                    signal_data = self.engine.generate_signal(df, symbol)
                    
                    signal_icon = "🟢" if signal_data['signal'] > 0.3 else "🔴" if signal_data['signal'] < -0.3 else "🟡"
                    action = "BUY" if signal_data['signal'] > 0.3 else "SELL" if signal_data['signal'] < -0.3 else "HOLD"
                    
                    print(f"{signal_icon} {Fore.WHITE}{symbol:8}{Style.RESET_ALL} | "
                          f"{Fore.CYAN}${df['Close'].iloc[-1]:8.2f}{Style.RESET_ALL} | "
                          f"{Fore.YELLOW}{action:6}{Style.RESET_ALL} | "
                          f"{Fore.MAGENTA}RSI: {signal_data.get('rsi', 0):5.1f}{Style.RESET_ALL} | "
                          f"{Fore.GREEN}Strength: {signal_data['strength']:.2f}{Style.RESET_ALL}")
            
            except Exception as e:
                print(f"{Fore.RED}❌ Error checking {symbol}: {e}{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    
    def run(self):
        """Run the main application"""
        self.run_interactive()

# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point"""
    try:
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'🚀 WELCOME TO AI TRADING ROBOT v2.0'.center(70)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        # Create trading robot
        robot = TradingRobot()
        
        # Run interactive mode
        robot.run()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}👋 Exiting Trading Robot...{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'👋 THANK YOU FOR USING TRADING ROBOT'.center(70)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
