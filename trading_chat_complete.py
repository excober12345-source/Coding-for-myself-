#!/usr/bin/env python3
"""
COMPLETE TRADING APPLICATION - ALL-IN-ONE FILE
MetaTrader 5 Style Trading Platform
"""

import sys
import os
import subprocess
import platform

# ==================== AUTO-INSTALL DEPENDENCIES ====================

def check_and_install_dependencies():
    """Automatically check and install missing packages"""
    required_packages = [
        'pandas',
        'numpy',
        'yfinance',
        'plotly',
        'tkinter'  # Usually comes with Python
    ]
    
    missing_packages = []
    
    print("=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package.replace('-', '_'))
            print(f"✓ {package:15} - Installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package:15} - Missing")
    
    if missing_packages:
        print(f"\nInstalling {len(missing_packages)} missing packages...")
        
        # Remove tkinter from install list (it's a system package)
        if 'tkinter' in missing_packages:
            missing_packages.remove('tkinter')
            print("Note: tkinter requires system installation")
            system = platform.system().lower()
            if 'linux' in system:
                print("On Ubuntu/Debian, run: sudo apt-get install python3-tk")
            elif 'darwin' in system:
                print("On macOS, tkinter should come with Python")
        
        if missing_packages:
            try:
                for package in missing_packages:
                    print(f"Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print("\n✅ All packages installed successfully!")
            except Exception as e:
                print(f"\n❌ Installation failed: {e}")
                print("Try running: pip install pandas numpy yfinance plotly")
                return False
    
    print("\n" + "=" * 60)
    print("DEPENDENCIES CHECK COMPLETE")
    print("=" * 60 + "\n")
    return True

# Run dependency check
if not check_and_install_dependencies():
    print("Failed to install dependencies. Please install manually:")
    print("pip install pandas numpy yfinance plotly")
    input("Press Enter to exit...")
    sys.exit(1)

# ==================== IMPORT PACKAGES ====================

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import tkinter
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
    TKINTER_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: tkinter not available. Running in console mode only.")
    TKINTER_AVAILABLE = False

# ==================== CORE TRADING ENGINE ====================

class TradingEngine:
    """Core trading engine with portfolio management"""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []  # Open positions
        self.trade_history = []  # Closed trades
        self.portfolio = {}  # Current holdings
        self.equity_curve = []
        self.max_drawdown = 0
        self.total_trades = 0
        
    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators for a dataframe"""
        if df.empty:
            return df
        
        # Copy to avoid warnings
        df = df.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14, min_periods=1).min()
        high_14 = df['High'].rolling(window=14, min_periods=1).max()
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3, min_periods=1).mean()
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14, min_periods=1).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price changes
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift())
        df['Volatility'] = df['Returns'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
        
        # Support and Resistance (simplified)
        df['Resistance'] = df['High'].rolling(window=20, min_periods=1).max()
        df['Support'] = df['Low'].rolling(window=20, min_periods=1).min()
        
        # Trend indicators
        df['Trend'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
        
        return df
    
    def generate_trading_signals(self, df):
        """Generate trading signals based on multiple indicators"""
        df = df.copy()
        df['Signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
        
        # Buy signals (multiple conditions)
        buy_conditions = (
            (df['SMA_20'] > df['SMA_50']) &  # Golden cross
            (df['RSI'] < 70) &  # Not overbought
            (df['MACD'] > df['MACD_Signal']) &  # MACD bullish
            (df['Close'] > df['BB_Middle']) &  # Above middle band
            (df['Volume_Ratio'] > 1)  # Above average volume
        )
        
        # Strong buy signals
        strong_buy_conditions = (
            (df['RSI'] < 30) &  # Oversold
            (df['Close'] < df['BB_Lower'])  # Below lower band
        )
        
        # Sell signals
        sell_conditions = (
            (df['SMA_20'] < df['SMA_50']) |  # Death cross
            (df['RSI'] > 70) |  # Overbought
            (df['MACD'] < df['MACD_Signal'])  # MACD bearish
        )
        
        # Strong sell signals
        strong_sell_conditions = (
            (df['RSI'] > 85) &  # Extremely overbought
            (df['Close'] > df['BB_Upper'])  # Above upper band
        )
        
        # Apply signals
        df.loc[buy_conditions, 'Signal'] = 1
        df.loc[strong_buy_conditions, 'Signal'] = 2  # Strong buy
        df.loc[sell_conditions, 'Signal'] = -1
        df.loc[strong_sell_conditions, 'Signal'] = -2  # Strong sell
        
        # Calculate signal strength
        df['Signal_Strength'] = 0
        df.loc[df['Signal'] == 1, 'Signal_Strength'] = 0.5
        df.loc[df['Signal'] == 2, 'Signal_Strength'] = 1.0
        df.loc[df['Signal'] == -1, 'Signal_Strength'] = -0.5
        df.loc[df['Signal'] == -2, 'Signal_Strength'] = -1.0
        
        return df
    
    def execute_order(self, symbol, order_type, quantity, price=None, 
                     stop_loss=None, take_profit=None, order_id=None):
        """Execute a trading order"""
        if order_id is None:
            order_id = f"{symbol}_{order_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate costs
        commission_rate = 0.001  # 0.1% commission
        current_price = price if price else self.get_current_price(symbol)
        
        if order_type.upper() in ['BUY', 'LONG']:
            total_cost = current_price * quantity
            commission = total_cost * commission_rate
            
            if total_cost + commission > self.balance:
                return False, "Insufficient balance"
            
            # Create position
            position = {
                'order_id': order_id,
                'symbol': symbol,
                'type': 'LONG',
                'entry_price': current_price,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'OPEN',
                'commission': commission
            }
            
            self.positions.append(position)
            self.portfolio[symbol] = self.portfolio.get(symbol, 0) + quantity
            self.balance -= (total_cost + commission)
            self.total_trades += 1
            
            # Log trade
            self.trade_history.append({
                **position,
                'action': 'OPEN_LONG'
            })
            
            return True, f"BUY order executed: {quantity} {symbol} @ ${current_price:.2f}"
        
        elif order_type.upper() in ['SELL', 'CLOSE']:
            # Check if we have the position
            position_to_close = None
            for pos in self.positions:
                if pos['symbol'] == symbol and pos['status'] == 'OPEN':
                    position_to_close = pos
                    break
            
            if not position_to_close:
                return False, f"No open position found for {symbol}"
            
            # Calculate P&L
            pnl = (current_price - position_to_close['entry_price']) * quantity
            pnl_percent = ((current_price - position_to_close['entry_price']) / 
                          position_to_close['entry_price']) * 100
            
            # Update position
            position_to_close['exit_price'] = current_price
            position_to_close['exit_time'] = datetime.now()
            position_to_close['status'] = 'CLOSED'
            position_to_close['pnl'] = pnl
            position_to_close['pnl_percent'] = pnl_percent
            
            # Calculate commission
            total_value = current_price * quantity
            commission = total_value * commission_rate
            
            # Update account
            self.balance += pnl - commission
            
            # Update portfolio
            self.portfolio[symbol] -= quantity
            if self.portfolio[symbol] <= 0:
                del self.portfolio[symbol]
            
            # Move to trade history
            self.trade_history.append(position_to_close.copy())
            self.positions.remove(position_to_close)
            
            # Update equity curve
            self.update_equity_curve()
            
            return True, (f"SELL order executed: {quantity} {symbol} @ ${current_price:.2f}\n"
                         f"P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")
        
        elif order_type.upper() == 'SHORT':
            # Short selling (simplified)
            total_value = current_price * quantity
            commission = total_value * commission_rate
            
            position = {
                'order_id': order_id,
                'symbol': symbol,
                'type': 'SHORT',
                'entry_price': current_price,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'OPEN',
                'commission': commission
            }
            
            self.positions.append(position)
            self.balance -= commission  # Only pay commission for short
            self.total_trades += 1
            
            return True, f"SHORT order executed: {quantity} {symbol} @ ${current_price:.2f}"
        
        return False, f"Unknown order type: {order_type}"
    
    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return data['Close'].iloc[-1]
        except:
            pass
        return 100.0  # Default fallback price
    
    def update_equity_curve(self):
        """Update equity curve with current portfolio value"""
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
            drawdown = ((peak - current) / peak) * 100
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def get_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'current_balance': self.balance,
                'max_drawdown': 0
            }
        
        # Filter closed trades
        closed_trades = [t for t in self.trade_history if t.get('status') == 'CLOSED']
        
        if not closed_trades:
            return {
                'total_trades': len(self.positions),
                'open_positions': len(self.positions),
                'current_balance': self.balance,
                'portfolio_value': self.balance + sum(
                    self.get_current_price(sym) * qty 
                    for sym, qty in self.portfolio.items()
                )
            }
        
        df_trades = pd.DataFrame(closed_trades)
        
        # Basic metrics
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        total_trades = len(df_trades)
        
        # Calculate advanced metrics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Risk metrics
        total_pnl = df_trades['pnl'].sum()
        sharpe_ratio = self.calculate_sharpe_ratio(df_trades)
        
        # Portfolio metrics
        portfolio_value = self.balance
        for symbol, quantity in self.portfolio.items():
            portfolio_value += self.get_current_price(symbol) * quantity
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'largest_win': df_trades['pnl'].max(),
            'largest_loss': df_trades['pnl'].min(),
            'sharpe_ratio': sharpe_ratio,
            'current_balance': self.balance,
            'portfolio_value': portfolio_value,
            'total_return': ((portfolio_value - self.initial_balance) / self.initial_balance) * 100,
            'max_drawdown': self.max_drawdown,
            'open_positions': len(self.positions),
            'avg_hold_time': self.calculate_avg_hold_time(df_trades)
        }
    
    def calculate_sharpe_ratio(self, df_trades, risk_free_rate=0.02):
        """Calculate Sharpe ratio from trades"""
        if len(df_trades) < 2:
            return 0
        
        returns = df_trades['pnl'] / (df_trades['entry_price'] * df_trades['quantity'])
        excess_returns = returns - (risk_free_rate / 252)
        
        if excess_returns.std() == 0:
            return 0
        
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        return sharpe
    
    def calculate_avg_hold_time(self, df_trades):
        """Calculate average holding time"""
        if df_trades.empty or 'exit_time' not in df_trades.columns:
            return "N/A"
        
        hold_times = []
        for _, trade in df_trades.iterrows():
            if 'entry_time' in trade and 'exit_time' in trade:
                if isinstance(trade['entry_time'], str):
                    entry = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                else:
                    entry = trade['entry_time']
                
                if isinstance(trade['exit_time'], str):
                    exit = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00'))
                else:
                    exit = trade['exit_time']
                
                hold_time = (exit - entry).total_seconds() / 3600  # in hours
                hold_times.append(hold_time)
        
        if hold_times:
            avg_hours = np.mean(hold_times)
            if avg_hours < 24:
                return f"{avg_hours:.1f} hours"
            else:
                return f"{avg_hours/24:.1f} days"
        
        return "N/A"
    
    def get_position_summary(self):
        """Get summary of all open positions"""
        positions_summary = []
        
        for position in self.positions:
            if position['status'] == 'OPEN':
                current_price = self.get_current_price(position['symbol'])
                pnl = (current_price - position['entry_price']) * position['quantity']
                pnl_percent = ((current_price - position['entry_price']) / 
                              position['entry_price']) * 100
                
                positions_summary.append({
                    'symbol': position['symbol'],
                    'type': position['type'],
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'value': current_price * position['quantity']
                })
        
        return positions_summary

# ==================== DATA MANAGER ====================

class DataManager:
    """Manages market data fetching and caching"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
        
    def fetch_data(self, symbol, period='1mo', interval='1d', use_cache=True):
        """Fetch market data with caching"""
        cache_key = f"{symbol}_{period}_{interval}"
        
        if use_cache and cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return cached_data.copy()
        
        try:
            print(f"Fetching data for {symbol}...")
            ticker = yf.Ticker(symbol)
            
            # Handle different intervals
            if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
                # For intraday, adjust period
                if period == '1mo':
                    period = '5d'
                elif period == '3mo':
                    period = '1mo'
            
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print(f"No data returned for {symbol}")
                return None
            
            # Clean column names
            df.columns = [col.capitalize() for col in df.columns]
            
            # Add symbol column
            df['Symbol'] = symbol
            
            # Cache the data
            self.cache[cache_key] = (df.copy(), datetime.now())
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_multiple_symbols(self, symbols, period='1mo', interval='1d'):
        """Fetch data for multiple symbols concurrently"""
        data = {}
        
        def fetch_symbol(symbol):
            df = self.fetch_data(symbol, period, interval)
            if df is not None:
                data[symbol] = df
        
        # Use threading for concurrent fetching
        threads = []
        for symbol in symbols:
            thread = threading.Thread(target=fetch_symbol, args=(symbol,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=10)
        
        return data
    
    def get_live_quotes(self, symbols):
        """Get live quotes for multiple symbols"""
        quotes = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_data(symbol, period='1d', interval='1m')
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    quotes[symbol] = {
                        'price': latest['Close'],
                        'change': latest['Close'] - df.iloc[-2]['Close'] if len(df) > 1 else 0,
                        'volume': latest['Volume'],
                        'high': latest['High'],
                        'low': latest['Low']
                    }
            except:
                quotes[symbol] = None
        
        return quotes

# ==================== GUI APPLICATION ====================

class TradingAppGUI:
    """Main GUI application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Python Trading Platform - MT5 Style")
        self.root.geometry("1400x800")
        
        # Initialize core components
        self.engine = TradingEngine(initial_balance=10000)
        self.data_manager = DataManager()
        
        # Current state
        self.current_symbol = "AAPL"
        self.current_data = None
        self.watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "BTC-USD", "ETH-USD"]
        
        # Auto-refresh
        self.auto_refresh = True
        self.refresh_interval = 5000  # 5 seconds
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        
        # Load initial data
        self.refresh_data()
        
        # Start auto-refresh
        self.schedule_refresh()
    
    def setup_styles(self):
        """Setup ttk styles"""
        if not TKINTER_AVAILABLE:
            return
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # Color scheme
        self.bg_color = '#1e1e1e'
        self.fg_color = '#ffffff'
        self.accent_color = '#007acc'
        self.profit_color = '#4CAF50'
        self.loss_color = '#F44336'
        
        # Configure styles
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabel', background=self.bg_color, foreground=self.fg_color)
        style.configure('TButton', background=self.accent_color, foreground=self.fg_color)
        style.configure('Buy.TButton', background=self.profit_color)
        style.configure('Sell.TButton', background=self.loss_color)
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
    
    def create_widgets(self):
        """Create all GUI widgets"""
        if not TKINTER_AVAILABLE:
            return
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left Panel - Controls and Watchlist
        left_panel = ttk.Frame(main_container, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Symbol Selection
        control_frame = ttk.LabelFrame(left_panel, text="Symbol Control", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.symbol_var = tk.StringVar(value=self.current_symbol)
        symbol_combo = ttk.Combobox(control_frame, textvariable=self.symbol_var, 
                                   values=self.watchlist, width=15)
        symbol_combo.grid(row=0, column=1, pady=5, padx=(5, 0))
        symbol_combo.bind('<<ComboboxSelected>>', self.on_symbol_change)
        
        # Quick add to watchlist
        ttk.Label(control_frame, text="Add Symbol:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.new_symbol_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.new_symbol_var, width=15).grid(row=1, column=1, pady=5, padx=(5, 0))
        ttk.Button(control_frame, text="Add", command=self.add_to_watchlist, width=8).grid(row=1, column=2, pady=5, padx=(5, 0))
        
        # Timeframe Selection
        ttk.Label(control_frame, text="Timeframe:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.timeframe_var = tk.StringVar(value="1d")
        timeframe_combo = ttk.Combobox(control_frame, textvariable=self.timeframe_var,
                                      values=["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"], width=15)
        timeframe_combo.grid(row=2, column=1, pady=5, padx=(5, 0))
        timeframe_combo.bind('<<ComboboxSelected>>', self.on_timeframe_change)
        
        # Period Selection
        ttk.Label(control_frame, text="Period:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.period_var = tk.StringVar(value="1mo")
        period_combo = ttk.Combobox(control_frame, textvariable=self.period_var,
                                   values=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], width=15)
        period_combo.grid(row=3, column=1, pady=5, padx=(5, 0))
        period_combo.bind('<<ComboboxSelected>>', self.on_period_change)
        
        # Trading Panel
        trade_frame = ttk.LabelFrame(left_panel, text="Trading", padding=10)
        trade_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Quantity
        ttk.Label(trade_frame, text="Quantity:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.quantity_var = tk.StringVar(value="10")
        ttk.Entry(trade_frame, textvariable=self.quantity_var, width=15).grid(row=0, column=1, pady=5, padx=(5, 0))
        
        # Price (display only)
        ttk.Label(trade_frame, text="Price:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.price_label = ttk.Label(trade_frame, text="$0.00")
        self.price_label.grid(row=1, column=1, pady=5, padx=(5, 0), sticky=tk.W)
        
        # Action buttons
        button_frame = ttk.Frame(trade_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="BUY", command=self.execute_buy,
                  style="Buy.TButton", width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="SELL", command=self.execute_sell,
                  style="Sell.TButton", width=10).pack(side=tk.LEFT)
        
        # Account Panel
        account_frame = ttk.LabelFrame(left_panel, text="Account", padding=10)
        account_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.balance_label = ttk.Label(account_frame, text=f"Balance: ${self.engine.balance:,.2f}")
        self.balance_label.pack(anchor=tk.W, pady=2)
        
        self.equity_label = ttk.Label(account_frame, text="Equity: $0.00")
        self.equity_label.pack(anchor=tk.W, pady=2)
        
        self.pnl_label = ttk.Label(account_frame, text="Today's P&L: $0.00")
        self.pnl_label.pack(anchor=tk.W, pady=2)
        
        # Watchlist Panel
        watchlist_frame = ttk.LabelFrame(left_panel, text="Watchlist", padding=10)
        watchlist_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create watchlist treeview
        columns = ('Symbol', 'Price', 'Change', 'Change %', 'Volume')
        self.watchlist_tree = ttk.Treeview(watchlist_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.watchlist_tree.heading(col, text=col)
            self.watchlist_tree.column(col, width=80)
        
        scrollbar = ttk.Scrollbar(watchlist_frame, orient=tk.VERTICAL, command=self.watchlist_tree.yview)
        self.watchlist_tree.configure(yscrollcommand=scrollbar.set)
        
        self.watchlist_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click to select symbol
        self.watchlist_tree.bind('<Double-1>', self.on_watchlist_select)
        
        # Right Panel - Main Content
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Chart Tab
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="Chart")
        
        # Market Data Tab
        self.market_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.market_frame, text="Market Data")
        
        # Positions Tab
        self.positions_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.positions_frame, text="Positions")
        
        # Orders Tab
        self.orders_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.orders_frame, text="Orders")
        
        # Analysis Tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")
        
        # Log Tab
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="Log")
        
        # Create log text widget
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=10, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # Status Bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Menu Bar
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Portfolio", command=self.save_portfolio)
        file_menu.add_command(label="Load Portfolio", command=self.load_portfolio)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Auto Refresh", variable=tk.BooleanVar(value=True),
                                 command=self.toggle_auto_refresh)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Backtest", command=self.open_backtester)
        tools_menu.add_command(label="Strategy Tester", command=self.open_strategy_tester)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)
    
    def log_message(self, message):
        """Add message to log"""
        if not TKINTER_AVAILABLE:
            print(f"[LOG] {message}")
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def update_status(self, message):
        """Update status bar"""
        if TKINTER_AVAILABLE:
            self.status_bar.config(text=message)
        print(f"[STATUS] {message}")
    
    def refresh_data(self):
        """Refresh all data"""
        try:
            # Update current symbol data
            self.current_data = self.data_manager.fetch_data(
                self.current_symbol,
                period=self.period_var.get(),
                interval=self.timeframe_var.get()
            )
            
            if self.current_data is not None:
                # Calculate indicators
                self.current_data = self.engine.calculate_technical_indicators(self.current_data)
                self.current_data = self.engine.generate_trading_signals(self.current_data)
                
                # Update display
                self.update_price_display()
                self.update_watchlist()
                self.update_positions_display()
                self.update_account_display()
                
                self.log_message(f"Data refreshed for {self.current_symbol}")
            
        except Exception as e:
            self.log_message(f"Error refreshing data: {e}")
    
    def update_price_display(self):
        """Update current price display"""
        if self.current_data is not None and not self.current_data.empty:
            latest = self.current_data.iloc[-1]
            current_price = latest['Close']
            
            if TKINTER_AVAILABLE:
                self.price_label.config(text=f"${current_price:.2f}")
            
            # Update chart
            self.update_chart()
            self.update_market_data_tab()
            self.update_analysis_tab()
    
    def update_chart(self):
        """Update the price chart"""
        if self.current_data is None or self.current_data.empty:
            return
        
        try:
            # Create Plotly figure
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.15, 0.15, 0.2],
                subplot_titles=(f'{self.current_symbol} Price Chart', 'Volume', 'RSI', 'MACD')
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=self.current_data.index,
                    open=self.current_data['Open'],
                    high=self.current_data['High'],
                    low=self.current_data['Low'],
                    close=self.current_data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            fig.add_trace(
                go.Scatter(x=self.current_data.index, y=self.current_data['SMA_20'],
                          name='SMA 20', line=dict(color='orange', width=1)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=self.current_data.index, y=self.current_data['SMA_50'],
                          name='SMA 50', line=dict(color='blue', width=1)),
                row=1, col=1
            )
            
            # Volume
            colors = ['red' if self.current_data['Close'].iloc[i] < self.current_data['Open'].iloc[i]
                     else 'green' for i in range(len(self.current_data))]
            fig.add_trace(
                go.Bar(x=self.current_data.index, y=self.current_data['Volume'],
                      name='Volume', marker_color=colors),
                row=2, col=1
            )
            
            # RSI
            fig.add_trace(
                go.Scatter(x=self.current_data.index, y=self.current_data['RSI'],
                          name='RSI', line=dict(color='purple')),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
            
            # MACD
            fig.add_trace(
                go.Scatter(x=self.current_data.index, y=self.current_data['MACD'],
                          name='MACD', line=dict(color='blue')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=self.current_data.index, y=self.current_data['MACD_Signal'],
                          name='Signal', line=dict(color='red')),
                row=4, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'{self.current_symbol} - {self.timeframe_var.get()} timeframe',
                xaxis_title='Date',
                yaxis_title='Price',
                showlegend=True,
                height=700,
                template='plotly_dark'
            )
            
            fig.update_xaxes(rangeslider_visible=False)
            
            # Save and open in browser
            chart_file = f"chart_{self.current_symbol}.html"
            fig.write_html(chart_file)
            
            self.log_message(f"Chart saved to {chart_file}")
            
            # Display chart info in GUI
            if TKINTER_AVAILABLE:
                for widget in self.chart_frame.winfo_children():
                    widget.destroy()
                
                label = ttk.Label(self.chart_frame, 
                                 text=f"Chart generated: {chart_file}\nOpen in browser to view interactive chart.",
                                 font=('Arial', 10))
                label.pack(pady=20)
                
                # Show last signal
                latest_signal = self.current_data['Signal'].iloc[-1]
                signal_text = {
                    2: "STRONG BUY",
                    1: "BUY",
                    0: "NEUTRAL",
                    -1: "SELL",
                    -2: "STRONG SELL"
                }.get(latest_signal, "UNKNOWN")
                
                color = self.profit_color if latest_signal > 0 else self.loss_color if latest_signal < 0 else self.fg_color
                signal_label = ttk.Label(self.chart_frame, text=f"Current Signal: {signal_text}",
                                        font=('Arial', 12, 'bold'), foreground=color)
                signal_label.pack(pady=10)
            
        except Exception as e:
            self.log_message(f"Error updating chart: {e}")
    
    def update_watchlist(self):
        """Update watchlist display"""
        if not TKINTER_AVAILABLE:
            return
        
        # Clear current items
        for item in self.watchlist_tree.get_children():
            self.watchlist_tree.delete(item)
        
        # Fetch quotes for all symbols
        quotes = self.data_manager.get_live_quotes(self.watchlist)
        
        # Add items to treeview
        for symbol in self.watchlist:
            if symbol in quotes and quotes[symbol] is not None:
                quote = quotes[symbol]
                change = quote['change']
                change_percent = (change / (quote['price'] - change)) * 100 if quote['price'] - change != 0 else 0
                
                # Determine color for change
                tags = ('profit',) if change >= 0 else ('loss',)
                
                self.watchlist_tree.insert('', tk.END, values=(
                    symbol,
                    f"${quote['price']:.2f}",
                    f"${change:+.2f}",
                    f"{change_percent:+.2f}%",
                    f"{quote['volume']:,.0f}"
                ), tags=tags)
            else:
                self.watchlist_tree.insert('', tk.END, values=(
                    symbol, "N/A", "N/A", "N/A", "N/A"
                ))
        
        # Configure tags for coloring
        self.watchlist_tree.tag_configure('profit', foreground=self.profit_color)
        self.watchlist_tree.tag_configure('loss', foreground=self.loss_color)
    
    def update_market_data_tab(self):
        """Update market data tab"""
        if not TKINTER_AVAILABLE or self.current_data is None:
            return
        
        # Clear frame
        for widget in self.market_frame.winfo_children():
            widget.destroy()
        
        latest = self.current_data.iloc[-1]
        
        # Create a grid of data
        data_frame = ttk.Frame(self.market_frame)
        data_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        data_points = [
            ("Symbol", self.current_symbol),
            ("Time", latest.name.strftime("%Y-%m-%d %H:%M:%S") if hasattr(latest.name, 'strftime') else str(latest.name)),
            ("Open", f"${latest['Open']:.2f}"),
            ("High", f"${latest['High']:.2f}"),
            ("Low", f"${latest['Low']:.2f}"),
            ("Close", f"${latest['Close']:.2f}"),
            ("Volume", f"{latest['Volume']:,.0f}"),
            ("", ""),
            ("SMA 20", f"${latest.get('SMA_20', 0):.2f}"),
            ("SMA 50", f"${latest.get('SMA_50', 0):.2f}"),
            ("RSI", f"{latest.get('RSI', 0):.2f}"),
            ("MACD", f"{latest.get('MACD', 0):.4f}"),
            ("BB Upper", f"${latest.get('BB_Upper', 0):.2f}"),
            ("BB Lower", f"${latest.get('BB_Lower', 0):.2f}"),
            ("ATR", f"${latest.get('ATR', 0):.2f}"),
            ("Volatility", f"{latest.get('Volatility', 0)*100:.2f}%")
        ]
        
        for i, (label, value) in enumerate(data_points):
            row = i // 2
            col = (i % 2) * 2
            
            ttk.Label(data_frame, text=label, font=('Arial', 10, 'bold')).grid(
                row=row, column=col, sticky=tk.W, pady=5, padx=(0, 10))
            ttk.Label(data_frame, text=value).grid(
                row=row, column=col+1, sticky=tk.W, pady=5)
    
    def update_positions_display(self):
        """Update positions tab"""
        if not TKINTER_AVAILABLE:
            return
        
        # Clear frame
        for widget in self.positions_frame.winfo_children():
            widget.destroy()
        
        positions = self.engine.get_position_summary()
        
        if not positions:
            label = ttk.Label(self.positions_frame, text="No open positions", font=('Arial', 12))
            label.pack(pady=50)
            return
        
        # Create positions table
        columns = ('Symbol', 'Type', 'Qty', 'Entry', 'Current', 'P&L', 'P&L %', 'Value')
        tree = ttk.Treeview(self.positions_frame, columns=columns, show='headings', height=10)
        
        column_widths = [80, 60, 60, 80, 80, 80, 80, 90]
        for col, width in zip(columns, column_widths):
            tree.heading(col, text=col)
            tree.column(col, width=width, anchor=tk.CENTER)
        
        # Add data
        total_pnl = 0
        total_value = 0
        
        for pos in positions:
            pnl_color = self.profit_color if pos['pnl'] >= 0 else self.loss_color
            tags = ('profit',) if pos['pnl'] >= 0 else ('loss',)
            
            tree.insert('', tk.END, values=(
                pos['symbol'],
                pos['type'],
                pos['quantity'],
                f"${pos['entry_price']:.2f}",
                f"${pos['current_price']:.2f}",
                f"${pos['pnl']:+.2f}",
                f"{pos['pnl_percent']:+.2f}%",
                f"${pos['value']:.2f}"
            ), tags=tags)
            
            total_pnl += pos['pnl']
            total_value += pos['value']
        
        # Configure tags
        tree.tag_configure('profit', foreground=self.profit_color)
        tree.tag_configure('loss', foreground=self.loss_color)
        
        # Pack tree
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add summary
        summary_frame = ttk.Frame(self.positions_frame)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(summary_frame, text=f"Total P&L: ${total_pnl:+.2f}",
                 foreground=self.profit_color if total_pnl >= 0 else self.loss_color,
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=20)
        
        ttk.Label(summary_frame, text=f"Total Value: ${total_value:.2f}",
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=20)
    
    def update_analysis_tab(self):
        """Update analysis tab with performance metrics"""
        if not TKINTER_AVAILABLE:
            return
        
        # Clear frame
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()
        
        metrics = self.engine.get_performance_metrics()
        
        # Create scrollable frame
        canvas = tk.Canvas(self.analysis_frame)
        scrollbar = ttk.Scrollbar(self.analysis_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Display metrics
        row = 0
        for key, value in metrics.items():
            # Format key for display
            display_key = ' '.join(word.capitalize() for word in key.split('_'))
            
            # Format value
            if isinstance(value, float):
                if 'rate' in key or 'ratio' in key or 'percent' in key:
                    display_value = f"{value:.2f}%"
                elif 'pnl' in key or 'balance' in key or 'value' in key or 'win' in key or 'loss' in key:
                    display_value = f"${value:,.2f}"
                else:
                    display_value = f"{value:.4f}"
            else:
                display_value = str(value)
            
            # Create labels
            ttk.Label(scrollable_frame, text=display_key + ":", 
                     font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=5, padx=20)
            ttk.Label(scrollable_frame, text=display_value).grid(row=row, column=1, sticky=tk.W, pady=5)
            
            row += 1
        
        # Add recommendation
        if self.current_data is not None and not self.current_data.empty:
            latest_signal = self.current_data['Signal'].iloc[-1]
            
            ttk.Label(scrollable_frame, text="Current Recommendation:", 
                     font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=2, pady=(20, 5), padx=20)
            row += 1
            
            if latest_signal == 2:
                recommendation = "STRONG BUY - Multiple indicators show bullish momentum"
                color = self.profit_color
            elif latest_signal == 1:
                recommendation = "BUY - Bullish indicators present"
                color = self.profit_color
            elif latest_signal == -1:
                recommendation = "SELL - Bearish indicators present"
                color = self.loss_color
            elif latest_signal == -2:
                recommendation = "STRONG SELL - Multiple indicators show bearish momentum"
                color = self.loss_color
            else:
                recommendation = "HOLD - No clear signal, wait for confirmation"
                color = self.fg_color
            
            ttk.Label(scrollable_frame, text=recommendation, 
                     foreground=color, font=('Arial', 10)).grid(row=row, column=0, columnspan=2, pady=5, padx=20)
        
        # Pack scrollable area
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def update_account_display(self):
        """Update account information"""
        if not TKINTER_AVAILABLE:
            return
        
        # Update balance
        self.balance_label.config(text=f"Balance: ${self.engine.balance:,.2f}")
        
        # Calculate equity
        equity = self.engine.balance
        for symbol, quantity in self.engine.portfolio.items():
            try:
                price = self.engine.get_current_price(symbol)
                equity += price * quantity
            except:
                pass
        
        self.equity_label.config(text=f"Equity: ${equity:,.2f}")
        
        # Update P&L (simplified - today's change)
        if self.current_data is not None and len(self.current_data) > 1:
            today_return = self.current_data['Returns'].iloc[-1] * 100
            pnl_color = self.profit_color if today_return >= 0 else self.loss_color
            self.pnl_label.config(text=f"Today's Return: {today_return:+.2f}%", foreground=pnl_color)
    
    def execute_buy(self):
        """Execute buy order"""
        try:
            symbol = self.symbol_var.get()
            quantity = int(self.quantity_var.get())
            
            if self.current_data is not None:
                current_price = self.current_data['Close'].iloc[-1]
                success, message = self.engine.execute_order(
                    symbol, 'BUY', quantity, current_price
                )
                
                if success:
                    messagebox.showinfo("Success", message)
                    self.log_message(f"BUY: {message}")
                else:
                    messagebox.showerror("Error", message)
                    self.log_message(f"BUY Failed: {message}")
                
                self.refresh_data()
        except ValueError:
            messagebox.showerror("Error", "Invalid quantity")
        except Exception as e:
            messagebox.showerror("Error", f"Trade execution failed: {e}")
    
    def execute_sell(self):
        """Execute sell order"""
        try:
            symbol = self.symbol_var.get()
            quantity = int(self.quantity_var.get())
            
            if self.current_data is not None:
                current_price = self.current_data['Close'].iloc[-1]
                success, message = self.engine.execute_order(
                    symbol, 'SELL', quantity, current_price
                )
                
                if success:
                    messagebox.showinfo("Success", message)
                    self.log_message(f"SELL: {message}")
                else:
                    messagebox.showerror("Error", message)
                    self.log_message(f"SELL Failed: {message}")
                
                self.refresh_data()
        except ValueError:
            messagebox.showerror("Error", "Invalid quantity")
        except Exception as e:
            messagebox.showerror("Error", f"Trade execution failed: {e}")
    
    def on_symbol_change(self, event=None):
        """Handle symbol change"""
        self.current_symbol = self.symbol_var.get()
        self.refresh_data()
        self.log_message(f"Symbol changed to {self.current_symbol}")
    
    def on_timeframe_change(self, event=None):
        """Handle timeframe change"""
        self.refresh_data()
        self.log_message(f"Timeframe changed to {self.timeframe_var.get()}")
    
    def on_period_change(self, event=None):
        """Handle period change"""
        self.refresh_data()
        self.log_message(f"Period changed to {self.period_var.get()}")
    
    def on_watchlist_select(self, event):
        """Handle watchlist item selection"""
        selection = self.watchlist_tree.selection()
        if selection:
            item = self.watchlist_tree.item(selection[0])
            symbol = item['values'][0]
            self.symbol_var.set(symbol)
            self.on_symbol_change()
    
    def add_to_watchlist(self):
        """Add symbol to watchlist"""
        symbol = self.new_symbol_var.get().strip().upper()
        if symbol and symbol not in self.watchlist:
            self.watchlist.append(symbol)
            self.symbol_var.set(symbol)
            self.new_symbol_var.set("")
            self.refresh_data()
            self.log_message(f"Added {symbol} to watchlist")
        elif symbol in self.watchlist:
            messagebox.showinfo("Info", f"{symbol} is already in watchlist")
    
    def schedule_refresh(self):
        """Schedule next auto-refresh"""
        if self.auto_refresh and TKINTER_AVAILABLE:
            self.refresh_data()
            self.root.after(self.refresh_interval, self.schedule_refresh)
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh"""
        self.auto_refresh = not self.auto_refresh
        status = "ON" if self.auto_refresh else "OFF"
        self.log_message(f"Auto-refresh turned {status}")
        
        if self.auto_refresh:
            self.schedule_refresh()
    
    def save_portfolio(self):
        """Save portfolio to file"""
        try:
            data = {
                'balance': self.engine.balance,
                'portfolio': self.engine.portfolio,
                'positions': self.engine.positions,
                'trade_history': self.engine.trade_history,
                'watchlist': self.watchlist
            }
            
            filename = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, default=str, indent=2)
            
            self.log_message(f"Portfolio saved to {filename}")
            messagebox.showinfo("Success", f"Portfolio saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save portfolio: {e}")
    
    def load_portfolio(self):
        """Load portfolio from file"""
        try:
            from tkinter import filedialog
            filename = filedialog.askopenfilename(
                title="Load Portfolio",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Load data
                self.engine.balance = data.get('balance', self.engine.balance)
                self.engine.portfolio = data.get('portfolio', {})
                self.engine.positions = data.get('positions', [])
                self.engine.trade_history = data.get('trade_history', [])
                self.watchlist = data.get('watchlist', self.watchlist)
                
                self.log_message(f"Portfolio loaded from {filename}")
                self.refresh_data()
                messagebox.showinfo("Success", "Portfolio loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load portfolio: {e}")
    
    def open_backtester(self):
        """Open backtesting window"""
        self.log_message("Backtester opened (placeholder)")
        messagebox.showinfo("Backtester", "Backtesting feature coming soon!")
    
    def open_strategy_tester(self):
        """Open strategy tester"""
        self.log_message("Strategy tester opened (placeholder)")
        messagebox.showinfo("Strategy Tester", "Strategy testing feature coming soon!")
    
    def show_user_guide(self):
        """Show user guide"""
        guide_text = """
        PYTHON TRADING PLATFORM - USER GUIDE
        
        BASIC OPERATIONS:
        1. Select a symbol from the dropdown or watchlist
        2. Choose timeframe and period for analysis
        3. View charts and technical indicators
        4. Execute trades using BUY/SELL buttons
        
        FEATURES:
        - Real-time market data (via Yahoo Finance)
        - Technical analysis with 15+ indicators
        - Portfolio tracking and management
        - Performance analytics
        - Trade logging and history
        
        TRADING:
        - Paper trading mode only (no real money)
        - 0.1% commission on all trades
        - Supports LONG positions only
        
        TIPS:
        - Double-click watchlist items to select
        - Charts are saved as HTML files
        - Auto-refresh updates every 5 seconds
        - Save/load portfolio from File menu
        """
        
        if TKINTER_AVAILABLE:
            top = tk.Toplevel(self.root)
            top.title("User Guide")
            top.geometry("600x500")
            
            text = scrolledtext.ScrolledText(top, width=70, height=30)
            text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            text.insert(tk.END, guide_text)
            text.config(state=tk.DISABLED)
        else:
            print(guide_text)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        Python Trading Platform
        Version 1.0
        
        A MetaTrader 5 style trading platform built with Python.
        
        Features:
        - Real-time market data
        - Technical analysis
        - Portfolio management
        - Backtesting framework
        
        Built with:
        - pandas, numpy for data analysis
        - yfinance for market data
        - plotly for charting
        - tkinter for GUI
        
        Disclaimer:
        This is for educational purposes only.
        Not financial advice. Paper trading only.
        """
        
        if TKINTER_AVAILABLE:
            messagebox.showinfo("About", about_text)
        else:
            print(about_text)

# ==================== CONSOLE MODE ====================

class ConsoleTradingApp:
    """Console-based trading app for environments without GUI"""
    
    def __init__(self):
        self.engine = TradingEngine()
        self.data_manager = DataManager()
        self.current_symbol = "AAPL"
        self.running = True
        
    def display_menu(self):
        """Display console menu"""
        print("\n" + "="*60)
        print("PYTHON TRADING PLATFORM - CONSOLE MODE")
        print("="*60)
        print("1. Analyze Symbol")
        print("2. View Watchlist")
        print("3. Buy Stock")
        print("4. Sell Stock")
        print("5. View Portfolio")
        print("6. View Performance")
        print("7. Change Symbol")
        print("8. Save Portfolio")
        print("9. Load Portfolio")
        print("0. Exit")
        print("="*60)
        
        return input("\nSelect option (0-9): ")
    
    def analyze_symbol(self):
        """Analyze a symbol"""
        symbol = input("Enter symbol (default: AAPL): ").strip().upper() or "AAPL"
        
        df = self.data_manager.fetch_data(symbol, period='1mo', interval='1d')
        if df is None:
            print(f"Failed to fetch data for {symbol}")
            return
        
        df = self.engine.calculate_technical_indicators(df)
        df = self.engine.generate_trading_signals(df)
        
        latest = df.iloc[-1]
        
        print(f"\n{'='*40}")
        print(f"ANALYSIS FOR {symbol}")
        print('='*40)
        print(f"Date: {latest.name}")
        print(f"Open: ${latest['Open']:.2f}")
        print(f"High: ${latest['High']:.2f}")
        print(f"Low: ${latest['Low']:.2f}")
        print(f"Close: ${latest['Close']:.2f}")
        print(f"Volume: {latest['Volume']:,.0f}")
        print(f"Change: {latest.get('Returns', 0)*100:.2f}%")
        print("\n--- TECHNICAL INDICATORS ---")
        print(f"SMA 20: ${latest.get('SMA_20', 0):.2f}")
        print(f"SMA 50: ${latest.get('SMA_50', 0):.2f}")
        print(f"RSI: {latest.get('RSI', 0):.2f}")
        print(f"MACD: {latest.get('MACD', 0):.4f}")
        print(f"Signal: {latest.get('Signal', 0)}")
        
        signal = latest.get('Signal', 0)
        if signal == 2:
            print("\n💪 STRONG BUY SIGNAL")
        elif signal == 1:
            print("\n✅ BUY SIGNAL")
        elif signal == -1:
            print("\n⚠️ SELL SIGNAL")
        elif signal == -2:
            print("\n🔻 STRONG SELL SIGNAL")
        else:
            print("\n➡️ NEUTRAL / HOLD")
    
    def view_watchlist(self):
        """View watchlist"""
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "BTC-USD"]
        
        print(f"\n{'='*70}")
        print(f"{'Symbol':<10} {'Price':<12} {'Change':<12} {'Change %':<12} {'Volume':<20}")
        print('='*70)
        
        quotes = self.data_manager.get_live_quotes(symbols)
        
        for symbol in symbols:
            if symbol in quotes and quotes[symbol] is not None:
                quote = quotes[symbol]
                change = quote['change']
                change_percent = (change / (quote['price'] - change)) * 100 if quote['price'] - change != 0 else 0
                
                change_str = f"${change:+.2f}"
                percent_str = f"{change_percent:+.2f}%"
                
                # Color coding for terminal
                if change >= 0:
                    change_str = f"\033[92m{change_str}\033[0m"  # Green
                    percent_str = f"\033[92m{percent_str}\033[0m"
                else:
                    change_str = f"\033[91m{change_str}\033[0m"  # Red
                    percent_str = f"\033[91m{percent_str}\033[0m"
                
                print(f"{symbol:<10} ${quote['price']:<11.2f} {change_str:<12} {percent_str:<12} {quote['volume']:<20,.0f}")
            else:
                print(f"{symbol:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<20}")
    
    def buy_stock(self):
        """Buy stock in console mode"""
        symbol = input("Enter symbol to buy: ").strip().upper()
        
        try:
            quantity = int(input("Enter quantity: "))
            
            df = self.data_manager.fetch_data(symbol, period='1d', interval='1m')
            if df is None or df.empty:
                print(f"Could not get price for {symbol}")
                return
            
            price = df['Close'].iloc[-1]
            success, message = self.engine.execute_order(symbol, 'BUY', quantity, price)
            
            if success:
                print(f"\n✅ {message}")
            else:
                print(f"\n❌ {message}")
                
        except ValueError:
            print("Invalid quantity")
        except Exception as e:
            print(f"Error: {e}")
    
    def sell_stock(self):
        """Sell stock in console mode"""
        symbol = input("Enter symbol to sell: ").strip().upper()
        
        try:
            quantity = int(input("Enter quantity: "))
            
            df = self.data_manager.fetch_data(symbol, period='1d', interval='1m')
            if df is None or df.empty:
                print(f"Could not get price for {symbol}")
                return
            
            price = df['Close'].iloc[-1]
            success, message = self.engine.execute_order(symbol, 'SELL', quantity, price)
            
            if success:
                print(f"\n✅ {message}")
            else:
                print(f"\n❌ {message}")
                
        except ValueError:
            print("Invalid quantity")
        except Exception as e:
            print(f"Error: {e}")
    
    def view_portfolio(self):
        """View portfolio in console mode"""
        print(f"\n{'='*70}")
        print("PORTFOLIO SUMMARY")
        print('='*70)
        print(f"Account Balance: ${self.engine.balance:,.2f}")
        
        positions = self.engine.get_position_summary()
        
        if positions:
            print(f"\n{'='*70}")
            print(f"{'Symbol':<10} {'Type':<8} {'Qty':<8} {'Entry':<10} {'Current':<10} {'P&L':<12} {'P&L %':<10}")
            print('='*70)
            
            total_pnl = 0
            total_value = self.engine.balance
            
            for pos in positions:
                pnl_str = f"${pos['pnl']:+.2f}"
                percent_str = f"{pos['pnl_percent']:+.2f}%"
                
                if pos['pnl'] >= 0:
                    pnl_str = f"\033[92m{pnl_str}\033[0m"
                    percent_str = f"\033[92m{percent_str}\033[0m"
                else:
                    pnl_str = f"\033[91m{pnl_str}\033[0m"
                    percent_str = f"\033[91m{percent_str}\033[0m"
                
                print(f"{pos['symbol']:<10} {pos['type']:<8} {pos['quantity']:<8} "
                      f"${pos['entry_price']:<9.2f} ${pos['current_price']:<9.2f} "
                      f"{pnl_str:<12} {percent_str:<10}")
                
                total_pnl += pos['pnl']
                total_value += pos['value']
            
            print('='*70)
            total_pnl_str = f"${total_pnl:+.2f}"
            if total_pnl >= 0:
                total_pnl_str = f"\033[92m{total_pnl_str}\033[0m"
            else:
                total_pnl_str = f"\033[91m{total_pnl_str}\033[0m"
            
            print(f"Total P&L: {total_pnl_str}")
            print(f"Portfolio Value: ${total_value:,.2f}")
        else:
            print("\nNo open positions")
    
    def view_performance(self):
        """View performance metrics"""
        metrics = self.engine.get_performance_metrics()
        
        print(f"\n{'='*60}")
        print("PERFORMANCE METRICS")
        print('='*60)
        
        for key, value in metrics.items():
            display_key = ' '.join(word.capitalize() for word in key.split('_'))
            
            if isinstance(value, float):
                if 'rate' in key or 'ratio' in key or 'percent' in key:
                    print(f"{display_key:<25} {value:>8.2f}%")
                elif 'pnl' in key or 'balance' in key or 'value' in key:
                    print(f"{display_key:<25} ${value:>8.2f}")
                else:
                    print(f"{display_key:<25} {value:>8.4f}")
            else:
                print(f"{display_key:<25} {value:>8}")
    
    def change_symbol(self):
        """Change current symbol"""
        symbol = input("Enter new symbol: ").strip().upper()
        self.current_symbol = symbol
        print(f"Current symbol changed to {symbol}")
    
    def save_portfolio_console(self):
        """Save portfolio in console mode"""
        try:
            data = {
                'balance': self.engine.balance,
                'portfolio': self.engine.portfolio,
                'positions': self.engine.positions,
                'trade_history': self.engine.trade_history
            }
            
            filename = input("Enter filename (default: portfolio.json): ").strip()
            if not filename:
                filename = "portfolio.json"
            
            with open(filename, 'w') as f:
                json.dump(data, f, default=str, indent=2)
            
            print(f"✅ Portfolio saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save portfolio: {e}")
    
    def load_portfolio_console(self):
        """Load portfolio in console mode"""
        try:
            filename = input("Enter filename to load: ").strip()
            if not filename:
                filename = "portfolio.json"
            
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Load data
            self.engine.balance = data.get('balance', self.engine.balance)
            self.engine.portfolio = data.get('portfolio', {})
            self.engine.positions = data.get('positions', [])
            self.engine.trade_history = data.get('trade_history', [])
            
            print(f"✅ Portfolio loaded from {filename}")
        except FileNotFoundError:
            print(f"❌ File {filename} not found")
        except Exception as e:
            print(f"❌ Failed to load portfolio: {e}")
    
    def run(self):
        """Run console application"""
        print("\n" + "="*60)
        print("WELCOME TO PYTHON TRADING PLATFORM")
        print("="*60)
        print("Starting with $10,000 paper trading account")
        print("Type 'help' for commands or '0' to exit")
        print("="*60)
        
        while self.running:
            choice = self.display_menu()
            
            if choice == '1':
                self.analyze_symbol()
            elif choice == '2':
                self.view_watchlist()
            elif choice == '3':
                self.buy_stock()
            elif choice == '4':
                self.sell_stock()
            elif choice == '5':
                self.view_portfolio()
            elif choice == '6':
                self.view_performance()
            elif choice == '7':
                self.change_symbol()
            elif choice == '8':
                self.save_portfolio_console()
            elif choice == '9':
                self.load_portfolio_console()
            elif choice == '0':
                self.running = False
                print("\nThank you for using the Trading Platform!")
                print(f"Final Balance: ${self.engine.balance:,.2f}")
            elif choice.lower() == 'help':
                print("\nQuick Help:")
                print("- Use option 1 to analyze stocks")
                print("- Use option 3/4 to buy/sell")
                print("- Use option 5 to view portfolio")
                print("- Start with analyzing AAPL or MSFT")
            else:
                print("Invalid choice. Please select 0-9 or 'help'.")
            
            if choice != '0':
                input("\nPress Enter to continue...")

# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point for the application"""
    print("\n" + "="*70)
    print("PYTHON TRADING PLATFORM - META TRADER 5 STYLE")
    print("="*70)
    
    # Check if we should run in GUI or console mode
    if TKINTER_AVAILABLE:
        print("GUI mode available.")
        mode = input("Run in (G)UI or (C)onsole mode? (G/C): ").strip().upper()
    else:
        print("GUI not available. Running in console mode.")
        mode = "C"
    
    if mode == "G" and TKINTER_AVAILABLE:
        try:
            root = tk.Tk()
            app = TradingAppGUI(root)
            print("\nStarting GUI application...")
            print("Check the opened window for the trading interface.")
            print("Charts will be saved as HTML files in the current directory.")
            root.mainloop()
        except Exception as e:
            print(f"GUI failed to start: {e}")
            print("Falling back to console mode...")
            app = ConsoleTradingApp()
            app.run()
    else:
        app = ConsoleTradingApp()
        app.run()

# ==================== BACKTESTING MODULE ====================

class Backtester:
    """Backtesting module for strategy testing"""
    
    @staticmethod
    def run_backtest(df, initial_capital=10000, commission=0.001):
        """Run a simple backtest on historical data"""
        print("\n" + "="*60)
        print("RUNNING BACKTEST")
        print("="*60)
        
        if df.empty:
            print("No data available for backtesting")
            return
        
        # Calculate signals
        engine = TradingEngine()
        df = engine.calculate_technical_indicators(df)
        df = engine.generate_trading_signals(df)
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            signal = df['Signal'].iloc[i]
            
            # Buy signal
            if signal > 0 and position == 0:
                position = capital / current_price
                capital = 0
                trades.append({
                    'type': 'BUY',
                    'date': df.index[i],
                    'price': current_price,
                    'shares': position
                })
            
            # Sell signal
            elif signal < 0 and position > 0:
                capital = position * current_price * (1 - commission)
                trades.append({
                    'type': 'SELL',
                    'date': df.index[i],
                    'price': current_price,
                    'shares': position,
                    'value': capital
                })
                position = 0
            
            # Calculate equity
            equity = capital + (position * current_price)
            equity_curve.append(equity)
        
        # Calculate final value
        if position > 0:
            final_value = position * df['Close'].iloc[-1] * (1 - commission)
        else:
            final_value = capital
        
        # Calculate metrics
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        total_trades = len(trades) // 2
        
        # Calculate Sharpe ratio
        returns = pd.Series(equity_curve).pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        
        if total_trades > 0:
            print(f"Avg Return per Trade: {total_return/total_trades:.2f}%")
        
        print("="*60)
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': total_trades,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades
        }

# ==================== STRATEGY LIBRARY ====================

class StrategyLibrary:
    """Collection of trading strategies"""
    
    @staticmethod
    def moving_average_crossover(df, short_window=20, long_window=50):
        """Moving average crossover strategy"""
        df = df.copy()
        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
        
        df['Signal'] = 0
        df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
        df.loc[df['SMA_Short'] < df['SMA_Long'], 'Signal'] = -1
        
        return df
    
    @staticmethod
    def mean_reversion(df, window=20, num_std=2):
        """Bollinger Bands mean reversion strategy"""
        df = df.copy()
        df['SMA'] = df['Close'].rolling(window=window).mean()
        df['STD'] = df['Close'].rolling(window=window).std()
        df['Upper'] = df['SMA'] + (df['STD'] * num_std)
        df['Lower'] = df['SMA'] - (df['STD'] * num_std)
        
        df['Signal'] = 0
        df.loc[df['Close'] < df['Lower'], 'Signal'] = 1  # Buy when below lower band
        df.loc[df['Close'] > df['Upper'], 'Signal'] = -1  # Sell when above upper band
        
        return df
    
    @staticmethod
    def rsi_strategy(df, oversold=30, overbought=70):
        """RSI-based strategy"""
        df = df.copy()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Signal'] = 0
        df.loc[df['RSI'] < oversold, 'Signal'] = 1  # Buy when oversold
        df.loc[df['RSI'] > overbought, 'Signal'] = -1  # Sell when overbought
        
        return df

# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    print("\nInitializing Trading Platform...")
    print("This may take a moment to install dependencies if needed.\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")
        print("Please report this issue.")
    finally:
        print("\nThank you for using the Python Trading Platform!")
