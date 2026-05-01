"""
Dynamic Position Sizing using Kelly Criterion
Improves risk-adjusted returns by sizing positions based on:
1. Model confidence (probability of winning)
2. Recent win rate (last N trades)
3. Market volatility (ATR-based)
4. Max risk per trade (2% of account)
"""

import numpy as np

class DynamicPositionSizer:
    """
    Kelly Criterion-based position sizing with safeguards
    """
    
    def __init__(self, max_risk_pct=0.02, max_position_pct=0.10, lookback_trades=10):
        """
        Args:
            max_risk_pct: Max account risk per trade (2% default)
            max_position_pct: Max position size as % of account (10% default)
            lookback_trades: Number of recent trades to calculate win rate
        """
        self.max_risk_pct = max_risk_pct
        self.max_position_pct = max_position_pct
        self.lookback_trades = lookback_trades
        self.trade_history = []  # List of {'pnl_pct': float, 'confidence': float}
        
    def calculate_position_size(self, account_balance, current_price, stop_loss_pct, 
                                confidence, atr_pct=None):
        """
        Calculate dynamic position size
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            stop_loss_pct: Stop loss as decimal (e.g., 0.015 for 1.5%)
            confidence: Model confidence (0.0 to 1.0)
            atr_pct: ATR as percentage of price (optional, for volatility adjustment)
        
        Returns:
            position_size: Number of units to trade
            risk_amount: Dollar amount at risk
        """
        # 1. Calculate historical win rate (last N trades)
        recent_trades = self.trade_history[-self.lookback_trades:] if self.trade_history else []
        if recent_trades:
            wins = sum(1 for t in recent_trades if t['pnl_pct'] > 0)
            win_rate = wins / len(recent_trades)
        else:
            win_rate = 0.5  # Default 50% if no history
        
        # 2. Calculate average win/loss ratio from history
        if recent_trades:
            wins = [t['pnl_pct'] for t in recent_trades if t['pnl_pct'] > 0]
            losses = [abs(t['pnl_pct']) for t in recent_trades if t['pnl_pct'] < 0]
            avg_win = np.mean(wins) if wins else 0.02
            avg_loss = np.mean(losses) if losses else 0.015
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 2.0
        else:
            win_loss_ratio = 1.5  # Default 1.5:1
            avg_win = 0.03
            avg_loss = 0.02
        
        # 3. Kelly Criterion: f* = (p * b - q) / b
        #    p = win probability, q = loss probability, b = win/loss ratio
        p = win_rate
        q = 1 - p
        b = win_loss_ratio
        
        kelly_fraction = (p * b - q) / b if b > 0 else 0
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% (safer than full Kelly)
        
        # 4. Adjust by confidence (scale Kelly by confidence)
        confidence_multiplier = np.clip(confidence, 0.5, 1.0)  # Between 0.5x and 1.0x
        adjusted_kelly = kelly_fraction * confidence_multiplier
        
        # 5. Adjust by volatility (if ATR provided)
        if atr_pct:
            # Higher volatility = smaller position
            volatility_factor = np.clip(0.02 / atr_pct, 0.5, 2.0)  # Normalize to 2% baseline
            adjusted_kelly *= volatility_factor
        
        # 6. Calculate position size
        risk_amount = account_balance * adjusted_kelly * self.max_risk_pct
        position_size = risk_amount / (current_price * stop_loss_pct)
        
        # 7. Apply max position size limit
        max_position_value = account_balance * self.max_position_pct
        max_position_size = max_position_value / current_price
        position_size = min(position_size, max_position_size)
        
        # 8. Minimum position size (0.01 lots for forex)
        min_position_size = 100  # $10,000 / current_price (approx 0.01 lots)
        position_size = max(position_size, min_position_size)
        
        return int(position_size), risk_amount
    
    def record_trade(self, pnl_pct, confidence):
        """Record trade result for future sizing decisions"""
        self.trade_history.append({
            'pnl_pct': pnl_pct,
            'confidence': confidence
        })
        # Keep only last 50 trades
        if len(self.trade_history) > 50:
            self.trade_history = self.trade_history[-50:]
    
    def get_stats(self):
        """Get current statistics"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 50.0,
                'avg_win': 3.0,
                'avg_loss': 2.0,
                'kelly_fraction': 0.0
            }
        
        recent = self.trade_history[-self.lookback_trades:]
        wins = [t for t in recent if t['pnl_pct'] > 0]
        losses = [t for t in recent if t['pnl_pct'] < 0]
        
        return {
            'total_trades': len(self.trade_history),
            'recent_trades': len(recent),
            'win_rate': len(wins) / len(recent) * 100 if recent else 50,
            'avg_win': np.mean([t['pnl_pct'] for t in wins]) * 100 if wins else 0,
            'avg_loss': np.mean([abs(t['pnl_pct']) for t in losses]) * 100 if losses else 0,
            'kelly_fraction': self._current_kelly()
        }
    
    def _current_kelly(self):
        """Calculate current Kelly fraction based on history"""
        recent = self.trade_history[-self.lookback_trades:]
        if not recent:
            return 0.0
        
        wins = sum(1 for t in recent if t['pnl_pct'] > 0)
        p = wins / len(recent)
        q = 1 - p
        
        wins_pct = [t['pnl_pct'] for t in recent if t['pnl_pct'] > 0]
        losses_pct = [abs(t['pnl_pct']) for t in recent if t['pnl_pct'] < 0]
        
        b = np.mean(wins_pct) / np.mean(losses_pct) if losses_pct else 2.0
        
        kelly = (p * b - q) / b if b > 0 else 0
        return max(0, min(kelly * 100, 25))  # Return as percentage, capped at 25%


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Dynamic Position Sizing - Kelly Criterion Test")
    print("="*60)
    
    sizer = DynamicPositionSizer(max_risk_pct=0.02, max_position_pct=0.10)
    
    # Simulate some trades
    np.random.seed(42)
    account_balance = 10000
    
    print("\nSimulating 20 trades...")
    for i in range(20):
        # Generate fake trade
        confidence = 0.7 + np.random.rand() * 0.25  # 0.7 - 0.95
        is_win = np.random.rand() < 0.6  # 60% win rate
        pnl_pct = (np.random.rand() * 0.04 + 0.01) if is_win else -(np.random.rand() * 0.02 + 0.005)
        
        # Record trade
        sizer.record_trade(pnl_pct, confidence)
        
        # Calculate position size for next trade
        current_price = 1.1000 + np.random.rand() * 0.01
        stop_loss_pct = 0.015
        
        size, risk = sizer.calculate_position_size(
            account_balance, current_price, stop_loss_pct, confidence
        )
        
        if (i+1) % 5 == 0:
            stats = sizer.get_stats()
            print(f"\nAfter {i+1} trades:")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Kelly Fraction: {stats['kelly_fraction']:.1f}%")
            print(f"  Next Position Size: {size} units")
            print(f"  Risk Amount: ${risk:.2f}")
    
    print("\n" + "="*60)
    print("Final Statistics:")
    stats = sizer.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")
    print("="*60)
