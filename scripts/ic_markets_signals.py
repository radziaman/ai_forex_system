"""
IC Markets cTrader Signal Generator
生成交易信号，供您在IC Markets平台手动执行
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_forex_system.data import DataFetcher
from ai_forex_system.features import FeatureEngineer
import tensorflow as tf


# flake8: noqa: E402

# IC Markets 支持的交易品种
IC_MARKETS_PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "XAUUSD=X",  # 黄金
    "AUDUSD": "AUDUSD=X",
}


def generate_signals(pairs=["AUDUSD"], timeframe="1h"):
    """生成交易信号"""

    print("=" * 60)
    print("  IC Markets cTrader Signal Generator")
    print("  生成时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    # 加载模型
    print("\n正在加载训练好的模型...")
    try:
        model = tf.keras.models.load_model("models/lstm_cnn_model.keras")
        print("✓ 模型加载成功!")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("请先训练模型: python main.py train")
        return

    # 初始化
    feature_engineer = FeatureEngineer(lookback=30)
    data_fetcher = DataFetcher(source="yfinance")

    signals = []

    for pair_name, symbol in pairs.items():
        print(f"\n{'='*60}")
        print(f"品种: {pair_name} ({symbol})")
        print(f"{'='*60}")

        try:
            # 获取最新数据
            print(f"正在获取{timeframe}数据...")
            df = data_fetcher.fetch_ohlcv(symbol, timeframe, "2024-01-01")

            if len(df) < 30:
                print(f"✗ 数据不足30根K线")
                continue

            # 生成特征
            df = feature_engineer.generate_all_features(df)

            # 获取最近30根K线用于预测
            window = df.iloc[-30:]
            features = window.values.reshape(1, 30, window.shape[1])

            # 模型预测
            prediction = model.predict(features, verbose=0)[0][0]
            current_price = window["close"].iloc[-1]

            # 计算信号
            price_diff = (prediction - current_price) / current_price * 100

            print(f"当前价格: {current_price:.5f}")
            print(f"预测价格: {prediction:.5f}")
            print(f"预期变化: {price_diff:+.3f}%")

            # 生成交易信号
            if price_diff > 0.05:  # 预测上涨超过0.05%
                signal = "BUY"
                confidence = min(price_diff * 10, 100)  # 简化置信度计算
                print(f"\n✓ 信号: {signal}")
                print(f"  入场价: ~{current_price:.5f}")
                print(f"  止损(SL): ~{current_price * 0.995:.5f} (50点)")
                print(f"  止盈(TP): ~{current_price * 1.01:.5f} (100点)")
                print(f"  置信度: {confidence:.1f}%")

                signals.append(
                    {
                        "pair": pair_name,
                        "symbol": symbol,
                        "signal": signal,
                        "entry": current_price,
                        "sl": current_price * 0.995,
                        "tp": current_price * 1.01,
                        "confidence": confidence,
                    }
                )

            elif price_diff < -0.05:  # 预测下跌超过0.05%
                signal = "SELL"
                confidence = min(abs(price_diff) * 10, 100)
                print(f"\n✓ 信号: {signal}")
                print(f"  入场价: ~{current_price:.5f}")
                print(f"  止损(SL): ~{current_price * 1.005:.5f} (50点)")
                print(f"  止盈(TP): ~{current_price * 0.99:.5f} (100点)")
                print(f"  置信度: {confidence:.1f}%")

                signals.append(
                    {
                        "pair": pair_name,
                        "symbol": symbol,
                        "signal": signal,
                        "entry": current_price,
                        "sl": current_price * 1.005,
                        "tp": current_price * 0.99,
                        "confidence": confidence,
                    }
                )
            else:
                print(f"\n- 信号: HOLD (预期变化太小)")

        except Exception as e:
            print(f"✗ 错误: {e}")
            continue

    # 输出总结
    print(f"\n{'='*60}")
    print("  交易信号总结")
    print(f"{'='*60}")

    if signals:
        print(f"\n找到 {len(signals)} 个交易信号:\n")
        for i, sig in enumerate(signals, 1):
            print(f"{i}. {sig['pair']} - {sig['signal']}")
            print(f"   入场: {sig['entry']:.5f}")
            print(f"   SL: {sig['sl']:.5f} | TP: {sig['tp']:.5f}")
            print(f"   置信度: {sig['confidence']:.1f}%")
            print()

        print("=" * 60)
        print("在IC Markets cTrader平台执行:")
        print("1. 登录您的IC Markets cTrader账户")
        print("2. 打开交易窗口 (F9)")
        print("3. 选择对应品种")
        print("4. 设置入场价、止损、止盈")
        print("5. 确认并执行交易")
        print("=" * 60)
    else:
        print("\n当前无交易信号 (所有品种都是HOLD)")
        print("建议: 等待更好的入场时机或调整阈值")

    return signals


if __name__ == "__main__":
    # 配置要监控的品种
    pairs_to_monitor = {
        "AUDUSD": "AUDUSD=X",
        "EURUSD": "EURUSD=X",
    }

    # 生成信号
    signals = generate_signals(pairs=pairs_to_monitor, timeframe="1h")

    # 可选：保存信号到文件
    if signals:
        import json

        with open(f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "w") as f:
            json.dump(signals, f, indent=2)
        print(f"\n信号已保存到文件")
