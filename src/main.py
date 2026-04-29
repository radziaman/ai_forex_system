"""
Institutional Forex AI Moneybot - Main Entry Point
Built with Python, Neural Networks, ML, and AI.
Integrates cTrader Open API (from C# COpenAPIClient.cs patterns).
"""

import asyncio
import signal
import sys
import os
import yaml
from loguru import logger
from typing import Optional
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.ctrader_client import CtraderClient, MarketDepth
from src.data.data_manager import DataManager
from src.ai.rl_agent import PPOAgent, TradingEnvironment
from src.risk.manager import RiskManager, RiskParameters
from src.execution.engine import ExecutionEngine
from src.dashboard.app import app, broadcast_update, update_state
import uvicorn


class InstitutionalForexMoneybot:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        logger.remove()
        logger.add(
            sys.stdout, level=self.config.get("logging", {}).get("level", "INFO")
        )
        logger.add("data/logs/moneybot.log", rotation="100 MB", retention="30 days")

        self._init_ctrader()
        self.data_manager = DataManager(
            historical_path=self.config["data"]["historical_path"],
            redis_url=self.config["data"].get("redis_url"),
        )
        self.risk_manager = RiskManager(
            RiskParameters(
                max_risk_per_trade=self.config["trading"]["max_risk_per_trade"],
                max_drawdown=self.config["trading"]["max_drawdown"],
                max_margin_usage=self.config["trading"]["max_margin_usage"],
                min_win_rate_live=self.config["trading"]["min_win_rate_live"],
                min_win_rate_paper=self.config["trading"]["min_win_rate_paper"],
            )
        )

        self.execution = ExecutionEngine(
            self.ctrader, self.risk_manager, self.data_manager
        )
        self.ai_agent = None
        self.trading_env = None
        self.feature_dim = 0
        self.is_running = False
        self.trade_count = 0
        self.last_ai_action = "HOLD"
        self.last_confidence = 0.0

        logger.success("=" * 60)
        logger.success("  INSTITUTIONAL FOREX AI MONEYBOT v3.0")
        logger.success("  Powered by Python, RL, Neural Networks & cTrader")
        logger.success("=" * 60)

    def _init_ctrader(self):
        ct = self.config["ctrader"]
        self.ctrader = CtraderClient(
            app_id=ct["app_id"],
            app_secret=ct["app_secret"],
            access_token=ct["access_token"],
            account_id=ct["account_id"],
            demo=ct["demo"],
            use_websocket=ct["use_websocket"],
        )
        self.ctrader.on_market_data = self._on_market_data
        self.ctrader.on_account_update = self._on_account_update
        self.ctrader.on_order_update = self._on_order_update
        self.ctrader.on_positions_update = self.execution._on_positions_update

    async def start(self):
        logger.info("Starting Institutional Forex AI Moneybot...")
        self.ctrader.start()
        await asyncio.sleep(3)

        logger.info("Loading historical data...")
        for tf in ["1m", "5m", "15m", "1h", "4h"]:
            self.data_manager.load_historical(tf, days=30)

        self.feature_dim = self.data_manager.get_feature_dim()
        logger.info(f"Feature dimension: {self.feature_dim}")

        self.ai_agent = PPOAgent(
            state_dim=self.feature_dim,
            n_actions=5,
            lr=self.config["ai"]["learning_rate"],
            gamma=self.config["ai"]["gamma"],
            gae_lambda=self.config["ai"]["gae_lambda"],
            clip_range=self.config["ai"]["clip_range"],
            ent_coef=self.config["ai"]["ent_coef"],
            vf_coef=self.config["ai"]["vf_coef"],
            device=self.config["ai"]["device"],
        )

        model_path = f"{self.config['data']['models_path']}/ppo_latest.pt"
        if os.path.exists(model_path):
            self.ai_agent.load(model_path)
            logger.info(f"Loaded model: {model_path}")

        self.trading_env = TradingEnvironment(
            state_dim=self.feature_dim,
            initial_balance=100000.0,
            spread_pips=self.config["trading"]["slippage_pips"],
            commission_per_lot=self.config["trading"]["commission_per_lot"],
        )

        d_cfg = self.config["dashboard"]
        threading.Thread(
            target=lambda: uvicorn.run(
                app, host=d_cfg["host"], port=d_cfg["port"], log_level="error"
            ),
            daemon=True,
        ).start()
        logger.info(f"Dashboard: http://{d_cfg['host']}:{d_cfg['port']}")

        acc_info = self.execution.get_account_info()
        self.risk_manager.reset_daily_stats(acc_info.get("balance", 100000))
        self.is_running = True
        logger.success("✅ Moneybot LIVE!")

        await self._main_loop()

    async def _main_loop(self):
        while self.is_running:
            try:
                await self._trading_cycle()
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(5)

    async def _trading_cycle(self):
        acc_info = self.execution.get_account_info()
        open_pos = self.execution.get_open_positions()
        snapshot = self.data_manager.get_snapshot(acc_info=acc_info, positions=open_pos)

        if snapshot is None or snapshot.features is None:
            return

        action, sl_mult, tp_mult, pos_size, ai_info = self.ai_agent.select_action(
            snapshot.features
        )

        atr = self._extract_atr(snapshot.features)
        sl_price = atr * sl_mult
        tp_price = atr * tp_mult

        confidence = abs(ai_info.get("value", 0.0))
        volume = self.risk_manager.calculate_position_size(
            balance=acc_info.get("balance", 100000),
            price=snapshot.bid if snapshot.bid > 0 else 1.1200,
            atr=atr,
            confidence=confidence,
        )

        action_map = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE_ALL", 4: "MODIFY"}
        action_str = action_map.get(action, "HOLD")

        if action == 1:
            await self.execution.open_position(
                "EURUSD", "BUY", volume, sl_price, tp_price, str(ai_info)
            )
        elif action == 2:
            await self.execution.open_position(
                "EURUSD", "SELL", volume, sl_price, tp_price, str(ai_info)
            )
        elif action == 3:
            await self.execution.close_all_positions("AI close all")
        elif action == 4:
            pass

        self.last_ai_action = action_str
        self.last_confidence = confidence

        if (
            self.trade_count % self.config["ai"]["finetune_freq"] == 0
            and self.trade_count > 0
        ):
            self._train_ai(snapshot.features, action, 0.0)

        self._update_dashboard(snapshot, acc_info, open_pos)
        self.trade_count += 1

    def _train_ai(self, state, action, reward):
        if self.ai_agent and self.trading_env:
            self.ai_agent.store_transition(reward, False)
            if len(self.ai_agent.states) >= 2048:
                metrics = self.ai_agent.train(n_epochs=10, batch_size=64)
                logger.info(f"AI train | Loss: {metrics.get('policy_loss',0):.4f}")
                self.ai_agent.save(
                    f"{self.config['data']['models_path']}/ppo_latest.pt"
                )

    def _update_dashboard(self, snapshot, acc_info, open_pos):
        update_state(
            balance=acc_info.get("balance", 100000),
            equity=acc_info.get("equity", 100000),
            margin=acc_info.get("margin", 0),
            free_margin=acc_info.get("free_margin", 100000),
            initial_balance=100000,
            total_trades=self.trade_count,
            win_rate=self.risk_manager.get_win_rate(),
            mode=self.risk_manager.mode,
            regime=snapshot.regime if snapshot else "unknown",
            open_positions=open_pos,
            trade_history=self.execution.get_trade_history(20),
            market_data=(
                {"bid": snapshot.bid, "ask": snapshot.ask, "spread": snapshot.spread}
                if snapshot
                else {}
            ),
            ai_metrics={
                "regime": snapshot.regime if snapshot else "",
                "confidence": self.last_confidence,
                "action": self.last_ai_action,
            },
        )
        try:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(broadcast_update(latest_state), loop)
        except:
            pass

    def _extract_atr(self, features):
        return abs(features[4]) + 0.001 if len(features) > 4 else 0.001

    def _on_market_data(self, depth):
        self.data_manager.update_tick(depth.symbol, depth.bid, depth.ask, depth.volume)

    def _on_account_update(self, acc_info):
        pass

    def _on_order_update(self, result):
        pass

    def stop(self):
        logger.info("Stopping Moneybot...")
        self.is_running = False
        asyncio.run(self.execution.close_all_positions("Shutdown"))
        self.ctrader.disconnect()
        if self.ai_agent:
            self.ai_agent.save(f"{self.config['data']['models_path']}/ppo_final.pt")
        logger.success("Moneybot stopped.")


def signal_handler(sig, frame):
    if "bot" in globals():
        bot.stop()


if __name__ == "__main__":
    import numpy as np

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("data/trades", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    bot = InstitutionalForexMoneybot("config.yaml")
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        bot.stop()
