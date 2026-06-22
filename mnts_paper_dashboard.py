from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, render_template_string

from breakout_v2_parameter_stability_4y import ParamConfig, prepare_breakout_columns, simulate_breakout_param
from engineer_pullback_breakout_v2 import gate_passed, state_streaks, strategy_grid
from mnts_15m_state_layer_v2 import apply_v2_model, fit_v2_model, prepare_features
from mnts_min_validation import KronosTokenizerAdapter, load_ohlcv_csv
from rolling_breakout_v2_full_system_walkforward import valid_training_rows


ROOT = Path(__file__).resolve().parent
BINANCE_TICKER_24H_URL = "https://api.binance.com/api/v3/ticker/24hr"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
PAPER_RUNTIME_DIR = ROOT / "runtime" / "paper_trading"

REFRESH_MS = 10_000
LIVE_SIGNAL_REFRESH_SECONDS = 60
LIVE_KLINE_LIMIT = 320
PAPER_RUNNER_MIN_SLEEP_SECONDS = 5.0
PAPER_RUNNER_MAX_SLEEP_SECONDS = 30.0
KRONOS_REPO_DIR = ROOT / "external" / "Kronos"
KRONOS_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_CHUNK_SIZE = 256
DEFAULT_FEE_BPS = 4.0
DEFAULT_SLIPPAGE_BPS = 1.0
BAR_INTERVAL = pd.Timedelta(minutes=15)
TRAIN_BARS = 11520
TEST_BARS = 2880
HISTORY_BOOTSTRAP_LOOKBACK_DAYS = 365 * 4 + 7
HISTORICAL_KLINE_BATCH_LIMIT = 1000
_TOKENIZER_ADAPTER: Optional[KronosTokenizerAdapter] = None
_LIVE_SIGNAL_CACHE: Dict[str, Any] = {
    "expires_at": 0.0,
    "updated_at": None,
    "snapshots": {},
    "refreshing": False,
    "last_error": None,
}
_LOCAL_HISTORY_CACHE: Dict[str, Any] = {}
_API_STATE_CACHE: Dict[str, Any] = {"state": None, "updated_at": None, "last_error": None}
_API_STATE_CACHE_LOCK = threading.Lock()
_PAPER_RUNNER_THREAD: Optional[threading.Thread] = None

SYMBOL_CONFIGS: List[Dict[str, Any]] = [
    {
        "symbol": "BTCUSDT",
        "anchor": "bw48_cohesion_hold8_cool0",
        "display_name": "BTC",
        "input_csv": ROOT / "data" / "btcusdt_15m_4y.csv",
        "summary_path": ROOT / "validation_outputs" / "btc_anchor_trade_outcomes_nextopen_1000u_2x" / "trade_outcomes_summary.json",
        "trades_path": ROOT / "validation_outputs" / "btc_anchor_trade_outcomes_nextopen_1000u_2x" / "trade_outcomes_fee4.csv",
    },
    {
        "symbol": "ETHUSDT",
        "anchor": "bw40_nonfracture_hold12_cool0",
        "display_name": "ETH",
        "input_csv": ROOT / "data" / "ethusdt_15m_4y.csv",
        "summary_path": ROOT / "validation_outputs" / "eth_anchor_trade_outcomes_nextopen_1000u_2x" / "trade_outcomes_summary.json",
        "trades_path": ROOT / "validation_outputs" / "eth_anchor_trade_outcomes_nextopen_1000u_2x" / "trade_outcomes_fee4.csv",
    },
    {
        "symbol": "SOLUSDT",
        "anchor": "bw40_cohesion_hold8_cool4",
        "display_name": "SOL",
        "input_csv": ROOT / "data" / "solusdt_15m_4y.csv",
        "summary_path": ROOT / "validation_outputs" / "sol_anchor_trade_outcomes_nextopen_1000u_2x" / "trade_outcomes_summary.json",
        "trades_path": ROOT / "validation_outputs" / "sol_anchor_trade_outcomes_nextopen_1000u_2x" / "trade_outcomes_fee4.csv",
    },
    {
        "symbol": "XRPUSDT",
        "anchor": "bw64_nonfracture_hold16_cool4",
        "display_name": "XRP",
        "input_csv": ROOT / "data" / "xrpusdt_15m_4y.csv",
        "summary_path": ROOT / "validation_outputs" / "xrp_anchor_trade_outcomes_nextopen_1000u_2x" / "trade_outcomes_summary.json",
        "trades_path": ROOT / "validation_outputs" / "xrp_anchor_trade_outcomes_nextopen_1000u_2x" / "trade_outcomes_fee4.csv",
    },
]

app = Flask(__name__)


TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MNTS 纸上实盘看板</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f172a;
      --panel: #111827;
      --panel-soft: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --line: #374151;
      --green: #10b981;
      --red: #ef4444;
      --amber: #f59e0b;
      --blue: #3b82f6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .wrap {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }
    h1, h2, h3 { margin: 0 0 12px; }
    .muted { color: var(--muted); }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 20px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
    }
    .panel.soft { background: var(--panel-soft); }
    .metric-label {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .metric-value {
      font-size: 24px;
      font-weight: 700;
      line-height: 1.2;
    }
    .metric-sub {
      font-size: 13px;
      color: var(--muted);
      margin-top: 6px;
    }
    .cards-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }
    .symbol-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 12px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 700;
      background: #1e3a8a;
      color: #dbeafe;
    }
    .green { color: var(--green); }
    .red { color: var(--red); }
    .amber { color: var(--amber); }
    .symbol-meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .meta-box {
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.05);
      border-radius: 10px;
      padding: 10px;
    }
    .meta-box .k {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .meta-box .v {
      font-size: 15px;
      font-weight: 700;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }
    th, td {
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .table-wrap {
      overflow-x: auto;
    }
    .footer-note {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <h1>MNTS 纸上实盘看板</h1>
        <div class="muted">BTC + ETH + SOL + XRP | 收盘发信号，下根开盘成交 | fee4 + slippage1 | 每币 1000U | 2 倍杠杆</div>
      </div>
      <div class="panel soft">
        <div class="metric-label">自动刷新</div>
        <div class="metric-value" id="refreshLabel">10s</div>
        <div class="metric-sub">最近更新时间：<span id="updatedAt">-</span></div>
      </div>
    </div>

    <div class="panel soft" style="margin-bottom: 16px;">
      <div class="metric-label">当前版本说明</div>
      <div class="metric-sub">上半部分展示历史回测基线；下半部分展示最新确认的 15 分钟纸面状态。当前实时层改为 180 天训练、45 天冻结运行，到期后再整体滚动重训。</div>
    </div>

    <h2>历史回测基线</h2>
    <div class="grid">
      <div class="panel">
        <div class="metric-label">历史基线初始本金</div>
        <div class="metric-value" id="portfolioInitial">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">历史基线终局权益</div>
        <div class="metric-value" id="portfolioFinal">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">历史累计净盈亏</div>
        <div class="metric-value" id="portfolioPnl">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">历史复利收益</div>
        <div class="metric-value" id="portfolioReturn">-</div>
      </div>
    </div>

    <h2>最新确认 15 分钟纸面状态</h2>
    <div class="grid">
      <div class="panel">
        <div class="metric-label">最新确认 K 线时间</div>
        <div class="metric-value" id="liveLastBar">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">当前持仓数量</div>
        <div class="metric-value" id="liveActiveCount">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">实时估算总权益</div>
        <div class="metric-value" id="liveMarkedEquity">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">当前多空分布</div>
        <div class="metric-value" id="liveExposureMix">-</div>
      </div>
    </div>

    <div class="grid">
      <div class="panel">
        <div class="metric-label">实时账户初始本金</div>
        <div class="metric-value" id="rtInitialCapital">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">实时账户已实现权益</div>
        <div class="metric-value" id="rtRealizedEquity">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">实时账户已实现净盈亏</div>
        <div class="metric-value" id="rtRealizedPnl">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">实时账户当前浮盈亏</div>
        <div class="metric-value" id="rtFloatingPnl">-</div>
      </div>
    </div>

    <div id="symbolCards" class="cards-2"></div>

    <div class="panel">
      <h2>实时信号流水</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>交易对</th>
              <th>动作</th>
              <th>信号时间</th>
              <th>成交时间</th>
              <th>成交价</th>
              <th>仓位变化</th>
              <th>状态</th>
              <th>突破</th>
              <th>处理后权益</th>
            </tr>
          </thead>
          <tbody id="recentSignalsTable"></tbody>
        </table>
      </div>
      <div class="footer-note">信号流水来自本地连续纸上账户账本，保存在 `runtime/paper_trading/` 目录下。</div>
    </div>

    <div class="panel">
      <h2>实时纸面已平仓交易</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>交易对</th>
              <th>方向</th>
              <th>开仓时间</th>
              <th>平仓时间</th>
              <th>开仓权益</th>
              <th>平仓权益</th>
              <th>净盈亏</th>
              <th>净收益率</th>
              <th>持仓 Bar 数</th>
            </tr>
          </thead>
          <tbody id="recentTradesTable"></tbody>
        </table>
      </div>
      <div class="footer-note">这里展示的是实时纸面账户运行过程中已经完成闭环的交易，不再混用历史回测结果。</div>
    </div>
  </div>

  <script>
    const REFRESH_MS = {{ refresh_ms }};

    function fmtUsd(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return `$${Number(value).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    }

    function fmtPct(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return `${(Number(value) * 100).toFixed(2)}%`;
    }

    function fmtText(value) {
      return value === null || value === undefined || value === "" ? "-" : String(value);
    }

    function pnlClass(value) {
      if (value > 0) return "green";
      if (value < 0) return "red";
      return "amber";
    }

    function fmtSide(value) {
      if (value === "long") return "多头";
      if (value === "short") return "空头";
      return value || "-";
    }

    function fmtState(value) {
      if (value === "cohesion") return "凝聚";
      if (value === "drift") return "漂移";
      if (value === "fracture") return "断裂";
      return value || "-";
    }

    function fmtPosition(value) {
      if (value > 0) return "多头";
      if (value < 0) return "空头";
      return "空仓";
    }

    function fmtStateDuration(state, cohesionStreak, fractureStreak) {
      if (state === "cohesion") return `已持续 ${fmtText(cohesionStreak)} 根`;
      if (state === "fracture") return `已持续 ${fmtText(fractureStreak)} 根`;
      if (state === "drift") return "中性漂移";
      return "-";
    }

    function fmtStateSummary(state, cohesionStreak, fractureStreak) {
      const stateName = fmtState(state);
      const duration = fmtStateDuration(state, cohesionStreak, fractureStreak);
      if (state === "drift") return `当前状态：${duration}`;
      if (!stateName || stateName === "-" || !duration || duration === "-") return "-";
      return `当前状态：${stateName}，${duration}`;
    }

    function fmtBool(value) {
      if (value === true) return "是";
      if (value === false) return "否";
      return "-";
    }

    function fmtPositionChange(previousValue, currentValue) {
      return `${fmtPosition(previousValue)} -> ${fmtPosition(currentValue)}`;
    }

    function buildSymbolCard(item) {
      const priceChangeClass = pnlClass(item.market.price_change_pct || 0);
      const pnlCss = pnlClass(item.paper.net_pnl_usd || 0);
      const lastTrade = item.last_trade || {};
      const live = item.live || {};
      const account = item.account || {};
      const livePnlCss = pnlClass(live.floating_pnl_usd || 0);
      const accountPnlCss = pnlClass(account.realized_net_pnl_usd || 0);
      const accountFloatingCss = pnlClass(account.current_floating_pnl_usd || 0);
      return `
        <div class="panel">
          <div class="symbol-header">
            <div>
              <h2>${item.display_name}</h2>
              <div class="muted">${item.symbol}</div>
            </div>
            <div class="badge">${item.anchor}</div>
          </div>
          <div class="symbol-meta">
            <div class="meta-box">
              <div class="k">最新价格</div>
              <div class="v">${fmtUsd(item.market.last_price)}</div>
            </div>
            <div class="meta-box">
              <div class="k">24 小时涨跌</div>
              <div class="v ${priceChangeClass}">${fmtPct(item.market.price_change_pct)}</div>
            </div>
            <div class="meta-box">
              <div class="k">初始本金</div>
              <div class="v">${fmtUsd(item.paper.initial_capital_usd)}</div>
            </div>
            <div class="meta-box">
              <div class="k">杠杆</div>
              <div class="v">${item.paper.leverage.toFixed(1)}x</div>
            </div>
            <div class="meta-box">
              <div class="k">历史终局权益</div>
              <div class="v">${fmtUsd(item.paper.final_equity_usd)}</div>
            </div>
            <div class="meta-box">
              <div class="k">历史净盈亏</div>
              <div class="v ${pnlCss}">${fmtUsd(item.paper.net_pnl_usd)}</div>
            </div>
            <div class="meta-box">
              <div class="k">历史复利收益</div>
              <div class="v">${fmtPct(item.paper.compounded_total_return)}</div>
            </div>
            <div class="meta-box">
              <div class="k">历史胜率 / 交易次数</div>
              <div class="v">${fmtPct(item.paper.win_rate)} / ${item.paper.trade_count}</div>
            </div>
            <div class="meta-box">
              <div class="k">实时已实现权益</div>
              <div class="v">${fmtUsd(account.realized_equity_usd)}</div>
            </div>
            <div class="meta-box">
              <div class="k">实时估算权益</div>
              <div class="v">${fmtUsd(account.marked_equity_usd)}</div>
            </div>
            <div class="meta-box">
              <div class="k">实时已实现净盈亏</div>
              <div class="v ${accountPnlCss}">${fmtUsd(account.realized_net_pnl_usd)}</div>
            </div>
            <div class="meta-box">
              <div class="k">实时当前浮盈亏</div>
              <div class="v ${accountFloatingCss}">${fmtUsd(account.current_floating_pnl_usd)}</div>
            </div>
            <div class="meta-box">
              <div class="k">已处理信号 / 已平仓交易</div>
              <div class="v">${fmtText(account.processed_signal_count)} / ${fmtText(account.closed_trade_count)}</div>
            </div>
            <div class="meta-box">
              <div class="k">当前挂账方向</div>
              <div class="v">${fmtText(account.position_label)}</div>
            </div>
            <div class="meta-box">
              <div class="k">模型训练区间</div>
              <div class="v">${fmtText(live.model_train_window_label)}</div>
            </div>
            <div class="meta-box">
              <div class="k">冻结运行区间</div>
              <div class="v">${fmtText(live.model_run_window_label)}</div>
            </div>
            <div class="meta-box">
              <div class="k">下次重训时间</div>
              <div class="v">${fmtText(live.next_retrain_timestamp)}</div>
            </div>
            <div class="meta-box">
              <div class="k">本轮运行进度</div>
              <div class="v">${fmtText(live.run_progress_label)}</div>
            </div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th colspan="4">实时纸面状态</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>确认时间</td>
                  <td>${fmtText(live.last_confirmed_timestamp)}</td>
                  <td>当前仓位</td>
                  <td>${fmtPosition(live.position)}</td>
                </tr>
                <tr>
                  <td>最新动作</td>
                  <td>${fmtText(live.latest_action)}</td>
                  <td>V2 状态</td>
                  <td>${fmtState(live.state)}</td>
                </tr>
                <tr>
                  <td>门控通过</td>
                  <td>${fmtBool(live.gate_passed)}</td>
                  <td>突破事件</td>
                  <td>${fmtText(live.breakout_label)}</td>
                </tr>
                <tr>
                  <td>入场成交价</td>
                  <td>${fmtUsd(live.entry_fill_price)}</td>
                  <td>最新价格</td>
                  <td>${fmtUsd(item.market.last_price)}</td>
                </tr>
                <tr>
                  <td>浮动盈亏</td>
                  <td class="${livePnlCss}">${fmtUsd(live.floating_pnl_usd)}</td>
                  <td>浮动收益率</td>
                  <td>${fmtPct(live.floating_return)}</td>
                </tr>
                <tr>
                  <td>状态说明</td>
                  <td>${fmtStateSummary(live.state, live.cohesion_streak, live.fracture_streak)}</td>
                  <td>当前 Bar 开盘</td>
                  <td>${fmtText(live.current_bar_open_timestamp)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th colspan="4">最近一笔已平仓交易</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>方向</td>
                  <td>${fmtSide(lastTrade.side)}</td>
                  <td>持仓 Bar 数</td>
                  <td>${lastTrade.bars_held || "-"}</td>
                </tr>
                <tr>
                  <td>开仓时间</td>
                  <td>${lastTrade.entry_time || "-"}</td>
                  <td>平仓时间</td>
                  <td>${lastTrade.exit_time || "-"}</td>
                </tr>
                <tr>
                  <td>开仓权益</td>
                  <td>${fmtUsd(lastTrade.entry_equity_usd)}</td>
                  <td>平仓权益</td>
                  <td>${fmtUsd(lastTrade.exit_equity_usd)}</td>
                </tr>
                <tr>
                  <td>净盈亏</td>
                  <td class="${pnlClass(lastTrade.net_pnl_usd || 0)}">${fmtUsd(lastTrade.net_pnl_usd)}</td>
                  <td>净收益率</td>
                  <td>${fmtPct(lastTrade.net_return)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      `;
    }

    function buildRecentTrades(rows) {
      if (!rows.length) {
        return `<tr><td colspan="9" class="muted">暂无交易记录。</td></tr>`;
      }
      return rows.map((row) => `
        <tr>
          <td>${row.symbol}</td>
          <td>${fmtSide(row.side)}</td>
          <td>${row.entry_time || "-"}</td>
          <td>${row.exit_time || "-"}</td>
          <td>${fmtUsd(row.entry_equity_usd)}</td>
          <td>${fmtUsd(row.exit_equity_usd)}</td>
          <td class="${pnlClass(row.net_pnl_usd || 0)}">${fmtUsd(row.net_pnl_usd)}</td>
          <td>${fmtPct(row.net_return)}</td>
          <td>${row.bars_held || "-"}</td>
        </tr>
      `).join("");
    }

    function buildRecentSignals(rows) {
      if (!rows.length) {
        return `<tr><td colspan="8" class="muted">暂无信号流水。</td></tr>`;
      }
      return rows.map((row) => `
        <tr>
          <td>${row.symbol}</td>
          <td>${fmtText(row.action)}</td>
          <td>${fmtText(row.signal_timestamp)}</td>
          <td>${fmtText(row.fill_timestamp)}</td>
          <td>${fmtUsd(row.fill_price)}</td>
          <td>${fmtPositionChange(row.previous_position, row.current_position)}</td>
          <td>${fmtState(row.state)}</td>
          <td>${fmtText(row.breakout_label)}</td>
          <td>${fmtUsd(row.equity_after_usd)}</td>
        </tr>
      `).join("");
    }

    async function refreshState() {
      const response = await fetch("/api/state", { cache: "no-store" });
      const data = await response.json();

      document.getElementById("updatedAt").textContent = data.updated_at || "-";
      document.getElementById("portfolioInitial").textContent = fmtUsd(data.portfolio.initial_capital_usd);
      document.getElementById("portfolioFinal").textContent = fmtUsd(data.portfolio.final_equity_usd);
      document.getElementById("portfolioPnl").innerHTML = `<span class="${pnlClass(data.portfolio.net_pnl_usd || 0)}">${fmtUsd(data.portfolio.net_pnl_usd)}</span>`;
      document.getElementById("portfolioReturn").textContent = fmtPct(data.portfolio.compounded_total_return);
      document.getElementById("liveLastBar").textContent = fmtText(data.live_portfolio.latest_confirmed_bar);
      document.getElementById("liveActiveCount").textContent = fmtText(data.live_portfolio.active_position_count);
      document.getElementById("liveMarkedEquity").textContent = fmtUsd(data.live_portfolio.marked_equity_usd);
      document.getElementById("liveExposureMix").textContent = fmtText(data.live_portfolio.position_mix);
      document.getElementById("rtInitialCapital").textContent = fmtUsd(data.realtime_portfolio.initial_capital_usd);
      document.getElementById("rtRealizedEquity").textContent = fmtUsd(data.realtime_portfolio.realized_equity_usd);
      document.getElementById("rtRealizedPnl").innerHTML = `<span class="${pnlClass(data.realtime_portfolio.realized_net_pnl_usd || 0)}">${fmtUsd(data.realtime_portfolio.realized_net_pnl_usd)}</span>`;
      document.getElementById("rtFloatingPnl").innerHTML = `<span class="${pnlClass(data.realtime_portfolio.current_floating_pnl_usd || 0)}">${fmtUsd(data.realtime_portfolio.current_floating_pnl_usd)}</span>`;

      document.getElementById("symbolCards").innerHTML = data.symbols.map(buildSymbolCard).join("");
      document.getElementById("recentTradesTable").innerHTML = buildRecentTrades(data.recent_trades);
      document.getElementById("recentSignalsTable").innerHTML = buildRecentSignals(data.recent_signals);
    }

    refreshState();
    setInterval(refreshState, REFRESH_MS);
  </script>
</body>
</html>
"""


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_recent_trades(path: Path, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    recent = df.tail(limit).copy()
    recent["symbol"] = symbol
    return recent.to_dict(orient="records")


def account_runtime_paths(symbol: str) -> Dict[str, Path]:
    base_dir = PAPER_RUNTIME_DIR / symbol.lower()
    base_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base_dir": base_dir,
        "model": base_dir / "model_cycle_state.json",
        "state": base_dir / "paper_account_state.json",
        "signals": base_dir / "signal_log.csv",
        "trades": base_dir / "closed_trades.csv",
        "equity": base_dir / "equity_curve.csv",
    }


def append_records(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def ensure_local_history_file(config: Dict[str, Any]) -> Path:
    path = Path(config["input_csv"]).resolve()
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    end_ts = pd.Timestamp.now(tz="UTC")
    start_ts = end_ts - pd.Timedelta(days=HISTORY_BOOTSTRAP_LOOKBACK_DAYS)
    next_start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    chunks: List[pd.DataFrame] = []

    while next_start_ms < end_ms:
        batch = fetch_klines(
            symbol=config["symbol"],
            limit=HISTORICAL_KLINE_BATCH_LIMIT,
            start_time_ms=next_start_ms,
            end_time_ms=end_ms,
        )
        if batch.empty:
            break

        closed_batch = batch[batch["close_timestamp"] < end_ts].copy().reset_index(drop=True)
        if closed_batch.empty:
            break
        chunks.append(closed_batch[["timestamp", "open", "high", "low", "close", "volume", "amount"]].copy())

        last_open_timestamp = pd.Timestamp(closed_batch.iloc[-1]["timestamp"])
        next_start_ms = int((last_open_timestamp + BAR_INTERVAL).timestamp() * 1000)
        if len(batch) < HISTORICAL_KLINE_BATCH_LIMIT:
            break

    if not chunks:
        raise FileNotFoundError(f"Unable to bootstrap local history for {config['symbol']} -> {path}")

    history = pd.concat(chunks, ignore_index=True)
    history = history.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    history.to_csv(path, index=False)
    _LOCAL_HISTORY_CACHE.pop(str(path), None)
    return path


def load_local_history(path: Path) -> pd.DataFrame:
    resolved = path.resolve()
    stat = resolved.stat()
    cache_key = str(resolved)
    cached = _LOCAL_HISTORY_CACHE.get(cache_key)
    if cached and cached["mtime_ns"] == stat.st_mtime_ns:
        return cached["frame"].copy()
    frame = load_ohlcv_csv(str(resolved))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    _LOCAL_HISTORY_CACHE[cache_key] = {"mtime_ns": stat.st_mtime_ns, "frame": frame.copy()}
    return frame


def load_symbol_history(config: Dict[str, Any]) -> pd.DataFrame:
    base = load_local_history(ensure_local_history_file(config))
    recent_remote = fetch_recent_klines(config["symbol"], limit=1000)
    recent_remote = recent_remote[recent_remote["close_timestamp"] < pd.Timestamp.now(tz="UTC")].copy()
    if recent_remote.empty:
        return base
    recent_remote = recent_remote[["timestamp", "open", "high", "low", "close", "volume", "amount"]].copy()
    merged = pd.concat([base, recent_remote], ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return merged


def timestamp_to_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.isoformat()


def shift_bar(ts: pd.Timestamp, bars: int) -> pd.Timestamp:
    return ts + BAR_INTERVAL * bars


def summarize_cycle_window(start_value: Any, end_value: Any) -> Optional[str]:
    start_text = timestamp_to_iso(start_value)
    end_text = timestamp_to_iso(end_value)
    if not start_text or not end_text:
        return None
    return f"{start_text} -> {end_text}"


def attach_state_streak_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cohesion_streak"] = state_streaks(out["state"], "cohesion")
    out["fracture_streak"] = state_streaks(out["state"], "fracture")
    return out


def describe_model_cycle(history: pd.DataFrame, cycle_index: int, active_cycle_index: Optional[int] = None) -> Dict[str, Any]:
    if len(history) <= TRAIN_BARS:
        raise ValueError(f"Not enough bars for walk-forward cycle: {len(history)}")
    if active_cycle_index is None:
        latest_idx = len(history) - 1
        active_cycle_index = max(0, int((latest_idx - TRAIN_BARS) // TEST_BARS))

    train_start_idx = cycle_index * TEST_BARS
    train_end_idx = train_start_idx + TRAIN_BARS
    run_start_idx = train_end_idx
    scheduled_run_end_exclusive = run_start_idx + TEST_BARS
    if cycle_index < active_cycle_index:
        available_run_end_exclusive = min(len(history), scheduled_run_end_exclusive)
    else:
        available_run_end_exclusive = len(history)

    if run_start_idx >= len(history):
        raise ValueError(f"No run window available for cycle index: {cycle_index}")

    train_start_ts = pd.Timestamp(history.iloc[train_start_idx]["timestamp"])
    train_end_ts = pd.Timestamp(history.iloc[train_end_idx - 1]["timestamp"])
    run_start_ts = pd.Timestamp(history.iloc[run_start_idx]["timestamp"])
    scheduled_run_end_ts = shift_bar(run_start_ts, TEST_BARS - 1)
    next_retrain_ts = shift_bar(run_start_ts, TEST_BARS)

    return {
        "cycle_index": cycle_index,
        "train_start_idx": train_start_idx,
        "train_end_idx": train_end_idx,
        "run_start_idx": run_start_idx,
        "scheduled_run_end_exclusive": scheduled_run_end_exclusive,
        "available_run_end_exclusive": available_run_end_exclusive,
        "train_start_timestamp": train_start_ts.isoformat(),
        "train_end_timestamp": train_end_ts.isoformat(),
        "run_start_timestamp": run_start_ts.isoformat(),
        "scheduled_run_end_timestamp": scheduled_run_end_ts.isoformat(),
        "next_retrain_timestamp": next_retrain_ts.isoformat(),
        "model_train_window_label": summarize_cycle_window(train_start_ts, train_end_ts),
        "model_run_window_label": summarize_cycle_window(run_start_ts, scheduled_run_end_ts),
        "run_progress_label": f"{available_run_end_exclusive - run_start_idx} / {TEST_BARS} 根",
    }


def describe_current_model_cycle(history: pd.DataFrame) -> Dict[str, Any]:
    latest_idx = len(history) - 1
    cycle_index = max(0, int((latest_idx - TRAIN_BARS) // TEST_BARS))
    return describe_model_cycle(history, cycle_index=cycle_index, active_cycle_index=cycle_index)


def extract_cycle_history_slice(history: pd.DataFrame, descriptor: Dict[str, Any]) -> pd.DataFrame:
    train_start_idx = int(descriptor["train_start_idx"])
    available_run_end_exclusive = int(descriptor["available_run_end_exclusive"])
    cycle_history = history.iloc[train_start_idx:available_run_end_exclusive].copy().reset_index(drop=True)
    if len(cycle_history) <= TRAIN_BARS:
        raise ValueError(f"Cycle history slice is too short: {len(cycle_history)}")
    return cycle_history


def prepare_strict_cycle_feature_splits(
    cycle_history: pd.DataFrame,
    token_ids: Any,
    embeddings: Any,
) -> Dict[str, pd.DataFrame]:
    train_history = cycle_history.iloc[:TRAIN_BARS].copy().reset_index(drop=True)
    train_token_ids = token_ids[:TRAIN_BARS]
    train_embeddings = embeddings[:TRAIN_BARS]
    train_prepared, center, scale = prepare_features(
        train_history,
        train_token_ids,
        train_embeddings,
        state_window=64,
        horizon=16,
    )

    full_prepared, _, _ = prepare_features(
        cycle_history.copy().reset_index(drop=True),
        token_ids,
        embeddings,
        state_window=64,
        horizon=16,
        center=center,
        scale=scale,
    )
    run_prepared = full_prepared.iloc[TRAIN_BARS:].copy().reset_index(drop=True)
    return {
        "train_prepared": train_prepared,
        "run_prepared": run_prepared,
    }


def same_model_cycle(cached: Dict[str, Any], config: Dict[str, Any], descriptor: Dict[str, Any]) -> bool:
    return bool(
        cached
        and cached.get("symbol") == config["symbol"]
        and cached.get("anchor") == config["anchor"]
        and cached.get("feature_fit_scope") == "train_window_only"
        and cached.get("cycle_index") == descriptor["cycle_index"]
        and cached.get("train_start_timestamp") == descriptor["train_start_timestamp"]
        and cached.get("train_end_timestamp") == descriptor["train_end_timestamp"]
        and cached.get("run_start_timestamp") == descriptor["run_start_timestamp"]
        and cached.get("next_retrain_timestamp") == descriptor["next_retrain_timestamp"]
        and isinstance(cached.get("model"), dict)
    )


def save_model_cycle_state(paths: Dict[str, Path], cycle_state: Dict[str, Any]) -> None:
    paths["model"].write_text(json.dumps(cycle_state, ensure_ascii=False, indent=2), encoding="utf-8")


def fit_cycle_state(
    config: Dict[str, Any],
    descriptor: Dict[str, Any],
    train_prepared: pd.DataFrame,
) -> Dict[str, Any]:
    train_for_v2 = valid_training_rows(train_prepared)
    min_required_rows = max(2000, TRAIN_BARS // 4)
    if len(train_for_v2) < min_required_rows:
        raise ValueError(
            f"Not enough valid training rows for {config['symbol']}: {len(train_for_v2)} < {min_required_rows}"
        )

    model = fit_v2_model(train_for_v2)
    return {
        "symbol": config["symbol"],
        "anchor": config["anchor"],
        "cycle_mode": "180d_train_45d_frozen",
        "feature_fit_scope": "train_window_only",
        "fitted_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "train_rows_for_v2": int(len(train_for_v2)),
        "model": model,
        **descriptor,
    }


def load_or_fit_model_cycle(
    config: Dict[str, Any],
    paths: Dict[str, Path],
    descriptor: Dict[str, Any],
    train_prepared: pd.DataFrame,
) -> Dict[str, Any]:
    cached = read_json(paths["model"])
    if same_model_cycle(cached, config, descriptor):
        out = dict(cached)
        out.update(descriptor)
        return out

    cycle_state = fit_cycle_state(config, descriptor, train_prepared)
    save_model_cycle_state(paths, cycle_state)
    return cycle_state


def compute_cycle_scored_frame(
    train_prepared: pd.DataFrame,
    run_prepared: pd.DataFrame,
    cycle: Dict[str, Any],
) -> pd.DataFrame:
    raw_train = train_prepared.copy().reset_index(drop=True)
    raw_run = run_prepared.copy().reset_index(drop=True)
    if raw_run.empty:
        raise ValueError("No bars available in active run window.")

    train_scored = attach_state_streak_columns(apply_v2_model(raw_train, cycle["model"]))
    run_scored = apply_v2_model(raw_run, cycle["model"])
    combined = pd.concat([train_scored.tail(256), run_scored], ignore_index=True)
    combined = attach_state_streak_columns(combined)
    return combined.iloc[len(combined) - len(run_scored) :].reset_index(drop=True)


def default_account_state(initial_capital_usd: float, leverage: float) -> Dict[str, Any]:
    return {
        "initial_capital_usd": float(initial_capital_usd),
        "leverage": float(leverage),
        "fee_bps": float(DEFAULT_FEE_BPS),
        "slippage_bps": float(DEFAULT_SLIPPAGE_BPS),
        "equity_usd": float(initial_capital_usd),
        "prev_pos": 0.0,
        "last_processed_signal_timestamp": None,
        "processed_signal_count": 0,
        "closed_trade_count": 0,
        "open_trade": None,
    }


def load_account_state(paths: Dict[str, Path], initial_capital_usd: float, leverage: float) -> Dict[str, Any]:
    state = default_account_state(initial_capital_usd, leverage)
    if paths["state"].exists():
        disk = read_json(paths["state"])
        state.update(disk)
    return state


def save_account_state(paths: Dict[str, Path], state: Dict[str, Any]) -> None:
    paths["state"].write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_closed_trade(
    state: Dict[str, Any],
    row: Dict[str, Any],
    fill_price: float,
    closed_rows: List[Dict[str, Any]],
) -> None:
    trade = state["open_trade"]
    if not trade:
        return
    entry_equity = float(trade["entry_equity_usd"])
    exit_equity = float(trade["exit_equity_usd"])
    closed_rows.append(
        {
            "side": trade["side"],
            "entry_time": trade["entry_time"],
            "entry_fill_price": float(trade["entry_fill_price"]),
            "entry_equity_usd": entry_equity,
            "exit_time": str(row["signal_timestamp"]),
            "exit_fill_price": float(fill_price),
            "exit_equity_usd": exit_equity,
            "bars_held": int(trade["bars_held"]),
            "gross_return": float(math.exp(float(trade["gross_log_ret"])) - 1.0),
            "net_return": float(math.exp(float(trade["net_log_ret"])) - 1.0),
            "gross_pnl_usd": float(entry_equity * (math.exp(float(trade["gross_log_ret"])) - 1.0)),
            "net_pnl_usd": float(exit_equity - entry_equity),
        }
    )
    state["closed_trade_count"] = int(state.get("closed_trade_count", 0)) + 1
    state["open_trade"] = None


def apply_realized_signal_rows(
    state: Dict[str, Any],
    rows: List[Dict[str, Any]],
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> Dict[str, List[Dict[str, Any]]]:
    cost_rate = leverage * (fee_rate + slippage_rate)
    signal_rows: List[Dict[str, Any]] = []
    closed_rows: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    for row in rows:
        prev_pos = float(state.get("prev_pos", 0.0) or 0.0)
        cur_pos = float(row["position"])
        gross_bar_ret = leverage * cur_pos * float(row["next_open_ret"])
        equity_usd = float(state["equity_usd"])
        fill_price = float(row["fill_price"])

        if prev_pos == 0.0 and cur_pos != 0.0:
            state["open_trade"] = {
                "side": "long" if cur_pos > 0 else "short",
                "entry_time": str(row["signal_timestamp"]),
                "entry_fill_price": fill_price,
                "entry_equity_usd": equity_usd,
                "exit_equity_usd": float("nan"),
                "bars_held": 1,
                "gross_log_ret": gross_bar_ret,
                "net_log_ret": gross_bar_ret - cost_rate,
            }
            equity_usd *= float(math.exp(gross_bar_ret - cost_rate))
            state["open_trade"]["exit_equity_usd"] = equity_usd
        elif prev_pos != 0.0 and cur_pos == prev_pos:
            trade = state["open_trade"]
            if trade:
                trade["bars_held"] = int(trade["bars_held"]) + 1
                trade["gross_log_ret"] = float(trade["gross_log_ret"]) + gross_bar_ret
                trade["net_log_ret"] = float(trade["net_log_ret"]) + gross_bar_ret
            equity_usd *= float(math.exp(gross_bar_ret))
            if trade:
                trade["exit_equity_usd"] = equity_usd
        elif prev_pos != 0.0 and cur_pos == 0.0:
            trade = state["open_trade"]
            if trade:
                trade["net_log_ret"] = float(trade["net_log_ret"]) - cost_rate
            equity_usd *= float(math.exp(-cost_rate))
            if trade:
                trade["exit_equity_usd"] = equity_usd
            state["equity_usd"] = equity_usd
            finalize_closed_trade(state, row, fill_price, closed_rows)
        elif prev_pos != 0.0 and cur_pos != prev_pos:
            trade = state["open_trade"]
            if trade:
                trade["net_log_ret"] = float(trade["net_log_ret"]) - cost_rate
            equity_usd *= float(math.exp(-cost_rate))
            if trade:
                trade["exit_equity_usd"] = equity_usd
            state["equity_usd"] = equity_usd
            finalize_closed_trade(state, row, fill_price, closed_rows)

            state["open_trade"] = {
                "side": "long" if cur_pos > 0 else "short",
                "entry_time": str(row["signal_timestamp"]),
                "entry_fill_price": fill_price,
                "entry_equity_usd": equity_usd,
                "exit_equity_usd": float("nan"),
                "bars_held": 1,
                "gross_log_ret": gross_bar_ret,
                "net_log_ret": gross_bar_ret - cost_rate,
            }
            equity_usd *= float(math.exp(gross_bar_ret - cost_rate))
            state["open_trade"]["exit_equity_usd"] = equity_usd

        state["equity_usd"] = equity_usd
        state["prev_pos"] = cur_pos
        state["last_processed_signal_timestamp"] = str(row["signal_timestamp"])
        state["processed_signal_count"] = int(state.get("processed_signal_count", 0)) + 1

        if prev_pos != cur_pos:
            signal_rows.append(
                {
                    "symbol": row["symbol"],
                    "action": latest_action_label(prev_pos, cur_pos),
                    "signal_timestamp": str(row["signal_timestamp"]),
                    "fill_timestamp": str(row["fill_timestamp"]),
                    "fill_price": fill_price,
                    "previous_position": prev_pos,
                    "current_position": cur_pos,
                    "state": row["state"],
                    "breakout_label": row["breakout_label"],
                    "equity_after_usd": float(state["equity_usd"]),
                }
            )

        equity_rows.append(
            {
                "timestamp": str(row["equity_timestamp"]),
                "equity_usd": float(state["equity_usd"]),
                "position": cur_pos,
                "state": row["state"],
            }
        )

    return {"signal_rows": signal_rows, "closed_rows": closed_rows, "equity_rows": equity_rows}


def build_live_account_view(
    state: Dict[str, Any],
    live_snapshot: Dict[str, Any],
    market_snapshot: Dict[str, Any],
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> Dict[str, Any]:
    realized_equity = float(state.get("equity_usd", 0.0) or 0.0)
    previous_position = float(state.get("prev_pos", 0.0) or 0.0)
    current_position = float(live_snapshot.get("position", 0.0) or 0.0)
    cost_rate = leverage * (fee_rate + slippage_rate)
    marked_equity = realized_equity
    current_floating_pnl_usd = 0.0
    position_label_text = position_label(current_position)

    latest_fill_price = live_snapshot.get("latest_fill_price")
    last_price = market_snapshot.get("last_price")
    pending_trade = state.get("open_trade")

    if previous_position != current_position:
        turnover = abs(current_position - previous_position)
        marked_equity *= float(math.exp(-turnover * cost_rate))

    reference_entry_price = None
    if current_position != 0.0:
        if previous_position == current_position and pending_trade:
            reference_entry_price = float(pending_trade.get("entry_fill_price"))
        elif latest_fill_price:
            reference_entry_price = float(latest_fill_price)

    if current_position != 0.0 and reference_entry_price and last_price:
        direction = 1.0 if current_position > 0 else -1.0
        mark_log_ret = leverage * direction * math.log(float(last_price) / max(reference_entry_price, 1e-12))
        marked_equity *= float(math.exp(mark_log_ret))
        current_floating_pnl_usd = marked_equity - realized_equity

    return {
        "initial_capital_usd": float(state.get("initial_capital_usd", 0.0) or 0.0),
        "realized_equity_usd": realized_equity,
        "realized_net_pnl_usd": realized_equity - float(state.get("initial_capital_usd", 0.0) or 0.0),
        "marked_equity_usd": marked_equity,
        "current_floating_pnl_usd": current_floating_pnl_usd,
        "processed_signal_count": int(state.get("processed_signal_count", 0) or 0),
        "closed_trade_count": int(state.get("closed_trade_count", 0) or 0),
        "position": current_position,
        "position_label": position_label_text,
    }


def update_paper_account(
    config: Dict[str, Any],
    live_snapshot: Dict[str, Any],
    market_snapshot: Dict[str, Any],
    initial_capital_usd: float,
    leverage: float,
) -> Dict[str, Any]:
    paths = account_runtime_paths(config["symbol"])
    is_first_bootstrap = not paths["state"].exists()
    state = load_account_state(paths, initial_capital_usd, leverage)
    fee_rate = float(state.get("fee_bps", DEFAULT_FEE_BPS)) / 10000.0
    slippage_rate = float(state.get("slippage_bps", DEFAULT_SLIPPAGE_BPS)) / 10000.0

    ledger_rows = live_snapshot.get("ledger_rows", [])
    if is_first_bootstrap and ledger_rows:
        state["last_processed_signal_timestamp"] = str(ledger_rows[-1]["signal_timestamp"])
        save_account_state(paths, state)

    last_processed = state.get("last_processed_signal_timestamp")
    if last_processed and ledger_rows and str(last_processed) < str(ledger_rows[0]["signal_timestamp"]):
        history = load_symbol_history(config)
        ledger_rows = replay_pending_ledger_rows(config, history, str(last_processed))
    pending_rows = [row for row in ledger_rows if not last_processed or str(row["signal_timestamp"]) > str(last_processed)]
    appended = apply_realized_signal_rows(
        state,
        pending_rows,
        leverage=leverage,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    append_records(paths["signals"], appended["signal_rows"])
    append_records(paths["trades"], appended["closed_rows"])
    append_records(paths["equity"], appended["equity_rows"])
    save_account_state(paths, state)

    signal_rows = read_recent_trades(paths["signals"], config["symbol"], limit=16)
    closed_rows = read_recent_trades(paths["trades"], config["symbol"], limit=12)
    account_view = build_live_account_view(
        state,
        live_snapshot,
        market_snapshot,
        leverage=leverage,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    account_view["runtime_dir"] = str(paths["base_dir"])
    return {
        "state": state,
        "account_view": account_view,
        "signal_rows": signal_rows,
        "closed_rows": closed_rows,
    }


def get_tokenizer_adapter() -> KronosTokenizerAdapter:
    global _TOKENIZER_ADAPTER
    if _TOKENIZER_ADAPTER is None:
        _TOKENIZER_ADAPTER = KronosTokenizerAdapter(
            repo_dir=str(KRONOS_REPO_DIR),
            tokenizer_name=KRONOS_TOKENIZER_NAME,
            chunk_size=KRONOS_CHUNK_SIZE,
            device="cpu",
        )
    return _TOKENIZER_ADAPTER


def fetch_klines(symbol: str, limit: int, start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None) -> pd.DataFrame:
    query_payload: Dict[str, Any] = {"symbol": symbol, "interval": "15m", "limit": limit}
    if start_time_ms is not None:
        query_payload["startTime"] = int(start_time_ms)
    if end_time_ms is not None:
        query_payload["endTime"] = int(end_time_ms)
    query = urllib.parse.urlencode(query_payload)
    url = f"{BINANCE_KLINES_URL}?{query}"
    with urllib.request.urlopen(url, timeout=15) as response:
        payload = response.read().decode("utf-8")
    rows = json.loads(payload)
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "trade_count",
            "taker_base_volume",
            "taker_quote_volume",
            "ignore",
        ],
    )
    for column in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_timestamp"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["amount"] = df["quote_asset_volume"].fillna(0.0)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_recent_klines(symbol: str, limit: int = LIVE_KLINE_LIMIT) -> pd.DataFrame:
    return fetch_klines(symbol=symbol, limit=limit)


def position_label(value: float) -> str:
    if value > 0:
        return "多头"
    if value < 0:
        return "空头"
    return "空仓"


def breakout_label(up_event: bool, down_event: bool) -> str:
    if up_event:
        return "上破"
    if down_event:
        return "下破"
    return "无"


def latest_action_label(previous: float, current: float) -> str:
    if previous == current:
        if current > 0:
            return "继续持有多头"
        if current < 0:
            return "继续持有空头"
        return "继续空仓"
    if previous == 0 and current > 0:
        return "新开多头"
    if previous == 0 and current < 0:
        return "新开空头"
    if previous > 0 and current == 0:
        return "多头平仓"
    if previous < 0 and current == 0:
        return "空头平仓"
    if previous > 0 and current < 0:
        return "多头反手为空头"
    if previous < 0 and current > 0:
        return "空头反手为多头"
    return "状态更新"


def build_live_ledger_rows(
    config: Dict[str, Any],
    scored: pd.DataFrame,
    positions: pd.Series,
    up_col: str,
    down_col: str,
) -> List[Dict[str, Any]]:
    ledger_rows: List[Dict[str, Any]] = []
    realizable_end = max(len(scored) - 1, 0)
    for i in range(realizable_end):
        signal_timestamp = pd.Timestamp(scored.iloc[i]["timestamp"]).isoformat()
        fill_timestamp_value = scored.iloc[i]["next_fill_timestamp"]
        equity_timestamp_value = scored.iloc[i]["next_equity_timestamp"]
        if pd.isna(scored.iloc[i]["next_open_ret"]) or pd.isna(scored.iloc[i]["next_open_fill_price"]):
            continue
        if pd.isna(fill_timestamp_value) or pd.isna(equity_timestamp_value):
            continue
        prev_bar_position = float(positions.iloc[i - 1]) if i > 0 else 0.0
        ledger_rows.append(
            {
                "symbol": config["symbol"],
                "signal_timestamp": signal_timestamp,
                "fill_timestamp": pd.Timestamp(fill_timestamp_value).isoformat(),
                "equity_timestamp": pd.Timestamp(equity_timestamp_value).isoformat(),
                "position": float(positions.iloc[i]),
                "previous_position": prev_bar_position,
                "next_open_ret": float(scored.iloc[i]["next_open_ret"]),
                "fill_price": float(scored.iloc[i]["next_open_fill_price"]),
                "state": str(scored.iloc[i]["state"]),
                "breakout_label": breakout_label(bool(scored.iloc[i][up_col]), bool(scored.iloc[i][down_col])),
            }
        )
    return ledger_rows


def cycle_index_for_signal_timestamp(history: pd.DataFrame, signal_timestamp: Any) -> int:
    if signal_timestamp is None:
        return 0
    ts = pd.Timestamp(signal_timestamp)
    if pd.isna(ts):
        return 0
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    timestamps = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    signal_idx = int(timestamps.searchsorted(ts, side="left"))
    if signal_idx < TRAIN_BARS:
        return 0
    return max(0, int((signal_idx - TRAIN_BARS) // TEST_BARS))


def build_cycle_snapshot_scored_frame(
    config: Dict[str, Any],
    history: pd.DataFrame,
    descriptor: Dict[str, Any],
) -> Dict[str, Any]:
    cycle_history = extract_cycle_history_slice(history, descriptor)
    adapter = get_tokenizer_adapter()
    token_input = cycle_history[["timestamp", "open", "high", "low", "close", "volume", "amount"]].copy()
    artifact = adapter.fit_transform(token_input)
    feature_splits = prepare_strict_cycle_feature_splits(
        artifact.frame.copy().reset_index(drop=True),
        artifact.token_ids,
        artifact.embeddings,
    )
    cycle_state = fit_cycle_state(config, descriptor, feature_splits["train_prepared"])
    scored = compute_cycle_scored_frame(feature_splits["train_prepared"], feature_splits["run_prepared"], cycle_state)

    cycle_run_history = cycle_history.iloc[TRAIN_BARS:].copy().reset_index(drop=True)
    future_open = cycle_run_history["open"].shift(-1).reset_index(drop=True)
    next_future_open = cycle_run_history["open"].shift(-2).reset_index(drop=True)
    next_fill_timestamp = cycle_run_history["timestamp"].shift(-1).reset_index(drop=True)
    next_equity_timestamp = cycle_run_history["timestamp"].shift(-2).reset_index(drop=True)
    scored["next_open_fill_price"] = future_open
    scored["next_open_ret"] = (next_future_open / future_open).map(
        lambda value: math.log(value) if pd.notna(value) and value > 0 else float("nan")
    )
    scored["next_fill_timestamp"] = next_fill_timestamp
    scored["next_equity_timestamp"] = next_equity_timestamp

    anchor_param = parse_anchor_param(config["anchor"])
    scored = prepare_breakout_columns(scored, [anchor_param.breakout_window])
    positions = simulate_breakout_param(
        scored,
        anchor_param,
        hard_stop_loss_pct=0.0,
        execution_model="close_to_next_open",
    )
    up_col = f"breakout_up_{anchor_param.breakout_window}"
    down_col = f"breakout_down_{anchor_param.breakout_window}"
    ledger_rows = build_live_ledger_rows(config, scored, positions, up_col, down_col)
    return {
        "cycle_history": cycle_history,
        "feature_splits": feature_splits,
        "cycle_state": cycle_state,
        "scored": scored,
        "positions": positions,
        "up_col": up_col,
        "down_col": down_col,
        "ledger_rows": ledger_rows,
    }


def replay_pending_ledger_rows(
    config: Dict[str, Any],
    history: pd.DataFrame,
    last_processed_signal_timestamp: Optional[str],
) -> List[Dict[str, Any]]:
    if last_processed_signal_timestamp is None:
        return []

    active_descriptor = describe_current_model_cycle(history)
    active_cycle_index = int(active_descriptor["cycle_index"])
    start_cycle_index = min(
        active_cycle_index,
        cycle_index_for_signal_timestamp(history, last_processed_signal_timestamp),
    )
    replay_rows: List[Dict[str, Any]] = []

    for cycle_index in range(start_cycle_index, active_cycle_index + 1):
        cycle_descriptor = describe_model_cycle(history, cycle_index=cycle_index, active_cycle_index=active_cycle_index)
        cycle_snapshot = build_cycle_snapshot_scored_frame(config, history, cycle_descriptor)
        replay_rows.extend(cycle_snapshot["ledger_rows"])

    filtered_rows = [
        row for row in replay_rows if row.get("signal_timestamp") and row["signal_timestamp"] > str(last_processed_signal_timestamp)
    ]
    deduped = {row["signal_timestamp"]: row for row in filtered_rows}
    return [deduped[key] for key in sorted(deduped.keys())]


def next_live_refresh_epoch(snapshot: Dict[str, Any], fallback_now: float) -> float:
    confirmed = snapshot.get("last_confirmed_timestamp")
    if not confirmed:
        return fallback_now + LIVE_SIGNAL_REFRESH_SECONDS
    try:
        confirmed_ts = pd.Timestamp(confirmed)
        if confirmed_ts.tzinfo is None:
            confirmed_ts = confirmed_ts.tz_localize("UTC")
        # A new confirmed 15m bar can only appear after two bar-open steps from the last confirmed bar.
        next_refresh_ts = shift_bar(confirmed_ts, 2) + pd.Timedelta(seconds=5)
        return max(next_refresh_ts.timestamp(), fallback_now + 5.0)
    except Exception:
        return fallback_now + LIVE_SIGNAL_REFRESH_SECONDS


def compute_all_live_signal_snapshots() -> Dict[str, Any]:
    now_ts = time.time()
    snapshots: Dict[str, Dict[str, Any]] = {}
    updated_at = pd.Timestamp.now(tz="UTC").isoformat()
    for config in SYMBOL_CONFIGS:
        try:
            snapshots[config["symbol"]] = compute_live_signal_snapshot(config)
        except Exception as exc:  # pragma: no cover - dashboard should degrade gracefully
            snapshots[config["symbol"]] = {
                "status": "error",
                "symbol": config["symbol"],
                "error": str(exc),
                "last_confirmed_timestamp": None,
                "current_bar_open_timestamp": None,
                "position": 0.0,
                "position_label": "空仓",
                "latest_action": "实时状态暂不可用",
                "state": None,
                "instability_score": None,
                "cohesion_streak": None,
                "fracture_streak": None,
                "gate_passed": None,
                "breakout_up_event": None,
                "breakout_down_event": None,
                "breakout_label": None,
                "entry_signal_timestamp": None,
                "entry_fill_timestamp": None,
                "entry_fill_price": None,
                "closed_bar_count": 0,
                "model_cycle_index": None,
                "cycle_mode": None,
                "model_fitted_at": None,
                "model_train_window_label": None,
                "model_run_window_label": None,
                "next_retrain_timestamp": None,
                "run_progress_label": None,
                "train_rows_for_v2": 0,
            }

    refresh_epochs = [next_live_refresh_epoch(snapshot, now_ts) for snapshot in snapshots.values()]
    return {
        "expires_at": min(refresh_epochs) if refresh_epochs else now_ts + LIVE_SIGNAL_REFRESH_SECONDS,
        "updated_at": updated_at,
        "snapshots": snapshots,
        "last_error": None,
    }


def refresh_live_signal_cache_async() -> None:
    try:
        fresh = compute_all_live_signal_snapshots()
        _LIVE_SIGNAL_CACHE["expires_at"] = fresh["expires_at"]
        _LIVE_SIGNAL_CACHE["updated_at"] = fresh["updated_at"]
        _LIVE_SIGNAL_CACHE["snapshots"] = fresh["snapshots"]
        _LIVE_SIGNAL_CACHE["last_error"] = None
    except Exception as exc:  # pragma: no cover - defensive fallback
        _LIVE_SIGNAL_CACHE["last_error"] = str(exc)
        # If background refresh fails, keep serving stale cache for a short grace period.
        _LIVE_SIGNAL_CACHE["expires_at"] = time.time() + 30.0
    finally:
        _LIVE_SIGNAL_CACHE["refreshing"] = False


def ensure_live_signal_cache_refresh_started() -> None:
    if _LIVE_SIGNAL_CACHE["refreshing"]:
        return
    _LIVE_SIGNAL_CACHE["refreshing"] = True
    worker = threading.Thread(target=refresh_live_signal_cache_async, daemon=True)
    worker.start()


def parse_anchor_param(label: str) -> ParamConfig:
    parts = label.split("_")
    if len(parts) < 4 or not parts[0].startswith("bw"):
        raise ValueError(f"Unsupported anchor label: {label}")
    breakout_window = int(parts[0][2:])
    gate_mode = parts[1]
    hold_part = next(part for part in parts if part.startswith("hold"))
    cool_part = next(part for part in parts if part.startswith("cool"))
    return ParamConfig(
        breakout_window=breakout_window,
        gate_mode=gate_mode,
        min_hold_bars=int(hold_part[4:]),
        cooldown_bars=int(cool_part[4:]),
    )


def compute_live_signal_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    anchor_param = parse_anchor_param(config["anchor"])
    strategy = anchor_param.to_strategy_config()
    paths = account_runtime_paths(config["symbol"])
    history = load_symbol_history(config)
    if history.empty:
        raise ValueError(f"No historical data for {config['symbol']}")
    descriptor = describe_current_model_cycle(history)
    cycle_history = extract_cycle_history_slice(history, descriptor)
    adapter = get_tokenizer_adapter()
    token_input = cycle_history[["timestamp", "open", "high", "low", "close", "volume", "amount"]].copy()
    artifact = adapter.fit_transform(token_input)
    feature_splits = prepare_strict_cycle_feature_splits(
        artifact.frame.copy().reset_index(drop=True),
        artifact.token_ids,
        artifact.embeddings,
    )
    train_prepared = feature_splits["train_prepared"]
    run_prepared = feature_splits["run_prepared"]
    cycle = load_or_fit_model_cycle(config, paths, descriptor, train_prepared)
    scored = compute_cycle_scored_frame(train_prepared, run_prepared, cycle)

    bars = fetch_recent_klines(config["symbol"], limit=LIVE_KLINE_LIMIT)
    if bars.empty:
        raise ValueError(f"No live kline data for {config['symbol']}")
    now_utc = pd.Timestamp.now(tz="UTC")
    closed_live = bars[bars["close_timestamp"] < now_utc].copy().reset_index(drop=True)
    cycle_run_history = cycle_history.iloc[TRAIN_BARS:].copy().reset_index(drop=True)
    future_open = cycle_run_history["open"].shift(-1).reset_index(drop=True)
    next_future_open = cycle_run_history["open"].shift(-2).reset_index(drop=True)
    next_fill_timestamp = cycle_run_history["timestamp"].shift(-1).reset_index(drop=True)
    next_equity_timestamp = cycle_run_history["timestamp"].shift(-2).reset_index(drop=True)
    scored["next_open_fill_price"] = future_open
    scored["next_open_ret"] = (next_future_open / future_open).map(lambda value: math.log(value) if pd.notna(value) and value > 0 else float("nan"))
    scored["next_fill_timestamp"] = next_fill_timestamp
    scored["next_equity_timestamp"] = next_equity_timestamp
    scored = prepare_breakout_columns(scored, [anchor_param.breakout_window])
    up_col = f"breakout_up_{anchor_param.breakout_window}"
    down_col = f"breakout_down_{anchor_param.breakout_window}"

    positions = simulate_breakout_param(
        scored,
        anchor_param,
        hard_stop_loss_pct=0.0,
        execution_model="close_to_next_open",
    )
    latest_idx = len(scored) - 1
    latest_row = scored.iloc[latest_idx]
    current_position = float(positions.iloc[latest_idx])
    previous_position = float(positions.iloc[latest_idx - 1]) if latest_idx > 0 else 0.0
    latest_fill_price = float(bars.iloc[len(closed_live)]["open"]) if len(bars) > len(closed_live) else None

    change_mask = positions.ne(positions.shift(fill_value=0.0))
    entry_signal_idx: Optional[int] = None
    entry_fill_price: Optional[float] = None
    entry_fill_timestamp: Optional[str] = None
    if current_position != 0.0:
        candidates = positions.index[(change_mask) & (positions == current_position)]
        if len(candidates) > 0:
            entry_signal_idx = int(candidates[-1])
            if pd.notna(scored.iloc[entry_signal_idx]["next_open_fill_price"]):
                entry_fill_price = float(scored.iloc[entry_signal_idx]["next_open_fill_price"])
            fill_timestamp_value = scored.iloc[entry_signal_idx]["next_fill_timestamp"]
            if pd.notna(fill_timestamp_value):
                entry_fill_timestamp = pd.Timestamp(fill_timestamp_value).isoformat()

    current_bar_open_timestamp = None
    if len(bars) > len(closed_live):
        current_bar_open_timestamp = pd.Timestamp(bars.iloc[len(closed_live)]["timestamp"]).isoformat()

    ledger_rows = build_live_ledger_rows(config, scored, positions, up_col, down_col)

    return {
        "status": "ok",
        "symbol": config["symbol"],
        "last_confirmed_timestamp": pd.Timestamp(latest_row["timestamp"]).isoformat(),
        "current_bar_open_timestamp": current_bar_open_timestamp,
        "position": current_position,
        "previous_position": previous_position,
        "position_label": position_label(current_position),
        "latest_action": latest_action_label(previous_position, current_position),
        "state": str(latest_row["state"]),
        "instability_score": float(latest_row["instability_score"]),
        "cohesion_streak": int(latest_row["cohesion_streak"]),
        "fracture_streak": int(latest_row["fracture_streak"]),
        "gate_passed": bool(gate_passed(latest_row, strategy)),
        "breakout_up_event": bool(latest_row[up_col]),
        "breakout_down_event": bool(latest_row[down_col]),
        "breakout_label": breakout_label(bool(latest_row[up_col]), bool(latest_row[down_col])),
        "entry_signal_timestamp": pd.Timestamp(scored.iloc[entry_signal_idx]["timestamp"]).isoformat() if entry_signal_idx is not None else None,
        "entry_fill_timestamp": entry_fill_timestamp,
        "entry_fill_price": entry_fill_price,
        "latest_fill_price": latest_fill_price,
        "closed_bar_count": int(len(history)),
        "model_cycle_index": int(cycle["cycle_index"]),
        "cycle_mode": str(cycle["cycle_mode"]),
        "model_fitted_at": cycle.get("fitted_at"),
        "model_train_window_label": cycle.get("model_train_window_label"),
        "model_run_window_label": cycle.get("model_run_window_label"),
        "next_retrain_timestamp": cycle.get("next_retrain_timestamp"),
        "run_progress_label": cycle.get("run_progress_label"),
        "train_rows_for_v2": int(cycle.get("train_rows_for_v2", 0) or 0),
        "ledger_rows": ledger_rows,
    }


def get_live_signal_snapshots(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    now_ts = time.time()
    if force_refresh:
        fresh = compute_all_live_signal_snapshots()
        _LIVE_SIGNAL_CACHE["expires_at"] = fresh["expires_at"]
        _LIVE_SIGNAL_CACHE["updated_at"] = fresh["updated_at"]
        _LIVE_SIGNAL_CACHE["snapshots"] = fresh["snapshots"]
        _LIVE_SIGNAL_CACHE["last_error"] = fresh["last_error"]
        _LIVE_SIGNAL_CACHE["refreshing"] = False
        return fresh["snapshots"]

    if _LIVE_SIGNAL_CACHE["snapshots"] and now_ts < float(_LIVE_SIGNAL_CACHE["expires_at"]):
        return _LIVE_SIGNAL_CACHE["snapshots"]

    if _LIVE_SIGNAL_CACHE["snapshots"]:
        ensure_live_signal_cache_refresh_started()
        return _LIVE_SIGNAL_CACHE["snapshots"]

    fresh = compute_all_live_signal_snapshots()
    _LIVE_SIGNAL_CACHE["expires_at"] = fresh["expires_at"]
    _LIVE_SIGNAL_CACHE["updated_at"] = fresh["updated_at"]
    _LIVE_SIGNAL_CACHE["snapshots"] = fresh["snapshots"]
    _LIVE_SIGNAL_CACHE["last_error"] = fresh["last_error"]
    return fresh["snapshots"]


def merge_live_market_state(
    live_snapshot: Dict[str, Any],
    market_snapshot: Dict[str, Any],
    initial_capital_usd: float,
    leverage: float,
) -> Dict[str, Any]:
    out = dict(live_snapshot)
    last_price = market_snapshot.get("last_price")
    out["last_price"] = last_price
    out["floating_return"] = None
    out["floating_pnl_usd"] = None

    entry_fill_price = out.get("entry_fill_price")
    position = float(out.get("position", 0.0) or 0.0)
    if last_price and entry_fill_price and position != 0.0:
        direction = 1.0 if position > 0 else -1.0
        raw_move = (float(last_price) / float(entry_fill_price) - 1.0) * direction
        floating_return = raw_move * leverage
        out["floating_return"] = floating_return
        out["floating_pnl_usd"] = initial_capital_usd * floating_return
    return out


def fetch_market_snapshot(symbol: str) -> Dict[str, Any]:
    query = urllib.parse.urlencode({"symbol": symbol})
    url = f"{BINANCE_TICKER_24H_URL}?{query}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = response.read().decode("utf-8")
        data = json.loads(payload)
        return {
            "last_price": float(data.get("lastPrice", 0.0) or 0.0),
            "price_change_pct": float(data.get("priceChangePercent", 0.0) or 0.0) / 100.0,
            "quote_volume": float(data.get("quoteVolume", 0.0) or 0.0),
            "trade_count_24h": int(float(data.get("count", 0.0) or 0.0)),
        }
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
        return {
            "last_price": None,
            "price_change_pct": None,
            "quote_volume": None,
            "trade_count_24h": None,
        }


def load_symbol_state(config: Dict[str, Any], live_snapshots: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    summary = read_json(config["summary_path"])
    fee_block = summary.get("4.0", {})
    trades = read_recent_trades(config["trades_path"], config["symbol"], limit=12)
    last_trade = trades[-1] if trades else {}
    market = fetch_market_snapshot(config["symbol"])
    if live_snapshots is None:
        live_snapshots = get_live_signal_snapshots()
    raw_live = live_snapshots.get(config["symbol"], {})
    live = merge_live_market_state(
        raw_live,
        market,
        initial_capital_usd=float(fee_block.get("initial_capital_usd", 0.0) or 0.0),
        leverage=float(fee_block.get("leverage", 0.0) or 0.0),
    )
    account_bundle = update_paper_account(
        config,
        raw_live,
        market,
        initial_capital_usd=float(fee_block.get("initial_capital_usd", 0.0) or 0.0),
        leverage=float(fee_block.get("leverage", 0.0) or 0.0),
    )
    live.pop("ledger_rows", None)
    return {
        "symbol": config["symbol"],
        "display_name": config["display_name"],
        "anchor": config["anchor"],
        "paper": {
            "initial_capital_usd": float(fee_block.get("initial_capital_usd", 0.0) or 0.0),
            "leverage": float(fee_block.get("leverage", 0.0) or 0.0),
            "final_equity_usd": float(fee_block.get("final_equity_usd", 0.0) or 0.0),
            "net_pnl_usd": float(fee_block.get("net_pnl_usd", 0.0) or 0.0),
            "compounded_total_return": float(fee_block.get("compounded_total_return", 0.0) or 0.0),
            "trade_count": int(fee_block.get("trade_count", 0) or 0),
            "win_rate": float(fee_block.get("win_rate", 0.0) or 0.0),
        },
        "market": market,
        "live": live,
        "account": account_bundle["account_view"],
        "last_trade": account_bundle["closed_rows"][-1] if account_bundle["closed_rows"] else {},
        "recent_trades": account_bundle["closed_rows"],
        "recent_signals": account_bundle["signal_rows"],
    }


def compute_api_state(force_live_refresh: bool = False) -> Dict[str, Any]:
    live_snapshots = get_live_signal_snapshots(force_refresh=force_live_refresh)
    symbols = [load_symbol_state(config, live_snapshots=live_snapshots) for config in SYMBOL_CONFIGS]
    recent_trades: List[Dict[str, Any]] = []
    recent_signals: List[Dict[str, Any]] = []
    initial_capital = 0.0
    final_equity = 0.0
    rt_initial_capital = 0.0
    rt_realized_equity = 0.0
    rt_marked_equity = 0.0

    for item in symbols:
        initial_capital += item["paper"]["initial_capital_usd"]
        final_equity += item["paper"]["final_equity_usd"]
        recent_trades.extend(item["recent_trades"])
        recent_signals.extend(item.get("recent_signals", []))
        rt_initial_capital += float(item["account"]["initial_capital_usd"])
        rt_realized_equity += float(item["account"]["realized_equity_usd"])
        rt_marked_equity += float(item["account"]["marked_equity_usd"])

    recent_trades = sorted(recent_trades, key=lambda row: str(row.get("exit_time", "")), reverse=True)[:18]
    recent_signals = sorted(recent_signals, key=lambda row: str(row.get("fill_timestamp", "")), reverse=True)[:18]
    live_items = [item.get("live", {}) for item in symbols]
    account_items = [item.get("account", {}) for item in symbols]
    active_count = sum(1 for item in live_items if float(item.get("position", 0.0) or 0.0) != 0.0)
    long_count = sum(1 for item in live_items if float(item.get("position", 0.0) or 0.0) > 0.0)
    short_count = sum(1 for item in live_items if float(item.get("position", 0.0) or 0.0) < 0.0)
    confirmed_times = [str(item.get("last_confirmed_timestamp")) for item in live_items if item.get("last_confirmed_timestamp")]
    current_floating_total = sum(float(item.get("current_floating_pnl_usd", 0.0) or 0.0) for item in account_items)
    return {
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "portfolio": {
            "initial_capital_usd": initial_capital,
            "final_equity_usd": final_equity,
            "net_pnl_usd": final_equity - initial_capital,
            "compounded_total_return": (final_equity / initial_capital - 1.0) if initial_capital > 0 else 0.0,
        },
        "live_portfolio": {
            "latest_confirmed_bar": max(confirmed_times) if confirmed_times else None,
            "active_position_count": active_count,
            "marked_equity_usd": rt_marked_equity,
            "position_mix": f"多头 {long_count} / 空头 {short_count} / 空仓 {len(symbols) - active_count}",
        },
        "realtime_portfolio": {
            "initial_capital_usd": rt_initial_capital,
            "realized_equity_usd": rt_realized_equity,
            "realized_net_pnl_usd": rt_realized_equity - rt_initial_capital,
            "marked_equity_usd": rt_marked_equity,
            "current_floating_pnl_usd": current_floating_total,
        },
        "symbols": symbols,
        "recent_trades": recent_trades,
        "recent_signals": recent_signals,
    }


def refresh_api_state_cache(force_live_refresh: bool = False) -> Dict[str, Any]:
    state = compute_api_state(force_live_refresh=force_live_refresh)
    with _API_STATE_CACHE_LOCK:
        _API_STATE_CACHE["state"] = state
        _API_STATE_CACHE["updated_at"] = state.get("updated_at")
        _API_STATE_CACHE["last_error"] = None
    return state


def get_cached_api_state() -> Optional[Dict[str, Any]]:
    with _API_STATE_CACHE_LOCK:
        state = _API_STATE_CACHE.get("state")
        if state is None:
            return None
        return json.loads(json.dumps(state))


def should_force_live_refresh() -> bool:
    if not _LIVE_SIGNAL_CACHE["snapshots"]:
        return True
    return time.time() >= float(_LIVE_SIGNAL_CACHE.get("expires_at", 0.0) or 0.0)


def next_paper_runner_sleep_seconds() -> float:
    if not _LIVE_SIGNAL_CACHE["snapshots"]:
        return PAPER_RUNNER_MIN_SLEEP_SECONDS
    seconds_until_live_refresh = float(_LIVE_SIGNAL_CACHE.get("expires_at", 0.0) or 0.0) - time.time()
    bounded = min(max(seconds_until_live_refresh, PAPER_RUNNER_MIN_SLEEP_SECONDS), PAPER_RUNNER_MAX_SLEEP_SECONDS)
    return max(bounded, PAPER_RUNNER_MIN_SLEEP_SECONDS)


def paper_runner_loop() -> None:
    while True:
        try:
            refresh_api_state_cache(force_live_refresh=should_force_live_refresh())
        except Exception as exc:  # pragma: no cover - daemon must stay alive
            with _API_STATE_CACHE_LOCK:
                _API_STATE_CACHE["last_error"] = str(exc)
        time.sleep(next_paper_runner_sleep_seconds())


def start_background_paper_runner() -> None:
    global _PAPER_RUNNER_THREAD
    if _PAPER_RUNNER_THREAD is not None and _PAPER_RUNNER_THREAD.is_alive():
        return
    _PAPER_RUNNER_THREAD = threading.Thread(target=paper_runner_loop, name="paper-runner", daemon=True)
    _PAPER_RUNNER_THREAD.start()


def build_api_state() -> Dict[str, Any]:
    cached = get_cached_api_state()
    if cached is not None:
        return cached
    return refresh_api_state_cache(force_live_refresh=True)


@app.get("/")
def index() -> str:
    return render_template_string(TEMPLATE, refresh_ms=REFRESH_MS)


@app.get("/api/state")
def api_state():
    return jsonify(build_api_state())


if __name__ == "__main__":
    try:
        refresh_api_state_cache(force_live_refresh=True)
    except Exception:
        pass
    start_background_paper_runner()
    app.run(host="0.0.0.0", port=8787, debug=False)
