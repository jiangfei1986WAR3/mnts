from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from flask import Flask, redirect, render_template_string, request, send_from_directory, url_for


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PSEUDO_OUTPUT_DIR = ROOT / "validation_outputs" / "pseudo_run"
KRONOS_OUTPUT_DIR = ROOT / "validation_outputs" / "kronos_run"
STATE15_OUTPUT_DIR = ROOT / "validation_outputs" / "15m_state_run"
CSV_PATH = DATA_DIR / "btcusdt_15m_1y.csv"
CSV_2Y_PATH = DATA_DIR / "btcusdt_15m_2y.csv"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

app = Flask(__name__)


TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MNTS 本地验证看板</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f7f7f9; color: #222; }
    h1, h2 { margin-bottom: 12px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 20px; }
    .card { background: #fff; border-radius: 10px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .actions { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
    button { padding: 10px 14px; border: 0; border-radius: 8px; background: #2563eb; color: white; cursor: pointer; }
    button.secondary { background: #475569; }
    .muted { color: #666; font-size: 14px; }
    table { width: 100%; border-collapse: collapse; background: white; }
    th, td { border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; font-size: 14px; }
    img { max-width: 100%; border-radius: 8px; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .section { margin-top: 24px; }
    .log { white-space: pre-wrap; background: #111827; color: #e5e7eb; padding: 12px; border-radius: 8px; font-size: 12px; }
    .pill { display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; margin-right: 6px; }
    .ok { background: #dcfce7; color: #166534; }
    .warn { background: #fef3c7; color: #92400e; }
    .danger { background: #fee2e2; color: #991b1b; }
  </style>
</head>
<body>
  <h1>MNTS 本地验证看板</h1>
  <p class="muted">目标：直接在本地看 BTC 15m 数据、最小验证结果和 tokenizer 路线是否值得继续。</p>

  <div class="actions">
    <form method="post" action="{{ url_for('run_fetch') }}">
      <button type="submit">1. 拉取币安 BTC 15m 数据</button>
    </form>
    <form method="post" action="{{ url_for('run_validate_pseudo') }}">
      <button type="submit">2. 跑最小验证（Pseudo）</button>
    </form>
    <form method="post" action="{{ url_for('run_validate_kronos') }}">
      <button type="submit">3. 跑真实验证（Kronos）</button>
    </form>
    <form method="post" action="{{ url_for('run_setup_kronos') }}">
      <button class="secondary" type="submit">4. 拉取官方 Kronos + Tokenizer</button>
    </form>
    <form method="post" action="{{ url_for('run_validate_15m_state') }}">
      <button class="secondary" type="submit">5. 跑 15分钟层 状态机</button>
    </form>
  </div>

  {% if message %}
  <div class="card">
    <strong>执行结果</strong>
    <div class="log">{{ message }}</div>
  </div>
  {% endif %}

  <div class="grid">
    <div class="card">
      <div class="muted">数据文件</div>
      <div>{{ data_status }}</div>
    </div>
    <div class="card">
      <div class="muted">数据行数</div>
      <div>{{ data_rows }}</div>
    </div>
    <div class="card">
      <div class="muted">时间范围</div>
      <div>{{ date_range }}</div>
    </div>
    <div class="card">
      <div class="muted">验证状态</div>
      <div>{{ validation_status }}</div>
    </div>
    <div class="card">
      <div class="muted">当前结果源</div>
      <div>{{ output_source }}</div>
    </div>
    <div class="card">
      <div class="muted">唯一 Token 数</div>
      <div>{{ unique_tokens }}</div>
    </div>
    <div class="card">
      <div class="muted">Top5 Token 占比</div>
      <div>{{ top5_share }}</div>
    </div>
  </div>

  {% if interpretation %}
  <div class="section">
    <h2>结果解读</h2>
    <div class="grid">
      <div class="card">
        <div class="muted">目标观察周期</div>
        <div>未来 {{ interpretation.target_horizon }} 根 K 线</div>
      </div>
      <div class="card">
        <div class="muted">最小样本门槛</div>
        <div>{{ interpretation.min_count }} 次</div>
      </div>
      <div class="card">
        <div class="muted">可解释收益 Token 数</div>
        <div>{{ interpretation.eligible_return_tokens }}</div>
      </div>
      <div class="card">
        <div class="muted">可解释风险 Token 数</div>
        <div>{{ interpretation.eligible_risk_tokens }}</div>
      </div>
      <div class="card">
        <div class="muted">收益区分度</div>
        <div>
          <span class="pill {{ 'ok' if interpretation.judgement.has_return_separation else 'warn' }}">
            {{ '存在' if interpretation.judgement.has_return_separation else '不足' }}
          </span>
        </div>
      </div>
      <div class="card">
        <div class="muted">风险区分度</div>
        <div>
          <span class="pill {{ 'ok' if interpretation.judgement.has_risk_separation else 'warn' }}">
            {{ '存在' if interpretation.judgement.has_risk_separation else '不足' }}
          </span>
        </div>
      </div>
    </div>
  </div>
  {% endif %}

  {% if risk_warning %}
  <div class="section">
    <h2>风险预警层</h2>
    <div class="grid">
      <div class="card">
        <div class="muted">当前预警级别</div>
        <div>
          <span class="pill {{ risk_level_class }}">{{ risk_warning.latest_warning_level }}</span>
        </div>
      </div>
      <div class="card">
        <div class="muted">最近高风险 Token 密度</div>
        <div>{{ risk_latest_ratio }}</div>
      </div>
      <div class="card">
        <div class="muted">最近分布漂移</div>
        <div>{{ risk_latest_shift }}</div>
      </div>
      <div class="card">
        <div class="muted">高风险 Token 集合</div>
        <div>{{ risk_tokens_text }}</div>
      </div>
    </div>
  </div>
  {% endif %}

  {% if risk_img %}
  <div class="section">
    <img src="{{ risk_img }}" alt="risk warning layer">
  </div>
  {% endif %}

  {% if state15_summary %}
  <div class="section">
    <h2>15分钟层状态机</h2>
    <div class="grid">
      <div class="card">
        <div class="muted">状态窗口</div>
        <div>{{ state15_summary.state_window }} 根</div>
      </div>
      <div class="card">
        <div class="muted">未来观察窗口</div>
        <div>{{ state15_summary.horizon }} 根</div>
      </div>
      <div class="card">
        <div class="muted">未来波动区分度</div>
        <div>
          <span class="pill {{ 'ok' if state15_summary.state_judgement.has_state_separation_on_future_rv else 'warn' }}">
            {{ '存在' if state15_summary.state_judgement.has_state_separation_on_future_rv else '不足' }}
          </span>
        </div>
      </div>
      <div class="card">
        <div class="muted">高波动事件区分度</div>
        <div>
          <span class="pill {{ 'ok' if state15_summary.state_judgement.has_state_separation_on_high_vol_event else 'warn' }}">
            {{ '存在' if state15_summary.state_judgement.has_state_separation_on_high_vol_event else '不足' }}
          </span>
        </div>
      </div>
      <div class="card">
        <div class="muted">验证期状态分布</div>
        <div>{{ state15_counts_text }}</div>
      </div>
    </div>
  </div>
  {% endif %}

  {% if state15_table %}
  <div class="section">
    <h2>15分钟层状态统计</h2>
    {{ state15_table | safe }}
  </div>
  {% endif %}

  {% if state15_img %}
  <div class="section">
    <img src="{{ state15_img }}" alt="15m state timeline">
  </div>
  {% endif %}

  {% if frequent_table %}
  <div class="section">
    <h2>最常见 Token</h2>
    {{ frequent_table | safe }}
  </div>
  {% endif %}

  {% if strong_table %}
  <div class="section">
    <h2>偏强 Token</h2>
    {{ strong_table | safe }}
  </div>
  {% endif %}

  {% if weak_table %}
  <div class="section">
    <h2>偏弱 Token</h2>
    {{ weak_table | safe }}
  </div>
  {% endif %}

  {% if risk_table %}
  <div class="section">
    <h2>高风险 Token</h2>
    {{ risk_table | safe }}
  </div>
  {% endif %}

  {% if calm_table %}
  <div class="section">
    <h2>低风险 Token</h2>
    {{ calm_table | safe }}
  </div>
  {% endif %}

  {% if preview_table %}
  <div class="section">
    <h2>BTC 15m 数据预览</h2>
    {{ preview_table | safe }}
  </div>
  {% endif %}

  {% if return_table %}
  <div class="section">
    <h2>Token 后续收益统计（前 20 行）</h2>
    {{ return_table | safe }}
  </div>
  {% endif %}

  {% if vol_table %}
  <div class="section">
    <h2>Token 后续波动率统计（前 20 行）</h2>
    {{ vol_table | safe }}
  </div>
  {% endif %}

  {% if token_usage_img %}
  <div class="section">
    <h2>图形输出</h2>
    <img src="{{ token_usage_img }}" alt="token usage">
  </div>
  {% endif %}

  {% if shift_img %}
  <div class="section">
    <img src="{{ shift_img }}" alt="distribution shift">
  </div>
  {% endif %}

  {% if pca_img %}
  <div class="section">
    <img src="{{ pca_img }}" alt="embedding pca">
  </div>
  {% endif %}
</body>
</html>
"""


def run_subprocess(command: list[str]) -> str:
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    prefix = f"exit_code={proc.returncode}\n"
    return prefix + output.strip()


def active_output_dir() -> Path:
    if (KRONOS_OUTPUT_DIR / "summary.json").exists():
        return KRONOS_OUTPUT_DIR
    return PSEUDO_OUTPUT_DIR


def read_summary() -> dict:
    summary_path = active_output_dir() / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path: Path, head: int = 20) -> str:
    if not path.exists():
        return ""
    df = pd.read_csv(path).head(head)
    return df.to_html(index=False, classes="table", border=0)


def read_interpretation_table(path: Path, category: str, head: int = 8) -> str:
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    if "category" not in df.columns:
        return ""
    df = df[df["category"] == category].copy().head(head)
    if df.empty:
        return ""
    if "category" in df.columns:
        df = df.drop(columns=["category"])
    if category == "frequent_tokens":
        keep_cols = [col for col in ["token_id", "occurrences", "share"] if col in df.columns]
        df = df[keep_cols]
    else:
        keep_cols = [
            col
            for col in ["token_id", "horizon", "count", "mean", "median", "std", "q10", "q90"]
            if col in df.columns
        ]
        df = df[keep_cols]
    df = df.dropna(axis=1, how="all")
    return df.to_html(index=False, classes="table", border=0)


def format_state_counts(state_summary: dict) -> str:
    counts = state_summary.get("validation_state_counts", {})
    if not counts:
        return "-"
    order = ["cohesion", "drift", "fracture"]
    parts = []
    for key in order:
        if key in counts:
            parts.append(f"{key}: {counts[key]}")
    for key, value in counts.items():
        if key not in order:
            parts.append(f"{key}: {value}")
    return ", ".join(parts)


def build_context(message: str = "") -> dict:
    data_status = "未拉取"
    data_rows = "-"
    date_range = "-"
    preview_table = ""

    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        data_status = str(CSV_PATH.name)
        data_rows = f"{len(df):,}"
        if "timestamp" in df.columns and len(df) > 0:
            date_range = f"{df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}"
        preview_table = df.head(15).to_html(index=False, border=0)

    summary = read_summary()
    output_dir = active_output_dir()
    interpretation = read_json(output_dir / "interpretation.json")
    risk_warning = read_json(output_dir / "risk_warning_summary.json")
    state15_summary = read_json(STATE15_OUTPUT_DIR / "15m_state_summary.json")
    validation_status = "已生成" if summary else "未运行"
    output_source = output_dir.name if summary else "-"
    unique_tokens = summary.get("num_unique_tokens", "-")
    top5_share = summary.get("token_top5_share", "-")
    if isinstance(top5_share, float):
        top5_share = f"{top5_share:.2%}"

    token_usage_img = (
        url_for("serve_output_file", filename="token_usage_distribution.png")
        if (output_dir / "token_usage_distribution.png").exists()
        else ""
    )
    shift_img = (
        url_for("serve_output_file", filename="rolling_distribution_shift.png")
        if (output_dir / "rolling_distribution_shift.png").exists()
        else ""
    )
    pca_img = (
        url_for("serve_output_file", filename="embedding_projection_pca.png")
        if (output_dir / "embedding_projection_pca.png").exists()
        else ""
    )
    risk_img = (
        url_for("serve_output_file", filename="risk_warning_layer.png")
        if (output_dir / "risk_warning_layer.png").exists()
        else ""
    )
    state15_img = (
        url_for("serve_state15_file", filename="validation_state_timeline.png")
        if (STATE15_OUTPUT_DIR / "validation_state_timeline.png").exists()
        else ""
    )

    level = str(risk_warning.get("latest_warning_level", "green"))
    risk_level_class = "ok"
    if level == "orange":
        risk_level_class = "warn"
    elif level == "red":
        risk_level_class = "danger"

    return {
        "message": message,
        "data_status": data_status,
        "data_rows": data_rows,
        "date_range": date_range,
        "validation_status": validation_status,
        "output_source": output_source,
        "unique_tokens": unique_tokens,
        "top5_share": top5_share,
        "interpretation": interpretation,
        "risk_warning": risk_warning,
        "risk_level_class": risk_level_class,
        "risk_latest_ratio": (
            f"{float(risk_warning.get('latest_high_risk_ratio', 0.0)):.2%}" if risk_warning else "-"
        ),
        "risk_latest_shift": (
            f"{float(risk_warning.get('latest_distribution_shift_l1', 0.0)):.4f}" if risk_warning else "-"
        ),
        "risk_tokens_text": (
            ", ".join(str(x) for x in risk_warning.get("high_risk_tokens", [])) if risk_warning else "-"
        ),
        "frequent_table": read_interpretation_table(output_dir / "interpretation_table.csv", "frequent_tokens"),
        "strong_table": read_interpretation_table(output_dir / "interpretation_table.csv", "strong_tokens"),
        "weak_table": read_interpretation_table(output_dir / "interpretation_table.csv", "weak_tokens"),
        "risk_table": read_interpretation_table(output_dir / "interpretation_table.csv", "risk_tokens"),
        "calm_table": read_interpretation_table(output_dir / "interpretation_table.csv", "calm_tokens"),
        "state15_summary": state15_summary,
        "state15_counts_text": format_state_counts(state15_summary),
        "state15_table": read_table(STATE15_OUTPUT_DIR / "validation_15m_state_summary.csv"),
        "state15_img": state15_img,
        "preview_table": preview_table,
        "return_table": read_table(output_dir / "token_future_return_summary.csv"),
        "vol_table": read_table(output_dir / "token_future_vol_summary.csv"),
        "token_usage_img": token_usage_img,
        "shift_img": shift_img,
        "pca_img": pca_img,
        "risk_img": risk_img,
    }


@app.route("/")
def index():
    return render_template_string(TEMPLATE, **build_context())


@app.post("/run/fetch")
def run_fetch():
    message = run_subprocess(
        [sys.executable, "fetch_binance_btc_15m.py", "--output", "data/btcusdt_15m_1y.csv"]
    )
    return render_template_string(TEMPLATE, **build_context(message))


@app.post("/run/validate/pseudo")
def run_validate_pseudo():
    if not CSV_PATH.exists():
        return render_template_string(
            TEMPLATE, **build_context("exit_code=1\n请先拉取币安 BTC 15m 数据。")
        )

    message = run_subprocess(
        [
            sys.executable,
            "mnts_min_validation.py",
            "--input-csv",
            str(CSV_PATH),
            "--output-dir",
            str(PSEUDO_OUTPUT_DIR),
            "--mode",
            "pseudo",
            "--num-tokens",
            "24",
            "--lookaheads",
            "4,8,16",
            "--rolling-window",
            "96",
        ]
    )
    return render_template_string(TEMPLATE, **build_context(message))


@app.post("/run/validate/kronos")
def run_validate_kronos():
    if not CSV_PATH.exists():
        return render_template_string(
            TEMPLATE, **build_context("exit_code=1\n请先拉取币安 BTC 15m 数据。")
        )
    python_exec = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    message = run_subprocess(
        [
            python_exec,
            "mnts_min_validation.py",
            "--input-csv",
            str(CSV_PATH),
            "--output-dir",
            str(KRONOS_OUTPUT_DIR),
            "--mode",
            "kronos",
            "--kronos-repo-dir",
            str(ROOT / "external" / "Kronos"),
            "--kronos-tokenizer-name",
            str(ROOT / "external" / "kronos_tokenizer_base"),
            "--chunk-size",
            "512",
            "--device",
            "cpu",
            "--lookaheads",
            "4,8,16",
            "--rolling-window",
            "96",
        ]
    )
    return render_template_string(TEMPLATE, **build_context(message))


@app.post("/run/setup-kronos")
def run_setup_kronos():
    message = run_subprocess([sys.executable, "setup_kronos_tokenizer.py"])
    return render_template_string(TEMPLATE, **build_context(message))


@app.post("/run/validate/15m-state")
def run_validate_15m_state():
    if not CSV_2Y_PATH.exists():
        return render_template_string(
            TEMPLATE, **build_context("exit_code=1\n请先准备两年 BTC 15m 数据文件 data/btcusdt_15m_2y.csv。")
        )
    python_exec = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    message = run_subprocess(
        [
            python_exec,
            "mnts_15m_state_layer.py",
            "--input-csv",
            str(CSV_2Y_PATH),
            "--output-dir",
            str(STATE15_OUTPUT_DIR),
            "--kronos-repo-dir",
            str(ROOT / "external" / "Kronos"),
            "--kronos-tokenizer-name",
            "NeoQuasar/Kronos-Tokenizer-base",
            "--chunk-size",
            "512",
            "--device",
            "cpu",
            "--state-window",
            "64",
            "--horizon",
            "16",
        ]
    )
    return render_template_string(TEMPLATE, **build_context(message))


@app.route("/outputs/<path:filename>")
def serve_output_file(filename: str):
    return send_from_directory(str(active_output_dir()), filename)


@app.route("/outputs/15m-state/<path:filename>")
def serve_state15_file(filename: str):
    return send_from_directory(str(STATE15_OUTPUT_DIR), filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, debug=False)
