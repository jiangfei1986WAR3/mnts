# MNTS V2 版本开发到当前状态完整复现说明

## 1. 文档目的

本文档不是概念讨论稿，而是面向复现和继续开发的工程交接文档。

目标是让一个新的开发者，或者另一个 AI 编程助手，在只看到当前项目目录和本文档的情况下，能够理解下面这些问题，并把当前 `V2` 主线完整复现出来：

1. `V2` 是怎么从 `15m` 状态层演化出来的。
2. 它依赖哪些数据、哪些脚本、哪些输出文件。
3. 哪些实验已经做过，结论是什么。
4. 当前最强主线为什么从 `V2 单独识别` 演化成了 `V2 + 趋势/突破策略`。
5. 当前最强的可复现路径是什么。
6. 如果要继续推进，应该从哪里接着写代码。

本文档默认工作目录为：

`d:\Projects\mnts`

---

## 2. 当前结论先说清楚

截至目前，项目得到的最重要结论是：

1. `Kronos tokenizer` 的 `token_id` 和 `pre_quant embedding` 都确实包含可用于市场状态建模的信息。
2. 纯粹让 `V2` 自己单打独斗，和经典指标比，并不稳定占优。
3. `V2` 最有价值的位置，不是独立 alpha 引擎，而是“市场环境过滤层 / 状态解释层 / 风险门控层”。
4. 当 `V2` 与趋势/突破类策略耦合时，效果明显增强。
5. 当前最强主线，不是 `MACD + V2`，而是 `breakout + V2`。
6. 经过更长历史、完整双滚动、参数稳定性测试后，`breakout + V2` 已经形成“强候选主策略”。

当前最关键的工程结论是：

- `V2` 与趋势/突破策略结合，确实能发挥很强威力。
- 当前最强稳定区域在：
  - `breakout window = 40 / 48 / 56`
  - `gate = cohesion`
  - `hold = 8 / 12`
  - `cool = 0 / 4 / 8 / 12`

当前参数稳定性测试里最强配置是：

- `bw48_cohesion_hold8_cool0`

它在 `4年历史` 的完整双滚动参数稳定性测试中，表现为：

- `1bps`: `+198.81%`, `Sharpe 3.26`, `MaxDD -10.51%`
- `4bps`: `+152.52%`, `Sharpe 2.76`, `MaxDD -10.57%`

注意：

- 这不是“4 年整段连续交易收益”。
- 这是在 `4年历史` 下做完整双滚动后，把样本外测试窗口拼接起来的结果。
- 拼接后的有效样本外长度约 `69118` 根 `15m` K 线，约等于 `1.9年`。

---

## 3. V2 主线是怎么演化出来的

### 3.1 阶段 0：先确认 Kronos tokenizer 能不能用于 MNTS

项目一开始并没有直接开发交易系统，而是先回答两个底层问题：

1. `Kronos tokenizer` 产出的东西到底是什么。
2. 它能不能作为市场“形态语义 token”的基础。

结论是：

1. `token_id` 不能直接当几何坐标用，不能直接做拓扑点云距离。
2. 但 `token_id` 可以用于状态统计、频率统计、转移统计、解释层。
3. `pre_quant embedding` 比 `token_id` 信息更丰富，可以做更细的连续异常度和状态特征。

这一步真正落地在：

- `mnts_min_validation.py`

这个文件定义了两个底层 tokenizer adapter：

1. `PseudoTokenizerAdapter`
2. `KronosTokenizerAdapter`

其中 `KronosTokenizerAdapter` 的关键逻辑是：

1. 从本地 `Kronos` 仓库导入 `KronosTokenizer`
2. 读入 `OHLCV + volume + amount`
3. 对每个 chunk 做归一化
4. 取 `tokenizer.quant_embed(z)` 的 `pre_quant` 表示作为 embedding
5. 取 `s1_ids, s2_ids`
6. 合并成 `combined token_id`

关键代码路径：

- `KronosTokenizerAdapter.fit_transform()` in `mnts_min_validation.py`

---

### 3.2 阶段 1：最小验证，先证明“有没有信息量”

最小验证脚本仍然是：

- `mnts_min_validation.py`

这一步的目标不是交易，而是回答：

1. 不同 token 是否对应不同的后续收益分布。
2. 不同 token 是否对应不同的后续波动率分布。
3. token 分布漂移是否能形成状态提示。

这一步的重要产出包括：

1. `token_id`
2. `token_s1`, `token_s2`
3. `embeddings`
4. `forward_returns`
5. `future_realized_vol`
6. `l1_distribution_shift`

如果重新从这里验证，可以先跑：

```powershell
.\.venv\Scripts\python.exe mnts_min_validation.py `
  --input-csv data/btcusdt_15m_1y.csv `
  --output-dir validation_outputs/kronos_run `
  --mode kronos `
  --kronos-repo-dir external/Kronos `
  --kronos-tokenizer-name NeoQuasar/Kronos-Tokenizer-base `
  --chunk-size 512 `
  --device cpu
```

这一步证明的是：

- `Kronos tokenizer` 这条路线值得继续。

---

### 3.3 阶段 2：15m 状态层 V1

文件：

- `mnts_15m_state_layer.py`

目标：

- 不再只看单个 token，而是把最近一个滚动窗口内的 token 分布动态，压成三态：
  - `cohesion`
  - `drift`
  - `fracture`

V1 的核心特征：

1. `distribution_shift_l1`
2. `token_entropy`
3. `dominant_token_share`

V1 的方法比较简单：

1. 用前半段数据做 discovery
2. discovery 上按分位数拟合阈值
3. 在后半段 validation 上冻结应用规则

V1 的意义是：

- 第一次把“15 分钟状态层”从概念变成了可跑、可观察的状态机。

---

### 3.4 阶段 3：15m 状态层 V2

文件：

- `mnts_15m_state_layer_v2.py`

这是 `V2` 的真正起点，也是当前主线的基础。

#### 3.4.1 V2 比 V1 多了什么

V2 不再只看三项简单统计，而是加入了更细的动态特征：

1. `distribution_shift_l1`
2. `token_entropy`
3. `entropy_delta`
4. `switch_rate`
5. `embedding_anomaly`
6. `dominant_token_share`

其中：

- `switch_rate` 用来衡量最近窗口 token 切换速度。
- `embedding_anomaly` 用 embedding 到训练中心的距离，衡量当前形态异常度。
- `dominant_token_share` 是反向特征，越高通常越稳定。

#### 3.4.2 V2 的状态模型如何拟合

核心函数：

- `fit_v2_model()`
- `compute_instability_score()`
- `apply_v2_model()`

V2 的拟合逻辑：

1. discovery 段计算未来波动率 `fwd_rv`
2. 取 `fwd_rv` 的 `90%` 分位，定义高波动事件
3. 比较高波动样本和普通样本在每个特征上的均值差
4. 对每个特征做 zscore
5. 用加权和得到 `instability_score`
6. 用 `score_q30 / score_q70` 把状态切成：
   - `cohesion`
   - `drift`
   - `fracture`

这意味着：

- `V2` 本质不是监督分类器，而是一个轻量的、面向高波动风险的状态评分器。

#### 3.4.3 运行命令

`2年` 版本：

```powershell
.\.venv\Scripts\python.exe mnts_15m_state_layer_v2.py `
  --input-csv data/btcusdt_15m_2y.csv `
  --output-dir validation_outputs/15m_state_v2_run `
  --kronos-repo-dir external/Kronos `
  --kronos-tokenizer-name NeoQuasar/Kronos-Tokenizer-base `
  --chunk-size 512 `
  --device cpu `
  --state-window 64 `
  --horizon 16
```

`4年` 版本：

```powershell
.\.venv\Scripts\python.exe mnts_15m_state_layer_v2.py `
  --input-csv data/btcusdt_15m_4y.csv `
  --output-dir validation_outputs/15m_state_v2_4y_run `
  --kronos-repo-dir external/Kronos `
  --kronos-tokenizer-name NeoQuasar/Kronos-Tokenizer-base `
  --chunk-size 512 `
  --device cpu `
  --state-window 64 `
  --horizon 16
```

#### 3.4.4 核心输出文件

以 `2年` 为例，关键输出是：

- `validation_outputs/15m_state_v2_run/discovery_year_15m_v2_tokenized.csv`
- `validation_outputs/15m_state_v2_run/validation_year_15m_v2_states.csv`
- `validation_outputs/15m_state_v2_run/15m_state_v2_summary.json`
- `validation_outputs/15m_state_v2_run/validation_15m_v2_state_summary.csv`
- `validation_outputs/15m_state_v2_run/validation_state_v2_timeline.png`

其中最重要的是：

- `discovery_year_15m_v2_tokenized.csv`
- `validation_year_15m_v2_states.csv`

因为后面大多数实验都直接复用这两个输出，不再重复跑 tokenizer。

---

## 4. 为什么 V2 没有停留在“单独打高波动识别”

### 4.1 同口径对比经典指标

文件：

- `compare_v2_vs_classic_indicators.py`

目标：

- 把 `V2` 和 `MA / RSI / MACD` 放在同样本外、同任务口径下比较。

任务定义：

- 未来 `16` 根 `15m` K 线是否进入高波动前 `10%`

结论：

1. `V2` 有信息量。
2. 但 `V2` 单独做这个任务时，不稳定优于经典指标。

这一步的重要意义是：

- 它迫使主线从“V2 自己打天下”转向“V2 给别的策略做环境过滤层”。

运行命令：

```powershell
.\.venv\Scripts\python.exe compare_v2_vs_classic_indicators.py `
  --input-csv data/btcusdt_15m_2y.csv `
  --v2-validation-csv validation_outputs/15m_state_v2_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/classic_compare_run `
  --horizon 16
```

输出目录：

- `validation_outputs/classic_compare_run`

---

### 4.2 多策略家族比较

文件：

- `compare_v2_across_strategy_families.py`

目标：

- 不再只拿 `MACD` 对比，而是看 `V2` 与多种策略家族耦合效果。

测试的策略包括：

1. `ma_cross`
2. `macd_regime`
3. `donchian_20`
4. `breakout_48`
5. `pullback_trend`
6. `rsi_reversion`

每个策略都比较四种模式：

1. `raw`
2. `no_fracture`
3. `cohesion_only`
4. `defensive_scale`

结论：

1. `V2` 对趋势 / 突破类更有帮助。
2. `V2` 对简单均值回归类帮助很弱，甚至可能有害。
3. `pullback_trend + V2`
4. `breakout_48 + V2`

是随后进入工程化推进的两条主线。

运行示例：

```powershell
.\.venv\Scripts\python.exe compare_v2_across_strategy_families.py `
  --input-csv data/btcusdt_15m_2y.csv `
  --v2-validation-csv validation_outputs/15m_state_v2_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/strategy_family_compare_run_fee1 `
  --fee-bps 1
```

```powershell
.\.venv\Scripts\python.exe compare_v2_across_strategy_families.py `
  --input-csv data/btcusdt_15m_2y.csv `
  --v2-validation-csv validation_outputs/15m_state_v2_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/strategy_family_compare_run_fee4 `
  --fee-bps 4
```

关键结论文件：

- `validation_outputs/strategy_family_compare_run_fee1/strategy_family_summary.json`
- `validation_outputs/strategy_family_compare_run_fee4/strategy_family_summary.json`

当时的核心结论是：

- `breakout_48` 和 `pullback_trend` 值得继续工程化。

---

## 5. 从“方向验证”进入“工程化策略”

文件：

- `engineer_pullback_breakout_v2.py`

这是把：

1. `pullback_trend + V2`
2. `breakout_48 + V2`

真正工程化为可执行交易规则的脚本。

### 5.1 pullback 分支怎么定义

1. 用 `EMA20 / EMA60` 定义趋势方向。
2. 上升趋势里，价格相对 `EMA20` 回撤足够多才考虑做多。
3. 下降趋势里，价格相对 `EMA20` 反弹足够多才考虑做空。
4. 再叠加 `V2 gate`、状态确认、最短持仓和冷却。

### 5.2 breakout 分支怎么定义

1. 看最近 `48` 根 `15m` K 线的最高/最低区间。
2. 向上突破区间上沿，做多。
3. 向下突破区间下沿，做空。
4. 再叠加：
   - `gate_mode`
   - `state_confirm_bars`
   - `exit_fracture_bars`
   - `min_hold_bars`
   - `cooldown_bars`

### 5.3 当前最关键的参数含义

以 `breakout48_cohesion_flip_exit2_hold12_cool8` 为例：

1. `breakout48`
   - 看最近 `48` 根 `15m` K 线突破
2. `cohesion`
   - 只有处于 `cohesion` 且连续满足状态确认时才允许交易
3. `flip`
   - 允许多空翻转
4. `exit2`
   - 连续 `2` 根进入 `fracture` 时退出
5. `hold12`
   - 最短持仓 `12` 根，即 `3 小时`
6. `cool8`
   - 平仓后冷却 `8` 根，即 `2 小时`

### 5.4 这一步的意义

这一步后，项目明确出现结论：

- `breakout_48 + V2` 强于 `pullback_trend + V2`

运行示例：

```powershell
.\.venv\Scripts\python.exe engineer_pullback_breakout_v2.py `
  --input-csv data/btcusdt_15m_2y.csv `
  --v2-validation-csv validation_outputs/15m_state_v2_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/engineered_pullback_breakout_run_fee1 `
  --fee-bps 1
```

---

## 6. 为什么后来重点变成 breakout + V2

原因很简单：

1. `breakout` 与 `V2` 的语义更一致。
2. `V2` 的本质是状态过滤层。
3. 突破策略本身更依赖市场叙事的凝聚与延续。
4. 所以 `cohesion` 状态更容易筛出高质量突破。

从那一步之后，主线基本固定为：

- `breakout + V2`

---

## 7. 完整双滚动：不只滚动策略，也滚动 V2 状态模型

文件：

- `rolling_breakout_v2_full_system_walkforward.py`

这是整个项目当前最关键的“系统级验证脚本”。

### 7.1 它解决了什么问题

之前只滚动交易策略配置，还不能完全排除未来信息。

这个脚本进一步要求：

1. 每个窗口只用过去 `180天` 数据
2. 先重拟合一版 `V2 model`
3. 用这版 `V2 model` 给未来 `45天` 打状态
4. 再在这个新状态上选择 breakout 执行配置
5. 再跑这 `45天` 的样本外

也就是说：

- `V2` 状态层和 breakout 执行层都一起滚动。

### 7.2 为什么这个脚本没有慢得不可接受

因为它复用了 `mnts_15m_state_layer_v2.py` 预先产出的缓存特征：

- `discovery_year_15m_v2_tokenized.csv`
- `validation_year_15m_v2_states.csv`

所以它没有在每个窗口里重新跑 tokenizer，而只做：

1. `V2` 重新拟合
2. 状态重打标
3. breakout 配置选择
4. 样本外执行

### 7.3 默认窗口配置

```text
train_bars = 11520   -> 180天
test_bars  = 2880    -> 45天
```

### 7.4 运行命令

`2年` 版：

```powershell
.\.venv\Scripts\python.exe rolling_breakout_v2_full_system_walkforward.py `
  --discovery-v2-csv validation_outputs/15m_state_v2_run/discovery_year_15m_v2_tokenized.csv `
  --validation-v2-csv validation_outputs/15m_state_v2_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/rolling_breakout_full_system_run_fee1 `
  --fee-bps 1
```

```powershell
.\.venv\Scripts\python.exe rolling_breakout_v2_full_system_walkforward.py `
  --discovery-v2-csv validation_outputs/15m_state_v2_run/discovery_year_15m_v2_tokenized.csv `
  --validation-v2-csv validation_outputs/15m_state_v2_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/rolling_breakout_full_system_run_fee4 `
  --fee-bps 4
```

`4年` 版：

```powershell
.\.venv\Scripts\python.exe rolling_breakout_v2_full_system_walkforward.py `
  --discovery-v2-csv validation_outputs/15m_state_v2_4y_run/discovery_year_15m_v2_tokenized.csv `
  --validation-v2-csv validation_outputs/15m_state_v2_4y_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/rolling_breakout_full_system_4y_run_fee1 `
  --fee-bps 1
```

```powershell
.\.venv\Scripts\python.exe rolling_breakout_v2_full_system_walkforward.py `
  --discovery-v2-csv validation_outputs/15m_state_v2_4y_run/discovery_year_15m_v2_tokenized.csv `
  --validation-v2-csv validation_outputs/15m_state_v2_4y_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/rolling_breakout_full_system_4y_run_fee4 `
  --fee-bps 4
```

### 7.5 当前 `4年` 完整双滚动结果

结果文件：

- `validation_outputs/rolling_breakout_full_system_4y_run_fee1/rolling_breakout_full_system_summary.json`
- `validation_outputs/rolling_breakout_full_system_4y_run_fee4/rolling_breakout_full_system_summary.json`

关键结果：

`1bps`

- `23` 个样本外窗口
- `18` 个正窗口
- `net_total_return = +266.60%`
- `net_sharpe = 2.67`
- `net_max_drawdown = -16.21%`

`4bps`

- `23` 个样本外窗口
- `16` 个正窗口
- `net_total_return = +145.34%`
- `net_sharpe = 1.91`
- `net_max_drawdown = -19.18%`

这一步证明：

- `breakout + V2` 不只是短样本漂亮，而是在更长历史、更严格双滚动里仍然成立。

---

## 8. 参数稳定性：证明不是只踩中一个幸运点

文件：

- `breakout_v2_parameter_stability_4y.py`

这是当前阶段用来压低过拟合嫌疑的关键脚本。

### 8.1 为什么要做参数稳定性

如果只有：

- `48 / hold12 / cool8`

这个点特别亮，而周围参数一改就塌，那就很可能是过拟合。

如果附近一圈参数都还活着，说明抓到的是一个“稳定参数区域”，不是“针尖”。

### 8.2 这个脚本测了什么

参数网格：

1. `breakout_window`: `40, 48, 56, 64`
2. `min_hold_bars`: `8, 12, 16`
3. `cooldown_bars`: `0, 4, 8, 12`
4. `gate_mode`: `nonfracture, cohesion`

总共：

```text
4 × 3 × 4 × 2 = 96 组
```

对每一组都跑：

1. `4年` 缓存特征
2. 完整双滚动
3. `1bps / 4bps`

### 8.3 运行命令

```powershell
.\.venv\Scripts\python.exe breakout_v2_parameter_stability_4y.py `
  --discovery-v2-csv validation_outputs/15m_state_v2_4y_run/discovery_year_15m_v2_tokenized.csv `
  --validation-v2-csv validation_outputs/15m_state_v2_4y_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/breakout_v2_param_stability_4y_run
```

### 8.4 输出文件

- `parameter_stability_summary.json`
- `parameter_stability_summary.csv`
- `parameter_stability_top20.csv`
- `parameter_stability_anchor_region.csv`
- `parameter_stability_window_results.csv`

### 8.5 当前参数稳定性结论

来自：

- `validation_outputs/breakout_v2_param_stability_4y_run/parameter_stability_summary.json`

结论如下：

`1bps`

- `96 / 96` 组参数为正收益
- `96 / 96` 组 `Sharpe > 1`
- `94 / 96` 组 `Sharpe > 1.5`

`4bps`

- `96 / 96` 组参数为正收益
- `82 / 96` 组 `Sharpe > 1`
- `64 / 96` 组 `Sharpe > 1.5`

两组费用下最强配置都是：

- `bw48_cohesion_hold8_cool0`

这说明：

- 当前主线不是单点运气，而是存在一片稳定参数区域。

### 8.6 当前最强参数区域怎么理解

最稳定的强区大体集中在：

1. `breakout_window = 40 / 48 / 56`
2. `gate_mode = cohesion`
3. `hold = 8 / 12`
4. `cool = 0 / 4 / 8 / 12`

这一步得到的核心工程结论是：

- `V2 + breakout` 的有效性，不是只靠一个幸运参数点撑起来。

---

## 9. 交易单层面的闭合交易分析

文件：

- `analyze_breakout_trade_outcomes.py`

这个脚本不是为了再做策略搜索，而是把某个固定配置拆成“闭合交易单”，统计：

1. 总共多少笔
2. 赢多少笔
3. 亏多少笔
4. 多头和空头谁贡献更大

### 9.1 运行命令

```powershell
.\.venv\Scripts\python.exe analyze_breakout_trade_outcomes.py `
  --discovery-v2-csv validation_outputs/15m_state_v2_4y_run/discovery_year_15m_v2_tokenized.csv `
  --validation-v2-csv validation_outputs/15m_state_v2_4y_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/breakout_trade_outcomes_bw48_cohesion_hold8_cool0 `
  --breakout-window 48 `
  --gate-mode cohesion `
  --min-hold-bars 8 `
  --cooldown-bars 0
```

### 9.2 当前最强参数的闭合交易结果

结果文件：

- `validation_outputs/breakout_trade_outcomes_bw48_cohesion_hold8_cool0/trade_outcomes_summary.json`

结果如下：

1. 闭合交易总数：`285`
2. 盈利：`138`
3. 亏损：`147`
4. 胜率：`48.42%`
5. 平均持有：`46.42` 根 `15m` K 线，约 `11.6小时`
6. 持有中位数：`37` 根，约 `9.25小时`

#### 多空贡献拆解

当前最强参数里：

- 空头明显强于多头

`1bps`

- 多头：
  - `158` 笔
  - 胜率 `43.04%`
  - 平均每笔净收益 `+0.0728%`
- 空头：
  - `127` 笔
  - 胜率 `55.12%`
  - 平均每笔净收益 `+0.8206%`

`4bps`

- 多头：
  - 平均每笔净收益 `+0.0346%`
- 空头：
  - 平均每笔净收益 `+0.7782%`

这意味着：

- 当前最强配置的利润主力是空头。

---

## 10. 数据、环境、依赖和目录准备

### 10.1 推荐运行环境

当前项目实际运行环境是：

1. Windows
2. Python 3.12
3. 本地虚拟环境 `.venv`
4. `torch` CPU 版可用

建议至少保证 `.venv` 中安装：

1. `numpy`
2. `pandas`
3. `matplotlib`
4. `torch`
5. `huggingface_hub`
6. `flask`

说明：

- 文档不强行写死一份 `requirements.txt`，因为当前项目是逐步搭出来的。
- 但从脚本依赖上看，上面这些是最低限度。

### 10.2 下载 Binance 数据

脚本：

- `fetch_binance_btc_15m.py`

示例：

拉 `2年`：

```powershell
python fetch_binance_btc_15m.py --months 24 --output data/btcusdt_15m_2y.csv
```

拉 `4年`：

```powershell
python fetch_binance_btc_15m.py --months 48 --output data/btcusdt_15m_4y.csv
```

当前已有数据文件：

- `data/btcusdt_15m_1y.csv`
- `data/btcusdt_15m_2y.csv`
- `data/btcusdt_15m_4y.csv`

### 10.3 下载 Kronos 代码和 tokenizer 资源

脚本：

- `setup_kronos_tokenizer.py`

它会：

1. 克隆官方 `Kronos` 仓库到 `external/Kronos`
2. 下载 `NeoQuasar/Kronos-Tokenizer-base` 到 `external/kronos_tokenizer_base`
3. 把 HF cache 导向当前目录而不是系统盘默认位置

运行：

```powershell
python setup_kronos_tokenizer.py
```

当前目录约定：

- `external/Kronos`
- `external/kronos_tokenizer_base`

---

## 11. 如果只想复现当前最强主线，最短路径是什么

如果新开发者不想重复所有历史实验，而是只想复现当前 `V2` 最强主线，建议按下面顺序执行。

### 第一步：准备 `4年 BTCUSDT 15m` 数据

确保存在：

- `data/btcusdt_15m_4y.csv`

### 第二步：准备 Kronos 和 tokenizer

确保存在：

- `external/Kronos`
- `external/kronos_tokenizer_base`

### 第三步：生成 `4年 V2` 状态缓存

```powershell
.\.venv\Scripts\python.exe mnts_15m_state_layer_v2.py `
  --input-csv data/btcusdt_15m_4y.csv `
  --output-dir validation_outputs/15m_state_v2_4y_run `
  --kronos-repo-dir external/Kronos `
  --kronos-tokenizer-name NeoQuasar/Kronos-Tokenizer-base `
  --chunk-size 512 `
  --device cpu `
  --state-window 64 `
  --horizon 16
```

### 第四步：跑 `4年` 完整双滚动主线

```powershell
.\.venv\Scripts\python.exe rolling_breakout_v2_full_system_walkforward.py `
  --discovery-v2-csv validation_outputs/15m_state_v2_4y_run/discovery_year_15m_v2_tokenized.csv `
  --validation-v2-csv validation_outputs/15m_state_v2_4y_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/rolling_breakout_full_system_4y_run_fee1 `
  --fee-bps 1
```

```powershell
.\.venv\Scripts\python.exe rolling_breakout_v2_full_system_walkforward.py `
  --discovery-v2-csv validation_outputs/15m_state_v2_4y_run/discovery_year_15m_v2_tokenized.csv `
  --validation-v2-csv validation_outputs/15m_state_v2_4y_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/rolling_breakout_full_system_4y_run_fee4 `
  --fee-bps 4
```

### 第五步：跑参数稳定性

```powershell
.\.venv\Scripts\python.exe breakout_v2_parameter_stability_4y.py `
  --discovery-v2-csv validation_outputs/15m_state_v2_4y_run/discovery_year_15m_v2_tokenized.csv `
  --validation-v2-csv validation_outputs/15m_state_v2_4y_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/breakout_v2_param_stability_4y_run
```

### 第六步：跑闭合交易分析

```powershell
.\.venv\Scripts\python.exe analyze_breakout_trade_outcomes.py `
  --discovery-v2-csv validation_outputs/15m_state_v2_4y_run/discovery_year_15m_v2_tokenized.csv `
  --validation-v2-csv validation_outputs/15m_state_v2_4y_run/validation_year_15m_v2_states.csv `
  --output-dir validation_outputs/breakout_trade_outcomes_bw48_cohesion_hold8_cool0 `
  --breakout-window 48 `
  --gate-mode cohesion `
  --min-hold-bars 8 `
  --cooldown-bars 0
```

如果只按这 6 步做，就能把当前最强 `V2` 主线完整复现出来。

---

## 12. 当前最重要的脚本和职责清单

### 数据与底层

1. `fetch_binance_btc_15m.py`
   - 从 Binance 抓取 `BTCUSDT 15m` 历史数据

2. `setup_kronos_tokenizer.py`
   - 下载官方 `Kronos` 仓库和 tokenizer 资源

3. `mnts_min_validation.py`
   - 最小验证总入口
   - 定义 tokenizer adapter
   - 提供 forward return / future RV / shift 等底层函数

### 状态层

4. `mnts_15m_state_layer.py`
   - 15m 状态层 V1

5. `mnts_15m_state_layer_v2.py`
   - 15m 状态层 V2
   - 当前主线的状态层基础

### 比较与方向选择

6. `compare_v2_vs_classic_indicators.py`
   - `V2` vs `MA / RSI / MACD`

7. `compare_v2_across_strategy_families.py`
   - `V2` 与不同策略家族的耦合比较

### 工程化策略

8. `engineer_pullback_breakout_v2.py`
   - `pullback + V2`
   - `breakout + V2`
   - 当前策略工程化原型

### 系统级验证

9. `rolling_breakout_v2_full_system_walkforward.py`
   - 完整双滚动验证
   - 当前系统级样本外验证核心脚本

10. `breakout_v2_parameter_stability_4y.py`
    - 参数稳定性地图
    - 当前“反过拟合”关键脚本

11. `analyze_breakout_trade_outcomes.py`
    - 闭合交易拆单
    - 统计胜率、多空贡献、持仓时长

---

## 13. 当前还没有解决的问题

虽然 `V2` 主线已经推进得很深，但下面这些问题还没有彻底解决：

1. 还没有加入更真实的滑点模型。
2. 还没有加入资金费率。
3. 还没有做分批成交或流动性限制。
4. 还没有做跨市场泛化测试。
5. 还没有真正进入更高层级的“信息几何 / 统计流形”工程化版本。
6. 当前最强配置空头贡献明显大于多头，多头侧还值得进一步拆解。

---

## 14. 如果让另一个 AI 接手，最重要的提示是什么

如果后续由另一个 AI 接手开发，建议明确告诉它下面这些约束：

1. 不要回到空谈 `V2 是否有信息量`，这一步已经验证过。
2. 当前最重要的主线是 `breakout + V2`，不是 `MACD + V2`。
3. 优先复用缓存，不要在每个滚动窗口重复跑 tokenizer。
4. 关键缓存文件是：
   - `discovery_year_15m_v2_tokenized.csv`
   - `validation_year_15m_v2_states.csv`
5. 当前最重要的验证顺序是：
   - 先状态层缓存
   - 再完整双滚动
   - 再参数稳定性
   - 再闭合交易分析
6. 当前最佳参数不是“唯一点”，而是一片稳定区域。
7. 当前最强参数 `bw48_cohesion_hold8_cool0` 的收益主力主要来自空头。

---

## 15. 当前建议的下一步

如果继续沿当前最强主线推进，最自然的下一步不是发散新策略，而是下面两项之一：

1. 给当前稳定参数区域加入更真实执行成本：
   - 滑点
   - 资金费率
   - 分批成交

2. 把多头和空头彻底拆开：
   - `long-only`
   - `short-only`
   - `long-short`

如果目标是让系统更接近真实可交易，我更建议先做第 1 项。

---

## 16. 一句话总结

截至当前，`MNTS V2` 已经从“基于 Kronos tokenizer 的 15m 状态层设想”，演化成了一条明确的工程主线：

- `Kronos tokenizer`
- `V2 state layer`
- `breakout + V2`
- `完整双滚动`
- `参数稳定性`
- `闭合交易分析`

而且目前数据支持：

- 这条路线不是短期巧合，
- 不只是一个幸运参数点，
- 并且在 `4年历史`、`完整双滚动`、`1bps / 4bps` 下仍然表现很强。

