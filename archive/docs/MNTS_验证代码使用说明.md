# MNTS 验证代码使用说明

## 文件说明

当前目录新增了以下文件：

- `MNTS_最小验证_Checklist.md`
- `mnts_min_validation.py`

其中：

- `Checklist` 用于判断这条路线是否值得继续
- `mnts_min_validation.py` 用于完成最小统计验证

---

## 建议运行方式

优先用 `D:` 盘保存数据、缓存和输出。

如果您只是快速验证“有没有规律”，建议先跑 `pseudo` 模式。

如果后面已经准备好本地 `Kronos` 仓库和依赖，再切换到 `kronos` 模式。

---

## 输入数据要求

CSV 至少包含以下列：

- `timestamp`
- `open`
- `high`
- `low`
- `close`

推荐再包含：

- `volume`

示例列名：

```text
timestamp,open,high,low,close,volume
```

---

## 先跑 Pseudo 模式

这是最省事的第一步。

示例命令：

```bash
python mnts_min_validation.py ^
  --input-csv D:\data\btc_15m_1y.csv ^
  --output-dir D:\mnts_outputs\pseudo_run ^
  --mode pseudo ^
  --num-tokens 24 ^
  --lookaheads 4,8,16 ^
  --rolling-window 96
```

说明：

- `--mode pseudo`：使用脚本内置的伪 tokenizer
- `--num-tokens 24`：伪 token 数量
- `--lookaheads 4,8,16`：统计未来 4/8/16 根 K 线行为
- `--rolling-window 96`：滚动窗口大小

---

## 再跑 Kronos 模式

只有在您本地已经准备好：

- `torch`
- `Kronos` 仓库代码
- `KronosTokenizer` 依赖

时再使用。

示例命令：

```bash
python mnts_min_validation.py ^
  --input-csv D:\data\btc_15m_1y.csv ^
  --output-dir D:\mnts_outputs\kronos_run ^
  --mode kronos ^
  --kronos-repo-dir D:\Projects\Kronos ^
  --kronos-tokenizer-name NeoQuasar/Kronos-Tokenizer-base ^
  --chunk-size 512 ^
  --device cpu
```

说明：

- `--kronos-repo-dir`：本地 Kronos 仓库路径
- `--kronos-tokenizer-name`：预训练 tokenizer 名称
- `--chunk-size 512`：分块处理长度
- `--device cpu`：建议轻量机器先用 `cpu`

注意：

- 当前脚本默认把连续表示定义为 `quantize` 前的 `pre_quant` 表示
- 这更适合做后续结构分析和拓扑候选输入

---

## 输出内容

脚本会输出以下结果到 `--output-dir`：

- `tokenized_dataset.csv`
- `token_future_return_summary.csv`
- `token_future_vol_summary.csv`
- `token_transition_matrix.csv`
- `embedding_projection_pca.csv`
- `summary.json`
- `token_usage_distribution.png`
- `rolling_distribution_shift.png`
- `embedding_projection_pca.png`

---

## 怎么判断值不值得继续

优先对照 `MNTS_最小验证_Checklist.md` 来看。

最关键的几个正信号：

- token 使用分布没有严重塌缩
- 不同 token 后续收益分布存在差异
- 不同 token 后续波动率分布存在差异
- 转移矩阵有结构
- 滚动分布变化能对应市场切换
- 连续表示有初步簇结构

如果满足其中 `3` 条左右，就值得继续深挖。

---

## 当前代码定位

这套代码的定位不是完整交易系统，而是：

- 最小研究验证工具
- 输入层验收工具
- 路线筛查工具

它回答的是：

- `tokenizer 是否真的把 BTC 15m 市场压缩成了一套有规律的状态语言`

它不直接回答：

- `MNTS 是否已经可以交易`

