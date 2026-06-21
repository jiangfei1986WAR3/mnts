# MNTS 项目迁移恢复与独立建仓说明

## 1. 文档目的

这份文档服务于两个目标：

1. 当项目迁移到新服务器时，可以按本文档把当前项目恢复到可继续开发、可继续回测的状态。
2. 当项目需要单独建一个远程仓库时，可以按本文档把当前目录整理成一个适合推送和长期维护的独立仓库。

本文档默认当前项目根目录为：

`D:\Projects\mnts`

---

## 2. 当前检查结果

### 2.1 Git / 远程仓库现状

我已经检查过当前目录：

- `D:\Projects\mnts` 目前 **不是 git 仓库**
- 当前目录下 **没有 `.git`**
- 因此当前目录也 **没有可检查的 remote**

这意味着：

1. 现在不能直接检查“当前项目绑定了哪个远程仓库”
2. 也不能直接把当前目录“推到现有 remote”
3. 如果要把这个项目单独上传，需要先把它初始化成一个新的独立仓库

### 2.2 体积判断

当前项目里几块主要目录体积如下：

- `data/` 约 `43.18 MB`
- `validation_outputs/` 约 `56.13 MB`
- `external/Kronos/` 约 `25.21 MB`
- `.venv/` 约 `818.79 MB`
- `.pydeps/` 约 `57.63 MB`

结论：

- 如果排除 `.venv/` 和 `.pydeps/`，当前项目整体做成一个单独仓库是 **可行的**
- 当前原始数据文件最大只有 `24.68 MB`
- 没有发现单个超大文件接近 GitHub 单文件 `100 MB` 限制

所以：

- **可以单独建仓**
- **不建议把 `.venv/` 和 `.pydeps/` 推上去**
- `data/`、`validation_outputs/`、代码和文档都可以按需纳入

### 2.3 一个必须注意的点

`external/Kronos/` 当前内部带有它自己的 `.git`。

这表示它是一个“嵌套 git 仓库”。

如果直接把 `D:\Projects\mnts` 初始化成父仓库，再直接 `git add .`，通常会出现下面两种问题之一：

1. Git 把 `external/Kronos` 当成嵌套仓库处理
2. Git 只记录一个 gitlink，而不是把 `Kronos` 代码内容真正纳入父仓库

因此如果要做“单独一个完整仓库”，必须先做下面二选一：

1. **推荐**：删除 `external/Kronos/.git`，把它当成普通目录纳入当前项目仓库
2. 或者把它改造成真正的 submodule

如果目标是“未来换服务器后拉下来就能直接恢复”，推荐第 1 种，最简单。

---

## 3. 建仓策略建议

### 3.1 推荐方案：自包含仓库

推荐把这个项目整理成一个“自包含仓库”。

意思是：

- 代码放进去
- 文档放进去
- 原始数据放进去
- 当前关键输出放进去
- `external/Kronos` 源码也放进去
- 但不放 Python 虚拟环境和本地依赖缓存

这样做的优点是：

1. 新服务器 `git clone` 后，不需要再到处找散落文件
2. 研究状态、历史输出和当前代码是同一份快照
3. 恢复文档能真正对应到一个完整可落地的目录

### 3.2 不推荐方案：只推代码，不推数据和输出

这种方案也能做，但恢复成本高很多，因为你还要额外找回：

- `data/*.csv`
- `validation_outputs/15m_state_v2_run`
- `validation_outputs/15m_state_v2_4y_run`
- 各实验结果输出

如果未来主要目的是“可恢复、可迁移、可继续研究”，不建议只传轻量代码壳子。

---

## 4. 建议纳入仓库的内容

### 4.1 必须纳入

这些内容建议一定纳入：

- 根目录所有正式研究脚本 `*.py`
- 根目录所有正式说明文档 `*.md`
- `requirements-mnts-validation.txt`
- `data/`
- `validation_outputs/`
- `external/Kronos/` 的源码内容
- `archive/docs/`

### 4.2 强烈建议保留的关键文件

以下内容对“恢复当前研究状态”特别关键：

- `mnts_min_validation.py`
- `mnts_15m_state_layer_v2.py`
- `compare_v2_across_strategy_families.py`
- `engineer_pullback_breakout_v2.py`
- `rolling_breakout_v2_walkforward.py`
- `rolling_breakout_v2_full_system_walkforward.py`
- `breakout_v2_parameter_stability_4y.py`
- `analyze_breakout_trade_outcomes.py`
- `adx_di_v2_experiment.py`
- `adx_di_v2_low_turnover_experiment.py`
- `rolling_adxdi_v2_low_turnover_walkforward.py`
- `rolling_adxdi_v2_full_system_walkforward.py`
- `keltner_channel_v2_experiment.py`
- `keltner_channel_v2_low_turnover_experiment.py`
- `rolling_keltner_channel_v2_low_turnover_walkforward.py`
- `vwap_anchored_vwap_v2_experiment.py`
- `supertrend_atr_regime_v2_experiment.py`
- `MNTS_V2_版本开发到当前状态完整复现说明.md`
- 本文档 `MNTS_项目迁移恢复与独立建仓说明.md`

### 4.3 可不纳入仓库

这些建议不要纳入：

- `.venv/`
- `.pydeps/`
- `__pycache__/`
- 本地 HF 缓存目录，比如 `.hf_cache/`
- 编辑器缓存、日志、临时文件

---

## 5. 新服务器恢复目标

换服务器后，恢复完成的标准不是“文件在”，而是下面 4 件事成立：

1. 能运行 `mnts_15m_state_layer_v2.py`
2. 能读取 `data/btcusdt_15m_2y.csv` 和 `data/btcusdt_15m_4y.csv`
3. 能直接使用现有 `validation_outputs/15m_state_v2_run/validation_year_15m_v2_states.csv`
4. 能继续运行趋势骨架实验脚本，例如 `breakout`、`ADX`、`Keltner`、`VWAP`、`Supertrend`

---

## 6. 新服务器恢复步骤

下面按最稳妥的方式恢复。

### 6.1 拉取仓库

```powershell
git clone <你的新仓库地址> mnts
cd mnts
```

### 6.2 创建 Python 虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 6.3 安装基础依赖

先装项目已经明确写在文件里的依赖：

```powershell
pip install -r requirements-mnts-validation.txt
```

`requirements-mnts-validation.txt` 当前包含：

- `numpy`
- `pandas`
- `matplotlib`
- `Flask`
- `huggingface_hub`

### 6.4 安装 Kronos 运行依赖

如果你要跑 `kronos` 路线，而不只是读现有输出，需要额外安装 `torch`。

CPU 版示例：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

如果新服务器有 NVIDIA GPU，可以按服务器 CUDA 版本安装对应 GPU 版 `torch`。

### 6.5 检查 `external/Kronos`

如果仓库里已经包含了 `external/Kronos/` 内容，这一步只需要确认它在：

```powershell
dir .\external\Kronos
```

如果你选择的是“轻量仓库方案”，没有把 `external/Kronos/` 放进仓库，则需要运行：

```powershell
python .\setup_kronos_tokenizer.py
```

这个脚本会：

1. 克隆官方 `Kronos` 仓库到 `external/Kronos`
2. 下载 tokenizer 相关资源

### 6.6 检查关键数据文件

确认下面文件存在：

```text
data/btcusdt_15m_1y.csv
data/btcusdt_15m_2y.csv
data/btcusdt_15m_4y.csv
```

### 6.7 检查关键状态输出

确认下面目录存在：

```text
validation_outputs/15m_state_v2_run/
validation_outputs/15m_state_v2_4y_run/
```

最关键的状态文件是：

```text
validation_outputs/15m_state_v2_run/validation_year_15m_v2_states.csv
validation_outputs/15m_state_v2_4y_run/validation_year_15m_v2_states.csv
```

如果这些文件已经在仓库中，就可以直接继续跑后续策略实验，而不必先重算 `V2` 状态。

### 6.8 最小验证恢复是否成功

先跑一个最小实验，确认环境正常：

```powershell
python .\supertrend_atr_regime_v2_experiment.py `
  --input-csv .\data\btcusdt_15m_2y.csv `
  --v2-validation-csv .\validation_outputs\15m_state_v2_run\validation_year_15m_v2_states.csv `
  --output-dir .\validation_outputs\supertrend_atr_regime_v2_run_fee1 `
  --fee-bps 1
```

如果这个能正常产出：

- `supertrend_atr_regime_comparison.csv`
- `supertrend_atr_regime_summary.json`

就说明：

- Python 环境正常
- 数据路径正常
- `V2` 状态输出可正常读取
- 当前研究主线已经恢复到可继续开发状态

---

## 7. 精确恢复当前项目的最小文件集

如果你的目标不是“完全原样镜像整个目录”，而是“最小代价恢复当前研究能力”，那么至少要保留下面这些目录：

```text
data/
validation_outputs/15m_state_v2_run/
validation_outputs/15m_state_v2_4y_run/
external/Kronos/
```

再加上根目录研究脚本和说明文档。

其他实验输出如果丢了，后面仍然可以重跑。

但是：

- `15m_state_v2_run`
- `15m_state_v2_4y_run`

这两块建议一定保留，因为它们是几乎所有后续实验的状态层基础缓存。

---

## 8. 单独建仓的推荐做法

### 8.1 第一步：清理不该进仓库的内容

确保下面内容不进入仓库：

- `.venv/`
- `.pydeps/`
- `__pycache__/`
- 其它本地缓存目录

本项目根目录已经额外提供了一份 `.gitignore` 模板，可直接使用。

### 8.2 第二步：处理 `external/Kronos/.git`

如果要把 `external/Kronos` 的源码一并纳入当前仓库，先执行：

```powershell
Remove-Item -Recurse -Force .\external\Kronos\.git
```

注意：

- 这一步只是在 `mnts` 当前目录里，把 `external/Kronos` 从“嵌套仓库”转成“普通目录”
- 它不会删除 `Kronos` 的源码内容
- 但会移除它自己的 git 历史

如果你不想去掉 `Kronos` 的独立历史，那就不要把它并进父仓库，而改用 submodule 方案。

### 8.3 第三步：初始化当前项目为新仓库

```powershell
git init
git branch -M main
git add .
git commit -m "Initial import of MNTS V2 project"
```

### 8.4 第四步：创建并绑定远程仓库

你可以在 GitHub / GitLab / Gitea 任意平台创建一个新的空仓库，例如：

- `mnts-v2-research`
- `mnts-state-layer-v2`
- `mnts-breakout-v2-research`

然后绑定远程：

```powershell
git remote add origin <你的新仓库地址>
git push -u origin main
```

---

## 9. 远程仓库命名建议

如果你想让仓库名字更准确，我建议这几个方向：

### 9.1 面向当前真实研究主线

- `mnts-v2-trend-research`
- `mnts-v2-breakout-research`
- `mnts-v2-state-layer-research`

### 9.2 面向长期演化

- `mnts-research`
- `mnts-v2-lab`
- `mnts-market-state-lab`

如果目标是以后继续扩更多骨架，而不是只围绕 `breakout`，推荐：

- `mnts-v2-trend-research`

---

## 10. 建仓时是否建议把所有输出都推上去

### 10.1 推荐做法

推荐保留：

- `15m_state_v2_run`
- `15m_state_v2_4y_run`
- 主要实验的 `summary.json`
- 关键 `comparison.csv`
- 滚动验证 `window_results.csv`
- 参数稳定性结果

### 10.2 可以后续再精简

如果以后仓库越来越大，可以再做第二轮整理：

1. 保留关键状态输出
2. 保留最重要实验结果
3. 把冗余中间结果移到 release 包、压缩包或对象存储

但就当前规模来看，还没有必要过度优化。

---

## 11. 当前最适合执行的建仓方案

如果你现在就想把当前项目单独推上去，我建议按下面方案执行：

### 方案

1. 使用当前根目录 `D:\Projects\mnts`
2. 保留：
   - 所有正式脚本
   - 所有正式文档
   - `data/`
   - `validation_outputs/`
   - `external/Kronos/`
3. 排除：
   - `.venv/`
   - `.pydeps/`
   - `__pycache__/`
4. 先删除 `external/Kronos/.git`
5. 再初始化新的父仓库并推送

### 这个方案的优点

1. 仓库足够完整
2. 新服务器 `clone` 后恢复最容易
3. 文档和代码、数据、输出完全一一对应

---

## 12. 我已经替你确认的结论

截至写本文档这一刻，可以明确确认：

1. 当前目录不是 git 仓库
2. 当前没有可直接检查的远程仓库
3. 当前项目完全可以独立建一个新仓库
4. 当前项目不应该把 `.venv` 和 `.pydeps` 放进仓库
5. 如果要做成真正“可直接拉取恢复”的仓库，推荐把 `data/`、`validation_outputs/`、`external/Kronos/` 一并纳入
6. 如果把 `external/Kronos/` 一并纳入，必须处理它内部的 `.git`

---

## 13. 后续如果要我继续做

如果你下一步希望我继续直接执行，我可以继续帮你做下面任一项：

1. 直接在当前目录生成适合建仓的 `.gitignore`
2. 帮你再做一轮“只保留恢复必需文件”的精简清单
3. 如果你给出新的远程仓库地址，我可以继续帮你把本地仓库初始化步骤准备好

