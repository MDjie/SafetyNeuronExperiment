# Claude-Mem 完整安装与修复工作流

适用于任何新环境中安装 claude-mem 插件并确保所有子系统正常工作。

---

## 一、环境前提

### 1.1 必需组件
- **Claude Code** (CLI) 已安装并登录
- **Python 3.12** (chroma-mcp 0.2.6 兼容，不要用 3.13)
- **pip** 可用
- **Node.js / Bun** (claude-mem worker 运行所需)

### 1.2 安装 Bun（claude-mem worker 运行时）

claude-mem worker 使用 Bun 运行。大多数服务器环境不自带 Bun，需要手动安装。

```bash
# 官方安装脚本
curl -fsSL https://bun.sh/install | bash

# 添加到 PATH（安装脚本自动写入 ~/.bashrc，手动 source 使其立即生效）
source ~/.bashrc

# 验证
bun --version  # 应输出 >= 1.0.0（本环境使用 1.3.13）
```

安装路径：`~/.bun/bin/bun`，通过 `~/.bashrc` 中 `export BUN_INSTALL="$HOME/.bun"` 加入 PATH。

### 1.3 安装 claude-mem 插件
```bash
# 在 Claude Code 中注册插件
claude plugins install thedotmack/claude-mem@13.0.1
```

---

## 二、三大关键修复

claude-mem 安装后默认无法工作，必须完成以下三个修复。

---

### 修复 1：SDK 认证 — 创建 `.env` 文件

**症状**：观察记录全部丢失，SessionStart 提示"no memory"，SDK 返回 "Not logged in · Please run /login"

**根因**：`CLAUDE_MEM_CLAUDE_AUTH_METHOD=subscription` 尝试 OAuth keychain 认证，但子进程中环境变量未传递。

**修复步骤**：

```bash
# 1. 修改认证方式为 api-key
# 编辑 ~/.claude-mem/settings.json，修改：
"CLAUDE_MEM_CLAUDE_AUTH_METHOD": "api-key"

# 2. 创建 ~/.claude-mem/.env 文件
cat > ~/.claude-mem/.env << 'EOF'
ANTHROPIC_AUTH_TOKEN=<你的API-KEY>
ANTHROPIC_BASE_URL=<你的API-BASE-URL>
EOF
```

**说明**：
- 如果使用 Anthropic 官方 API：`ANTHROPIC_BASE_URL` 不需要设置
- 如果使用 DeepSeek 等兼容 API：设置 `ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic`
- 该文件会在 worker 启动时被读取，认证信息会传递给 SDK 子进程

---

### 修复 2：ChromaDB MCP — 绕过 uvx 冷启动超时

**症状**：语义搜索超时、ChromaDB 连接失败、`uvx` 首次运行下载 300MB+ 依赖超过 30s MCP 握手超时

**根因**：`uvx chroma-mcp` 首次调用需要下载 onnxruntime、chromadb、openai 等数十个包（~300MB），30 秒握手超时远不够。

**修复步骤**：

```bash
# 步骤 1：确保 Python 版本为 3.12（不是 3.13）
# 修改 ~/.claude-mem/settings.json：
"CLAUDE_MEM_PYTHON_VERSION": "3.12"

# 步骤 2：直接用 pip 安装 chroma-mcp 及其依赖
pip install chroma-mcp==0.2.6

# 步骤 3：创建 uvx 包装脚本，拦截 chroma-mcp 调用
cat > /usr/local/bin/uvx << 'SCRIPT'
#!/bin/bash
# Wrapper: 将 chroma-mcp 调用重定向到 pip 安装的版本
for arg in "$@"; do
    if [[ "$arg" == chroma-mcp* ]]; then
        data_dir="/root/.claude-mem/chroma"
        for ((i=0; i<$#; i++)); do
            if [[ "${!i}" == "--data-dir" ]]; then
                next=$((i+1))
                data_dir="${!next}"
                break
            fi
        done
        export CHROMA_DATA_DIR="$data_dir"
        exec /usr/local/bin/chroma-mcp --client-type persistent --data-dir "$data_dir" 2>>/root/.claude-mem/chroma/stderr.log
    fi
done
# 其他命令正常走 uvx
exec /root/.local/bin/uvx "$@"
SCRIPT

chmod +x /usr/local/bin/uvx
```

**关键点**：
- `/usr/local/bin/uvx` 必须在 PATH 中优先于 `/root/.local/bin/uvx`
- `/root/.local/bin/uvx` 是真正的 uvx 二进制文件（不要覆盖它）
- 包装脚本只拦截含 `chroma-mcp` 的参数，其他 uvx 调用透传

---

### 修复 3：ONNX 嵌入模型 — 预下载正确格式模型

**症状**：ChromaDB 启动后反复尝试从 AWS S3 下载 `all-MiniLM-L6-v2` ONNX 模型（79.3MB），下载速度 <20 KiB/s，每次超时崩溃。ModelScope 导出的 ONNX 模型格式不兼容（INVALID_PROTOBUF 错误）。

**根因**：chroma ONNX embedding function 要求特定格式的模型，且通过 SHA256 硬编码校验。必须使用 chroma 官方发布的 ONNX 模型。

**修复步骤**：

```bash
# 步骤 1：从 chroma 官方 S3 下载正确的模型包
# URL: https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz
# 如果 S3 下载慢，可以用其他渠道下载，但必须 SHA256 匹配

# 步骤 2：验证 SHA256（必须完全匹配）
echo "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3  onnx.tar.gz" | sha256sum -c

# 步骤 3：放到 chroma 缓存目录
mkdir -p /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/
cp onnx.tar.gz /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/

# 步骤 4（重要）：写保护缓存目录，防止 chroma 重新下载导致损坏
chmod -R 555 /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/
```

**tar.gz 内容结构**（供参考）：
```
onnx/
  model.onnx          # 86.2 MB — ONNX 模型文件
  config.json         # 650 B
  tokenizer.json      # 695 KB
  vocab.txt           # 226 KB
  tokenizer_config.json
  special_tokens_map.json
```

---

## 三、重启与验证

### 3.1 重启 Worker 服务
```bash
# 杀掉所有相关进程
pkill -f "worker-service" 2>/dev/null
pkill -f "chroma-mcp" 2>/dev/null

# Claude Code 会自动重启 worker，或重新打开 Claude Code 会话
```

### 3.2 验证四个子系统

| 子系统 | 验证方法 | 预期结果 |
|--------|---------|---------|
| **观察记录** | `claude-mem status` | 显示 observation count > 0，无 "Not logged in" 错误 |
| **AI 分析 (SDK)** | 执行任意任务后检查观察 | 自动生成 title/subtitle/facts/narrative/concepts |
| **关键词搜索 (FTS5)** | `/claude-mem:mem-search` 搜索内容 | 返回匹配结果 |
| **语义搜索 (ChromaDB)** | 检查 stderr.log 无 ONNX 错误 | `tail /root/.claude-mem/chroma/stderr.log` 无 INVALID_PROTOBUF |
| **智能回填** | 重启后观察日志 | 自动从 watermark=0 开始同步历史数据 |

### 3.3 快速健康检查命令
```bash
# 检查必需运行时
bun --version
python3.12 --version

# 检查 worker 是否运行
ps aux | grep -E "(chroma-mcp|worker-service)" | grep -v grep

# 检查 ChromaDB 数据
ls -lh /root/.claude-mem/chroma/

# 检查 ONNX 模型完整性
sha256sum /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz

# 检查 chroma-mcp 是否可用
chroma-mcp --help

# 检查 uvx 包装脚本
which uvx && file /usr/local/bin/uvx
```

---

## 四、关键文件速查

| 文件 | 作用 | 修改内容 |
|------|------|---------|
| `~/.bun/bin/bun` | Bun 运行时 | 官方安装脚本，`~/.bashrc` 自动配置 PATH |
| `~/.claude-mem/settings.json` | 主配置 | `AUTH_METHOD: api-key`, `PYTHON_VERSION: 3.12` |
| `~/.claude-mem/.env` | SDK 认证凭据 | `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL` |
| `/usr/local/bin/uvx` | 包装脚本 | 拦截 chroma-mcp 调用，重定向到 pip 安装版本 |
| `/root/.local/bin/uvx` | 真正的 uvx 二进制 | 不要覆盖（约 36MB 二进制） |
| `/root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz` | ONNX 嵌入模型 | SHA256: `913d7300...`，写保护 |

---

## 五、常见问题排查

### Q1: 重启后 observation 仍为 0
- 检查 `~/.claude-mem/.env` 是否存在，API key 是否正确
- 检查 `~/.claude-mem/settings.json` 中 `CLAUDE_MEM_CLAUDE_AUTH_METHOD` 是否为 `"api-key"`
- 确认 `CLAUDE_MEM_PROVIDER` 为 `"claude"`

### Q2: 语义搜索不工作
- 检查 `chroma-mcp` 是否 pip 安装：`pip show chroma-mcp`
- 检查 `/usr/local/bin/uvx` 包装脚本是否存在且可执行
- 检查 ONNX 模型 SHA256 是否匹配
- 检查 `/root/.claude-mem/chroma/stderr.log` 看是否有错误

### Q3: ONNX 模型反复下载
- 确认缓存目录已写保护：`chmod 555 /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/`
- 确认 `onnx.tar.gz` 在正确路径且有正确的 SHA256

### Q4: 其他环境可能缺少的依赖
```bash
pip install chromadb cohere httpx mcp openai pillow python-dotenv voyageai onnxruntime
```

### Q5: pip 镜像加速（中国环境）
```bash
pip install chroma-mcp==0.2.6 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

---

## 六、修复总结

| # | 修复 | 症状 | 难度 | 影响范围 |
|---|------|------|------|---------|
| 1 | SDK 认证 `.env` | 所有观察丢失 | 低 | 观察记录 + AI 分析 |
| 2 | uvx 包装 + pip 安装 | 语义搜索超时 | 中 | 语义搜索 |
| 3 | ONNX 模型预下载 | ChromaDB 反复崩溃 | 中 | 语义搜索 |

三个修复相互独立但有层级依赖：修复 1 恢复基本观察功能，修复 2 恢复 ChromaDB 连接，修复 3 恢复嵌入向量功能。

最终状态：**观察记录、AI 分析、关键词搜索、语义搜索四个子系统全部正常，智能回填自动同步历史数据。**
