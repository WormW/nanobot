

---

## Task 6-20: 三层记忆管理器实现

由于实施计划较长，以下是各任务的概要。完整的分步代码请参考设计文档。

### Task 6: WorkingMemoryManager 基础实现
**Files:**
- Create: `nanobot/agent/memory/tiers/working.py`
- Test: `tests/agent/memory/test_tiers/test_working.py`

- [ ] 编写 WorkingMemoryManager 测试（add_turn, get_recent）
- [ ] 实现 WorkingMemoryManager 类
- [ ] 运行测试确认通过
- [ ] 提交

### Task 7: EpisodicMemoryManager 基础实现
**Files:**
- Create: `nanobot/agent/memory/tiers/episodic.py`
- Test: `tests/agent/memory/test_tiers/test_episodic.py`

- [ ] 编写 EpisodicMemoryManager 测试（create_summary, search）
- [ ] 实现 EpisodicMemoryManager 类
- [ ] 运行测试确认通过
- [ ] 提交

### Task 8: SemanticMemoryManager 基础实现
**Files:**
- Create: `nanobot/agent/memory/tiers/semantic.py`
- Test: `tests/agent/memory/test_tiers/test_semantic.py`

- [ ] 编写 SemanticMemoryManager 测试（store_knowledge, search）
- [ ] 实现 SemanticMemoryManager 类
- [ ] 运行测试确认通过
- [ ] 提交

### Task 9: 混合模式上下文构建器
**Files:**
- Create: `nanobot/agent/memory/context_builder.py`
- Test: `tests/agent/memory/test_context_builder.py`

- [ ] 编写 MixedContextBuilder 测试
- [ ] 实现 RetrievalContext 和 MixedContextBuilder
- [ ] 运行测试确认通过
- [ ] 提交

### Task 10: 记忆巩固引擎
**Files:**
- Create: `nanobot/agent/memory/consolidation.py`
- Test: `tests/agent/memory/test_consolidation.py`

- [ ] 编写 ConsolidationEngine 测试
- [ ] 实现 ConsolidationEngine 类
- [ ] 运行测试确认通过
- [ ] 提交

### Task 11: 记忆协调器
**Files:**
- Create: `nanobot/agent/memory/orchestrator.py`
- Test: `tests/agent/memory/test_orchestrator.py`

- [ ] 编写 MemoryOrchestrator 测试
- [ ] 实现 MemoryOrchestrator 类
- [ ] 运行测试确认通过
- [ ] 提交

### Task 12: Loop Hook 集成
**Files:**
- Create: `nanobot/agent/memory/hook.py`
- Test: `tests/agent/memory/test_hook.py`

- [ ] 编写 MemoryHook 测试
- [ ] 实现 MemoryHook 类（集成到 AgentLoop）
- [ ] 运行测试确认通过
- [ ] 提交

### Task 13-15: 可选存储后端实现（SQLite, Chroma）
- [ ] 实现 SQLiteBackend
- [ ] 实现 ChromaBackend（向量存储）

### Task 16-18: 可选嵌入提供者实现
- [ ] 实现 LocalEmbeddingProvider（sentence-transformers）
- [ ] 实现 OpenAIEmbeddingProvider

### Task 19: 集成测试
- [ ] 编写完整集成测试
- [ ] 验证端到端记忆流转

### Task 20: 性能优化和边缘情况处理
- [ ] 添加异步锁防止并发问题
- [ ] 实现 Token 预算管理
- [ ] 添加降级处理（嵌入失败时）

---

## 快速启动命令

```bash
# 安装依赖
pip install pytest pytest-asyncio sentence-transformers chromadb

# 运行所有记忆系统测试
python -m pytest tests/agent/memory/ -v

# 运行特定测试
python -m pytest tests/agent/memory/test_types.py -v
```

---

## Self-Review 检查清单

- [ ] **Spec coverage:** 所有设计文档中的组件都有对应的任务
- [ ] **Placeholder scan:** 无 TBD/TODO，所有代码完整
- [ ] **Type consistency:** 接口定义在各任务中保持一致
- [ ] **Test coverage:** 每个实现类都有对应的测试
- [ ] **Integration path:** Task 12 的 Hook 集成路径清晰

---

**Plan complete and saved to `docs/superpowers/plans/2025-04-04-layered-memory-implementation.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
