# Nanobot 分层记忆系统设计文档

**日期**: 2025-04-04  
**版本**: 1.0  
**状态**: 设计评审阶段

---

## 1. 设计目标

为 nanobot 构建一个功能丰富、可扩展、低耦合的分层记忆系统，实现更可靠的对话连续性和拟人化的交流体验。

### 核心目标
- **功能丰富**: 支持语义搜索、自动记忆注入、对话连续性、记忆巩固
- **可扩展**: 插件化设计，支持多种存储后端和嵌入提供者
- **低耦合**: 通过 Hook 机制与核心 Loop 解耦集成
- **拟人感**: 流畅的自然语言记忆注入，用户无感的来源标识

---

## 2. 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    AgentLoop                             │
│  ┌─────────────────────────────────────────────────────┐│
│  │  MemoryHook (集成层)                                 ││
│  │  ├── before_iteration: 注入记忆                      ││
│  │  └── after_iteration:  保存记忆                      ││
│  └─────────────────────────────────────────────────────┘│
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              MemoryOrchestrator (协调层)                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │   Working   │  │   Episodic   │  │    Semantic    │ │
│  │   Memory    │→ │    Memory    │→ │     Memory     │ │
│  │  (最近对话)  │  │  (对话摘要)   │  │   (知识向量)    │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
│         ↓                  ↓                 ↓          │
│  ┌─────────────────────────────────────────────────────┐│
│  │        ConsolidationEngine (巩固引擎)                ││
│  │   Token阈值 / 时间周期 / 显式命令                     ││
│  └─────────────────────────────────────────────────────┘│
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│         MemoryBackend (存储抽象层)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │  FileSystem │  │  SQLite  │  │  Chroma  │  │  Custom │ │
│  │  (默认)   │  │ (轻量)   │  │ (向量)   │  │ (扩展)  │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 核心组件设计

### 3.1 存储后端抽象 (MemoryBackend)

```python
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class MemoryTier(Enum):
    WORKING = "working"      # 工作记忆：最近 N 轮对话
    EPISODIC = "episodic"    # 情节记忆：对话摘要
    SEMANTIC = "semantic"    # 语义记忆：向量化的知识

@dataclass
class MemoryEntry:
    id: str
    content: str                    # 记忆内容
    tier: MemoryTier
    created_at: datetime
    updated_at: datetime
    source_session: Optional[str]   # 来源会话
    metadata: dict                  # 扩展元数据
    embedding: Optional[list[float]] = None  # 语义记忆使用

@dataclass
class RetrievalResult:
    entry: MemoryEntry
    relevance_score: float          # 0-1 相关性分数
    retrieval_method: str           # "exact" | "semantic" | "temporal"

class MemoryBackend(ABC):
    """存储后端抽象，支持同步和异步实现"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化存储连接"""
        pass
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """存储记忆条目"""
        pass
    
    @abstractmethod
    async def retrieve(
        self, 
        query: str,
        tier: Optional[MemoryTier] = None,
        limit: int = 10,
        embedding: Optional[list[float]] = None
    ) -> list[RetrievalResult]:
        """检索记忆，支持语义和关键词匹配"""
        pass
    
    @abstractmethod
    async def consolidate(
        self, 
        source_tier: MemoryTier, 
        target_tier: MemoryTier,
        entries: list[MemoryEntry]
    ) -> list[MemoryEntry]:
        """跨层记忆迁移（用于巩固）"""
        pass
    
    @abstractmethod
    async def delete_expired(self, max_age_days: dict[MemoryTier, int]) -> int:
        """清理过期记忆"""
        pass
```

### 3.2 嵌入提供者抽象 (EmbeddingProvider)

```python
class EmbeddingProvider(ABC):
    """嵌入向量生成抽象"""
    
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成嵌入"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """嵌入维度"""
        pass
    
    @property
    @abstractmethod
    def max_tokens_per_text(self) -> int:
        """单次最大 token 数"""
        pass
```

### 3.3 记忆协调器 (MemoryOrchestrator)

```python
class MemoryOrchestrator:
    """
    记忆系统中央协调器
    职责：管理三层记忆生命周期，协调检索和注入
    """
    
    def __init__(
        self,
        backend: MemoryBackend,
        embedder: EmbeddingProvider,
        config: MemoryConfig
    ):
        self.backend = backend
        self.embedder = embedder
        self.config = config
        
        # 三层记忆管理器
        self.working = WorkingMemoryManager(backend, config.working)
        self.episodic = EpisodicMemoryManager(backend, config.episodic)
        self.semantic = SemanticMemoryManager(backend, embedder, config.semantic)
        
        # 巩固调度器
        self.consolidator = ConsolidationEngine(
            self.working, self.episodic, self.semantic, config.consolidation
        )
    
    async def on_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        token_usage: TokenUsage
    ) -> None:
        """每轮对话后调用，更新工作记忆"""
        # 1. 存入工作记忆
        await self.working.add_turn(session_id, user_message, assistant_response)
        
        # 2. 检查是否需要巩固
        if await self.consolidator.should_consolidate(token_usage):
            await self.consolidator.run(session_id)
    
    async def retrieve_for_context(
        self,
        current_query: str,
        recent_context: list[dict],
        max_tokens: int
    ) -> RetrievalContext:
        """
        检索相关记忆，构建注入上下文
        返回混合格式：结构化 + 自然语言
        """
        # 1. 生成查询嵌入
        query_embedding = await self.embedder.embed([current_query])
        
        # 并行检索三层记忆
        working_memories = await self.working.get_recent()
        episodic_results = await self.episodic.search(current_query, limit=5)
        semantic_results = await self.semantic.search(
            query=current_query,
            embedding=query_embedding[0],
            limit=5
        )
        
        # 去重和相关性排序
        all_results = self._deduplicate_and_rank(
            episodic_results + semantic_results
        )
        
        # 构建混合格式输出
        return self._build_mixed_context(
            working_memories,
            all_results,
            max_tokens
        )
```

---

## 4. 三层记忆管理器

### 4.1 工作记忆 (WorkingMemoryManager)

```python
@dataclass
class WorkingMemoryConfig:
    max_turns: int = 10           # 保留最近 N 轮
    max_tokens: int = 4000        # 软上限
    ttl_seconds: int = 3600       # 1小时无活动则下沉

class WorkingMemoryManager:
    """
    工作记忆：最近对话的完整保留
    特点：无嵌入，线性存储，快速访问
    """
    
    async def add_turn(
        self,
        session_id: str,
        user: str,
        assistant: str
    ) -> None:
        entry = MemoryEntry(
            id=generate_id(),
            content=json.dumps({
                "user": user,
                "assistant": assistant,
                "timestamp": now_iso()
            }),
            tier=MemoryTier.WORKING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session=session_id,
            metadata={"type": "turn"}
        )
        await self.backend.store(entry)
        
        # 检查溢出，触发下沉
        if await self._is_overflow():
            await self._overflow_to_episodic(session_id)
    
    async def get_recent(self, n: Optional[int] = None) -> list[MemoryEntry]:
        """获取最近 N 轮对话"""
        return await self.backend.retrieve(
            query="",  # 空查询表示时间排序
            tier=MemoryTier.WORKING,
            limit=n or self.config.max_turns
        )
```

### 4.2 情节记忆 (EpisodicMemoryManager)

```python
@dataclass
class EpisodicMemoryConfig:
    summary_model: str = "default"  # 用于生成摘要的模型
    max_entries: int = 1000
    consolidation_batch: int = 5    # 几轮工作记忆合并为一个情节

class EpisodicMemoryManager:
    """
    情节记忆：对话摘要，保留时间线和上下文关系
    特点：轻量级嵌入，时间 + 语义检索
    """
    
    async def create_summary(
        self,
        session_id: str,
        turns: list[MemoryEntry]
    ) -> MemoryEntry:
        """将多轮对话总结为情节摘要"""
        # 调用 LLM 生成摘要
        summary_text = await self._summarize_turns(turns)
        
        entry = MemoryEntry(
            id=generate_id(),
            content=summary_text,
            tier=MemoryTier.EPISODIC,
            created_at=turns[0].created_at,
            updated_at=datetime.now(),
            source_session=session_id,
            metadata={
                "type": "summary",
                "turn_count": len(turns),
                "original_turn_ids": [t.id for t in turns]
            }
        )
        await self.backend.store(entry)
        return entry
    
    async def search(self, query: str, limit: int = 5) -> list[RetrievalResult]:
        """关键词 + 时间邻近检索"""
        return await self.backend.retrieve(
            query=query,
            tier=MemoryTier.EPISODIC,
            limit=limit
        )
```

### 4.3 语义记忆 (SemanticMemoryManager)

```python
@dataclass
class SemanticMemoryConfig:
    embedding_dimension: int = 768
    similarity_threshold: float = 0.75
    max_entries: int = 10000

class SemanticMemoryManager:
    """
    语义记忆：知识性内容，支持深度语义检索
    特点：完整嵌入向量，相似度匹配
    """
    
    async def store_knowledge(
        self,
        content: str,
        source_session: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> MemoryEntry:
        """存储知识性记忆，自动生成嵌入"""
        embedding = await self.embedder.embed([content])
        
        entry = MemoryEntry(
            id=generate_id(),
            content=content,
            tier=MemoryTier.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_session=source_session,
            metadata=metadata or {},
            embedding=embedding[0]
        )
        await self.backend.store(entry)
        return entry
    
    async def search(
        self,
        query: str,
        embedding: list[float],
        limit: int = 5
    ) -> list[RetrievalResult]:
        """纯语义检索，按相似度排序"""
        return await self.backend.retrieve(
            query=query,
            tier=MemoryTier.SEMANTIC,
            limit=limit,
            embedding=embedding
        )
```

---

## 5. 混合模式记忆注入

```python
@dataclass
class RetrievalContext:
    """
    检索结果混合格式：
    - 重要记忆：自然语言段落（更拟人）
    - 相关事实：结构化数据（更准确）
    """
    natural_section: str
    structured_facts: list[dict]
    debug_sources: list[dict]
    
    def to_system_prompt_addition(self) -> str:
        """生成 System Prompt 附加内容"""
        sections = []
        
        if self.natural_section:
            sections.append(f"## 相关背景\n\n{self.natural_section}")
        
        if self.structured_facts:
            facts_text = "\n".join([
                f"- [{f['category']}] {f['content']}"
                for f in self.structured_facts
            ])
            sections.append(f"## 参考信息\n\n{facts_text}")
        
        return "\n\n".join(sections)


class MixedContextBuilder:
    """构建混合格式的记忆上下文"""
    
    def build(
        self,
        working_memories: list[MemoryEntry],
        retrieved_results: list[RetrievalResult],
        max_tokens: int
    ) -> RetrievalContext:
        # 分类处理
        high_priority = [r for r in retrieved_results if r.relevance_score > 0.85]
        medium_priority = [r for r in retrieved_results if 0.7 < r.relevance_score <= 0.85]
        
        # 生成自然语言段落
        natural_section = self._generate_natural_paragraph(working_memories, high_priority)
        
        # 构建结构化事实
        structured_facts = [
            {
                "category": self._categorize_fact(r.entry),
                "content": r.entry.content,
                "relevance": round(r.relevance_score, 2)
            }
            for r in medium_priority[:5]
        ]
        
        # 记录调试来源
        debug_sources = [
            {
                "entry_id": r.entry.id,
                "tier": r.entry.tier.value,
                "score": r.relevance_score,
                "retrieval_method": r.retrieval_method,
                "source_session": r.entry.source_session
            }
            for r in retrieved_results
        ]
        
        return RetrievalContext(
            natural_section=natural_section,
            structured_facts=structured_facts,
            debug_sources=debug_sources
        )
    
    def _generate_natural_paragraph(
        self,
        working: list[MemoryEntry],
        high_priority: list[RetrievalResult]
    ) -> str:
        """流畅的自然语言段落生成"""
        parts = []
        
        if working:
            recent_turns = [json.loads(r.content) for r in working[-3:]]
            user_concerns = [t['user'][:80] for t in recent_turns if len(t['user']) > 10]
            if user_concerns:
                context = "，随后".join(user_concerns)
                parts.append(f"你们刚才聊到{context}。")
        
        episodic_memories = [r for r in high_priority if r.entry.tier == MemoryTier.EPISODIC]
        semantic_memories = [r for r in high_priority if r.entry.tier == MemoryTier.SEMANTIC]
        
        if episodic_memories:
            summaries = [r.entry.content for r in episodic_memories[:2]]
            if len(summaries) == 1:
                parts.append(f"我记得之前你们聊过{summaries[0]}。")
            else:
                joined = "，还有".join(summaries)
                parts.append(f"之前你们讨论过{joined}。")
        
        if semantic_memories:
            facts = [r.entry.content for r in semantic_memories[:2]]
            if len(facts) == 1:
                parts.append(f"对了，{facts[0]}这部分内容可能有关联。")
            else:
                parts.append(f"顺便提一下，{'；'.join(facts)}——这些也许对现在有帮助。")
        
        return "".join(parts) if parts else ""
```

---

## 6. Loop Hook 集成

```python
from nanobot.agent.hook import AgentHook, AgentHookContext

class MemoryHook(AgentHook):
    """记忆系统与 AgentLoop 的集成钩子"""
    
    def __init__(self, orchestrator: MemoryOrchestrator):
        self.orchestrator = orchestrator
        self.session_memories: dict[str, list[MemoryEntry]] = {}
    
    async def before_iteration(self, context: AgentHookContext) -> None:
        """每轮迭代开始前：注入相关记忆"""
        if not context.messages:
            return
        
        last_user_msg = self._extract_last_user_message(context.messages)
        if not last_user_msg:
            return
        
        retrieval_context = await self.orchestrator.retrieve_for_context(
            current_query=last_user_msg,
            recent_context=context.messages[-5:],
            max_tokens=1500
        )
        
        if retrieval_context.has_content():
            memory_prompt = retrieval_context.to_system_prompt_addition()
            self._inject_memory_to_system(context, memory_prompt)
            logger.debug(f"Memory injection: {retrieval_context.debug_sources}")
    
    async def after_iteration(self, context: AgentHookContext) -> None:
        """每轮迭代结束后：保存对话"""
        session_id = self._extract_session_id(context)
        if not session_id or not context.response:
            return
        
        token_usage = self._calculate_token_usage(context)
        await self.orchestrator.on_conversation_turn(
            session_id=session_id,
            user_message=self._extract_last_user_message(context.messages),
            assistant_response=context.response.content or "",
            token_usage=token_usage
        )
    
    def _inject_memory_to_system(
        self, 
        context: AgentHookContext, 
        memory_prompt: str
    ) -> None:
        """将记忆注入到系统提示"""
        system_msg_idx = None
        for i, msg in enumerate(context.messages):
            if msg.get("role") == "system":
                system_msg_idx = i
                break
        
        if system_msg_idx is not None:
            original = context.messages[system_msg_idx].get("content", "")
            context.messages[system_msg_idx]["content"] = (
                f"{original}\n\n{memory_prompt}"
            )
        else:
            context.messages.insert(0, {
                "role": "system", 
                "content": memory_prompt
            })
```

---

## 7. 记忆巩固引擎

```python
@dataclass
class ConsolidationConfig:
    working_memory_token_threshold: int = 3000
    episodic_memory_count_threshold: int = 50
    auto_consolidate_interval_minutes: int = 30
    enable_explicit_consolidation: bool = True

class ConsolidationEngine:
    """记忆巩固引擎"""
    
    def __init__(
        self,
        working: WorkingMemoryManager,
        episodic: EpisodicMemoryManager,
        semantic: SemanticMemoryManager,
        config: ConsolidationConfig
    ):
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.config = config
        self._last_consolidation: dict[str, datetime] = {}
    
    async def should_consolidate(self, token_usage: TokenUsage) -> bool:
        """检查是否满足巩固条件"""
        if token_usage.prompt_tokens > self.config.working_memory_token_threshold:
            return True
        
        last_time = self._last_consolidation.get(token_usage.session_id)
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds() / 60
            if elapsed > self.config.auto_consolidate_interval_minutes:
                return True
        
        return False
    
    async def run(self, session_id: str) -> ConsolidationResult:
        """执行记忆巩固流程"""
        result = ConsolidationResult(session_id=session_id)
        
        # 工作记忆 → 情节记忆
        working_entries = await self.working.get_all_for_session(session_id)
        if len(working_entries) >= self.config.episodic_memory_count_threshold:
            to_summarize = working_entries[:-10]
            summary = await self.episodic.create_summary(session_id, to_summarize)
            result.episodic_created.append(summary)
            await self.working.archive_entries([e.id for e in to_summarize])
        
        # 情节记忆 → 语义记忆
        episodic_entries = await self.episodic.get_for_session(session_id)
        for entry in episodic_entries:
            knowledge_chunks = await self._extract_knowledge(entry)
            for chunk in knowledge_chunks:
                semantic_entry = await self.semantic.store_knowledge(
                    content=chunk,
                    source_session=session_id,
                    metadata={"extracted_from": entry.id}
                )
                result.semantic_created.append(semantic_entry)
        
        self._last_consolidation[session_id] = datetime.now()
        return result
```

---

## 8. 配置示例

```yaml
# nanobot.yaml
memory:
  backend:
    type: "filesystem"  # filesystem | sqlite | chroma
    path: "./memory"
  
  embedding:
    provider: "local"  # local | openai
    model: "nomic-embed-text"  # 本地模型名称或 API 模型
    dimension: 768
  
  working:
    max_turns: 10
    max_tokens: 4000
    ttl_seconds: 3600
  
  episodic:
    summary_model: "default"
    max_entries: 1000
    consolidation_batch: 5
  
  semantic:
    similarity_threshold: 0.75
    max_entries: 10000
  
  consolidation:
    token_threshold: 3000
    time_interval_minutes: 30
    enable_explicit: true
```

---

## 9. 存储后端实现建议

### 9.1 FileSystemBackend（默认）
- 工作记忆：`memory/working/{session_id}.jsonl`
- 情节记忆：`memory/episodic/{session_id}/{timestamp}.md`
- 语义记忆：`memory/semantic/index.json` + 可选的向量缓存

### 9.2 SQLiteBackend（轻量）
- 单文件：`memory.db`
- 表结构：`working_memory`, `episodic_memory`, `semantic_memory`
- 向量使用 BLOB 存储，应用层计算相似度

### 9.3 ChromaBackend（向量优先）
- 集成 ChromaDB 或 Qdrant
- 专门用于语义记忆的向量检索
- 其他层回退到 SQLite

---

## 10. 测试策略

1. **单元测试**: 各管理器的独立测试
2. **集成测试**: 完整记忆流转测试
3. **性能测试**: 大规模记忆的检索性能
4. **降级测试**: 嵌入服务不可用时的行为

---

## 11. 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 嵌入模型加载失败 | 中 | 高 | 降级为关键词检索 |
| 向量数据库性能问题 | 低 | 中 | 本地缓存 + 异步更新 |
| 记忆注入 token 超限 | 中 | 中 | 动态预算管理 |
| 隐私泄露 | 低 | 高 | 敏感信息过滤 + 用户确认 |

---

## 12. 后续优化方向

1. **记忆重要性评估**: 基于用户交互频率自动调整记忆权重
2. **跨会话关联**: 识别不同会话中的相关主题
3. **主动记忆提示**: 在适当时候主动提及相关历史
4. **记忆可视化**: 为用户提供记忆管理界面

---

**文档版本**: 1.0  
**最后更新**: 2025-04-04  
**评审状态**: 待评审
