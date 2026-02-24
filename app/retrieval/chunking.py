"""
app/retrieval/chunking.py
=========================
文本分块策略模块：将长文本切割为适合向量化的固定/滑动窗口切片。

策略说明：
  - 采用固定大小 + 重叠的滑动窗口分块（Sliding Window Chunking）。
  - 可按字符数或 Token 数控制切片粒度。
  - 支持按句子边界优先对齐，避免切断语义。
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# 默认分块参数
DEFAULT_CHUNK_SIZE = 512       # 每个切片的目标字符数
DEFAULT_CHUNK_OVERLAP = 64     # 相邻切片的重叠字符数
DEFAULT_MIN_CHUNK_SIZE = 50    # 切片最小字符数（过短则丢弃）


@dataclass
class TextChunk:
    """
    单个文本切片的数据结构。

    Attributes:
        content:    切片文本内容。
        chunk_idx:  在原文中的切片序号（从 0 开始）。
        start_char: 在原文中的起始字符位置。
        end_char:   在原文中的结束字符位置。
        metadata:   附加元数据字典（如来源、时间戳等）。
    """
    content: str
    chunk_idx: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """
    滑动窗口文本分块器。

    支持：
      - 固定字符数分块（默认）
      - 句子边界对齐（优先在句号/换行处分割）
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    ) -> None:
        """
        初始化分块器参数。

        Args:
            chunk_size:     每个切片的目标字符数。
            chunk_overlap:  相邻切片间的重叠字符数（保持上下文连贯）。
            min_chunk_size: 过滤掉低于此长度的切片，避免噪声。
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._step = chunk_size - chunk_overlap

        logger.debug(
            "TextChunker 初始化 | chunk_size={} | overlap={} | step={}",
            chunk_size, chunk_overlap, self._step,
        )

    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> List[TextChunk]:
        """
        将输入文本切割为切片列表。

        Args:
            text:     待分块的原始文本。
            metadata: 附加元数据（将复制到每个切片）。

        Returns:
            TextChunk 列表，按原文顺序排列。
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        base_metadata = metadata or {}
        chunks: List[TextChunk] = []

        # 优先在自然句子边界分割
        sentences = self._split_into_sentences(text)
        current_chunk: List[str] = []
        current_len = 0
        start_char = 0
        global_pos = 0  # 追踪在原文中的字符位置

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_len + sentence_len > self.chunk_size and current_chunk:
                # 当前切片已满，输出
                chunk_text = "".join(current_chunk)
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunks.append(
                        TextChunk(
                            content=chunk_text.strip(),
                            chunk_idx=len(chunks),
                            start_char=start_char,
                            end_char=start_char + len(chunk_text),
                            metadata={**base_metadata},
                        )
                    )

                # 保留重叠部分：从末尾回溯 chunk_overlap 字符
                overlap_text = chunk_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)
                start_char = global_pos - len(overlap_text)

            current_chunk.append(sentence)
            current_len += sentence_len
            global_pos += sentence_len

        # 处理末尾剩余内容
        if current_chunk:
            chunk_text = "".join(current_chunk)
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(
                    TextChunk(
                        content=chunk_text.strip(),
                        chunk_idx=len(chunks),
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        metadata={**base_metadata},
                    )
                )

        logger.debug("文本分块完成 | input_len={} | chunks={}", len(text), len(chunks))
        return chunks

    def chunk_texts(
        self,
        texts: List[str],
        metadata: Optional[dict] = None,
    ) -> List[TextChunk]:
        """
        对多段文本批量分块，自动合并结果并重新编号切片索引。

        Args:
            texts:    文本列表。
            metadata: 共享元数据。

        Returns:
            合并后的 TextChunk 列表。
        """
        all_chunks: List[TextChunk] = []
        for text in texts:
            chunks = self.chunk(text, metadata=metadata)
            # 重新编号，确保全局索引连续
            for chunk in chunks:
                chunk.chunk_idx = len(all_chunks)
                all_chunks.append(chunk)
        return all_chunks

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """
        按句子边界（句号、换行符等）将文本拆分为句子列表。
        每个元素保留末尾分隔符，以便重新拼接时不丢失信息。

        Args:
            text: 输入文本。

        Returns:
            句子字符串列表。
        """
        # 在句子结束标点后分割，保留分隔符
        pattern = r'(?<=[。！？.!?\n])'
        parts = re.split(pattern, text)
        # 过滤空字符串，保留有效句子片段
        return [p for p in parts if p]