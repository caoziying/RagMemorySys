"""
app/retrieval/bm25.py
=====================
轻量级 BM25 稀疏向量生成器。

Milvus 的 SPARSE_FLOAT_VECTOR 字段接受 {token_id: weight} 格式的稀疏向量。
本模块实现：
  - 简单的中英文分词（jieba 可选，不可用时退回空格/字符 n-gram 分词）
  - BM25 权重计算（基于 TF-IDF 近似，适合单文档打分）
  - 输出 {int: float} 稀疏向量，可直接传给 pymilvus

⚠️ 此处的 BM25 是"文档级别"的稀疏表示（每个 token 的 BM25 权重），
   而非标准的跨语料库 IDF，因为插入时没有全局统计。
   这与 Milvus 官方 BM25 全文索引的定位一致：
   在 hybrid_search 中与稠密向量互补，提供词汇匹配信号。
"""

import hashlib
import math
import re
from typing import Dict, List

# 尝试导入 jieba（中文分词），不可用则使用内置分词
try:
    import jieba
    _USE_JIEBA = True
except ImportError:
    _USE_JIEBA = False

# BM25 超参数
BM25_K1 = 1.5   # 词频饱和系数（1.2~2.0 常用）
BM25_B  = 0.75  # 文档长度归一化系数

# 平均文档长度（无全局统计时使用经验值）
AVG_DOC_LEN = 128.0


def tokenize(text: str) -> List[str]:
    """
    对文本进行分词，输出 token 列表。

    优先使用 jieba（中文），不可用时使用空格分割 + 字符 n-gram 兜底。

    Args:
        text: 待分词的原始文本。

    Returns:
        token 字符串列表。
    """
    text = text.strip().lower()
    if not text:
        return []

    if _USE_JIEBA:
        tokens = [t for t in jieba.cut(text) if t.strip() and t not in _STOPWORDS]
    else:
        # 按空格/标点分割英文，同时提取中文字符 bigram
        en_tokens = re.findall(r'[a-z0-9]+', text)
        zh_chars  = re.findall(r'[\u4e00-\u9fff]', text)
        # 中文字符 bigram（相邻两字组成一个 token）
        zh_tokens = [zh_chars[i] + zh_chars[i+1] for i in range(len(zh_chars)-1)]
        tokens = [t for t in en_tokens + zh_tokens if t not in _STOPWORDS]

    return tokens


def text_to_sparse_vector(text: str) -> Dict[int, float]:
    """
    将文本转换为 BM25 稀疏向量，格式为 {token_id: bm25_weight}。

    token_id 由 token 字符串的 SHA-256 哈希截断为 32 位无符号整数，
    保证跨文档一致性（相同 token → 相同 id）。

    Args:
        text: 待转换的文本。

    Returns:
        稀疏向量字典 {token_id: float}；文本为空时返回 {}。
    """
    tokens = tokenize(text)
    if not tokens:
        return {}

    # 统计词频
    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    doc_len = len(tokens)
    sparse: Dict[int, float] = {}

    for token, freq in tf.items():
        # BM25 TF 部分（不使用全局 IDF，以 log(2) 作为常数因子近似）
        # score = IDF * (tf * (k1+1)) / (tf + k1 * (1 - b + b * dl/avgdl))
        # 此处 IDF ≈ log(2) ≈ 0.693（无语料统计时的中性值）
        idf = math.log(2.0)
        tf_norm = (freq * (BM25_K1 + 1)) / (
            freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / AVG_DOC_LEN)
        )
        score = float(idf * tf_norm)
        if score > 0:
            token_id = _token_to_id(token)
            # 若哈希冲突，取较大权重
            if token_id in sparse:
                sparse[token_id] = max(sparse[token_id], score)
            else:
                sparse[token_id] = score

    return sparse


def _token_to_id(token: str) -> int:
    """
    将 token 字符串映射为稳定的 32 位无符号整数 ID。
    使用 SHA-256 前 4 字节，碰撞概率极低（~1/2^32）。

    Args:
        token: token 字符串。

    Returns:
        [0, 2^32) 范围内的整数。
    """
    h = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


# 简单停用词表（中英混合）
_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
    "着", "没有", "看", "好", "自己", "这", "那", "它", "他", "她",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "must", "can",
    "could", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "then", "once", "and", "but", "or", "nor", "so", "yet",
    "both", "either", "not", "only", "own", "same", "than", "too",
    "very", "just", "because", "if", "while", "i", "me", "my",
    "we", "our", "you", "your", "it", "its", "this", "that",
}
