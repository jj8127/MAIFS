"""
Knowledge Base Module
각 전문가 에이전트를 위한 도메인 지식 관리
"""
from pathlib import Path
from typing import Optional


class KnowledgeBase:
    """도메인 지식 베이스 로더"""

    KNOWLEDGE_DIR = Path(__file__).parent

    KNOWLEDGE_FILES = {
        "frequency": "frequency_domain_knowledge.md",
        "noise": "noise_domain_knowledge.md",
        "fatformer": "fatformer_domain_knowledge.md",
        "spatial": "spatial_domain_knowledge.md",
    }

    _cache = {}  # 메모리 캐시

    @classmethod
    def load(cls, domain: str) -> str:
        """
        도메인 지식 로드

        Args:
            domain: "frequency", "noise", "fatformer", "spatial"

        Returns:
            str: 도메인 지식 전체 내용
        """
        if domain in cls._cache:
            return cls._cache[domain]

        if domain not in cls.KNOWLEDGE_FILES:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(cls.KNOWLEDGE_FILES.keys())}")

        knowledge_file = cls.KNOWLEDGE_DIR / cls.KNOWLEDGE_FILES[domain]

        if not knowledge_file.exists():
            raise FileNotFoundError(f"Knowledge file not found: {knowledge_file}")

        with open(knowledge_file, 'r', encoding='utf-8') as f:
            content = f.read()

        cls._cache[domain] = content
        return content

    @classmethod
    def get_summary(cls, domain: str, max_chars: int = 2000) -> str:
        """
        도메인 지식 요약 (LLM 토큰 제한용)

        Args:
            domain: 도메인 이름
            max_chars: 최대 문자 수

        Returns:
            str: 요약된 지식
        """
        full_knowledge = cls.load(domain)

        # 섹션 추출: ## 📚 과학적 근거 ~ ## 📊 메트릭 해석 가이드
        lines = full_knowledge.split('\n')

        summary_lines = []
        in_summary = False

        for line in lines:
            # 핵심 섹션 시작
            if line.startswith('## 📚 과학적 근거') or line.startswith('## 🔬 분석 원리'):
                in_summary = True

            # 예시 섹션은 제외 (너무 길어짐)
            if line.startswith('## 💡 해석 예시') or line.startswith('## 🔍 특수 케이스'):
                in_summary = False

            if in_summary:
                summary_lines.append(line)

        summary = '\n'.join(summary_lines)

        # 길이 제한
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n\n... (전체 지식은 파일 참조)"

        return summary


__all__ = ["KnowledgeBase"]
