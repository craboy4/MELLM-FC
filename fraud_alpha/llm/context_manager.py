"""Assemble prompt sections within the configured context budget."""
import re
import logging
from typing import List, Dict, Tuple
logger = logging.getLogger(__name__)

class ContextManager:

    def __init__(self, token_budget: int=20000):
        self.token_budget = token_budget
        self.last_token_estimate = 0

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        chinese_chars = re.findall('[\\u4e00-\\u9fff\\u3400-\\u4dbf]', text)
        n_chinese = len(chinese_chars)
        text_no_chinese = re.sub('[\\u4e00-\\u9fff\\u3400-\\u4dbf]', ' ', text)
        english_words = [w for w in text_no_chinese.split() if w.strip()]
        n_english = len(english_words)
        total = int(n_chinese * 1.5 + n_english)
        return total

    def build_prompt(self, sections: List[Dict]) -> str:
        if not sections:
            self.last_token_estimate = 0
            return ''
        sorted_sections = sorted(sections, key=lambda s: s['priority'])
        kept_parts = []
        used_tokens = 0
        trimmed = False
        for section in sorted_sections:
            content = section['content']
            section_tokens = self.estimate_tokens(content)
            if used_tokens + section_tokens <= self.token_budget:
                kept_parts.append(content)
                used_tokens += section_tokens
            else:
                trimmed = True
                logger.warning(f"Token budget insufficient; trimming section '{section['name']}' (priority={section['priority']}, requires {section_tokens} tokens, remaining {self.token_budget - used_tokens}tokens)")
        if trimmed:
            logger.warning(f'Prompt trimmed: retained {len(kept_parts)}/{len(sections)} sections, estimated {used_tokens} tokens (budget {self.token_budget})')
        result = '\n\n'.join(kept_parts)
        self.last_token_estimate = self.estimate_tokens(result)
        return result

    def trim_factor_display(self, pool_size: int) -> Tuple[int, int]:
        if pool_size <= 20:
            return (pool_size, pool_size)
        else:
            return (10, 5)
