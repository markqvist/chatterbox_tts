"""
Text preprocessing pipeline for Chatterbox TTS.

Extensible architecture for current cleanup needs and future expansion
to sentiment analysis and paralinguistic tag insertion.
"""

from __future__ import annotations

import re
from typing import Callable
from dataclasses import dataclass, field


# Paralinguistic tags supported by Chatterbox Turbo
PARALINGUISTIC_TAGS = [
    "[clear throat]",
    "[sigh]",
    "[shush]",
    "[cough]",
    "[groan]",
    "[sniff]",
    "[gasp]",
    "[chuckle]",
    "[laugh]",
]


@dataclass
class PreprocessingContext:
    """Context passed through preprocessing pipeline stages."""
    
    original_text: str
    processed_text: str = ""
    metadata: dict = field(default_factory=dict)
    applied_transforms: list[str] = field(default_factory=list)


class TextPreprocessor:
    """
    Extensible text preprocessing pipeline.
    
    Current capabilities:
    - Strip markdown code blocks
    - Strip markdown tables
    - Replace with meaningful indicators
    - Normalize whitespace
    
    Future expansion hooks:
    - Sentiment analysis
    - Paralinguistic tag insertion
    - Emotion detection
    - Template-based styling
    """
    
    def __init__(self):
        self._stages: list[tuple[str, Callable[[PreprocessingContext], PreprocessingContext]]] = []
        self._register_default_stages()
    
    def _register_default_stages(self):
        """Register default preprocessing stages."""
        self._stages = [
            ("normalize_whitespace", self._normalize_whitespace),
            ("strip_code_blocks", self._strip_code_blocks),
            ("strip_tables", self._strip_tables),
            ("normalize_punctuation", self._normalize_punctuation),
        ]
    
    def preprocess(self, text: str) -> PreprocessingContext:
        """
        Run text through the preprocessing pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            PreprocessingContext with processed text and metadata
        """
        ctx = PreprocessingContext(original_text=text, processed_text=text)
        
        for stage_name, stage_func in self._stages:
            ctx = stage_func(ctx)
            ctx.applied_transforms.append(stage_name)
        
        return ctx
    
    def _normalize_whitespace(self, ctx: PreprocessingContext) -> PreprocessingContext:
        """Normalize whitespace (multiple spaces, tabs, newlines)."""
        text = ctx.processed_text
        # Replace tabs with spaces
        text = text.replace("\t", " ")
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)
        # Replace multiple newlines with single newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        ctx.processed_text = text.strip()
        return ctx
    
    def _strip_code_blocks(self, ctx: PreprocessingContext) -> PreprocessingContext:
        """
        Strip markdown code blocks and replace with indicator.
        
        Matches:
        ```language
code
```
        or
        ```
code
```
        """
        text = ctx.processed_text
        pattern = r"```[\w]*\n?[\s\S]*?\n?```"
        
        def replace_code_block(match: re.Match) -> str:
            block = match.group(0)
            lines = block.count("\n")
            ctx.metadata["code_blocks_removed"] = ctx.metadata.get("code_blocks_removed", 0) + 1
            ctx.metadata["code_lines_removed"] = ctx.metadata.get("code_lines_removed", 0) + lines
            return " (see code in content) "
        
        ctx.processed_text = re.sub(pattern, replace_code_block, text)
        return ctx
    
    def _strip_tables(self, ctx: PreprocessingContext) -> PreprocessingContext:
        """
        Strip markdown tables and replace with indicator.
        
        Matches simple pipe-delimited tables with header separator.
        """
        text = ctx.processed_text
        
        # Pattern for markdown tables
        # Looks for rows starting with | and containing header separator |---|
        table_pattern = r"(\|[^\n]+\|\n\|[-:\| ]+\|\n?(\|[^\n]+\|\n?)+)"
        
        def replace_table(match: re.Match) -> str:
            table = match.group(0)
            rows = table.count("\n") + 1
            ctx.metadata["tables_removed"] = ctx.metadata.get("tables_removed", 0) + 1
            ctx.metadata["table_rows_removed"] = ctx.metadata.get("table_rows_removed", 0) + rows
            return " (see table in content) "
        
        ctx.processed_text = re.sub(table_pattern, replace_table, text)
        return ctx
    
    def _normalize_punctuation(self, ctx: PreprocessingContext) -> PreprocessingContext:
        """
        Normalize common punctuation issues.
        
        Note: Chatterbox has its own punc_norm() that handles many cases,
        but we can do some cleanup here for better input.
        """
        text = ctx.processed_text
        
        # Replace ellipsis with comma + space
        text = text.replace("…", ", ")
        
        # Replace em/en dashes with hyphen
        text = text.replace("—", "-").replace("–", "-")
        
        # Normalize quotes
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")
        
        ctx.processed_text = text
        return ctx
    
    # =========================================================================
    # Future Expansion Hooks (not yet implemented)
    # =========================================================================
    
    def _analyze_sentiment(self, ctx: PreprocessingContext) -> PreprocessingContext:
        """
        Future: Analyze sentiment for paralinguistic tag insertion.
        
        Would detect:
        - Humor/sarcasm -> [chuckle], [laugh]
        - Frustration -> [sigh], [groan]
        - Surprise -> [gasp]
        - Contemplation -> [clear throat], [um]
        """
        # Placeholder for future implementation
        ctx.metadata["sentiment_analysis"] = "not_implemented"
        return ctx
    
    def _insert_paralinguistic_tags(self, ctx: PreprocessingContext) -> PreprocessingContext:
        """
        Future: Auto-insert paralinguistic tags based on analysis.
        
        Would use sentiment analysis results to strategically insert
        tags for more natural speech patterns.
        """
        # Placeholder for future implementation
        return ctx
    
    def _apply_template_style(self, ctx: PreprocessingContext, template: str) -> PreprocessingContext:
        """
        Future: Apply predefined speaking style templates.
        
        Templates:
        - casual: Add fillers like [um], [uh]
        - professional: Minimal paralinguistics
        - expressive: Aggressive tag insertion
        - dramatic: Sighs, gasps, emotional markers
        """
        # Placeholder for future implementation
        ctx.metadata["template_applied"] = template
        return ctx
    
    # =========================================================================
    # Pipeline Management
    # =========================================================================
    
    def add_stage(
        self,
        name: str,
        func: Callable[[PreprocessingContext], PreprocessingContext],
        position: int | None = None
    ):
        """Add a custom preprocessing stage."""
        if position is None:
            self._stages.append((name, func))
        else:
            self._stages.insert(position, (name, func))
    
    def remove_stage(self, name: str):
        """Remove a preprocessing stage by name."""
        self._stages = [(n, f) for n, f in self._stages if n != name]
    
    def clear_stages(self):
        """Clear all preprocessing stages."""
        self._stages = []
    
    def list_stages(self) -> list[str]:
        """List all registered stage names."""
        return [name for name, _ in self._stages]


# Global preprocessor instance
PREPROCESSOR = TextPreprocessor()


def preprocess_text(text: str) -> str:
    """
    Convenience function for quick preprocessing.
    
    Args:
        text: Raw input text
        
    Returns:
        Processed text ready for TTS
    """
    ctx = PREPROCESSOR.preprocess(text)
    return ctx.processed_text
