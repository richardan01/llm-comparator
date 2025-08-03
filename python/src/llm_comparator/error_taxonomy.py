# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Error taxonomy and classification system for LLM Comparator."""

from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import re


class ErrorCategory(Enum):
    """Primary error categories for LLM responses."""
    FACTUAL = "factual"
    REASONING = "reasoning"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    BIAS = "bias"
    SAFETY = "safety"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    FORMATTING = "formatting"
    HALLUCINATION = "hallucination"


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"      # Completely wrong or harmful
    HIGH = "high"             # Significantly impacts quality
    MEDIUM = "medium"         # Noticeable but manageable
    LOW = "low"              # Minor issues
    NEGLIGIBLE = "negligible" # Barely noticeable


@dataclass
class ErrorType:
    """Detailed error type definition."""
    name: str
    category: ErrorCategory
    description: str
    keywords: List[str]
    patterns: List[str]  # Regex patterns for detection
    severity_default: ErrorSeverity


class ErrorTaxonomy:
    """Comprehensive error taxonomy for LLM evaluation."""
    
    def __init__(self):
        self.error_types = self._initialize_error_types()
        self.category_weights = self._initialize_category_weights()
    
    def _initialize_error_types(self) -> Dict[str, ErrorType]:
        """Initialize the comprehensive error taxonomy."""
        error_types = {}
        
        # Factual Errors
        error_types["factual_incorrect"] = ErrorType(
            name="Factual Incorrectness",
            category=ErrorCategory.FACTUAL,
            description="Information that is objectively wrong or inaccurate",
            keywords=["incorrect", "wrong", "false", "inaccurate", "mistaken", "error"],
            patterns=[r"(?i)(incorrect|wrong|false|inaccurate|mistaken).*fact"],
            severity_default=ErrorSeverity.HIGH
        )
        
        error_types["factual_outdated"] = ErrorType(
            name="Outdated Information",
            category=ErrorCategory.FACTUAL,
            description="Information that was correct but is now outdated",
            keywords=["outdated", "old", "obsolete", "deprecated", "superseded"],
            patterns=[r"(?i)(outdated|obsolete|deprecated).*information"],
            severity_default=ErrorSeverity.MEDIUM
        )
        
        # Reasoning Errors
        error_types["logical_fallacy"] = ErrorType(
            name="Logical Fallacy",
            category=ErrorCategory.REASONING,
            description="Flawed logical reasoning or argumentation",
            keywords=["fallacy", "illogical", "contradictory", "inconsistent logic"],
            patterns=[r"(?i)(logical.*fallacy|flawed.*reasoning|contradictory)"],
            severity_default=ErrorSeverity.HIGH
        )
        
        error_types["causal_confusion"] = ErrorType(
            name="Causal Confusion",
            category=ErrorCategory.REASONING,
            description="Incorrect cause-effect relationships",
            keywords=["cause", "effect", "because", "therefore", "leads to"],
            patterns=[r"(?i)(incorrect.*causation|wrong.*cause|flawed.*reasoning)"],
            severity_default=ErrorSeverity.MEDIUM
        )
        
        # Coherence Errors
        error_types["internal_contradiction"] = ErrorType(
            name="Internal Contradiction",
            category=ErrorCategory.COHERENCE,
            description="Statements that contradict each other within the response",
            keywords=["contradicts", "inconsistent", "conflicts", "opposite"],
            patterns=[r"(?i)(contradicts|conflicts.*with|inconsistent)"],
            severity_default=ErrorSeverity.HIGH
        )
        
        error_types["unclear_structure"] = ErrorType(
            name="Unclear Structure",
            category=ErrorCategory.COHERENCE,
            description="Poor organization or flow of ideas",
            keywords=["unclear", "confusing", "disorganized", "jumbled"],
            patterns=[r"(?i)(unclear.*structure|disorganized|confusing.*flow)"],
            severity_default=ErrorSeverity.MEDIUM
        )
        
        # Relevance Errors
        error_types["off_topic"] = ErrorType(
            name="Off-topic Response",
            category=ErrorCategory.RELEVANCE,
            description="Response doesn't address the question or prompt",
            keywords=["off-topic", "irrelevant", "unrelated", "doesn't answer"],
            patterns=[r"(?i)(off.topic|irrelevant|unrelated.*to.*question)"],
            severity_default=ErrorSeverity.HIGH
        )
        
        error_types["partial_relevance"] = ErrorType(
            name="Partial Relevance",
            category=ErrorCategory.RELEVANCE,
            description="Response only partially addresses the prompt",
            keywords=["partially", "incomplete", "misses", "doesn't fully"],
            patterns=[r"(?i)(partially.*addresses|incomplete.*answer|misses.*point)"],
            severity_default=ErrorSeverity.MEDIUM
        )
        
        # Bias Errors
        error_types["demographic_bias"] = ErrorType(
            name="Demographic Bias",
            category=ErrorCategory.BIAS,
            description="Unfair treatment based on demographic characteristics",
            keywords=["biased", "stereotyping", "discrimination", "unfair"],
            patterns=[r"(?i)(biased.*against|stereotyping|discriminatory)"],
            severity_default=ErrorSeverity.CRITICAL
        )
        
        error_types["cultural_bias"] = ErrorType(
            name="Cultural Bias",
            category=ErrorCategory.BIAS,
            description="Assumptions based on specific cultural perspectives",
            keywords=["cultural", "western", "assumption", "perspective"],
            patterns=[r"(?i)(cultural.*bias|western.*centric|narrow.*perspective)"],
            severity_default=ErrorSeverity.HIGH
        )
        
        # Safety Errors
        error_types["harmful_content"] = ErrorType(
            name="Harmful Content",
            category=ErrorCategory.SAFETY,
            description="Content that could cause harm or promote dangerous activities",
            keywords=["harmful", "dangerous", "unsafe", "risky", "toxic"],
            patterns=[r"(?i)(harmful|dangerous|unsafe|toxic.*content)"],
            severity_default=ErrorSeverity.CRITICAL
        )
        
        error_types["misinformation"] = ErrorType(
            name="Misinformation",
            category=ErrorCategory.SAFETY,
            description="False information that could mislead users",
            keywords=["misinformation", "misleading", "false claim", "conspiracy"],
            patterns=[r"(?i)(misinformation|misleading.*claim|false.*information)"],
            severity_default=ErrorSeverity.CRITICAL
        )
        
        # Completeness Errors
        error_types["incomplete_answer"] = ErrorType(
            name="Incomplete Answer",
            category=ErrorCategory.COMPLETENESS,
            description="Response doesn't fully address all aspects of the question",
            keywords=["incomplete", "missing", "partial", "doesn't cover"],
            patterns=[r"(?i)(incomplete.*answer|missing.*information|doesn't.*cover)"],
            severity_default=ErrorSeverity.MEDIUM
        )
        
        error_types["lacks_detail"] = ErrorType(
            name="Lacks Detail",
            category=ErrorCategory.COMPLETENESS,
            description="Response is too superficial or lacks necessary detail",
            keywords=["superficial", "lacks detail", "too brief", "vague"],
            patterns=[r"(?i)(lacks.*detail|too.*superficial|insufficient.*depth)"],
            severity_default=ErrorSeverity.LOW
        )
        
        # Consistency Errors
        error_types["format_inconsistency"] = ErrorType(
            name="Format Inconsistency",
            category=ErrorCategory.CONSISTENCY,
            description="Inconsistent formatting or style within the response",
            keywords=["inconsistent", "formatting", "style", "mixed"],
            patterns=[r"(?i)(inconsistent.*format|mixed.*style|formatting.*error)"],
            severity_default=ErrorSeverity.LOW
        )
        
        # Hallucination Errors
        error_types["fabricated_facts"] = ErrorType(
            name="Fabricated Facts",
            category=ErrorCategory.HALLUCINATION,
            description="Made-up information presented as factual",
            keywords=["fabricated", "made up", "invented", "fictional"],
            patterns=[r"(?i)(fabricated|made.*up|invented.*fact|fictional.*claim)"],
            severity_default=ErrorSeverity.CRITICAL
        )
        
        error_types["false_citations"] = ErrorType(
            name="False Citations",
            category=ErrorCategory.HALLUCINATION,
            description="Non-existent or incorrect citations and references",
            keywords=["false citation", "fake reference", "non-existent", "made up source"],
            patterns=[r"(?i)(false.*citation|fake.*reference|non.existent.*source)"],
            severity_default=ErrorSeverity.HIGH
        )
        
        return error_types
    
    def _initialize_category_weights(self) -> Dict[ErrorCategory, float]:
        """Initialize weights for different error categories."""
        return {
            ErrorCategory.SAFETY: 1.0,          # Highest priority
            ErrorCategory.FACTUAL: 0.9,
            ErrorCategory.HALLUCINATION: 0.9,
            ErrorCategory.BIAS: 0.8,
            ErrorCategory.REASONING: 0.7,
            ErrorCategory.RELEVANCE: 0.6,
            ErrorCategory.COHERENCE: 0.5,
            ErrorCategory.COMPLETENESS: 0.4,
            ErrorCategory.CONSISTENCY: 0.3,
            ErrorCategory.FORMATTING: 0.2      # Lowest priority
        }
    
    def get_error_types_by_category(self, category: ErrorCategory) -> List[ErrorType]:
        """Get all error types for a specific category."""
        return [error_type for error_type in self.error_types.values() 
                if error_type.category == category]
    
    def get_all_categories(self) -> List[ErrorCategory]:
        """Get all error categories."""
        return list(ErrorCategory)
    
    def get_category_weight(self, category: ErrorCategory) -> float:
        """Get the weight for a specific error category."""
        return self.category_weights.get(category, 0.5)
    
    def search_error_types(self, query: str) -> List[ErrorType]:
        """Search for error types based on keywords or description."""
        query_lower = query.lower()
        matching_types = []
        
        for error_type in self.error_types.values():
            # Check if query matches keywords
            if any(keyword.lower() in query_lower for keyword in error_type.keywords):
                matching_types.append(error_type)
                continue
            
            # Check if query matches description
            if query_lower in error_type.description.lower():
                matching_types.append(error_type)
                continue
            
            # Check if query matches patterns
            for pattern in error_type.patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    matching_types.append(error_type)
                    break
        
        return matching_types


# Global taxonomy instance
ERROR_TAXONOMY = ErrorTaxonomy()
