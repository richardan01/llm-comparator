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
"""Error classification system for LLM Comparator."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import re
import math

from llm_comparator import error_taxonomy
from llm_comparator import types
from llm_comparator import _logging

_logger = _logging.logger


@dataclass
class DetectedError:
    """Represents a detected error in an LLM response."""
    error_type: error_taxonomy.ErrorType
    confidence: float  # 0.0 to 1.0
    evidence: str  # Text that triggered the detection
    location: Optional[str] = None  # Where in the response the error was found
    severity_override: Optional[error_taxonomy.ErrorSeverity] = None
    
    @property
    def severity(self) -> error_taxonomy.ErrorSeverity:
        """Get the effective severity of this error."""
        return self.severity_override or self.error_type.severity_default


@dataclass
class ErrorAnalysisResult:
    """Complete error analysis result for a single response."""
    response_id: str
    detected_errors: List[DetectedError]
    overall_error_score: float  # Weighted error score
    category_scores: Dict[error_taxonomy.ErrorCategory, float]
    confidence_score: float  # Overall confidence in the analysis
    
    @property
    def error_count_by_category(self) -> Dict[error_taxonomy.ErrorCategory, int]:
        """Count of errors by category."""
        counts = defaultdict(int)
        for error in self.detected_errors:
            counts[error.error_type.category] += 1
        return dict(counts)
    
    @property
    def has_critical_errors(self) -> bool:
        """Check if any critical errors were detected."""
        return any(error.severity == error_taxonomy.ErrorSeverity.CRITICAL 
                  for error in self.detected_errors)


class ErrorClassifier:
    """Automated error classification system."""
    
    def __init__(self, 
                 generation_model_helper=None,
                 taxonomy: Optional[error_taxonomy.ErrorTaxonomy] = None):
        """Initialize the error classifier.
        
        Args:
            generation_model_helper: Optional LLM for advanced error detection
            taxonomy: Error taxonomy to use (defaults to global taxonomy)
        """
        self.taxonomy = taxonomy or error_taxonomy.ERROR_TAXONOMY
        self.generation_model_helper = generation_model_helper
        
        # Compile regex patterns for efficiency
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[Tuple[re.Pattern, error_taxonomy.ErrorType]]]:
        """Compile regex patterns for efficient matching."""
        compiled = defaultdict(list)
        
        for error_type in self.taxonomy.error_types.values():
            for pattern in error_type.patterns:
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    compiled[error_type.category.value].append((compiled_pattern, error_type))
                except re.error as e:
                    _logger.warning(f"Invalid regex pattern for {error_type.name}: {pattern} - {e}")
        
        return dict(compiled)
    
    def _detect_pattern_based_errors(self, text: str) -> List[DetectedError]:
        """Detect errors using regex patterns."""
        detected_errors = []
        
        for category, patterns in self._compiled_patterns.items():
            for pattern, error_type in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    # Calculate confidence based on match quality
                    confidence = self._calculate_pattern_confidence(match, text)
                    
                    detected_errors.append(DetectedError(
                        error_type=error_type,
                        confidence=confidence,
                        evidence=match.group(0),
                        location=f"Position {match.start()}-{match.end()}"
                    ))
        
        return detected_errors
    
    def _calculate_pattern_confidence(self, match: re.Match, full_text: str) -> float:
        """Calculate confidence score for a pattern match."""
        # Base confidence
        confidence = 0.6
        
        # Boost confidence for longer matches
        match_length = len(match.group(0))
        if match_length > 20:
            confidence += 0.1
        elif match_length > 50:
            confidence += 0.2
        
        # Boost confidence if match is in a sentence with error keywords
        sentence_start = max(0, match.start() - 100)
        sentence_end = min(len(full_text), match.end() + 100)
        context = full_text[sentence_start:sentence_end].lower()
        
        error_keywords = ["wrong", "incorrect", "error", "mistake", "false", "inaccurate"]
        keyword_count = sum(1 for keyword in error_keywords if keyword in context)
        confidence += min(0.2, keyword_count * 0.05)
        
        return min(1.0, confidence)
    
    def _calculate_overall_error_score(self, detected_errors: List[DetectedError]) -> float:
        """Calculate an overall error score based on detected errors."""
        if not detected_errors:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for error in detected_errors:
            # Get category weight
            category_weight = self.taxonomy.get_category_weight(error.error_type.category)
            
            # Get severity multiplier
            severity_multipliers = {
                error_taxonomy.ErrorSeverity.CRITICAL: 1.0,
                error_taxonomy.ErrorSeverity.HIGH: 0.8,
                error_taxonomy.ErrorSeverity.MEDIUM: 0.6,
                error_taxonomy.ErrorSeverity.LOW: 0.4,
                error_taxonomy.ErrorSeverity.NEGLIGIBLE: 0.2,
            }
            severity_multiplier = severity_multipliers.get(error.severity, 0.5)
            
            # Calculate weighted score for this error
            error_score = error.confidence * category_weight * severity_multiplier
            total_weighted_score += error_score
            total_weight += category_weight * severity_multiplier
        
        # Normalize to 0-1 range
        if total_weight == 0:
            return 0.0
        
        normalized_score = total_weighted_score / total_weight
        return min(1.0, normalized_score)
    
    def _calculate_category_scores(self, detected_errors: List[DetectedError]) -> Dict[error_taxonomy.ErrorCategory, float]:
        """Calculate error scores by category."""
        category_scores = {}
        category_errors = defaultdict(list)
        
        # Group errors by category
        for error in detected_errors:
            category_errors[error.error_type.category].append(error)
        
        # Calculate score for each category
        for category in self.taxonomy.get_all_categories():
            errors_in_category = category_errors.get(category, [])
            
            if not errors_in_category:
                category_scores[category] = 0.0
                continue
            
            # Average confidence weighted by severity
            total_score = 0.0
            total_weight = 0.0
            
            for error in errors_in_category:
                severity_weight = {
                    error_taxonomy.ErrorSeverity.CRITICAL: 1.0,
                    error_taxonomy.ErrorSeverity.HIGH: 0.8,
                    error_taxonomy.ErrorSeverity.MEDIUM: 0.6,
                    error_taxonomy.ErrorSeverity.LOW: 0.4,
                    error_taxonomy.ErrorSeverity.NEGLIGIBLE: 0.2,
                }.get(error.severity, 0.5)
                
                total_score += error.confidence * severity_weight
                total_weight += severity_weight
            
            category_scores[category] = total_score / total_weight if total_weight > 0 else 0.0
        
        return category_scores
    
    def classify_errors(self, 
                       response_id: str,
                       prompt: str, 
                       response: str,
                       use_llm_analysis: bool = False) -> ErrorAnalysisResult:
        """Classify errors in an LLM response.
        
        Args:
            response_id: Unique identifier for this response
            prompt: The original prompt/question
            response: The LLM's response to analyze
            use_llm_analysis: Whether to use LLM-based error detection (disabled for now)
            
        Returns:
            ErrorAnalysisResult containing all detected errors and scores
        """
        detected_errors = []
        
        # Pattern-based detection
        pattern_errors = self._detect_pattern_based_errors(response)
        detected_errors.extend(pattern_errors)
        
        # Remove duplicates and merge similar errors
        detected_errors = self._deduplicate_errors(detected_errors)
        
        # Calculate scores
        overall_score = self._calculate_overall_error_score(detected_errors)
        category_scores = self._calculate_category_scores(detected_errors)
        
        # Calculate overall confidence
        if detected_errors:
            confidence_score = sum(error.confidence for error in detected_errors) / len(detected_errors)
        else:
            confidence_score = 1.0  # High confidence in "no errors found"
        
        return ErrorAnalysisResult(
            response_id=response_id,
            detected_errors=detected_errors,
            overall_error_score=overall_score,
            category_scores=category_scores,
            confidence_score=confidence_score
        )
    
    def _deduplicate_errors(self, errors: List[DetectedError]) -> List[DetectedError]:
        """Remove duplicate errors and merge similar ones."""
        if not errors:
            return []
        
        # Group errors by type and evidence similarity
        error_groups = defaultdict(list)
        
        for error in errors:
            # Create a key based on error type and evidence similarity
            key = (error.error_type.name, error.evidence[:50])  # First 50 chars of evidence
            error_groups[key].append(error)
        
        # Keep the highest confidence error from each group
        deduplicated = []
        for group in error_groups.values():
            best_error = max(group, key=lambda e: e.confidence)
            deduplicated.append(best_error)
        
        return deduplicated