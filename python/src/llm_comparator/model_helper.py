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
"""Classes for calling generating LLMs and embedding models."""

import abc
from collections.abc import Sequence
import time
from typing import Optional

from google import genai
from google.genai import types as genai_types
import tqdm.auto

from llm_comparator import _logging


MAX_NUM_RETRIES = 5
DEFAULT_MAX_OUTPUT_TOKENS = 256

_logger = _logging.logger


class GenerationModelHelper(abc.ABC):
  """Class for managing calling LLMs."""

  def predict(self, prompt: str, **kwargs) -> str:
    raise NotImplementedError()

  def predict_batch(self, prompts: Sequence[str], **kwargs) -> Sequence[str]:
    raise NotImplementedError()


class VertexGenerationModelHelper(GenerationModelHelper):
  """Vertex AI text generation model calls via the Google Gen AI SDK."""

  def __init__(
      self,
      model_name: str = 'gemini-2.5-flash',
      project: Optional[str] = None,
      location: Optional[str] = None,
      thinking_budget: Optional[int] = 0,
  ):
    """Initializes the generation model helper.

    Args:
      model_name: Name of a Gemini model available on Vertex AI.
      project: Google Cloud project ID. Falls back to the
        GOOGLE_CLOUD_PROJECT environment variable.
      location: Google Cloud region, e.g. 'us-central1'. Falls back to the
        GOOGLE_CLOUD_LOCATION environment variable.
      thinking_budget: Thinking token budget for reasoning models. Defaults
        to 0 (thinking disabled) so that short judge/bulletize/cluster
        responses are not consumed by thinking tokens. Set to None to use the
        model's default dynamic thinking; models that cannot disable thinking
        (e.g. gemini-2.5-pro) require None or a positive budget.
    """
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    self.client = genai.Client(
        vertexai=True, project=project, location=location
    )

  def predict(
      self,
      prompt: str,
      temperature: Optional[float] = None,
      max_output_tokens: Optional[int] = DEFAULT_MAX_OUTPUT_TOKENS,
  ) -> str:
    if not prompt:
      return ''
    num_attempts = 0
    response = None

    thinking_config = None
    if self.thinking_budget is not None:
      thinking_config = genai_types.ThinkingConfig(
          thinking_budget=self.thinking_budget
      )

    while num_attempts < MAX_NUM_RETRIES and response is None:
      num_attempts += 1

      try:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                candidate_count=1,
                max_output_tokens=max_output_tokens,
                thinking_config=thinking_config,
            ),
        )
      except Exception as e:  # pylint: disable=broad-except
        if 'quota' in str(e).lower():
          _logger.info('\033[31mQuota limit exceeded.\033[0m')
        wait_time = 2**num_attempts
        _logger.info('\033[31mWaiting %ds to retry...\033[0m', wait_time)
        time.sleep(wait_time)

    if response is None or response.text is None:
      return ''
    return response.text

  def predict_batch(
      self,
      prompts: Sequence[str],
      temperature: Optional[float] = None,
      max_output_tokens: Optional[int] = DEFAULT_MAX_OUTPUT_TOKENS,
  ) -> Sequence[str]:
    outputs = []
    for i in tqdm.auto.tqdm(range(0, len(prompts))):
      # TODO(b/344631023): Implement multiprocessing.
      outputs.append(self.predict(prompts[i], temperature, max_output_tokens))
    return outputs


class EmbeddingModelHelper(abc.ABC):
  """Class for managing calling text embedding models."""

  def embed(self, text: str) -> Sequence[float]:
    raise NotImplementedError()

  def embed_batch(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
    raise NotImplementedError()


class VertexEmbeddingModelHelper(EmbeddingModelHelper):
  """Vertex AI text embedding model calls via the Google Gen AI SDK."""

  def __init__(
      self,
      model_name: str = 'gemini-embedding-001',
      project: Optional[str] = None,
      location: Optional[str] = None,
      output_dimensionality: Optional[int] = None,
  ):
    """Initializes the embedding model helper.

    Args:
      model_name: Name of a text embedding model available on Vertex AI.
      project: Google Cloud project ID. Falls back to the
        GOOGLE_CLOUD_PROJECT environment variable.
      location: Google Cloud region, e.g. 'us-central1'. Falls back to the
        GOOGLE_CLOUD_LOCATION environment variable.
      output_dimensionality: Optional embedding size override, e.g. 768.
        Defaults to the model's native size (3072 for gemini-embedding-001).
    """
    self.model_name = model_name
    self.output_dimensionality = output_dimensionality
    self.client = genai.Client(
        vertexai=True, project=project, location=location
    )

  def embed(self, text: str) -> Sequence[float]:
    """Embeds a string into the model's embedding space."""
    # On Vertex AI, gemini-embedding-001 accepts only one input per request,
    # so embedding requests are issued per-text rather than in batches.
    num_attempts = 0
    embeddings = None

    config = None
    if self.output_dimensionality is not None:
      config = genai_types.EmbedContentConfig(
          output_dimensionality=self.output_dimensionality
      )

    while num_attempts < MAX_NUM_RETRIES and embeddings is None:
      num_attempts += 1
      try:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=config,
        )
        embeddings = response.embeddings
      except Exception as e:  # pylint: disable=broad-except
        wait_time = 2**num_attempts
        _logger.info('Waiting %ds to retry... (%s)', wait_time, e)
        time.sleep(wait_time)

    if not embeddings:
      return []

    return embeddings[0].values

  def embed_batch(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
    return [self.embed(text) for text in tqdm.auto.tqdm(texts)]
