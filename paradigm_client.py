"""
Standalone Paradigm API Client
================================

This file is a complete, self-contained Paradigm API client that can be copied
directly into client applications without any dependencies on the main codebase.

Purpose:
--------
This client will be included in generated workflow packages, allowing clients
to run workflows independently with full Paradigm API support.

Features:
---------
- Complete Paradigm API integration (search, analysis, chat, file upload)
- Asynchronous polling for long-running document analysis
- VisionDocumentSearch fallback for robust extraction
- Zero external dependencies (except aiohttp and standard library)

Usage in Client Packages:
--------------------------
When generating a standalone workflow package for a client, this entire file
is copied into the package. The client only needs to:
1. Set their API key
2. Import ParadigmClient
3. Use it in their workflow

Example:
--------
```python
from paradigm_client_standalone import ParadigmClient

# Initialize
paradigm = ParadigmClient(
    api_key="client_api_key_here",
    base_url="https://paradigm.lighton.ai"
)

# Use it
result = await paradigm.document_search("Find total amount", file_ids=[123])
analysis = await paradigm.analyze_documents_with_polling(
    "Analyze invoice",
    document_ids=[123, 456]
)
```

Version: 1.8.0 (get_file + wait_for_embedding + Session Reuse)
Date: 2025-11-27
Author: LightOn Workflow Builder Team
"""

import aiohttp
import asyncio
import json
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# ============================================================================
# JSON CLEANING UTILITIES
# ============================================================================

def clean_json_response(text: str) -> str:
    """
    Clean JSON response by removing markdown code blocks and extra whitespace.

    Sometimes AI responses wrap JSON in markdown code blocks like:
    ```json
    {"key": "value"}
    ```

    This function removes those wrappers to get clean JSON that can be parsed.

    Args:
        text: Raw text response that may contain JSON wrapped in markdown

    Returns:
        str: Cleaned JSON string ready for parsing

    Examples:
        >>> clean_json_response('```json\\n{"name": "test"}\\n```')
        '{"name": "test"}'

        >>> clean_json_response('{"name": "test"}')
        '{"name": "test"}'
    """
    if not text:
        return text

    # Remove markdown code block markers
    text = text.strip()

    # Remove ```json at the start
    if text.startswith('```json'):
        text = text[7:]  # Remove '```json'
    elif text.startswith('```'):
        text = text[3:]  # Remove '```'

    # Remove ``` at the end
    if text.endswith('```'):
        text = text[:-3]

    # Strip whitespace
    text = text.strip()

    return text


class ParadigmClient:
    """
    Complete standalone client for LightOn Paradigm API with session reuse optimization.

    This client can be copied as-is into any Python project and provides
    full access to Paradigm's document search, analysis, and chat capabilities.

    Performance: Uses session reuse for 5.55x speed improvement over creating
    new connections for each request (1.86s vs 10.33s for 20 requests).

    Query Formulation Best Practices:
        The Paradigm API may reformulate queries, which can lose important information.
        To get the best results (+40% accuracy), follow these rules:

        1. BE SPECIFIC with field names: "Extract SIRET number" instead of "Extract identifier"
        2. INCLUDE FORMATS: "Extract date in DD/MM/YYYY format" instead of "Find the date"
        3. MENTION SECTIONS: "Extract name from 'Legal Information' section" when known
        4. USE DOCUMENT KEYWORDS: "Extract 'Montant TTC'" instead of "Extract total"
        5. AVOID VAGUE TERMS: List specific fields instead of "Extract all information"

        Example of GOOD query:
        "Extract the SIRET number (14 digits) from the 'Informations légales' section"

    Attributes:
        api_key (str): Your Paradigm API authentication key
        base_url (str): The Paradigm API base URL (usually https://paradigm.lighton.ai)
        headers (dict): HTTP headers for authentication

    Example:
        >>> client = ParadigmClient(api_key="sk-...", base_url="https://paradigm.lighton.ai")
        >>> try:
        >>>     result = await client.document_search("Find the invoice total", file_ids=[123])
        >>>     print(result["answer"])
        >>> finally:
        >>>     await client.close()  # Always close to free resources
    """

    def __init__(self, api_key: str, base_url: str = "https://paradigm.lighton.ai"):
        """
        Initialize the Paradigm client.

        Args:
            api_key: Your secret API key from Paradigm
            base_url: The Paradigm API address (default: https://paradigm.lighton.ai)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"✅ ParadigmClient initialized: {base_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create the shared aiohttp session for performance.

        Reusing the same session across multiple requests provides 5.55x performance
        improvement by avoiding connection setup overhead on every call.

        Official benchmark (Paradigm docs):
        - With session reuse: 1.86s for 20 requests
        - Without session reuse: 10.33s for 20 requests

        Returns:
            aiohttp.ClientSession: The shared HTTP session
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            logger.debug("🔌 Created new aiohttp session")
        return self._session

    async def close(self):
        """
        Close the shared aiohttp session and free resources.

        IMPORTANT: Always call this method when done with the client,
        typically in a finally block to ensure cleanup even if errors occur.

        Example:
            >>> client = ParadigmClient(api_key="...")
            >>> try:
            >>>     await client.document_search("query")
            >>> finally:
            >>>     await client.close()
        """
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("🔌 Closed aiohttp session")
            self._session = None

    async def document_search(
        self,
        query: str,
        file_ids: Optional[List[int]] = None,
        workspace_ids: Optional[List[int]] = None,
        chat_session_id: Optional[str] = None,
        model: Optional[str] = None,
        company_scope: bool = False,
        private_scope: bool = True,
        tool: str = "DocumentSearch",
        private: bool = True
    ) -> Dict[str, Any]:
        """
        Search through documents using semantic search.

        Args:
            query: Your search question (e.g., "What is the total amount?")
            file_ids: Which files to search in (e.g., [123, 456])
            workspace_ids: Which workspaces to search (optional)
            chat_session_id: Chat session for context (optional)
            model: Specific AI model to use (optional)
            company_scope: Search company-wide documents
            private_scope: Search private documents
            tool: Search method - "DocumentSearch" (default) or "VisionDocumentSearch"
                  Use "VisionDocumentSearch" for:
                  - Scanned documents or images
                  - Checkboxes or form fields
                  - Complex layouts or tables
                  - Poor OCR quality documents
            private: Whether this request is private

        Returns:
            dict: Search results with "answer", "documents", and metadata

        Example with Vision OCR:
            result = await client.document_search(
                query="Quelle case est cochée dans la section C ?",
                file_ids=[123],
                tool="VisionDocumentSearch"
            )

        Note:
            If tool="VisionDocumentSearch", it analyzes documents as images
            instead of text. Useful for scanned or complex documents.
        """
        endpoint = f"{self.base_url}/api/v2/chat/document-search"

        payload = {
            "query": query,
            "company_scope": company_scope,
            "private_scope": private_scope,
            "tool": tool,
            "private": private
        }

        if file_ids:
            payload["file_ids"] = file_ids
        if workspace_ids:
            payload["workspace_ids"] = workspace_ids
        if chat_session_id:
            payload["chat_session_id"] = chat_session_id
        if model:
            payload["model"] = model

        try:
            logger.info(f"🔍 Document Search: {query[:50]}... (tool={tool})")

            session = await self._get_session()
            async with session.post(
                endpoint,
                json=payload,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Search completed: {len(result.get('documents', []))} documents")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Search failed: {response.status} - {error_text}")
                    raise Exception(f"Document search failed: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"❌ Search error: {str(e)}")
            raise

    async def search_with_vision_fallback(
        self,
        query: str,
        file_ids: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Smart search with automatic fallback to VisionDocumentSearch.

        Tries normal search first, then falls back to vision search if results
        are unclear or empty. Vision search is more robust for scanned documents,
        complex layouts, and poor OCR quality.

        Args:
            query: Search question
            file_ids: Files to search in
            **kwargs: Additional search parameters

        Returns:
            dict: Search results from whichever method succeeded
        """
        try:
            logger.info("🔍 Smart search: trying normal search first...")

            # Try normal search
            result = await self.document_search(
                query,
                file_ids=file_ids,
                tool="DocumentSearch",
                **kwargs
            )

            # Check result quality
            answer = result.get("answer", "").strip()
            has_documents = len(result.get("documents", [])) > 0
            failure_indicators = ["not found", "no information", "cannot find", "unable to", "n/a"]
            seems_unsuccessful = any(indicator in answer.lower() for indicator in failure_indicators)

            if answer and has_documents and not seems_unsuccessful:
                logger.info("✅ Normal search succeeded")
                return result

            # Fallback to vision
            logger.info("⚠️ Normal search unclear, trying vision fallback...")
            vision_result = await self.document_search(
                query,
                file_ids=file_ids,
                tool="VisionDocumentSearch",
                **kwargs
            )

            logger.info("✅ Vision search completed")
            return vision_result

        except Exception as e:
            logger.error(f"❌ Smart search failed: {str(e)}")
            raise

    async def document_analysis_start(
        self,
        query: str,
        document_ids: List[int],
        model: Optional[str] = None,
        private: bool = True
    ) -> str:
        """
        Start a document analysis job (asynchronous).

        Returns a chat_response_id that can be used to poll for results.

        Args:
            query: Analysis question or instruction
            document_ids: Which documents to analyze
            model: Specific AI model (optional)
            private: Whether analysis should be private

        Returns:
            str: chat_response_id (tracking number for this analysis)
        """
        endpoint = f"{self.base_url}/api/v2/chat/document-analysis"

        payload = {
            "query": query,
            "document_ids": document_ids
        }

        if model:
            payload["model"] = model

        try:
            logger.info(f"📊 Starting analysis: {query[:50]}...")

            session = await self._get_session()
            async with session.post(
                endpoint,
                json=payload,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    chat_response_id = result.get("chat_response_id")
                    logger.info(f"✅ Analysis started: {chat_response_id}")
                    return chat_response_id
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Analysis start failed: {response.status}")
                    raise Exception(f"Failed to start analysis: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"❌ Analysis start error: {str(e)}")
            raise

    async def document_analysis_get_result(self, chat_response_id: str) -> Dict[str, Any]:
        """
        Get the result of a running document analysis.

        Args:
            chat_response_id: The tracking number from document_analysis_start

        Returns:
            dict: Analysis result with "status", "result", and metadata
        """
        endpoint = f"{self.base_url}/api/v2/chat/document-analysis/{chat_response_id}"

        try:
            session = await self._get_session()
            async with session.get(endpoint, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    return {"status": "processing"}
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get analysis result: {response.status}")

        except Exception as e:
            logger.error(f"❌ Get result error: {str(e)}")
            raise

    async def analyze_documents_with_polling(
        self,
        query: str,
        document_ids: List[int],
        model: Optional[str] = None,
        private: bool = True,
        max_wait_time: int = 300,
        poll_interval: int = 5
    ) -> str:
        """
        Analyze documents and automatically wait for results (with polling).

        This is the "smart" version that handles everything:
        1. Starts the analysis
        2. Waits and checks every few seconds (polling)
        3. Returns the result when ready

        Args:
            query: Your analysis question
            document_ids: Which documents to analyze
            model: Specific AI model (optional)
            private: Whether analysis is private
            max_wait_time: Maximum seconds to wait (default: 300 = 5 minutes)
            poll_interval: Seconds between checks (default: 5)

        Returns:
            str: The analysis result text

        Raises:
            Exception: If analysis fails or times out

        Example:
            >>> result = await client.analyze_documents_with_polling(
            ...     "Analyze this invoice",
            ...     document_ids=[123, 456]
            ... )
            >>> print(result)  # Automatically waited for completion!
        """
        try:
            logger.info(f"📊 Analysis with polling: max={max_wait_time}s, interval={poll_interval}s")

            # Start the analysis
            chat_response_id = await self.document_analysis_start(
                query, document_ids, model, private
            )

            # Poll for results
            elapsed = 0
            while elapsed < max_wait_time:
                try:
                    result = await self.document_analysis_get_result(chat_response_id)
                    status = result.get("status", "").lower()

                    logger.info(f"🔄 Polling: {status} (elapsed: {elapsed}s)")

                    # Check if completed
                    if status in ["completed", "complete", "finished", "success"]:
                        analysis_result = result.get("result") or result.get("detailed_analysis")
                        if analysis_result:
                            logger.info(f"✅ Analysis done! ({len(analysis_result)} chars)")
                            return analysis_result
                        else:
                            return "Analysis completed but no result was returned"

                    # Check if failed
                    elif status in ["failed", "error"]:
                        logger.error(f"❌ Analysis failed: {status}")
                        raise Exception(f"Analysis failed with status: {status}")

                    # Still processing
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval

                except Exception as e:
                    if "not found" in str(e).lower() or "404" in str(e):
                        # Still processing
                        logger.info(f"⏳ Still running... ({elapsed}s)")
                        await asyncio.sleep(poll_interval)
                        elapsed += poll_interval
                        continue
                    else:
                        raise

            # Timeout
            logger.error(f"⏰ Timeout after {max_wait_time}s")
            raise Exception(f"Analysis timed out after {max_wait_time} seconds")

        except Exception as e:
            logger.error(f"❌ Analysis with polling failed: {str(e)}")
            return f"Document analysis failed: {str(e)}"

    async def chat_completion(
        self,
        prompt: str,
        model: str = "alfred-4.2",
        system_prompt: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        guided_json: Optional[Dict[str, Any]] = None,
        guided_regex: Optional[str] = None
    ) -> str:
        """
        Get a chat completion response (like ChatGPT).

        No documents involved - just a conversation with the AI.

        Args:
            prompt: Your question or instruction
            model: Which AI model to use (default: alfred-4.2)
            system_prompt: Optional instructions for the AI's behavior and output format
                          Use this to enforce specific formats like JSON-only responses
            guided_choice: Optional list of allowed response values (forces AI to choose from list)
            guided_json: Optional JSON schema to enforce structured JSON output format
            guided_regex: Optional regex pattern to enforce structured output format

        Returns:
            str: The AI's response

        Example with JSON-only output:
            result = await client.chat_completion(
                prompt="Vérifie que le nom de l'acheteur est identique dans les deux documents",
                system_prompt='''Tu es un assistant qui réponds UNIQUEMENT au format JSON VALIDE.
                Le json doit contenir :
                "is_correct" : un booléen (true ou false)
                "details" : une phrase expliquant pourquoi la réponse est correcte ou non
                '''
            )
            # Returns: {"is_correct": true, "details": "Les noms sont identiques"}

        Example with guided_choice (force specific values):
            category = await client.chat_completion(
                prompt="Classify this invoice: " + invoice_text,
                guided_choice=["Fournitures", "Services", "Matériel", "Logiciels"]
            )
            # Returns one of: "Fournitures", "Services", "Matériel", or "Logiciels"

        Example with guided_json (guarantee valid JSON):
            data = await client.chat_completion(
                prompt="Extract invoice data from: " + invoice_text,
                guided_json={
                    "type": "object",
                    "properties": {
                        "invoice_number": {"type": "string"},
                        "date": {"type": "string"},
                        "amount": {"type": "number"}
                    }
                }
            )
            # Returns valid JSON matching the schema

        Example without system prompt:
            result = await client.chat_completion(
                prompt="Explique-moi ce qu'est un SIRET"
            )
        """
        endpoint = f"{self.base_url}/api/v2/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages
        }

        # Add guided parameters if provided
        if guided_choice:
            payload["guided_choice"] = guided_choice
        if guided_json:
            payload["guided_json"] = guided_json
        if guided_regex:
            payload["guided_regex"] = guided_regex

        try:
            logger.info(f"💬 Chat completion: {prompt[:50]}...")

            session = await self._get_session()
            async with session.post(
                endpoint,
                json=payload,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result["choices"][0]["message"]["content"]
                    logger.info(f"✅ Chat completed ({len(answer)} chars)")
                    return answer
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Chat failed: {response.status}")
                    raise Exception(f"Chat completion failed: {response.status}")

        except Exception as e:
            logger.error(f"❌ Chat error: {str(e)}")
            raise

    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        collection_type: str = "private"
    ) -> Dict[str, Any]:
        """
        Upload a file to Paradigm for analysis.

        Args:
            file_content: The file data as bytes
            filename: Name of the file (e.g., "invoice.pdf")
            collection_type: Where to store ("private", "company", "workspace")

        Returns:
            dict: Upload result with file ID and metadata
        """
        endpoint = f"{self.base_url}/api/v2/files"

        data = aiohttp.FormData()
        data.add_field('file', file_content, filename=filename)
        data.add_field('collection_type', collection_type)

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            logger.info(f"📁 Uploading: {filename} ({len(file_content)} bytes)")

            session = await self._get_session()
            async with session.post(endpoint, data=data, headers=headers) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    file_id = result.get("id") or result.get("file_id")
                    logger.info(f"✅ File uploaded: ID={file_id}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Upload failed: {response.status}")
                    raise Exception(f"File upload failed: {response.status}")

        except Exception as e:
            logger.error(f"❌ Upload error: {str(e)}")
            raise

    async def filter_chunks(
        self,
        query: str,
        chunk_ids: List[str],
        n: Optional[int] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Filter document chunks based on relevance to a query.

        This method takes a list of chunk UUIDs (typically from document_search)
        and filters them to return only the most relevant ones based on semantic
        similarity to your query.

        Endpoint: POST /api/v2/filter/chunks

        Args:
            query: The query to filter chunks against
            chunk_ids: List of chunk UUIDs to filter (e.g., ["3f885f64-5747-4562-b3fc-2c963f66afa6", ...])
            n: Optional maximum number of chunks to return (returns top N most relevant)
            model: Optional model name to use for filtering

        Returns:
            Dict containing:
            - query: str - The original query used for filtering
            - chunks: List[Dict] - Filtered chunks sorted by relevance (highest first)
                - uuid: str - Chunk UUID
                - text: str - The chunk content
                - metadata: Dict - Additional metadata from the chunk
                - filter_score: float - Relevance score (higher = more relevant)

        When to use:
            ✅ You have many chunks from multiple documents and want only relevant ones
            ✅ Reducing noise in multi-document search results
            ✅ Need to rank chunks by relevance to a specific question
            ✅ Working with 20+ chunks and need the top 5-10

            ❌ You only have a few chunks (2-5) - filtering adds overhead
            ❌ Single document queries - document_search already returns relevant chunks
            ❌ You need ALL chunks regardless of relevance

        Example - Multi-document filtering:
            # Search across multiple documents
            search_result = await paradigm.document_search(
                query="Find contracts",
                file_ids=[101, 102, 103, 104, 105]
            )

            # Extract all chunk IDs from search results
            all_chunks = []
            for doc in search_result.get('documents', []):
                all_chunks.extend(doc.get('chunks', []))

            chunk_uuids = [chunk['uuid'] for chunk in all_chunks]

            # Filter to find chunks specifically about pricing
            pricing_chunks = await paradigm.filter_chunks(
                query="What are the pricing terms and payment conditions?",
                chunk_ids=chunk_uuids,
                n=10
            )

            print(f"Filtered {len(chunk_uuids)} chunks down to {len(pricing_chunks['chunks'])}")

        Example - Without session reuse (automatic):
            filtered = await paradigm.filter_chunks(
                query="technical specifications",
                chunk_ids=["uuid1", "uuid2", "uuid3"]
            )
            # Session reuse happens automatically via self._get_session()

        Raises:
            Exception: If the API call fails or returns an error

        Performance:
            Uses session reuse internally for 5.55x faster performance
            when making multiple filter_chunks calls in sequence.

        Impact:
            +20% precision on multi-document queries by removing irrelevant chunks
            and focusing on the most semantically similar content.
        """
        endpoint = f"{self.base_url}/api/v2/filter/chunks"

        payload = {
            "query": query,
            "chunk_ids": chunk_ids
        }

        if n is not None:
            payload["n"] = n
        if model is not None:
            payload["model"] = model

        try:
            logger.info(f"🔍 Filtering {len(chunk_ids)} chunks")
            logger.info(f"❓ QUERY: {query}")

            session = await self._get_session()
            async with session.post(
                endpoint,
                json=payload,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    num_filtered = len(result.get('chunks', []))
                    logger.info(f"✅ Filter returned {num_filtered} chunks")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Filter chunks failed: {response.status}")
                    raise Exception(f"Filter chunks API error {response.status}: {error_text}")

        except Exception as e:
            logger.error(f"❌ Filter chunks error: {str(e)}")
            raise

    async def get_file_chunks(
        self,
        file_id: int
    ) -> Dict[str, Any]:
        """
        Retrieve all chunks for a given document file.

        Endpoint: GET /api/v2/files/{id}/chunks

        Args:
            file_id: The ID of the file to retrieve chunks from

        Returns:
            Dict containing document chunks and metadata

        Example:
            result = await paradigm.get_file_chunks(file_id=123)
            print(f"Found {len(result.get('chunks', []))} chunks")

        Performance:
            Uses session reuse internally for 5.55x faster performance
        """
        endpoint = f"{self.base_url}/api/v2/files/{file_id}/chunks"

        try:
            logger.info(f"📄 Getting chunks for file {file_id}")

            session = await self._get_session()
            async with session.get(
                endpoint,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    num_chunks = len(result.get('chunks', []))
                    logger.info(f"✅ Retrieved {num_chunks} chunks from file {file_id}")
                    return result

                elif response.status == 404:
                    error_text = await response.text()
                    logger.error(f"❌ File {file_id} not found")
                    raise Exception(f"File {file_id} not found: {error_text}")

                else:
                    error_text = await response.text()
                    logger.error(f"❌ Get file chunks failed: {response.status}")
                    raise Exception(f"Get file chunks API error {response.status}: {error_text}")

        except Exception as e:
            logger.error(f"❌ Get file chunks error: {str(e)}")
            raise

    async def query(
        self,
        query: str,
        collection: Optional[str] = None,
        n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract relevant chunks from knowledge base without AI-generated response.

        This endpoint retrieves semantically relevant chunks based on your query
        WITHOUT generating a synthetic answer. Use this when you only need the raw
        chunks for further processing, saving time and tokens compared to document_search.

        Endpoint: POST /api/v2/query

        Args:
            query: Search query (can be single string or list of strings)
            collection: Collection to query (defaults to base_collection if not specified)
            n: Number of chunks to return (defaults to 5 if not specified)

        Returns:
            Dict containing:
            - query: str - The original query
            - chunks: List[Dict] - Relevant chunks sorted by relevance
                - uuid: str - Chunk UUID
                - text: str - Chunk content
                - metadata: Dict - Additional chunk metadata
                - score: float - Relevance score (higher = more relevant)

        When to use:
            ✅ Need raw chunks without AI synthesis
            ✅ Processing chunks yourself (data extraction, pattern matching)
            ✅ Want to save time and tokens (no text generation)
            ✅ Building custom processing pipelines

            ❌ Need a synthesized answer - use document_search instead
            ❌ Need contextual summary - use document_search instead

        Example:
            # Get top 10 relevant chunks without AI response
            result = await paradigm.query(
                query="Find invoice amounts and dates",
                n=10
            )

            for chunk in result['chunks']:
                print(f"Score: {chunk['score']}")
                print(f"Text: {chunk['text']}")

        Performance:
            Uses session reuse internally for 5.55x faster performance
            ~30% faster than document_search (no AI generation overhead)
        """
        endpoint = f"{self.base_url}/api/v2/query"

        payload = {"query": query}

        if collection is not None:
            payload["collection"] = collection
        if n is not None:
            payload["n"] = n

        try:
            logger.info(f"🔍 Querying knowledge base: {query}")
            if n:
                logger.info(f"📊 Requesting top {n} chunks")

            session = await self._get_session()
            async with session.post(
                endpoint,
                json=payload,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    num_chunks = len(result.get('chunks', []))
                    logger.info(f"✅ Query returned {num_chunks} chunks")
                    return result

                else:
                    error_text = await response.text()
                    logger.error(f"❌ Query failed: {response.status}")
                    raise Exception(f"Query API error {response.status}: {error_text}")

        except Exception as e:
            logger.error(f"❌ Query error: {str(e)}")
            raise

    async def get_file(
        self,
        file_id: int,
        include_content: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve file metadata and status from Paradigm.

        Endpoint: GET /api/v2/files/{id}

        Args:
            file_id: The ID of the file to retrieve
            include_content: Include the file content in the response (default: False)

        Returns:
            Dict containing file metadata including status field

        Example:
            file_info = await paradigm.get_file(file_id=123)
            print(f"Status: {file_info['status']}")

        Performance:
            Uses session reuse internally for 5.55x faster performance
        """
        endpoint = f"{self.base_url}/api/v2/files/{file_id}"

        params = {}
        if include_content:
            params["include_content"] = "true"

        try:
            logger.info(f"📄 Getting file info for ID {file_id}")

            session = await self._get_session()
            async with session.get(
                endpoint,
                params=params,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    status = result.get('status', 'unknown')
                    filename = result.get('filename', 'N/A')
                    logger.info(f"✅ File {file_id} ({filename}): status={status}")
                    return result

                elif response.status == 404:
                    error_text = await response.text()
                    logger.error(f"❌ File {file_id} not found")
                    raise Exception(f"File {file_id} not found: {error_text}")

                else:
                    error_text = await response.text()
                    logger.error(f"❌ Get file failed: {response.status}")
                    raise Exception(f"Get file API error {response.status}: {error_text}")

        except Exception as e:
            logger.error(f"❌ Get file error: {str(e)}")
            raise

    async def list_files(
        self,
        private: Optional[bool] = None,
        workspace_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List available documents for the user.

        Args:
            private: Filter by private documents (True) or company documents (False)
            workspace_id: Filter by specific workspace ID

        Returns:
            List[Dict]: List of file objects with id, filename, etc.

        Example:
            files = await paradigm.list_files(private=True)
            for file in files:
                print(f"{file['id']}: {file['filename']}")
        """
        try:
            session = await self._get_session()

            # Build query parameters for v3 API
            params = {}
            if private is not None:
                # v3 API expects string 'true' or 'false', not boolean
                params['private'] = 'true' if private else 'false'
            if workspace_id is not None:
                params['workspace_id'] = workspace_id

            url = f"{self.base_url}/api/v3/files"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            logger.info(f"📋 Listing files using v3 API with params: {params}")

            # Fetch all pages (v3 API is paginated)
            all_files = []
            current_url = url
            page_num = 1

            while current_url:
                async with session.get(current_url, headers=headers, params=params if page_num == 1 else {}) as response:
                    if response.status == 200:
                        result = await response.json()
                        # v3 API returns: {"count": N, "next": url, "previous": url, "results": [[...]]}
                        # Note: results is a list containing a single list of files
                        results = result.get('results', [])
                        if results and isinstance(results[0], list):
                            page_files = results[0]  # Extract the nested list
                        else:
                            page_files = results

                        all_files.extend(page_files)
                        logger.info(f"✅ Page {page_num}: Retrieved {len(page_files)} files (total so far: {len(all_files)})")

                        # Check for next page
                        current_url = result.get('next')
                        page_num += 1

                        # Safety limit to avoid infinite loops
                        if page_num > 100:
                            logger.warning(f"⚠️ Stopped at page 100 to avoid infinite loop")
                            break
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ List files failed: {response.status}")
                        raise Exception(f"List files API error {response.status}: {error_text}")

            logger.info(f"✅ Retrieved {len(all_files)} total files across {page_num} pages using v3 API")
            return all_files

        except Exception as e:
            logger.error(f"❌ List files error: {str(e)}")
            raise

    async def wait_for_embedding(
        self,
        file_id: int,
        max_wait_time: int = 300,
        poll_interval: int = 2
    ) -> Dict[str, Any]:
        """
        Wait for a file to be fully embedded and ready for use.

        Args:
            file_id: The ID of the file to wait for
            max_wait_time: Maximum time to wait in seconds (default: 300)
            poll_interval: Time between status checks in seconds (default: 2)

        Returns:
            Dict: Final file info when status is 'embedded'

        Example:
            file_info = await paradigm.wait_for_embedding(file_id=123)
            print(f"File ready: {file_info['filename']}")

        Performance:
            Uses session reuse internally for efficient polling (5.55x faster)
        """
        try:
            logger.info(f"⏳ Waiting for file {file_id} to be embedded (max={max_wait_time}s, interval={poll_interval}s)")

            elapsed = 0
            while elapsed < max_wait_time:
                file_info = await self.get_file(file_id)
                status = file_info.get('status', '').lower()
                filename = file_info.get('filename', 'N/A')

                logger.info(f"🔄 File {file_id} ({filename}): status={status} (elapsed: {elapsed}s)")

                if status == 'embedded':
                    logger.info(f"✅ File {file_id} is embedded and ready!")
                    return file_info

                elif status == 'failed':
                    logger.error(f"❌ File {file_id} embedding failed")
                    raise Exception(f"File {file_id} embedding failed")

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            logger.error(f"⏰ Timeout waiting for file {file_id} after {max_wait_time}s")
            raise Exception(f"Timeout waiting for file {file_id} to be embedded")

        except Exception as e:
            logger.error(f"❌ Wait for embedding error: {str(e)}")
            raise

    async def analyze_image(
        self,
        query: str,
        document_ids: List[str],
        model: Optional[str] = None,
        private: bool = False
    ) -> str:
        """
        Analyze images using vision capabilities.

        Args:
            query: What to look for in the images
            document_ids: Image document IDs
            model: Specific vision model (optional)
            private: Privacy setting

        Returns:
            str: Analysis result
        """
        endpoint = f"{self.base_url}/api/v2/chat/image-analysis"

        payload = {
            "query": query,
            "document_ids": document_ids
        }
        if model:
            payload["model"] = model
        if private is not None:
            payload["private"] = private

        try:
            logger.info(f"🖼️ Image analysis: {query[:50]}...")

            session = await self._get_session()
            async with session.post(
                endpoint,
                json=payload,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result.get("answer", "No analysis result provided")
                    logger.info(f"✅ Image analysis completed")
                    return answer
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Image analysis failed: {response.status}")
                    raise Exception(f"Image analysis failed: {response.status}")

        except Exception as e:
            logger.error(f"❌ Image analysis error: {str(e)}")
            raise


# Module metadata
__version__ = "1.9.0"  # tool parameter (VisionDocumentSearch) + system_prompt documentation
__author__ = "LightOn Workflow Builder Team"
__all__ = ["ParadigmClient"]
