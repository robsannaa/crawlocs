import os
import time
import asyncio
import argparse
import sys
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.docstore.document import Document

# Import crawl4ai components
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    DomainFilter,
    ContentTypeFilter,
)
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "docs-mcp")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", dimensions=1536, request_timeout=30
)


def add_documents(library: str, docs: list[Document]):
    """Adds documents to the specified library (namespace) in Pinecone."""
    namespace = library.lower()
    print(f"📚 Adding {len(docs)} documents to namespace '{namespace}'...")
    print(f"   🎯 Target namespace: {namespace}")
    print(f"   📦 Total documents to process: {len(docs)}")

    # Optimal batch size for upserting vectors
    batch_size = 100
    total_batches = (len(docs) + batch_size - 1) // batch_size
    index = pc.Index(index_name)
    
    print(f"   🔧 Using batch size: {batch_size}")
    print(f"   📊 Total batches to process: {total_batches}")
    print(f"   🗄️  Target Pinecone index: {index_name}")
    print("")

    for i in range(0, len(docs), batch_size):
        batch_num = (i // batch_size) + 1
        batch_docs = docs[i : i + batch_size]

        print(f"   📦 Processing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)...")
        print(f"      🎯 Batch range: {i+1} to {min(i+batch_size, len(docs))}")
        print(f"      📄 Documents in this batch: {len(batch_docs)}")

        # Generate embeddings for the batch
        texts_to_embed = [doc.page_content for doc in batch_docs]
        try:
            print(f"      🧠 Generating embeddings for {len(texts_to_embed)} documents...")
            print(f"         📏 Average text length: {sum(len(text) for text in texts_to_embed) // len(texts_to_embed)} characters")
            print(f"         📏 Total text to embed: {sum(len(text) for text in texts_to_embed)} characters")
            
            doc_embeddings = embeddings.embed_documents(texts_to_embed)
            print(f"         ✅ Embeddings generated successfully!")
            print(f"         🎯 Embedding dimensions: {len(doc_embeddings[0]) if doc_embeddings else 0}")
        except Exception as e:
            print(f"         ❌ Failed to generate embeddings for batch {batch_num}: {e}")
            raise

        # Prepare vectors for upsert
        vectors = []
        print(f"      🔧 Preparing vectors for Pinecone upsert...")
        for j, doc in enumerate(batch_docs):
            # Truncate chunk_text to stay within Pinecone's 40KB metadata limit
            # Keep first 1000 characters for context, which should be ~1-2KB
            chunk_text = doc.page_content
            if len(chunk_text) > 1000:
                chunk_text = chunk_text[:1000] + "... [truncated]"
                print(f"         ⚠️  Truncated chunk {j+1} from {len(doc.page_content)} to {len(chunk_text)} characters")
            
            metadata = {
                "chunk_text": chunk_text,
                "source": doc.metadata.get("source", ""),
                "title": doc.metadata.get("title", ""),
                "depth": doc.metadata.get("depth", 0),
                "library": library,
            }
            
            # Calculate metadata size to ensure we stay within limits
            import json
            metadata_size = len(json.dumps(metadata))
            if metadata_size > 35000:  # Leave some buffer below 40KB limit
                print(f"         ⚠️  Metadata size ({metadata_size} bytes) approaching limit, further truncating...")
                # Further truncate chunk_text if needed
                max_chunk_length = 500
                chunk_text = doc.page_content[:max_chunk_length] + "... [truncated]"
                metadata["chunk_text"] = chunk_text
                metadata_size = len(json.dumps(metadata))
                print(f"         ✅ Final metadata size: {metadata_size} bytes")
            
            vector = {
                "id": f"{library}_{i + j}_{int(time.time())}",
                "values": doc_embeddings[j],
                "metadata": metadata,
            }
            vectors.append(vector)
        
        print(f"         ✅ Prepared {len(vectors)} vectors")
        print(f"         🆔 Vector ID format: {library}_<index>_<timestamp>")

        success = False
        for attempt in range(3):
            try:
                start_time = time.time()
                print(f"      🔄 Attempt {attempt + 1}/3: Upserting batch {batch_num} with {len(vectors)} vectors...")
                print(f"         🗄️  Target namespace: {namespace}")
                print(f"         📊 Vector count: {len(vectors)}")
                print(f"         🎯 Embedding model: {embeddings.model}")

                # Use integrated embedding - Pinecone will handle the embedding generation
                index.upsert(vectors=vectors, namespace=namespace)

                elapsed = time.time() - start_time
                print(f"         ✅ Batch {batch_num} completed successfully in {elapsed:.1f}s")
                print(f"         ⚡ Average time per vector: {elapsed/len(vectors):.3f}s")
                success = True
                break

            except Exception as e:
                print(f"         ❌ Attempt {attempt + 1} failed for batch {batch_num}: {e}")
                if attempt < 2:
                    print(f"         🔄 Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"         💥 All attempts failed for batch {batch_num}")
                    raise

        if not success:
            raise Exception(f"Failed to process batch {batch_num} after all attempts")
        
        print(f"      🎉 Batch {batch_num} completed successfully!")
        print("")

    print(f"✅ Successfully added all {len(docs)} documents to namespace '{namespace}'")
    print(f"   📊 Total documents processed: {len(docs)}")
    print(f"   📦 Total batches completed: {total_batches}")
    print(f"   🗄️  Target namespace: {namespace}")
    print(f"   🎯 Target index: {index_name}")
    print("")


def log_message(message: str, library: str = None):
    """Log a message to both console and a log file for real-time monitoring."""
    timestamp = time.strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"

    # Print to console with enhanced formatting
    print(formatted_message, flush=True)

    # Write to log file for detailed monitoring
    if library:
        log_file = f"crawl_{library}.log"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(formatted_message + "\n")
                f.flush()
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")

    # Also write to temp file for Streamlit
    if library:
        temp_log_file = f"/tmp/crawl_{library}.log"
        try:
            with open(temp_log_file, "a", encoding="utf-8") as f:
                f.write(formatted_message + "\n")
                f.flush()
        except Exception:
            pass  # Don't fail if we can't write to temp log file


def split_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """Splits markdown documents based on headers for better semantic structure."""
    print(f"📄 Starting markdown document splitting...")
    print(f"   📊 Input documents: {len(documents)}")
    print(f"   🔧 Using MarkdownHeaderTextSplitter")
    print(f"   📏 Chunk size: {chunk_size}")
    print(f"   🔄 Chunk overlap: {chunk_overlap}")
    
    # Define headers to split on - this preserves the documentation structure
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"), 
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]
    
    print(f"   🎯 Headers to split on: {len(headers_to_split_on)} levels")
    for i, (header, name) in enumerate(headers_to_split_on):
        print(f"      {i+1}. {header} → {name}")
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line=False,
        strip_headers=True,
    )
    
    all_chunks = []
    print(f"   🔄 Processing {len(documents)} documents...")
    
    for doc_idx, doc in enumerate(documents):
        print(f"      📄 Document {doc_idx + 1}/{len(documents)}: {doc.metadata.get('title', 'Untitled')}")
        print(f"         📏 Content length: {len(doc.page_content)} characters")
        print(f"         🔗 Source: {doc.metadata.get('source', 'Unknown')}")
        
        try:
            # Split the markdown content based on headers
            print(f"         🔧 Splitting with MarkdownHeaderTextSplitter...")
            chunks = markdown_splitter.split_text(doc.page_content)
            print(f"         ✅ Split into {len(chunks)} chunks")
            
            # Preserve the original metadata and add header information
            for chunk_idx, chunk in enumerate(chunks):
                chunk.metadata.update(doc.metadata)
                # Add header context if available
                if hasattr(chunk.metadata, 'Header 1'):
                    chunk.metadata['section'] = chunk.metadata['Header 1']
                elif hasattr(chunk.metadata, 'Header 2'):
                    chunk.metadata['section'] = chunk.metadata['Header 2']
                elif hasattr(chunk.metadata, 'Header 3'):
                    chunk.metadata['section'] = chunk.metadata['Header 3']
                
                print(f"            Chunk {chunk_idx + 1}: {len(chunk.page_content)} chars")
                if hasattr(chunk.metadata, 'section'):
                    print(f"               📍 Section: {chunk.metadata['section']}")
            
            all_chunks.extend(chunks)
            print(f"         🎉 Document {doc_idx + 1} processed successfully!")
            
        except Exception as e:
            print(f"         ❌ Warning: Failed to split document with markdown splitter: {e}")
            print(f"         🔄 Falling back to original document")
            # Fallback: keep the original document if splitting fails
            all_chunks.append(doc)
    
    print(f"   📊 Splitting complete!")
    print(f"      📄 Input documents: {len(documents)}")
    print(f"      📦 Output chunks: {len(all_chunks)}")
    print(f"      📈 Expansion ratio: {len(all_chunks)/len(documents):.1f}x")
    print("")
    
    return all_chunks


async def crawl_entire_documentation(library: str, start_url: str):
    """Crawl the ENTIRE documentation site using crawl4ai deep crawling and progressively index."""
    log_message(
        f"🚀 Starting COMPREHENSIVE crawl for library '{library}' at {start_url}...",
        library,
    )
    log_message(f"📋 Configuration:", library)
    log_message(f"   - Target: ALL pages on the domain", library)
    log_message(f"   - Using crawl4ai deep crawling for maximum coverage", library)
    log_message(f"   - BFS strategy for systematic, level-by-level crawling", library)
    log_message(f"   - MAXIMUM coverage with no artificial limits", library)
    log_message(f"   - Progressive indexing enabled", library)
    log_message("", library)

    # Parse domain for filtering
    from urllib.parse import urlparse

    base_domain = urlparse(start_url).netloc
    log_message(f"🌐 Base domain: {base_domain}", library)

    # Create MINIMAL filter chain to maximize discovery
    filter_chain = FilterChain(
        [
            # Stay within the same domain but be more permissive
            DomainFilter(allowed_domains=[base_domain], blocked_domains=[]),
            # Accept ALL content types to maximize discovery
            ContentTypeFilter(allowed_types=["text/html", "text/plain", "application/xhtml+xml"]),
        ]
    )

    # Use BFS strategy with MAXIMUM coverage settings
    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=20,  # Maximum depth for comprehensive coverage
        include_external=False,  # Stay within domain
        filter_chain=filter_chain,
        max_pages=10000,  # Allow up to 10,000 pages
    )

    # Configure crawler for MAXIMUM documentation discovery
    config = CrawlerRunConfig(
        deep_crawl_strategy=deep_crawl_strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        css_selector="body",  # Extract content from the entire page body
        stream=True,  # Process results as they come
        verbose=True,  # Detailed logging
        cache_mode=CacheMode.BYPASS,  # Don't cache for fresh content
        word_count_threshold=50,  # Minimum 50 words to ensure meaningful content
        process_iframes=True,  # Include iframe content
        exclude_external_links=False,  # Allow internal link discovery
        remove_overlay_elements=False,  # Keep all elements for link discovery
        excluded_tags=[],  # Don't exclude any tags - keep all for link discovery
    )

    page_count = 0
    failed_count = 0
    processed_pages = 0
    total_chunks = 0

    log_message(f"🕷️ Starting MAXIMUM coverage crawl4ai deep crawl...", library)
    log_message(f"   📊 Max depth: {deep_crawl_strategy.max_depth}", library)
    log_message(f"   📊 Max pages: {deep_crawl_strategy.max_pages}", library)
    log_message(f"   📊 Word threshold: {config.word_count_threshold}", library)
    log_message("", library)

    try:
        async with AsyncWebCrawler() as crawler:
            log_message(f"🔍 Starting crawl with MAXIMUM coverage settings...", library)
            
            # Remove timeout to allow full crawl completion
            try:
                async for result in await crawler.arun(start_url, config=config):
                    page_count += 1
                    depth = result.metadata.get("depth", 0)

                    # Enhanced logging for each page
                    log_message(f"🔄 [{page_count}] CRAWLING: {result.url}", library)
                    log_message(f"   📊 Depth: {depth}", library)
                    log_message(f"   ✅ Success: {result.success}", library)
                    log_message(f"   📝 Has markdown: {bool(result.markdown)}", library)
                    log_message(f"   📏 Markdown length: {len(result.markdown) if result.markdown else 0}", library)
                    
                    # Show discovered links for debugging
                    try:
                        if hasattr(result, 'links') and result.links:
                            log_message(f"   🔗 Discovered {len(result.links)} links", library)
                            # Show first few links
                            for i, link in enumerate(result.links[:3]):
                                log_message(f"      {i+1}. {link}", library)
                            if len(result.links) > 3:
                                log_message(f"      ... and {len(result.links) - 3} more", library)
                    except Exception as e:
                        log_message(f"   🔗 Link discovery error: {e}", library)
                    
                    # Add more detailed information
                    if result.markdown:
                        # Show first 100 characters of content
                        preview = result.markdown[:100].replace('\n', ' ').strip()
                        log_message(f"   📄 Preview: {preview}...", library)
                        
                        # Pretty print the full markdown content
                        print(f"\n{'='*80}")
                        print(f"📄 FULL MARKDOWN CONTENT - Page {page_count}")
                        print(f"🌐 URL: {result.url}")
                        print(f"📝 Title: {result.metadata.get('title', 'No title')}")
                        print(f"📊 Depth: {depth}")
                        print(f"📏 Length: {len(result.markdown)} characters")
                        print(f"{'='*80}")
                        print(result.markdown)
                        print(f"{'='*80}")
                        print()
                        
                        # Analyze content structure
                        lines = result.markdown.split('\n')
                        headers = [line for line in lines if line.strip().startswith('#')]
                        log_message(f"   📊 Content analysis:", library)
                        log_message(f"      📏 Total lines: {len(lines)}", library)
                        log_message(f"      📏 Non-empty lines: {len([l for l in lines if l.strip()])}", library)
                        log_message(f"      🎯 Headers found: {len(headers)}", library)
                        if headers:
                            log_message(f"      📍 Header levels: {[h.count('#') for h in headers[:5]]}", library)
                            # Show the actual headers
                            log_message(f"      📋 Headers:", library)
                            for i, header in enumerate(headers[:10]):  # Show first 10 headers
                                log_message(f"         {i+1}. {header.strip()}", library)
                            if len(headers) > 10:
                                log_message(f"         ... and {len(headers) - 10} more headers", library)
                    
                    # Show any error details
                    if not result.success:
                        error_msg = getattr(result, 'error_message', 'Unknown error')
                        log_message(f"   ❌ Error: {error_msg}", library)

                    if result.success and result.markdown:
                        # Filter out gallery/index pages and ensure meaningful content
                        markdown_content = result.markdown.strip()
                        
                        # Skip pages that are mostly links/images (gallery pages)
                        link_count = markdown_content.count('[') + markdown_content.count('![')
                        image_count = markdown_content.count('![')
                        word_count = len(markdown_content.split())
                        
                        # Skip if too many links relative to content
                        if link_count > word_count * 0.3:  # More than 30% links
                            log_message(f"   ⏭️  Skipping gallery/index page (too many links: {link_count} links, {word_count} words)", library)
                            continue
                        
                        # Skip if too many images relative to content
                        if image_count > word_count * 0.1:  # More than 10% images
                            log_message(f"   ⏭️  Skipping image gallery page (too many images: {image_count} images, {word_count} words)", library)
                            continue
                        
                        # Skip if content is too short (likely navigation/index)
                        if word_count < 100:
                            log_message(f"   ⏭️  Skipping short content page (only {word_count} words)", library)
                            continue
                        
                        # Skip common index/gallery URLs
                        skip_patterns = [
                            '/index.html', '/gallery/', '/examples/', '/demo/',
                            'thumb.png', 'sphx_glr_', '_thumb.png'
                        ]
                        if any(pattern in result.url.lower() for pattern in skip_patterns):
                            log_message(f"   ⏭️  Skipping gallery/index page (URL pattern match)", library)
                            continue
                        
                        processed_pages += 1
                        title = result.metadata.get("title", "No title")
                        if not title or title == "No title":
                            title = result.url.split("/")[-1].replace("-", " ").title()

                        metadata = {
                            "source": result.url,
                            "title": title,
                            "depth": depth,
                            "library": library,
                        }
                        doc = Document(page_content=result.markdown, metadata=metadata)

                        log_message(f"   ✅ Content length: {len(result.markdown)} characters", library)
                        log_message(f"   📝 Title: {title}", library)
                        log_message(f"   🎯 Processing document for indexing...", library)

                        # --- Progressive Indexing ---
                        log_message(f"   🔄 Splitting document into chunks...", library)
                        chunks = split_documents([doc])
                        log_message(f"   📦 Created {len(chunks)} chunks.", library)
                        total_chunks += len(chunks)

                        # Pretty print the chunks
                        print(f"\n{'='*80}")
                        print(f"📦 CHUNKS CREATED - Page {page_count}")
                        print(f"🌐 URL: {result.url}")
                        print(f"📊 Total chunks: {len(chunks)}")
                        print(f"{'='*80}")
                        
                        for i, chunk in enumerate(chunks):
                            print(f"\n--- CHUNK {i+1}/{len(chunks)} ---")
                            print(f"📏 Length: {len(chunk.page_content)} characters")
                            print(f"📍 Section: {chunk.metadata.get('section', 'No section')}")
                            print(f"📝 Title: {chunk.metadata.get('title', 'No title')}")
                            print(f"🔗 Source: {chunk.metadata.get('source', 'Unknown')}")
                            print(f"📊 Depth: {chunk.metadata.get('depth', 0)}")
                            print(f"📚 Library: {chunk.metadata.get('library', 'Unknown')}")
                            print("-" * 40)
                            print(chunk.page_content)
                            print("-" * 40)
                        
                        print(f"{'='*80}")
                        print()

                        if chunks:
                            log_message(f"   📤 Adding {len(chunks)} chunks to Pinecone...", library)
                            # Use the optimized function with integrated embedding
                            await asyncio.to_thread(add_documents, library, chunks)
                            log_message(f"   ✅ Successfully added chunks to Pinecone.", library)
                            log_message(f"   🎉 Page {page_count} fully processed and indexed!", library)

                    else:
                        failed_count += 1
                        error_msg = (
                            result.error_message
                            if hasattr(result, "error_message")
                            else "Unknown error"
                        )
                        log_message(f"   ❌ Failed: {error_msg}", library)
                        log_message(f"   💔 Page {page_count} failed to process", library)

                    log_message("", library)
                    
                    # Show progress every 10 pages
                    if page_count % 10 == 0:
                        log_message(f"📊 PROGRESS UPDATE - Page {page_count}:", library)
                        log_message(f"   ✅ Processed: {processed_pages} pages", library)
                        log_message(f"   ❌ Failed: {failed_count} pages", library)
                        log_message(f"   📦 Total chunks: {total_chunks}", library)
                        log_message(f"   🎯 Success rate: {(processed_pages/page_count)*100:.1f}%", library)
                        log_message("", library)
                    
            except asyncio.TimeoutError:
                log_message(f"⏰ Crawl timed out - but this shouldn't happen with no timeout", library)
            except Exception as e:
                log_message(f"❌ Crawl iteration error: {str(e)}", library)

    except Exception as e:
        log_message(f"❌ Crawl error: {str(e)}", library)
        import traceback

        log_message(f"❌ Traceback: {traceback.format_exc()}", library)

    log_message(f"📊 COMPREHENSIVE Crawling Summary:", library)
    log_message(
        f"   ✅ Successfully processed and indexed: {processed_pages} pages", library
    )
    log_message(f"   🔗 Total URLs visited: {page_count}", library)
    log_message(f"   ❌ Failed to process: {failed_count} pages", library)
    log_message(f"   📦 Total chunks created and indexed: {total_chunks}", library)
    log_message("", library)

    return processed_pages, failed_count, total_chunks


def main():
    """Main function to handle command-line execution."""
    print("🚀 =========================================")
    print("🚀 DOCUMENTATION CRAWLER & INDEXER")
    print("🚀 =========================================")
    print("")
    
    parser = argparse.ArgumentParser(
        description="Crawl and index documentation websites into Pinecone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python populate_db.py --library_name pandas --url https://pandas.pydata.org/docs/
  python populate_db.py --library_name numpy --url https://numpy.org/doc/
  python populate_db.py --library_name matplotlib --url https://matplotlib.org/stable/
        """
    )
    
    parser.add_argument(
        "--library_name",
        required=True,
        help="Name of the library to crawl (e.g., pandas, numpy, matplotlib)"
    )
    
    parser.add_argument(
        "--url",
        required=True,
        help="Starting URL for the documentation crawl"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for Pinecone upserts (default: 100)"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Size of text chunks (default: 1000)"
    )
    
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )
    
    args = parser.parse_args()
    
    print("📋 PARSING COMMAND LINE ARGUMENTS...")
    print(f"   📚 Library name: {args.library_name}")
    print(f"   🌐 Target URL: {args.url}")
    print(f"   📦 Batch size: {args.batch_size}")
    print(f"   📄 Chunk size: {args.chunk_size}")
    print(f"   🔄 Chunk overlap: {args.chunk_overlap}")
    print("")
    
    # Validate inputs
    print("🔍 VALIDATING INPUTS...")
    if not args.library_name.strip():
        print("❌ Error: library_name cannot be empty")
        sys.exit(1)
    
    if not args.url.strip():
        print("❌ Error: url cannot be empty")
        sys.exit(1)
    
    print("✅ Input validation passed!")
    print("")
    
    # Convert library name to lowercase for consistency
    library_name = args.library_name.lower().strip()
    start_url = args.url.strip()
    
    print("🔧 INITIALIZING CRAWLER...")
    print(f"   📚 Normalized library name: '{library_name}'")
    print(f"   🌐 Normalized URL: '{start_url}'")
    print(f"   🗄️  Target Pinecone index: {os.getenv('PINECONE_INDEX_NAME', 'docs-mcp')}")
    print(f"   🧠 Embedding model: {os.getenv('OPENAI_MODEL', 'text-embedding-3-small')}")
    print("")
    
    print("🚀 STARTING DOCUMENTATION CRAWL")
    print("=" * 60)
    
    try:
        # Run the crawl
        print("🔄 Launching async crawl process...")
        processed_pages, failed_count, total_chunks = asyncio.run(
            crawl_entire_documentation(library_name, start_url)
        )
        
        # Final summary
        print("=" * 60)
        print("🎉 CRAWL COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📚 Library: {library_name}")
        print(f"🌐 URL: {start_url}")
        print(f"✅ Successfully processed: {processed_pages} pages")
        print(f"❌ Failed to process: {failed_count} pages")
        print(f"📦 Total chunks indexed: {total_chunks}")
        print(f"🎯 Success rate: {(processed_pages/(processed_pages+failed_count))*100:.1f}%" if (processed_pages+failed_count) > 0 else "🎯 Success rate: 0%")
        print(f"📊 Total pages visited: {processed_pages + failed_count}")
        print("")
        
        if processed_pages > 0:
            print("🎉 SUCCESS! Your documentation has been indexed!")
            print(f"📖 You can now search for '{library_name}' documentation through your MCP server!")
            print(f"🔍 Try searching for topics related to '{library_name}' in your chat interface.")
        else:
            print("⚠️  WARNING: No pages were successfully processed.")
            print("🔍 Check the URL and ensure the documentation site is accessible.")
            print("💡 Common issues:")
            print("   - Invalid URL format")
            print("   - Site requires authentication")
            print("   - Site blocks automated crawling")
            print("   - Network connectivity issues")
        
        print("")
        print("🏁 Crawl process finished!")
            
    except KeyboardInterrupt:
        print("\n⏹️  CRAWL INTERRUPTED BY USER")
        print("🔄 Gracefully shutting down...")
        sys.exit(1)
    except Exception as e:
        print("\n❌ CRAWL FAILED WITH ERROR")
        print(f"💥 Error: {e}")
        print("📋 Full traceback:")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
