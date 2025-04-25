import json
import os
import sys
import time
import psycopg
from psycopg.rows import dict_row
from google.cloud import aiplatform
from google.api_core import exceptions
from vertexai.language_models import TextEmbeddingModel

# --- Configuration ---
# AlloyDB Connection Details (Replace with your actual details or use environment variables)
DB_HOST = os.environ.get("DB_HOST", "10.28.128.2") # Or Cluster URI
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "ch4ng3m3")

# Vertex AI Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "maplequad-11859276-cmf-dev")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west1")
MODEL_ID = "text-embedding-005" # Or specify the publisher endpoint path if needed

# Input File
JSONL_FILE = "chunks-round03.json"

# Batching (Adjust as needed for performance/API limits)
EMBEDDING_BATCH_SIZE = 100  # Vertex AI API batch limit can be higher, but smaller batches provide more frequent feedback
DB_BATCH_SIZE = 100

# --- Vertex AI Initialization ---
aiplatform.init(project=PROJECT_ID, location=LOCATION)
#embedding_model = TextEmbeddingModel.from_pretrained(MODEL_ID)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
# --- Embedding Function ---
def get_embeddings_batch(texts):
    try:
        response = embedding_model.get_embeddings(texts)
        embeddings = [r.values for r in response]
        return embeddings
    except Exception as e:
        print(f"An error occurred while generating embeddings: {e}")
        return None


# --- Main Ingestion Logic ---
def main():
    db_conn_str = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
    chunks_batch = []
    db_batch_data = []
    total_processed = 0
    total_ingested = 0
    error_count = 0

    try:
        with psycopg.connect(db_conn_str, row_factory=dict_row) as conn, conn.cursor() as cur:
            print("Successfully connected to AlloyDB.")
            print(f"Reading data from {JSONL_FILE}...")

            with open(JSONL_FILE, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        chunk_data = json.loads(line.strip())
                        chunks_batch.append(chunk_data)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON on line {line_num + 1}", file=sys.stderr)
                        error_count += 1
                        continue

                    # --- Process batch for embeddings ---
                    if len(chunks_batch) >= EMBEDDING_BATCH_SIZE:
                        texts_to_embed = [chunk['contents'] for chunk in chunks_batch]
                        print(f"Generating embeddings for batch of {len(texts_to_embed)}...")
                        start_time = time.time()
                        embeddings = get_embeddings_batch(texts_to_embed)
                        end_time = time.time()
                        print(f"Embedding generation took {end_time - start_time:.2f} seconds.")

                        if embeddings and len(embeddings) == len(chunks_batch):
                            for i, chunk in enumerate(chunks_batch):
                                meta = chunk.get('metadata', {})
                                # Prepare data tuple for insertion
                                # Ensure vector is correctly formatted for psycopg/pgvector ('[1.2, 3.4,...]')
                                embedding_str = '[' + ','.join(map(str, embeddings[i])) + ']'
                                db_batch_data.append((
                                    chunk.get('contents'),
                                    embedding_str,
                                    meta.get('page_number'),
                                    meta.get('document_title'),
                                    meta.get('report_year'),
                                    meta.get('organization'),
                                    meta.get('fund_name'),
                                    meta.get('section_title'),
                                    meta.get('subsection_title'),
                                    meta.get('content_type'),
                                    meta.get('keywords') # Pass list directly for TEXT[]
                                ))
                            total_processed += len(chunks_batch)
                        else:
                            print(f"Error generating embeddings for batch starting line {line_num + 1 - len(chunks_batch)}. Skipping batch.", file=sys.stderr)
                            error_count += len(chunks_batch)

                        chunks_batch = [] # Clear the batch

                    # --- Process batch for DB ingestion ---
                    if len(db_batch_data) >= DB_BATCH_SIZE:
                        print(f"Ingesting batch of {len(db_batch_data)} records into AlloyDB...")
                        start_time = time.time()
                        try:
                            # Use execute_values for efficient batch insertion
                            insert_query = """
                                INSERT INTO document_embeddings (
                                    content, embedding, page_number, document_title, report_year,
                                    organization, fund_name, section_title, subsection_title,
                                    content_type, keywords
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
                            cur.executemany(insert_query, db_batch_data)
                            conn.commit() # Commit after each DB batch
                            total_ingested += len(db_batch_data)
                            db_batch_data = [] # Clear the batch
                            end_time = time.time()
                            print(f"DB ingestion took {end_time - start_time:.2f} seconds.")
                        except (Exception, psycopg.Error) as db_error:
                            print(f"Database Error during batch insert: {db_error}", file=sys.stderr)
                            conn.rollback() # Rollback failed batch
                            error_count += len(db_batch_data)
                            db_batch_data = [] # Clear the batch to avoid retrying failed data

                # --- Process any remaining items in batches ---
                if chunks_batch: # Remaining embeddings
                    texts_to_embed = [chunk['contents'] for chunk in chunks_batch]
                    print(f"Generating embeddings for final batch of {len(texts_to_embed)}...")
                    embeddings = get_embeddings_batch(texts_to_embed)
                    if embeddings and len(embeddings) == len(chunks_batch):
                        for i, chunk in enumerate(chunks_batch):
                            meta = chunk.get('metadata', {})
                            embedding_str = '[' + ','.join(map(str, embeddings[i])) + ']'
                            db_batch_data.append((
                                chunk.get('contents'), embedding_str, meta.get('page_number'),
                                meta.get('document_title'), meta.get('report_year'), meta.get('organization'),
                                meta.get('fund_name'), meta.get('section_title'), meta.get('subsection_title'),
                                meta.get('content_type'), meta.get('keywords')
                            ))
                        total_processed += len(chunks_batch)
                    else:
                         print(f"Error generating embeddings for final batch. Skipping.", file=sys.stderr)
                         error_count += len(chunks_batch)

                if db_batch_data: # Remaining DB inserts
                    print(f"Ingesting final batch of {len(db_batch_data)} records into AlloyDB...")
                    try:
                        insert_query = """
                            INSERT INTO document_embeddings (
                                content, embedding, page_number, document_title, report_year,
                                organization, fund_name, section_title, subsection_title,
                                content_type, keywords
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        cur.executemany(insert_query, db_batch_data)
                        conn.commit()
                        total_ingested += len(db_batch_data)
                    except (Exception, psycopg.Error) as db_error:
                         print(f"Database Error during final batch insert: {db_error}", file=sys.stderr)
                         conn.rollback()
                         error_count += len(db_batch_data)


    except FileNotFoundError:
        print(f"Error: Input file '{JSONL_FILE}' not found.", file=sys.stderr)
        sys.exit(1)
    except psycopg.Error as e:
        print(f"Database connection error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Ingestion Summary ---")
    print(f"Total lines processed from file: {line_num + 1}")
    print(f"Total chunks successfully embedded: {total_processed}")
    print(f"Total records successfully ingested into AlloyDB: {total_ingested}")
    print(f"Total errors/skipped records: {error_count}")
    print("Ingestion complete.")

if __name__ == "__main__":
    # --- Check configuration ---
    if "your-gcp-project-id" in PROJECT_ID or "your_alloydb" in DB_HOST:
        print("ERROR: Please update placeholders for DB connection details and GCP Project ID in the script.", file=sys.stderr)
        sys.exit(1)
    if not DB_PASSWORD:
         print("ERROR: Database password not set. Please set the DB_PASSWORD environment variable.", file=sys.stderr)
         sys.exit(1)

    main()