import gzip
import json
import os
import ast
import random
from collections import defaultdict, Counter
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DATA_DIR = os.path.join(PROJECT_ROOT, "sourceData")


def load_json_gz(filename, format='auto'):
    """
    Load a .json.gz file from the sourceData directory.
    
    Args:
        filename: Name of the file (e.g., "beeradvocate.json.gz")
        format: 'auto' (detect), 'json' (standard JSON), 'jsonl' (JSON Lines), 
                or 'python' (Python dict syntax, one per line)
    
    Returns:
        The parsed data (list of dicts for JSONL/Python format, or dict/list for JSON)
    """
    file_path = os.path.join(SOURCE_DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            # Try to detect format by reading first line
            if format == 'auto':
                first_line = f.readline()
                f.seek(0)  # Reset to beginning
                
                # Check if it's Python dict syntax (single quotes)
                if first_line.strip().startswith("{'") or first_line.strip().startswith("{u'"):
                    format = 'python'
                # Check if it's JSONL (valid JSON on first line)
                elif first_line.strip().startswith('{"'):
                    try:
                        json.loads(first_line.strip())
                        format = 'jsonl'
                    except:
                        format = 'json'  # Might be single JSON object
                else:
                    format = 'json'  # Try as single JSON object
            
            # Load based on detected/selected format
            if format == 'python':
                # Python dict syntax, one per line (JSONL-like)
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            record = ast.literal_eval(line)
                            data.append(record)
                        except (ValueError, SyntaxError) as e:
                            print(f"Warning: Could not parse line {line_num}: {e}")
                            continue
                print(f"File '{filename}' successfully read and parsed ({len(data)} records).")
                return data
            
            elif format == 'jsonl':
                # Standard JSON Lines format
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            record = json.loads(line)
                            data.append(record)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse line {line_num}: {e}")
                            continue
                print(f"File '{filename}' successfully read and parsed ({len(data)} records).")
                return data
            
            else:  # format == 'json'
                # Standard JSON format (single object)
                data = json.load(f)
                print(f"File '{filename}' successfully read and parsed.")
                return data
                
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        raise
    except UnicodeDecodeError as e:
        print(f"Error decoding text: {e}. Try a different encoding if 'utf-8' is not correct.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def explore_dataset(data, dataset_name="Dataset", num_samples=5):
    """
    Explore and display summary statistics about a loaded dataset.
    Similar to pandas .head(), .info(), and .describe() but for raw dict data.
    
    Args:
        data: List of dictionaries (records)
        dataset_name: Name of the dataset for display
        num_samples: Number of sample records to display
    """
    print("\n" + "="*80)
    print(f"EXPLORING: {dataset_name}")
    print("="*80)
    
    if not isinstance(data, list) or len(data) == 0:
        print("Dataset is empty or not a list of records.")
        return
    
    print(f"\n📊 BASIC STATISTICS")
    print(f"   Total records: {len(data):,}")
    
    # Get all unique keys across records
    all_keys = set()
    for record in data[:1000]:  # Sample first 1000 to get key structure
        if isinstance(record, dict):
            all_keys.update(record.keys())
    
    print(f"   Number of fields: {len(all_keys)}")
    print(f"   Field names: {sorted(all_keys)}")
    
    # Analyze each field
    print(f"\n📋 FIELD ANALYSIS")
    for key in sorted(all_keys):
        values = []
        missing_count = 0
        for record in data[:10000]:  # Sample first 10k records for analysis
            if isinstance(record, dict):
                if key in record:
                    values.append(record[key])
                else:
                    missing_count += 1
        
        if values:
            # Try to determine data type
            sample_val = values[0]
            val_type = type(sample_val).__name__
            
            # Count unique values (for categorical)
            unique_vals = len(set(str(v) for v in values))
            
            # For numeric fields, show range
            if isinstance(sample_val, (int, float)) or (isinstance(sample_val, str) and sample_val.replace('.', '').replace('-', '').isdigit()):
                try:
                    numeric_vals = [float(v) if isinstance(v, str) else v for v in values if v]
                    if numeric_vals:
                        print(f"   {key:30s} | Type: {val_type:10s} | Unique: {unique_vals:6,} | "
                              f"Range: [{min(numeric_vals):.2f}, {max(numeric_vals):.2f}] | "
                              f"Missing: {missing_count}")
                    else:
                        print(f"   {key:30s} | Type: {val_type:10s} | Unique: {unique_vals:6,} | Missing: {missing_count}")
                except:
                    print(f"   {key:30s} | Type: {val_type:10s} | Unique: {unique_vals:6,} | Missing: {missing_count}")
            else:
                # For text/categorical fields
                avg_len = sum(len(str(v)) for v in values if v) / len(values) if values else 0
                print(f"   {key:30s} | Type: {val_type:10s} | Unique: {unique_vals:6,} | "
                      f"Avg length: {avg_len:.1f} | Missing: {missing_count}")
    
    # Show sample records (like head())
    print(f"\n📄 SAMPLE RECORDS (first {num_samples}):")
    print("-"*80)
    for i, record in enumerate(data[:num_samples], 1):
        print(f"\nRecord {i}:")
        if isinstance(record, dict):
            for key, value in sorted(record.items()):
                # Truncate long text values
                val_str = str(value)
                if len(val_str) > 100:
                    val_str = val_str[:97] + "..."
                print(f"   {key:30s}: {val_str}")
        else:
            print(f"   {record}")
    
    print("\n" + "="*80)


def load_sample(filename, sample_size=50000, random_seed=42):
    """
    Load a random sample from a dataset file.
    
    Args:
        filename: Name of the .json.gz file
        sample_size: Number of records to sample
        random_seed: Random seed for reproducibility
    
    Returns:
        List of sampled records
    """
    print(f"\n📥 Loading sample from {filename}...")
    data = load_json_gz(filename)
    
    if len(data) <= sample_size:
        print(f"   Dataset has {len(data):,} records, using all.")
        return data
    
    random.seed(random_seed)
    sampled = random.sample(data, sample_size)
    print(f"   Sampled {sample_size:,} records from {len(data):,} total records.")
    return sampled


def extract_beeradvocate_triples(data):
    """
    Extract user-item-rating triples from BeerAdvocate dataset.
    
    Returns:
        List of dicts with keys: user_id, item_id, rating
    """
    triples = []
    for record in data:
        if not isinstance(record, dict):
            continue
        
        user_id = record.get('review/profileName', '').strip()
        item_id = record.get('beer/beerId', '').strip()
        rating_str = record.get('review/overall', '').strip()
        
        if user_id and item_id and rating_str:
            try:
                rating = float(rating_str)
                if 0 <= rating <= 5:  # Valid rating range
                    triples.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'rating': rating
                    })
            except (ValueError, TypeError):
                continue
    
    return triples


def extract_goodreads_triples(data):
    """
    Extract user-item-rating triples from Goodreads dataset.
    
    Returns:
        List of dicts with keys: user_id, item_id, rating
    """
    triples = []
    for record in data:
        if not isinstance(record, dict):
            continue
        
        user_id = str(record.get('user_id', '')).strip()
        item_id = str(record.get('book_id', '')).strip()
        rating = record.get('rating', None)
        
        # Rating might be missing or 0, check if it exists
        if user_id and item_id and rating is not None:
            try:
                rating = float(rating)
                if 1 <= rating <= 5:  # Valid rating range
                    triples.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'rating': rating
                    })
            except (ValueError, TypeError):
                continue
    
    return triples


def extract_steam_triples(data):
    """
    Extract user-item-rating triples from Steam reviews dataset.
    
    Returns:
        List of dicts with keys: user_id, item_id, rating
    """
    triples = []
    for record in data:
        if not isinstance(record, dict):
            continue
        
        user_id = str(record.get('user_id', '')).strip()
        reviews = record.get('reviews', [])
        
        if not user_id or not reviews:
            continue
        
        for review in reviews:
            if not isinstance(review, dict):
                continue
            
            item_id = str(review.get('item_id', '')).strip()
            recommend = review.get('recommend', None)
            
            if item_id and recommend is not None:
                # Convert boolean to rating: True = 5, False = 1
                rating = 5.0 if recommend else 1.0
                triples.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': rating
                })
    
    return triples


def clean_triples(triples, min_user_reviews=3, min_item_reviews=3):
    """
    Clean user-item-rating triples:
    1. Remove duplicates
    2. Filter users/items with minimum reviews
    3. Reindex user and item IDs to consecutive integers
    
    Args:
        triples: List of dicts with user_id, item_id, rating
        min_user_reviews: Minimum number of reviews per user
        min_item_reviews: Minimum number of reviews per item
    
    Returns:
        Cleaned triples with reindexed IDs, and mapping dictionaries
    """
    print(f"\n🧹 Cleaning triples...")
    print(f"   Initial records: {len(triples):,}")
    
    # Remove duplicates (same user-item pair)
    seen = set()
    unique_triples = []
    for triple in triples:
        key = (triple['user_id'], triple['item_id'])
        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)
    
    print(f"   After removing duplicates: {len(unique_triples):,}")
    
    # Count reviews per user and item
    user_counts = Counter(t['user_id'] for t in unique_triples)
    item_counts = Counter(t['item_id'] for t in unique_triples)
    
    # Filter by minimum reviews
    valid_users = {uid for uid, count in user_counts.items() if count >= min_user_reviews}
    valid_items = {iid for iid, count in item_counts.items() if count >= min_item_reviews}
    
    filtered_triples = [
        t for t in unique_triples
        if t['user_id'] in valid_users and t['item_id'] in valid_items
    ]
    
    print(f"   After filtering (min {min_user_reviews} user reviews, min {min_item_reviews} item reviews): {len(filtered_triples):,}")
    
    # Reindex user and item IDs to consecutive integers
    unique_users = sorted(set(t['user_id'] for t in filtered_triples))
    unique_items = sorted(set(t['item_id'] for t in filtered_triples))
    
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    
    cleaned_triples = []
    for triple in filtered_triples:
        cleaned_triples.append({
            'user_id': user_to_idx[triple['user_id']],
            'item_id': item_to_idx[triple['item_id']],
            'rating': triple['rating']
        })
    
    print(f"   Final cleaned records: {len(cleaned_triples):,}")
    print(f"   Unique users: {len(unique_users):,}")
    print(f"   Unique items: {len(unique_items):,}")
    
    return cleaned_triples, user_to_idx, item_to_idx


def compute_dataset_metrics(triples, dataset_name):
    """
    Compute comparable metrics for a cleaned dataset.
    
    Args:
        triples: List of cleaned triples with reindexed IDs
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary of metrics
    """
    if not triples:
        return {}
    
    ratings = [t['rating'] for t in triples]
    user_counts = Counter(t['user_id'] for t in triples)
    item_counts = Counter(t['item_id'] for t in triples)
    
    metrics = {
        'dataset': dataset_name,
        'total_interactions': len(triples),
        'num_users': len(user_counts),
        'num_items': len(item_counts),
        'avg_rating': sum(ratings) / len(ratings),
        'min_rating': min(ratings),
        'max_rating': max(ratings),
        'avg_reviews_per_user': len(triples) / len(user_counts) if user_counts else 0,
        'avg_reviews_per_item': len(triples) / len(item_counts) if item_counts else 0,
        'min_reviews_per_user': min(user_counts.values()) if user_counts else 0,
        'max_reviews_per_user': max(user_counts.values()) if user_counts else 0,
        'min_reviews_per_item': min(item_counts.values()) if item_counts else 0,
        'max_reviews_per_item': max(item_counts.values()) if item_counts else 0,
        'sparsity': 1 - (len(triples) / (len(user_counts) * len(item_counts))) if user_counts and item_counts else 1.0,
    }
    
    # Rating distribution
    rating_dist = Counter(ratings)
    metrics['rating_distribution'] = dict(sorted(rating_dist.items()))
    
    return metrics


def print_comparison_table(metrics_list):
    """
    Print a comparison table of metrics across datasets.
    
    Args:
        metrics_list: List of metric dictionaries
    """
    print("\n" + "="*100)
    print("DATASET COMPARISON METRICS")
    print("="*100)
    
    # Define metrics to display
    metric_keys = [
        ('total_interactions', 'Total Interactions'),
        ('num_users', 'Num Users'),
        ('num_items', 'Num Items'),
        ('avg_rating', 'Avg Rating'),
        ('min_rating', 'Min Rating'),
        ('max_rating', 'Max Rating'),
        ('avg_reviews_per_user', 'Avg Reviews/User'),
        ('avg_reviews_per_item', 'Avg Reviews/Item'),
        ('sparsity', 'Sparsity'),
    ]
    
    # Print header
    print(f"\n{'Metric':<25}", end="")
    for m in metrics_list:
        print(f"{m['dataset']:>20}", end="")
    print()
    print("-" * (25 + 20 * len(metrics_list)))
    
    # Print each metric
    for key, label in metric_keys:
        print(f"{label:<25}", end="")
        for m in metrics_list:
            value = m.get(key, 0)
            if isinstance(value, float):
                print(f"{value:>20.4f}", end="")
            else:
                print(f"{value:>20,}", end="")
        print()
    
    # Print rating distributions (separate rows for clarity)
    print(f"\n{'Rating Distribution':<25}")
    for m in metrics_list:
        dist = m.get('rating_distribution', {})
        dist_str = ", ".join([f"{k}:{v:,}" for k, v in sorted(dist.items())])
        print(f"{'':<25}{m['dataset']:>20}")
        # Wrap long distributions
        if len(dist_str) > 75:
            words = dist_str.split(", ")
            line = ""
            for word in words:
                if len(line + word) > 75:
                    print(f"{'':<25}{line:>20}")
                    line = word + ", "
                else:
                    line += word + ", " if line else word + ", "
            if line:
                print(f"{'':<25}{line.rstrip(', '):>20}")
        else:
            print(f"{'':<25}{dist_str:>20}")
        print()
    
    print("="*100)


def save_cleaned_dataset(triples, dataset_name, output_dir="sampleSets"):
    """
    Save cleaned dataset to a JSON file.
    
    Args:
        triples: List of cleaned triples
        dataset_name: Name of the dataset
        output_dir: Directory to save the file
    """
    # Create output directory if it doesn't exist
    output_path = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Save as JSON
    filename = os.path.join(output_path, f"{dataset_name}_cleaned.json")
    with open(filename, 'w') as f:
        json.dump(triples, f, indent=2)
    
    print(f"   💾 Saved {len(triples):,} cleaned triples to {filename}")
    return filename


def process_all_datasets(sample_size=50000, min_user_reviews=3, min_item_reviews=3, random_seed=42, save_output=True):
    """
    Process all three datasets with consistent parameters.
    
    Args:
        sample_size: Number of records to sample from each dataset
        min_user_reviews: Minimum reviews per user
        min_item_reviews: Minimum reviews per item
        random_seed: Random seed for reproducibility
        save_output: Whether to save cleaned datasets to sampleSets folder
    
    Returns:
        Dictionary mapping dataset names to cleaned triples and metrics
    """
    results = {}
    
    # Process BeerAdvocate
    print("\n" + "="*100)
    print("PROCESSING BEERADVOCATE DATASET")
    print("="*100)
    try:
        beer_sample = load_sample("beeradvocate.json.gz", sample_size, random_seed)
        beer_triples = extract_beeradvocate_triples(beer_sample)
        beer_cleaned, _, _ = clean_triples(beer_triples, min_user_reviews, min_item_reviews)
        beer_metrics = compute_dataset_metrics(beer_cleaned, "BeerAdvocate")
        if save_output:
            save_cleaned_dataset(beer_cleaned, "beeradvocate")
        results['beeradvocate'] = {
            'triples': beer_cleaned,
            'metrics': beer_metrics
        }
    except Exception as e:
        print(f"Error processing BeerAdvocate: {e}")
        import traceback
        traceback.print_exc()
    
    # Process Goodreads
    print("\n" + "="*100)
    print("PROCESSING GOODREADS DATASET")
    print("="*100)
    try:
        goodreads_sample = load_sample("goodreads_reviews_spoiler.json.gz", sample_size, random_seed)
        goodreads_triples = extract_goodreads_triples(goodreads_sample)
        goodreads_cleaned, _, _ = clean_triples(goodreads_triples, min_user_reviews, min_item_reviews)
        goodreads_metrics = compute_dataset_metrics(goodreads_cleaned, "Goodreads")
        if save_output:
            save_cleaned_dataset(goodreads_cleaned, "goodreads")
        results['goodreads'] = {
            'triples': goodreads_cleaned,
            'metrics': goodreads_metrics
        }
    except Exception as e:
        print(f"Error processing Goodreads: {e}")
        import traceback
        traceback.print_exc()
    
    # Process Steam
    print("\n" + "="*100)
    print("PROCESSING STEAM DATASET")
    print("="*100)
    try:
        steam_sample = load_sample("australian_user_reviews.json.gz", sample_size, random_seed)
        steam_triples = extract_steam_triples(steam_sample)
        steam_cleaned, _, _ = clean_triples(steam_triples, min_user_reviews, min_item_reviews)
        steam_metrics = compute_dataset_metrics(steam_cleaned, "Steam")
        if save_output:
            save_cleaned_dataset(steam_cleaned, "steam")
        results['steam'] = {
            'triples': steam_cleaned,
            'metrics': steam_metrics
        }
    except Exception as e:
        print(f"Error processing Steam: {e}")
        import traceback
        traceback.print_exc()
    
    # Print comparison table
    metrics_list = [results[k]['metrics'] for k in results.keys() if 'metrics' in results[k]]
    if metrics_list:
        print_comparison_table(metrics_list)
    
    return results


# Example usage:
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        # Run full data cleaning pipeline
        print("="*100)
        print("DATA CLEANING PIPELINE")
        print("="*100)
        print("\nParameters:")
        print("  Sample size per dataset: 50,000")
        print("  Min reviews per user: 3")
        print("  Min reviews per item: 3")
        print("  Random seed: 42")
        
        results = process_all_datasets(
            sample_size=50000,
            min_user_reviews=3,
            min_item_reviews=3,
            random_seed=42,
            save_output=True
        )
        
        print("\n✅ Data cleaning complete!")
        print(f"\nProcessed {len(results)} datasets:")
        for name, data in results.items():
            if 'metrics' in data:
                print(f"  - {name}: {data['metrics']['total_interactions']:,} interactions, "
                      f"{data['metrics']['num_users']:,} users, {data['metrics']['num_items']:,} items")
    else:
        # Default: Just explore one dataset
        try:
            print("Loading beeradvocate dataset...")
            beer_data = load_json_gz("beeradvocate.json.gz")
            
            # Explore the dataset
            explore_dataset(beer_data, dataset_name="BeerAdvocate Reviews", num_samples=3)
                
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()