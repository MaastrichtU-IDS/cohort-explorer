"""
Data analysis endpoints for downloading and analyzing cohort data.
"""
import json
import logging
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Set

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from src.auth import get_current_user
from src.cohort_cache import get_cohorts_from_cache, is_cache_initialized, cohort_to_dict
from src.utils import retrieve_cohorts_metadata

router = APIRouter()


def compare_cache_objects(cache_a: Dict, cache_b: Dict) -> Dict[str, Any]:
    """
    Compare two cache objects and generate a detailed comparison report.
    
    Args:
        cache_a: First cache object (dict with 'cohorts' key)
        cache_b: Second cache object (dict with 'cohorts' key)
    
    Returns:
        Detailed comparison report
    """
    cohorts_a = cache_a.get("cohorts", {})
    cohorts_b = cache_b.get("cohorts", {})
    
    # 1. Compare cohorts between the two caches
    cohort_ids_a = set(cohorts_a.keys())
    cohort_ids_b = set(cohorts_b.keys())
    
    cohorts_only_in_a = cohort_ids_a - cohort_ids_b
    cohorts_only_in_b = cohort_ids_b - cohort_ids_a
    cohorts_in_both = cohort_ids_a & cohort_ids_b
    
    # Initialize report structure
    report = {
        "summary": {
            "total_cohorts_a": len(cohort_ids_a),
            "total_cohorts_b": len(cohort_ids_b),
            "cohorts_only_in_a": len(cohorts_only_in_a),
            "cohorts_only_in_b": len(cohorts_only_in_b),
            "cohorts_in_both": len(cohorts_in_both),
            "cohorts_with_differences": 0
        },
        "cohorts_only_in_a": sorted(list(cohorts_only_in_a)),
        "cohorts_only_in_b": sorted(list(cohorts_only_in_b)),
        "cohort_comparisons": {}
    }
    
    # 2-4. For cohorts in both, compare keys, values, and variables
    cohorts_with_diffs = 0
    
    for cohort_id in sorted(cohorts_in_both):
        cohort_a = cohorts_a[cohort_id]
        cohort_b = cohorts_b[cohort_id]
        
        # Get all keys (excluding 'variables' for now)
        keys_a = set(k for k in cohort_a.keys() if k != "variables")
        keys_b = set(k for k in cohort_b.keys() if k != "variables")
        
        # 2. Compare keys
        keys_only_in_a = keys_a - keys_b
        keys_only_in_b = keys_b - keys_a
        keys_in_both = keys_a & keys_b
        
        # 3. Compare values for matching keys
        value_differences = {}
        for key in sorted(keys_in_both):
            val_a = cohort_a.get(key)
            val_b = cohort_b.get(key)
            
            if val_a != val_b:
                value_differences[key] = {
                    "value_in_a": val_a,
                    "value_in_b": val_b
                }
        
        # 4. Compare variables
        vars_a = cohort_a.get("variables", {})
        vars_b = cohort_b.get("variables", {})
        
        var_ids_a = set(vars_a.keys())
        var_ids_b = set(vars_b.keys())
        
        vars_only_in_a = var_ids_a - var_ids_b
        vars_only_in_b = var_ids_b - var_ids_a
        vars_in_both = var_ids_a & var_ids_b
        
        # Compare variable details for matching variables
        variable_differences = {}
        for var_id in sorted(vars_in_both):
            var_a = vars_a[var_id]
            var_b = vars_b[var_id]
            
            # Compare variable attributes
            var_keys_a = set(k for k in var_a.keys() if k != "categories")
            var_keys_b = set(k for k in var_b.keys() if k != "categories")
            
            var_attr_diffs = {}
            for var_key in (var_keys_a | var_keys_b):
                val_a = var_a.get(var_key)
                val_b = var_b.get(var_key)
                
                if val_a != val_b:
                    var_attr_diffs[var_key] = {
                        "value_in_a": val_a,
                        "value_in_b": val_b
                    }
            
            # Compare categories if present
            cats_a = var_a.get("categories", {})
            cats_b = var_b.get("categories", {})
            
            cat_ids_a = set(cats_a.keys())
            cat_ids_b = set(cats_b.keys())
            
            category_diff = None
            if cat_ids_a != cat_ids_b or any(cats_a.get(c) != cats_b.get(c) for c in (cat_ids_a & cat_ids_b)):
                category_diff = {
                    "categories_only_in_a": sorted(list(cat_ids_a - cat_ids_b)),
                    "categories_only_in_b": sorted(list(cat_ids_b - cat_ids_a)),
                    "category_value_differences": {
                        cat_id: {
                            "value_in_a": cats_a[cat_id],
                            "value_in_b": cats_b[cat_id]
                        }
                        for cat_id in sorted(cat_ids_a & cat_ids_b)
                        if cats_a[cat_id] != cats_b[cat_id]
                    }
                }
            
            if var_attr_diffs or category_diff:
                variable_differences[var_id] = {}
                if var_attr_diffs:
                    variable_differences[var_id]["attribute_differences"] = var_attr_diffs
                if category_diff:
                    variable_differences[var_id]["category_differences"] = category_diff
        
        # Build cohort comparison report
        has_differences = (
            len(keys_only_in_a) > 0 or
            len(keys_only_in_b) > 0 or
            len(value_differences) > 0 or
            len(vars_only_in_a) > 0 or
            len(vars_only_in_b) > 0 or
            len(variable_differences) > 0
        )
        
        if has_differences:
            cohorts_with_diffs += 1
            report["cohort_comparisons"][cohort_id] = {
                "metadata_keys_only_in_a": sorted(list(keys_only_in_a)),
                "metadata_keys_only_in_b": sorted(list(keys_only_in_b)),
                "metadata_value_differences": value_differences,
                "variables_summary": {
                    "total_variables_a": len(var_ids_a),
                    "total_variables_b": len(var_ids_b),
                    "variables_only_in_a": len(vars_only_in_a),
                    "variables_only_in_b": len(vars_only_in_b),
                    "variables_with_differences": len(variable_differences)
                },
                "variables_only_in_a": sorted(list(vars_only_in_a)),
                "variables_only_in_b": sorted(list(vars_only_in_b)),
                "variable_differences": variable_differences
            }
    
    report["summary"]["cohorts_with_differences"] = cohorts_with_diffs
    
    return report


@router.get("/download-cohorts-cache")
async def download_cohorts_cache(user: Any = Depends(get_current_user)) -> FileResponse:
    """
    Download the complete cohorts metadata cache as JSON file.
    Returns all cohorts with their metadata and variables in the same structure as the explore page.
    Filename includes timestamp of download.
    """
    user_email = user["email"]
    
    # Try to get cohorts from the cache
    if is_cache_initialized():
        logging.info("Retrieving cohorts from cache for download")
        cohorts = get_cohorts_from_cache(user_email)
        if cohorts:
            # Convert Cohort objects to dictionaries for JSON serialization
            cohorts_dict = {cid: cohort_to_dict(c) for cid, c in cohorts.items()}
            cache_status = "from_cache"
        else:
            logging.warning("Cache is initialized but empty, falling back to SPARQL queries")
            cohorts = retrieve_cohorts_metadata(user_email)
            cohorts_dict = {cid: cohort_to_dict(c) for cid, c in cohorts.items()}
            cache_status = "from_sparql"
    else:
        logging.info("Cache not initialized, falling back to SPARQL queries")
        cohorts = retrieve_cohorts_metadata(user_email)
        cohorts_dict = {cid: cohort_to_dict(c) for cid, c in cohorts.items()}
        cache_status = "from_sparql"
    
    # Prepare data for download
    data = {
        "cohorts": cohorts_dict,
        "userEmail": user_email,
        "cache_status": cache_status,
        "cohort_count": len(cohorts_dict),
        "download_timestamp": datetime.now().isoformat()
    }
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"cohorts_cache_{timestamp}.json"
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(data, tmp_file, indent=2)
        tmp_path = tmp_file.name
    
    return FileResponse(
        path=tmp_path,
        filename=filename,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.post("/compare-cohorts-cache")
async def compare_cohorts_cache(
    cache_file_a: UploadFile = File(..., description="First cache JSON file"),
    cache_file_b: UploadFile = File(..., description="Second cache JSON file"),
    user: Any = Depends(get_current_user)
) -> FileResponse:
    """
    Compare two cohorts cache files and generate a detailed comparison report.
    
    Accepts two cache JSON files and returns a comprehensive report showing:
    1. Cohorts present in A but not in B, and vice versa
    2. Metadata keys present in one cache but not the other for matching cohorts
    3. Value differences for matching metadata keys
    4. Variable differences including added/removed variables and attribute changes
    
    Returns a timestamped JSON report file.
    """
    try:
        # Read and parse both cache files
        content_a = await cache_file_a.read()
        content_b = await cache_file_b.read()
        
        cache_a = json.loads(content_a)
        cache_b = json.loads(content_b)
        
        # Perform comparison
        comparison_report = compare_cache_objects(cache_a, cache_b)
        
        # Add metadata to report
        comparison_report["comparison_metadata"] = {
            "file_a_name": cache_file_a.filename,
            "file_b_name": cache_file_b.filename,
            "comparison_timestamp": datetime.now().isoformat(),
            "performed_by": user["email"]
        }
        
        # Generate timestamped filename for report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"cache_comparison_report_{timestamp}.json"
        
        # Write report to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            json.dump(comparison_report, tmp_file, indent=2)
            tmp_path = tmp_file.name
        
        return FileResponse(
            path=tmp_path,
            filename=report_filename,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{report_filename}"'
            }
        )
        
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing cache files: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON in cache file: {str(e)}")
    except Exception as e:
        logging.error(f"Error comparing cache files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing cache files: {str(e)}")
