import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from PIL import Image

from src.auth import get_current_user
from src.config import settings
from src.demo.assets import asset_root, contained_asset_path, validate_asset_component
from src.mapping_artifacts import MappingArtifactStore
from src.metadata_providers.factory import (
    get_concept_search_provider,
    get_mapping_generation_provider,
)

router = APIRouter()


def _mapping_store() -> MappingArtifactStore:
    return MappingArtifactStore(
        Path(settings.mapping_output_dir),
        cohorts_root=Path(settings.cohort_folder),
    )


@router.post("/check-mapping-cache")
async def check_mapping_cache(
    source_study: str = Body(...),
    target_studies: list = Body(...),
    user: Any = Depends(get_current_user),
):
    """Check canonical dictionary mtimes against configured mapping artifacts."""
    _ = user
    store = _mapping_store()
    source = source_study.strip().casefold()
    try:
        targets = [str(target[0]).strip().casefold() for target in target_studies]
    except (IndexError, TypeError):
        raise HTTPException(status_code=422, detail="Invalid target study request")

    dictionary_mtimes_ns = {
        cohort: store.dictionary_mtime_ns(cohort)
        for cohort in {source, *targets}
    }
    dictionary_timestamps = {
        cohort: mtime_ns / 1_000_000_000
        for cohort, mtime_ns in dictionary_mtimes_ns.items()
        if mtime_ns
    }
    cached_pairs: list[dict[str, Any]] = []
    uncached_pairs: list[dict[str, Any]] = []
    outdated_pairs: list[dict[str, Any]] = []
    for target in targets:
        artifact = store.find_pair(source, target, include_stale=True)
        if artifact is None:
            uncached_pairs.append({"source": source, "target": target})
            continue
        pair_info: dict[str, Any] = {
            "source": source,
            "target": target,
            "timestamp": artifact.timestamp,
        }
        if store.cache_status(artifact).fresh:
            cached_pairs.append(pair_info)
            continue
        source_mtime = dictionary_mtimes_ns.get(source, 0)
        target_mtime = dictionary_mtimes_ns.get(target, 0)
        pair_info["outdated_cohort"] = source if source_mtime >= target_mtime else target
        outdated_pairs.append(pair_info)

    from CohortVarLinker.main import _check_graphs_need_recreate

    will_recreate_graph = _check_graphs_need_recreate(
        {source, *targets},
        settings.cohort_folder,
    )

    return JSONResponse(content={
        "cached_pairs": cached_pairs,
        "uncached_pairs": uncached_pairs,
        "outdated_pairs": outdated_pairs,
        "dictionary_timestamps": dictionary_timestamps,
        "will_recreate_graph": will_recreate_graph,
    })


@router.post("/get-available-mapping-files")
async def get_available_mapping_files(
    cohort_ids: list[str] = Body(...),
    user: Any = Depends(get_current_user)
):
    """List fresh mapping CSVs whose cohorts are all currently selected."""
    _ = user
    deduped = [
        artifact.to_api_dict()
        for artifact in _mapping_store().list_for_cohorts(set(cohort_ids))
    ]
    return JSONResponse(content={
        "available_mappings": deduped,
        "cohort_count": len(cohort_ids),
    })


@router.get("/get-cached-mapping-file/{filename}")
async def get_cached_mapping_file(
    filename: str,
    user: Any = Depends(get_current_user),
):
    """Return the JSON content of a cached mapping file by filename.
    
    Returns the raw JSON directly (not wrapped in another JSON envelope)
    to avoid double-serialization overhead on large files.
    The filename is returned via the X-Filename response header.
    """
    _ = user
    store = _mapping_store()
    try:
        filepath = store.safe_path(filename)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Mapping file not found")
    artifact = store.artifact_for(filepath)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Mapping file not found")
    if not store.cache_status(artifact).fresh:
        raise HTTPException(status_code=409, detail="Mapping file is outdated")
    file_content = filepath.read_text(encoding="utf-8")
    if filepath.suffix.casefold() == ".csv":
        return Response(
            content=file_content,
            media_type="text/csv",
            headers={
                "X-Filename": filepath.name,
                "Access-Control-Expose-Headers": "X-Filename",
            },
        )
    
    # Sanitize NaN values server-side so clients don't have to
    file_content = file_content.replace("NaN", "null")
    
    return Response(
        content=file_content,
        media_type="application/json",
        headers={
            "X-Filename": filepath.name,
            "Access-Control-Expose-Headers": "X-Filename",
        },
    )


@router.post("/generate-mapping")
async def generate_mapping(
    source_study: str = Body(...),
    target_studies: list = Body(...),
    user: Any = Depends(get_current_user),
):
    """Generate through the selected provider, then serve its fresh JSON artifact."""
    try:
        normalized_targets = sorted(target_studies, key=lambda target: str(target[0]).casefold())
        target_names = [str(target[0]) for target in normalized_targets]
    except (IndexError, TypeError):
        raise HTTPException(status_code=422, detail="Invalid target study request")

    store = _mapping_store()
    run_context = {
        "source": source_study,
        "targets": target_names,
        "user": user.get("email", "unknown"),
    }
    started_at = time.monotonic()
    store.record_activity(
        "run_started",
        "Mapping run started",
        context=run_context,
    )
    provider = get_mapping_generation_provider(settings)
    try:
        result = provider.generate(source_study, normalized_targets)
    except Exception as error:
        store.record_activity(
            "run_failed",
            f"Mapping run failed: {error}",
            context={**run_context, "error": str(error)},
        )
        status_code = getattr(error, "status_code", None)
        detail = getattr(error, "detail", str(error))
        if isinstance(status_code, int):
            raise HTTPException(status_code=status_code, detail=detail) from error
        raise

    artifact = store.result_for_request(source_study, target_names)
    if artifact is not None:
        elapsed = round(time.monotonic() - started_at, 2)
        store.record_activity(
            "result_file_served",
            f"Serving mapping result: {artifact.filename}",
            context={
                "filename": artifact.filename,
                "file_size_bytes": artifact.file_size,
            },
            level="DETAIL",
            depth=1,
        )
        store.record_activity(
            "run_completed",
            f"Mapping run completed in {elapsed}s",
            context={**run_context, "elapsed_s": elapsed},
        )
        return JSONResponse(content={
            "cache_info": result.cache_info,
            "file_content": artifact.path.read_text(encoding="utf-8"),
            "filename": artifact.filename,
        })
    store.record_activity(
        "result_not_found",
        f"No fresh mapping output file found for {source_study}",
        context=run_context,
    )
    return JSONResponse(status_code=404, content={"error": "Cache error. Mapping file not found."})


@router.get("/mapping-activity-log")
async def get_mapping_activity_log(
    limit: int = Query(default=200, ge=1, le=5000),
    level: str | None = Query(default=None, description="Filter by level: MAIN or DETAIL"),
    process: str | None = Query(default=None, description="Filter by process: cohort_var_linker or standard_code_mapping"),
    user: Any = Depends(get_current_user),
):
    """Return mapping activity from the same configured artifact directory."""
    _ = user
    entries, total = _mapping_store().read_activity(
        limit=limit,
        level=level,
        process=process,
    )
    return JSONResponse(content={"entries": entries, "total": total})


@router.get("/search-concepts")
async def search_concepts(
    query: str, domain: list[str] | None = Query(default=None), user: Any = Depends(get_current_user)
):
    """Search the selected concept provider and enrich results with local usage."""
    provider = get_concept_search_provider(settings)
    return [asdict(item) for item in await provider.search(query, domain or [])]


def find_dcr_output_folder(cohort_id: str) -> str | None:
    """
    Find the actual dcr_output folder for a cohort, handling case-insensitive matching.
    Returns the actual folder name if found, None otherwise.
    """
    try:
        cohort_id = validate_asset_component(cohort_id)
    except ValueError:
        return None
    data_folder = asset_root(settings).resolve()
    if not data_folder.is_dir():
        return None
    
    # Try exact match first
    exact_folder = f"dcr_output_{cohort_id}"
    if (data_folder / exact_folder).is_dir():
        return exact_folder
    
    # Try case-insensitive search
    target_prefix = f"dcr_output_{cohort_id.lower()}"
    for folder in data_folder.iterdir():
        if folder.name.lower() == target_prefix and folder.is_dir():
            return folder.name
    
    return None


def _route_asset_component(value: str) -> str:
    try:
        return validate_asset_component(value)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _route_asset_path(*parts: str) -> Path:
    try:
        return contained_asset_path(asset_root(settings), *parts)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@router.get("/compare-eda/{source_cohort}/{source_var}/{target_cohort}/{target_var}")
async def compare_eda(
    source_cohort: str,
    source_var: str,
    target_cohort: str,
    target_var: str
):
    """
    Merge two EDA PNG files vertically (source on top, target on bottom) and return the merged image.
    """
    import io

    source_cohort = _route_asset_component(source_cohort)
    source_var = _route_asset_component(source_var)
    target_cohort = _route_asset_component(target_cohort)
    target_var = _route_asset_component(target_var)
    
    # Find the actual folder names (case-insensitive)
    source_folder = find_dcr_output_folder(source_cohort)
    target_folder = find_dcr_output_folder(target_cohort)
    
    # Collect detailed error messages
    errors = []
    
    # Check source cohort
    if not source_folder:
        errors.append(f"Source cohort '{source_cohort}': Exploratory Data Analysis has not yet been run on this cohort")
    else:
        source_image_path = _route_asset_path(
            source_folder,
            f"{source_var.lower()}.png",
        )
        if not source_image_path.is_file():
            errors.append(f"Source variable '{source_var}' in cohort '{source_cohort}': This variable was excluded from the EDA analysis")
    
    # Check target cohort
    if not target_folder:
        errors.append(f"Target cohort '{target_cohort}': Exploratory Data Analysis has not yet been run on this cohort")
    else:
        target_image_path = _route_asset_path(
            target_folder,
            f"{target_var.lower()}.png",
        )
        if not target_image_path.is_file():
            errors.append(f"Target variable '{target_var}' in cohort '{target_cohort}': This variable was excluded from the EDA analysis")
    
    # If any errors, raise with detailed message
    if errors:
        error_detail = "Cannot compare EDA images:\n" + "\n".join(f"• {err}" for err in errors)
        raise HTTPException(
            status_code=404,
            detail=error_detail
        )
    
    # Both files exist, construct paths
    source_image_path = _route_asset_path(
        source_folder,
        f"{source_var.lower()}.png",
    )
    target_image_path = _route_asset_path(
        target_folder,
        f"{target_var.lower()}.png",
    )
    
    try:
        # Load both images
        source_image = Image.open(source_image_path)
        target_image = Image.open(target_image_path)
        
        # Convert images to RGB if they have transparency
        if source_image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', source_image.size, 'white')
            if source_image.mode == 'P':
                source_image = source_image.convert('RGBA')
            background.paste(source_image, mask=source_image.split()[-1] if source_image.mode in ('RGBA', 'LA') else None)
            source_image = background
        elif source_image.mode != 'RGB':
            source_image = source_image.convert('RGB')
            
        if target_image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', target_image.size, 'white')
            if target_image.mode == 'P':
                target_image = target_image.convert('RGBA')
            background.paste(target_image, mask=target_image.split()[-1] if target_image.mode in ('RGBA', 'LA') else None)
            target_image = background
        elif target_image.mode != 'RGB':
            target_image = target_image.convert('RGB')
        
        # Resize images to 75% of original size
        source_image = source_image.resize(
            (int(source_image.width * 0.75), int(source_image.height * 0.75)),
            Image.Resampling.LANCZOS
        )
        target_image = target_image.resize(
            (int(target_image.width * 0.75), int(target_image.height * 0.75)),
            Image.Resampling.LANCZOS
        )
        
        # Calculate dimensions for the merged image
        max_width = max(source_image.width, target_image.width)
        total_height = source_image.height + target_image.height
        
        # Create a new image with white background
        merged_image = Image.new('RGB', (max_width, total_height), 'white')
        
        # Paste source image on top (centered if narrower)
        source_x = (max_width - source_image.width) // 2
        merged_image.paste(source_image, (source_x, 0))
        
        # Paste target image on bottom (centered if narrower)
        target_x = (max_width - target_image.width) // 2
        merged_image.paste(target_image, (target_x, source_image.height))
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        merged_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to merge EDA images: {str(e)}"
        )
