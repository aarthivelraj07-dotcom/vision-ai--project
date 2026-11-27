"""
safe_extract_isl.py
===================
Production-Grade Hardened & Audited Extraction Pipeline for ISL Dataset
Task 5: Indian Sign Language Recognition (Plan A: Static Alphabet & Digits)

Defense-in-Depth Security Guarantees:
1. Strict Whitelist Filtering:
   - Extracts ONLY 12 authorized classes: {A, B, C, L, V, W, Y, 1, 2, 3, 4, 5}.
   - Rejects unapproved image extensions; accepts only {.jpg, .jpeg, .png}.
2. Structural Layout & Parsing Order Enforcement:
   - Validates directory entries ending in exactly one separator ('/' or '\\')
     for path traversal, drive letters, invalid characters, and reserved names
     BEFORE safely skipping them.
   - Enforces a strictly consistent layout across the entire archive:
       Either Layout A: <class>/<filename> (depth 2)
       Or     Layout B: <single_common_root>/<class>/<filename> (depth 3)
     Mixing layouts or detecting multiple common roots triggers immediate abort.
   - Rejects ambiguous paths containing multiple target-class components.
   - Rejects mixed separators ('/' and '\\') and repeated separators ('//', '\\\\').
3. Windows Path Traversal, Dangling Symlink, and Collision Defense:
   - Preserves final destination component unresolved so os.path.lexists() detects
     existing directories, regular files, symbolic links, and dangling symlinks.
   - Delays parent directory creation until archive preflights pass.
   - Case-insensitive Unicode NFKC collision tracking prevents Windows NTFS overwrites.
   - Comprehensive Windows reserved device name check (CON, PRN, AUX, NUL,
     COM1-9, COM¹-³, LPT1-9, LPT¹-³).
4. Resource Limiting & Decompression Bomb Prevention:
   - Caps total archive entries (MAX_ARCHIVE_ENTRIES = 100,000).
   - Caps per-file uncompressed size (MAX_FILE_BYTES = 10 MB).
   - Caps total extracted dataset size (MAX_TOTAL_EXTRACTED_BYTES = 500 MB).
   - Rejects anomalous compression ratios (> 100.0) and zero-compressed entries with file_size > 0.
   - Rejects PKZIP traditional, strong, and WinZip AES encryption.
5. Mandatory Two-Stage Image Integrity & Decoding:
   - Stage 1 (Pillow): Parses image headers (IHDR/SOF) to verify dimensions (16 to 4096 px)
     without allocating pixel raster.
   - Stage 2 (OpenCV): Always decompresses full raster buffer in memory; verifies 3 channels (BGR)
     and exact dimension agreement with header.
   - Scoped Image.MAX_IMAGE_PIXELS restoration in finally block.
     NOTE: Pillow MAX_IMAGE_PIXELS is process-global and not thread-safe across concurrent threads.
6. Robust Nested Cleanup & Atomic Publication:
   - Extraction, threshold evaluation, and publication run under a unified lifecycle.
   - Nested try/finally structure guarantees ZIP closure and staging cleanup attempts
     even if Image.MAX_IMAGE_PIXELS restoration fails.
   - Staging cleanup failures report the exact error and lingering filesystem path.
   - Promotes staging directory to destination atomically via os.rename().
   - Published destination is never deleted upon success.
7. Strict Training Approval Gate & Threshold Reporting:
   - Aborts and removes staging if any class has fewer than MIN_PER_CLASS_THRESHOLD (100 images).
   - Flags PARTIAL or EXCEEDED review if class counts deviate from EXPECTED_PER_CLASS (1,200 images).
   - Sets authorized_for_training = False always; training requires explicit user review.
"""

import hashlib
import io
import os
import shutil
import sys
import unicodedata
import uuid
import zipfile
import cv2
import numpy as np
from PIL import Image

# ==============================================================================
# CONFIGURATION & SECURITY CONSTANTS
# ==============================================================================

TARGET_CLASSES = frozenset(
    {"A", "B", "C", "L", "V", "W", "Y", "1", "2", "3", "4", "5"}
)
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})

MAX_ARCHIVE_ENTRIES = 100_000
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB per file limit
MAX_TOTAL_EXTRACTED_BYTES = 500 * 1024 * 1024  # 500 MB total limit
MAX_COMPRESSION_RATIO = 100.0  # Suspect decompression bomb ratio

MIN_IMAGE_DIM = 16
MAX_IMAGE_DIM = 4096
MAX_IMAGE_PIXELS = 16_777_216  # 4096 x 4096 maximum pixel count

MIN_PER_CLASS_THRESHOLD = 100  # Minimum usable images required to permit publication
EXPECTED_PER_CLASS = 1_200  # Official expected count per class

# Comprehensive Windows reserved device names, including superscript variants
WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "COM¹",
        "COM²",
        "COM³",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
        "LPT¹",
        "LPT²",
        "LPT³",
    }
)


def compute_sha256(filepath: str) -> str:
  """Computes SHA-256 hash of a file streaming in 64KB blocks."""
  h = hashlib.sha256()
  with open(filepath, "rb") as f:
    while chunk := f.read(65536):
      h.update(chunk)
  return h.hexdigest()


def safe_extract_isl(archive_path: str, destination_dir: str) -> dict:
  """Extracts, audits, and publishes the 12 authorized ISL classes from a ZIP archive.

  Parameters
  ----------
  archive_path : str
      Filesystem path to the downloaded dataset ZIP archive.
  destination_dir : str
      Target directory path where extracted classes will be published.

  Returns
  -------
  dict
      Manifest dictionary containing extracted metrics, class counts, and
      validation status.
  """
  # --------------------------------------------------------------------------
  # 1. Preflight Validation: Archive Path & File Type
  # --------------------------------------------------------------------------
  archive_abs = os.path.realpath(os.path.abspath(archive_path))
  if not os.path.isfile(archive_abs):
    raise FileNotFoundError(
        f"Archive file not found or is not a regular file: '{archive_path}'"
    )

  print("\n" + "=" * 74)
  print("STARTING AUDITED SAFE EXTRACTION: ISL DATASET (PLAN A: 12 CLASSES)")
  print("=" * 74)
  print(f"Archive Path:                  {archive_abs}")
  print(f"Archive Size:                  {os.path.getsize(archive_abs):,} bytes")

  print("Computing local archive SHA-256 (recording local archive identity)...")
  archive_hash = compute_sha256(archive_abs)
  print(f"Local Archive SHA-256:         {archive_hash}")
  print(
      "Notice: SHA-256 records local identity; Kaggle does not publish static"
      " hashes."
  )

  # --------------------------------------------------------------------------
  # 2. Preflight Validation: Destination Path & Symlink Inspection
  #    (Parent directory is NOT created yet to avoid premature filesystem changes)
  # --------------------------------------------------------------------------
  dest_normalized = os.path.normpath(os.path.abspath(destination_dir))
  dest_basename = os.path.basename(dest_normalized)
  parent_dir = os.path.dirname(dest_normalized)

  if not dest_basename:
    raise ValueError(f"Invalid destination directory path: '{destination_dir}'")

  # Inspect parent directory if it exists; resolve its symlinks
  if os.path.lexists(parent_dir):
    if not os.path.isdir(parent_dir):
      raise NotADirectoryError(
          f"Destination parent path exists and is not a directory:"
          f" '{parent_dir}'"
      )
    parent_real = os.path.realpath(parent_dir)
  else:
    # Parent does not exist yet; preserve uncreated path without resolving
    parent_real = parent_dir

  # Construct dest_path preserving final component without resolving it
  dest_path = os.path.join(parent_real, dest_basename)

  # Preflight check: abort immediately if destination or dangling symlink exists
  if os.path.lexists(dest_path):
    raise FileExistsError(
        f"Destination path already exists (directory, file, or dangling"
        f" symlink): '{dest_path}'"
    )

  # --------------------------------------------------------------------------
  # 3. Outer Protected Lifecycle: Resource & Sandbox Management
  # --------------------------------------------------------------------------
  staging_dir = None
  is_published = False
  zf = None
  prev_max_pixels = Image.MAX_IMAGE_PIXELS

  try:
    # Open Archive Under Protected Lifecycle
    try:
      zf = zipfile.ZipFile(archive_abs, mode="r")
    except Exception as open_err:
      raise ValueError(
          f"Failed to open archive as a valid ZIP file: '{archive_abs}'."
          f" Error: {open_err}"
      ) from open_err

    total_entries = len(zf.filelist)
    print(
        f"Initial Archive Check:         Central directory parsed"
        f" ({total_entries:,} entries)"
    )

    if total_entries > MAX_ARCHIVE_ENTRIES:
      raise ValueError(
          f"Archive entry count ({total_entries:,}) exceeds security ceiling "
          f"({MAX_ARCHIVE_ENTRIES:,}). Aborting."
      )

    print(
        f"Archive Preflight Status:      Initial checks passed ({total_entries:,}"
        f" <= {MAX_ARCHIVE_ENTRIES:,} max entries)"
    )
    print(
        "Notice: Full entry validation and image decoding occur during"
        " extraction."
    )

    # Prepare Staging Sandbox (Parent directory created only after preflights pass)
    os.makedirs(parent_real, exist_ok=True)
    parent_real = os.path.realpath(parent_real)
    dest_path = os.path.join(parent_real, dest_basename)

    if os.path.lexists(dest_path):
      raise FileExistsError(
          f"Destination path appeared during preflight initialization:"
          f" '{dest_path}'"
      )

    staging_name = f".staging_isl_{os.getpid()}_{uuid.uuid4().hex}"
    staging_dir = os.path.join(parent_real, staging_name)
    os.makedirs(staging_dir, exist_ok=False)
    print(f"Isolated Staging Sandbox:      {staging_dir}")

    # Extraction verification state
    class_counts = {cls: 0 for cls in sorted(TARGET_CLASSES)}
    total_extracted_bytes = 0
    total_extracted_images = 0
    skipped_out_of_scope = 0

    seen_archive_entries = set()
    seen_dest_paths = set()

    # Hierarchy consistency trackers
    detected_layout = None  # 'flat' (depth 2) or 'rooted' (depth 3)
    common_root_name = None  # exact root folder name if 'rooted'

    # Apply process-global Pillow max pixels setting
    # NOTE: Image.MAX_IMAGE_PIXELS is a process-global variable. Modifying it is
    # NOT thread-safe across concurrent threads running in the same Python process.
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

    # ------------------------------------------------------------------------
    # 4. Entry-by-Entry Validation & Extraction Loop
    # ------------------------------------------------------------------------
    for zip_info in zf.filelist:
      member = zip_info.filename

      # --- A. Check for Mixed and Repeated Separators ---
      if "/" in member and "\\" in member:
        raise ValueError(f"Mixed path separators detected in entry: '{member}'")
      if "//" in member or "\\\\" in member:
        raise ValueError(
            f"Repeated path separators detected in entry: '{member}'"
        )

      # --- B. Check for Drive Letters, Absolute Paths, and Null Bytes ---
      if len(member) >= 2 and member[1] == ":" and member[0].isalpha():
        raise ValueError(f"Absolute drive path rejected: '{member}'")
      if member.startswith(("/", "\\")):
        raise ValueError(
            f"Leading path separator (absolute path) rejected: '{member}'"
        )
      if "\0" in member:
        raise ValueError(f"Null byte detected in archive entry: '{member}'")

      # --- C. Identify Directory Entry vs. File Entry ---
      is_dir_entry = zip_info.is_dir() or member.endswith(("/", "\\"))
      clean_path = member.replace("\\", "/")

      if is_dir_entry:
        path_to_validate = clean_path.rstrip("/")
        if not path_to_validate:
          raise ValueError(f"Empty root directory entry rejected: '{member}'")
      else:
        path_to_validate = clean_path

      parts = path_to_validate.split("/")
      if any(len(p) == 0 for p in parts):
        raise ValueError(f"Empty path component detected in entry: '{member}'")

      # --- D. Component-Level Traversal, Unicode, and Reserved Name Auditing ---
      for p in parts:
        if p in (".", ".."):
          raise ValueError(
              f"Path traversal component '{p}' rejected in entry: '{member}'"
          )
        if "\0" in p:
          raise ValueError(
              f"Null byte in path component '{p}' in entry: '{member}'"
          )
        if p.endswith((".", " ")):
          raise ValueError(
              f"Path component ending with dot or space rejected: '{p}' in"
              f" entry: '{member}'"
          )
        if any(c in '<>:"|?*' or ord(c) < 32 for c in p):
          raise ValueError(
              f"Invalid character in path component '{p}' in entry: '{member}'"
          )

        # Canonical Unicode normalization check (NFKC)
        norm_p = unicodedata.normalize("NFKC", p)
        if norm_p != p:
          raise ValueError(
              f"Non-canonical Unicode component rejected: '{p}' != '{norm_p}'"
              f" in entry: '{member}'"
          )

        # Windows reserved device name check (audits base stem + superscript variants)
        clean_stem = p.split(".")[0].rstrip(". ")
        if (
            clean_stem.upper() in WINDOWS_RESERVED_NAMES
            or unicodedata.normalize("NFKC", clean_stem).upper()
            in WINDOWS_RESERVED_NAMES
        ):
          raise ValueError(
              f"Windows reserved device name rejected: '{p}' in entry:"
              f" '{member}'"
          )

      # --- E. Safe Directory Skipping (AFTER path validation has passed) ---
      if is_dir_entry:
        continue

      # --- F. File Entry Hierarchy & Consistency Enforcement ---
      filename = parts[-1]
      parent_dirs = parts[:-1]

      # Find target class matches in directory hierarchy
      class_matches = [
          (i, part)
          for i, part in enumerate(parent_dirs)
          if part in TARGET_CLASSES
      ]

      if not class_matches:
        # File does not belong to any authorized target class; skip as out-of-scope
        skipped_out_of_scope += 1
        continue

      # Reject ambiguous paths containing multiple target-class components
      if len(class_matches) > 1:
        matched_names = [name for _, name in class_matches]
        raise ValueError(
            f"Ambiguous path with multiple target-class components"
            f" {matched_names} in entry: '{member}'"
        )

      class_idx, target_class = class_matches[0]

      # Verify that matched class is the immediate parent directory of filename
      if target_class != parent_dirs[-1]:
        raise ValueError(
            f"Target class '{target_class}' is not immediate parent of file in"
            f" entry: '{member}'"
        )

      # Enforce uniform hierarchy across all target entries in archive:
      # Layout A: <class>/<filename> (depth 2)
      # Layout B: <single_common_root>/<class>/<filename> (depth 3)
      if len(parts) == 2:
        current_layout = "flat"
        current_root = None
      elif len(parts) == 3:
        current_layout = "rooted"
        current_root = parts[0]
      else:
        raise ValueError(
            f"Unexpected nested path depth ({len(parts)}) for target class in"
            f" entry: '{member}'. Expected '<class>/<file>' or"
            " '<single_common_root>/<class>/<file>'."
        )

      if detected_layout is None:
        detected_layout = current_layout
        common_root_name = current_root
      else:
        if current_layout != detected_layout:
          raise ValueError(
              f"Inconsistent ZIP hierarchy: archive mixes '{detected_layout}'"
              f" layout with '{current_layout}' layout in entry: '{member}'"
          )
        if detected_layout == "rooted" and current_root != common_root_name:
          raise ValueError(
              "Inconsistent ZIP hierarchy: multiple common roots detected"
              f" ('{common_root_name}' vs '{current_root}') in entry: '{member}'"
          )

      # Validate allowed image extensions
      _, ext = os.path.splitext(filename)
      if ext.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Disallowed file extension '{ext}' in entry: '{member}'. Allowed:"
            f" {sorted(ALLOWED_EXTENSIONS)}"
        )

      # --- G. Duplicate Entry Detection ---
      member_norm = unicodedata.normalize("NFKC", member).casefold()
      if member_norm in seen_archive_entries:
        raise ValueError(f"Duplicate archive entry detected: '{member}'")
      seen_archive_entries.add(member_norm)

      # --- H. Windows Case-Insensitive Collision Detection ---
      dest_rel = os.path.join(target_class, filename)
      dest_rel_norm = unicodedata.normalize("NFKC", dest_rel).casefold()
      if dest_rel_norm in seen_dest_paths:
        raise ValueError(
            f"Destination collision (case/Unicode fold) detected: '{dest_rel}'"
        )
      seen_dest_paths.add(dest_rel_norm)

      # --- I. ZIP Metadata Pre-Read Inspection & Encryption Defense ---
      # Check PKZIP Bit 0 (traditional encryption) and Bit 6 (strong encryption)
      if bool(zip_info.flag_bits & 0x1) or bool(zip_info.flag_bits & 0x40):
        raise ValueError(f"Encrypted ZIP entry rejected: '{member}'")
      # Check WinZip AES encryption compression type (99)
      if zip_info.compress_type == 99:
        raise ValueError(f"AES-encrypted ZIP entry rejected: '{member}'")

      # Detect anomalous zero-compressed bytes with declared positive uncompressed size
      if zip_info.file_size > 0 and zip_info.compress_size == 0:
        raise ValueError(
            f"Anomalous ZIP metadata (file_size={zip_info.file_size} but"
            f" compress_size=0): '{member}'"
        )

      # Resource limit: maximum uncompressed file size
      if zip_info.file_size > MAX_FILE_BYTES:
        raise ValueError(
            f"Entry exceeds per-file limit ({zip_info.file_size:,} >"
            f" {MAX_FILE_BYTES:,} bytes): '{member}'"
        )

      # Resource limit: cumulative extracted dataset size
      if total_extracted_bytes + zip_info.file_size > MAX_TOTAL_EXTRACTED_BYTES:
        raise ValueError(
            f"Cumulative extracted size exceeds ceiling"
            f" ({MAX_TOTAL_EXTRACTED_BYTES:,} bytes). Aborting."
        )

      # Resource limit: compression ratio check (zip bomb defense)
      if zip_info.compress_size > 0:
        ratio = zip_info.file_size / zip_info.compress_size
        if ratio > MAX_COMPRESSION_RATIO:
          raise ValueError(
              f"Suspicious compression ratio ({ratio:.1f} >"
              f" {MAX_COMPRESSION_RATIO}): '{member}'"
          )

      # --- J. Read & Verify Raw File Payload ---
      file_bytes = zf.read(zip_info)
      if len(file_bytes) != zip_info.file_size:
        raise ValueError(
            f"Extracted byte count mismatch for '{member}': expected"
            f" {zip_info.file_size}, got {len(file_bytes)}"
        )

      # --- K. Mandatory Dual-Stage Image Validation ---
      # Stage 1: Pillow header-only inspection
      try:
        with Image.open(io.BytesIO(file_bytes)) as pil_img:
          w, h = pil_img.size
          if not (
              MIN_IMAGE_DIM <= w <= MAX_IMAGE_DIM
              and MIN_IMAGE_DIM <= h <= MAX_IMAGE_DIM
          ):
            raise ValueError(
                f"Image dimensions ({w}x{h}) outside permitted bounds"
                f" [{MIN_IMAGE_DIM}, {MAX_IMAGE_DIM}]: '{member}'"
            )
          pil_img.verify()
      except Exception as e:
        raise ValueError(
            f"Corrupted or invalid image header in '{member}': {e}"
        ) from e

      # Stage 2: Mandatory OpenCV raster decompression & 3-channel BGR check
      nparr = np.frombuffer(file_bytes, np.uint8)
      cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
      if cv_img is None:
        raise ValueError(f"OpenCV failed to decode image raster: '{member}'")
      if len(cv_img.shape) != 3 or cv_img.shape[2] != 3:
        raise ValueError(
            f"Image raster must have exactly 3 channels (BGR), got shape"
            f" {cv_img.shape}: '{member}'"
        )
      if cv_img.shape[0] != h or cv_img.shape[1] != w:
        raise ValueError(
            f"OpenCV raster dimensions ({cv_img.shape[1]}x{cv_img.shape[0]})"
            f" mismatch Pillow header ({w}x{h}): '{member}'"
        )

      # --- L. Stage File Write & Containment Verification ---
      class_staging_dir = os.path.join(staging_dir, target_class)
      os.makedirs(class_staging_dir, exist_ok=True)

      target_file_path = os.path.join(class_staging_dir, filename)

      # Verify staged path remains strictly contained within staging sandbox
      target_abs = os.path.realpath(os.path.abspath(target_file_path))
      staging_real = os.path.realpath(staging_dir)
      if not (
          target_abs == staging_real
          or target_abs.startswith(staging_real + os.sep)
      ):
        raise ValueError(
            f"Staged path containment violation: '{target_abs}' escapes sandbox"
            f" '{staging_real}'"
        )

      if os.path.exists(target_file_path):
        raise ValueError(
            f"Target file already exists in staging (collision):"
            f" '{target_file_path}'"
        )

      with open(target_file_path, "wb") as out_f:
        out_f.write(file_bytes)

      class_counts[target_class] += 1
      total_extracted_bytes += len(file_bytes)
      total_extracted_images += 1

    # Close ZIP handle as soon as extraction loop finishes successfully
    zf.close()
    zf = None

    # ------------------------------------------------------------------------
    # 5. Audit Threshold Assessment (Under Protected Lifecycle)
    # ------------------------------------------------------------------------
    below_minimum_classes = [
        cls
        for cls, count in class_counts.items()
        if count < MIN_PER_CLASS_THRESHOLD
    ]
    partial_classes = [
        cls
        for cls, count in class_counts.items()
        if MIN_PER_CLASS_THRESHOLD <= count < EXPECTED_PER_CLASS
    ]
    exceeded_classes = [
        cls for cls, count in class_counts.items() if count > EXPECTED_PER_CLASS
    ]

    # Criterion A: Minimum usable data threshold (failure aborts publication)
    if below_minimum_classes:
      raise RuntimeError(
          "Extraction aborted: The following target classes failed minimum"
          f" threshold (< {MIN_PER_CLASS_THRESHOLD} images):"
          f" {below_minimum_classes}. Staging sandbox cleanup scheduled."
      )

    meets_minimum_threshold = True

    # Criterion B: Completeness assessment
    is_dataset_complete = (
        len(partial_classes) == 0 and len(exceeded_classes) == 0
    )

    if is_dataset_complete:
      publication_status = "PUBLISHED (FULLY COMPLETE - 1,200/class)"
    else:
      publication_status = "PUBLISHED (PARTIAL / REVIEW REQUIRED)"

    # ------------------------------------------------------------------------
    # 6. Atomic Publication (Under Protected Lifecycle)
    # ------------------------------------------------------------------------
    if os.path.lexists(dest_path):
      raise FileExistsError(
          "Destination path appeared concurrently immediately prior to"
          f" publication: '{dest_path}'"
      )

    # Atomic rename on the same filesystem volume
    os.rename(staging_dir, dest_path)
    is_published = True  # Marks that staging has been atomically promoted to destination!

    print(f"\nExtraction Successfully Published To: {dest_path}")
    print(f"Publication Status:            {publication_status}")

  finally:
    # Nested try/finally structure guarantees that ZIP handle closure and staging
    # cleanup attempts always execute, even if restoring Image.MAX_IMAGE_PIXELS fails.
    try:
      try:
        Image.MAX_IMAGE_PIXELS = prev_max_pixels
      except Exception as img_err:
        print(f"Warning: Failed to restore Image.MAX_IMAGE_PIXELS: {img_err}")
    finally:
      try:
        if zf is not None:
          try:
            zf.close()
          except Exception as zf_err:
            print(f"Warning: Failed to close ZIP handle: {zf_err}")
      finally:
        # Staging sandbox cleanup attempt (never deletes published destination)
        if staging_dir and not is_published:
          try:
            if os.path.exists(staging_dir) and (
                not os.path.exists(dest_path)
                or os.path.realpath(staging_dir) != os.path.realpath(dest_path)
            ):
              shutil.rmtree(staging_dir)
          except Exception as cleanup_err:
            print(
                "CRITICAL WARNING: Staging directory cleanup attempt failed:"
                f" {cleanup_err}. Lingering staging directory remains on disk"
                f" at: '{staging_dir}'"
            )

  # --------------------------------------------------------------------------
  # 7. Audit Manifest & Human Review Reporting Table
  # --------------------------------------------------------------------------
  print("\n" + "=" * 74)
  print("DATASET EXTRACTION AUDIT REPORT (TASK 5 – PLAN A)")
  print("=" * 74)
  print(
      f"{'Class':<8} | {'Extracted':<12} | {'Expected':<12} | {'Audit Status'}"
  )
  print("-" * 74)

  discrepancies_found = False
  for cls in sorted(TARGET_CLASSES):
    count = class_counts[cls]
    if count < EXPECTED_PER_CLASS:
      status = f"PARTIAL ({count} < {EXPECTED_PER_CLASS}) - REVIEW REQUIRED"
      discrepancies_found = True
    elif count == EXPECTED_PER_CLASS:
      status = "VERIFIED COMPLETE (1,200 images)"
    else:
      status = f"EXCEEDED ({count} > {EXPECTED_PER_CLASS}) - REVIEW REQUIRED"
      discrepancies_found = True
    print(f"{cls:<8} | {count:<12} | {EXPECTED_PER_CLASS:<12} | {status}")

  print("-" * 74)
  print(f"Total Valid Extracted Images:  {total_extracted_images}")
  print(
      f"Total Measured Extracted Size: {total_extracted_bytes:,} bytes"
      f" ({total_extracted_bytes / (1024*1024):.2f} MB)"
  )
  print(f"Skipped Out-of-Scope Classes:  {skipped_out_of_scope} entries")
  print(
      "Minimum Threshold (>= 100/cls):"
      f" {'MET' if meets_minimum_threshold else 'NOT MET'}"
  )
  print(
      "Completeness Target (1,200):   "
      f" {'FULLY COMPLETE' if is_dataset_complete else 'INCOMPLETE / REVIEW REQUIRED'}"
  )
  print(
      "Training Gate Status:          LOCKED (Explicit user review & approval"
      " required before training)"
  )
  print("Archive Retention Status:      Preserved unmodified")
  print("=" * 74 + "\n")

  return {
      "archive_sha256": archive_hash,
      "class_counts": dict(class_counts),
      "total_images": total_extracted_images,
      "total_bytes": total_extracted_bytes,
      "meets_minimum_threshold": meets_minimum_threshold,
      "is_dataset_complete": is_dataset_complete,
      "publication_status": publication_status,
      "authorized_for_training": (
          False  # Training gate remains locked; requires explicit user approval
      ),
      "discrepancies_found": discrepancies_found,
  }


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python safe_extract_isl.py <archive_path> <destination_dir>")
    sys.exit(1)
  archive_file = sys.argv[1]
  dest_directory = sys.argv[2]
  safe_extract_isl(archive_file, dest_directory)
