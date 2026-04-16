import os
import shutil


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DELIV_ROOT = os.path.join(PROJECT_ROOT, "Deliv")

SRC_DIR = os.path.join(DELIV_ROOT, "src")
NB_DIR = os.path.join(DELIV_ROOT, "notebooks")
REPORT_DIR = os.path.join(DELIV_ROOT, "report")

ALLOWED_EXTS = {".py", ".cl", ".ipynb", ".pdf"}
EXCLUDED_EXTS = {".csv", ".npy", ".npz", ".log", ".pyc"}
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".ipynb_checkpoints",
    "venv",
    ".env",
    "Deliv",
    ".git",
}

REQUIRED_SCRIPT_NAMES = {
    "matmul_multi_device_split.py",
    "matmul_multi_device_split_optimized.py",
    "sgemm_rtx3050_benchmark.py",
    "sgemm_kernel6_advanced.py",
}

SOURCE_HINT_DIRS = [
    os.path.normpath(os.path.join("opencl_examples", "multi_device_analysis")),
    os.path.normpath(os.path.join("opencl_examples", "multi_device_optimized_both")),
    os.path.normpath(os.path.join("opencl_examples", "rtx3050_benchmark")),
    os.path.normpath("opencl_sgemm_advanced"),
]

NOTEBOOK_HINT_DIRS = [
    os.path.normpath(os.path.join("opencl_examples", "multi_device_analysis")),
    os.path.normpath(os.path.join("opencl_examples", "multi_device_optimized_both")),
    os.path.normpath(os.path.join("opencl_examples", "tp0_benchmark")),
]


PROF_FOLDER_TOKEN = os.path.normpath(os.path.join("opencl_examples", "prof_files"))


def normalize_rel(path):
    return os.path.normpath(path)


def is_under(rel_path, rel_dir):
    rel_path = normalize_rel(rel_path)
    rel_dir = normalize_rel(rel_dir)
    return rel_path == rel_dir or rel_path.startswith(rel_dir + os.sep)


def ensure_dirs():
    os.makedirs(SRC_DIR, exist_ok=True)
    os.makedirs(NB_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def is_student_source(rel_path, file_name, ext):
    rel_path_norm = normalize_rel(rel_path)

    if is_under(rel_path_norm, PROF_FOLDER_TOKEN):
        return False

    if file_name.endswith("_runtime.py"):
        return False

    if file_name == "organize_delivery.py":
        return True

    if file_name in REQUIRED_SCRIPT_NAMES:
        return True

    for hint in SOURCE_HINT_DIRS:
        if is_under(rel_path_norm, hint):
            return True

    return False


def is_student_notebook(rel_path):
    rel_path_norm = normalize_rel(rel_path)

    if is_under(rel_path_norm, PROF_FOLDER_TOKEN):
        return False

    for hint in NOTEBOOK_HINT_DIRS:
        if is_under(rel_path_norm, hint):
            return True

    return False


def is_student_report(file_name, rel_path):
    rel_path_norm = normalize_rel(rel_path)
    name = file_name.lower()

    if is_under(rel_path_norm, PROF_FOLDER_TOKEN):
        return False

    if "instructions" in name:
        return False

    return "report" in name


def target_base_for(ext):
    if ext in {".py", ".cl"}:
        return SRC_DIR
    if ext == ".ipynb":
        return NB_DIR
    return REPORT_DIR


def copy_file(src_abs, rel_path, copied_destinations):
    ext = os.path.splitext(src_abs)[1].lower()
    base = target_base_for(ext)

    dest_abs = os.path.join(base, rel_path)
    dest_dir = os.path.dirname(dest_abs)
    os.makedirs(dest_dir, exist_ok=True)

    dest_abs_norm = os.path.normcase(os.path.abspath(dest_abs))
    if dest_abs_norm in copied_destinations:
        print(f"[SKIP] duplicate target: {dest_abs}")
        return False

    if os.path.exists(dest_abs):
        print(f"[SKIP] exists (not overwritten): {dest_abs}")
        copied_destinations.add(dest_abs_norm)
        return False

    shutil.copy2(src_abs, dest_abs)
    copied_destinations.add(dest_abs_norm)
    print(f"[COPY] {src_abs} -> {dest_abs}")
    return True


def should_include(rel_path, file_name):
    ext = os.path.splitext(file_name)[1].lower()

    if ext in EXCLUDED_EXTS:
        return False

    if ext not in ALLOWED_EXTS:
        return False

    if ext in {".py", ".cl"}:
        return is_student_source(rel_path, file_name, ext)

    if ext == ".ipynb":
        return is_student_notebook(rel_path)

    if ext == ".pdf":
        return is_student_report(file_name, rel_path)

    return False


def main():
    ensure_dirs()

    copied = 0
    skipped = 0
    copied_destinations = set()

    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIR_NAMES]

        for file_name in filenames:
            src_abs = os.path.join(dirpath, file_name)
            rel_path = os.path.relpath(src_abs, PROJECT_ROOT)

            if not should_include(rel_path, file_name):
                continue

            if copy_file(src_abs, rel_path, copied_destinations):
                copied += 1
            else:
                skipped += 1

    print("\nSummary")
    print(f"Copied: {copied}")
    print(f"Skipped: {skipped}")
    print(f"Delivery folder: {DELIV_ROOT}")


if __name__ == "__main__":
    main()
