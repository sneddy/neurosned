#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <release-folder>" >&2
  echo "Example: $0 release_v3" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
arg="$1"

if [[ -d "$arg" ]]; then
  target_dir="$(cd "$arg" && pwd -P)"
elif [[ -d "$script_dir/$arg" ]]; then
  target_dir="$(cd "$script_dir/$arg" && pwd -P)"
else
  echo "Error: folder not found: $arg" >&2
  exit 1
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "Error: latexmk is not installed or not on PATH." >&2
  exit 1
fi

mapfile -t tex_files < <(find "$target_dir" -maxdepth 1 -type f -name '*.tex' -printf '%f\n' | sort)

if [[ ${#tex_files[@]} -eq 0 ]]; then
  echo "Error: no .tex file found in $target_dir" >&2
  exit 1
fi

if [[ ${#tex_files[@]} -eq 1 ]]; then
  tex_file="${tex_files[0]}"
else
  mapfile -t main_tex_files < <(printf '%s\n' "${tex_files[@]}" | grep -E '^main.*\.tex$' || true)
  if [[ ${#main_tex_files[@]} -eq 1 ]]; then
    tex_file="${main_tex_files[0]}"
  else
    echo "Error: multiple .tex files found in $target_dir; keep one file or one main*.tex file." >&2
    printf '  %s\n' "${tex_files[@]}" >&2
    exit 1
  fi
fi

parent_dir="$(cd "$target_dir/.." && pwd -P)"

if [[ ! -d "$parent_dir/img" ]]; then
  echo "Warning: expected image directory not found: $parent_dir/img" >&2
fi

# Compile from inside the release folder so paths such as ../img and
# ../references.bib resolve naturally. TEXINPUTS also lets files include
# img/... from the parent overleaf folder when the .tex file is less explicit.
(
  cd "$target_dir"
  export TEXINPUTS=".:..//:${TEXINPUTS:-}"
  export BIBINPUTS=".:..:${BIBINPUTS:-}"
  export BSTINPUTS=".:..:${BSTINPUTS:-}"
  latexmk -pdf -silent -file-line-error -interaction=nonstopmode -halt-on-error "$tex_file"

  base="${tex_file%.tex}"
  rm -f \
    "$base.aux" \
    "$base.bbl" \
    "$base.blg" \
    "$base.fdb_latexmk" \
    "$base.fls" \
    "$base.out" \
    "$base.toc" \
    "$base.lof" \
    "$base.lot" \
    "$base.synctex.gz"
)

echo "PDF written to: $target_dir/${tex_file%.tex}.pdf"
echo "Build log kept at: $target_dir/${tex_file%.tex}.log"
