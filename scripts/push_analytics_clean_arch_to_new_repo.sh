#!/usr/bin/env bash
set -euo pipefail

NEW_REMOTE_URL="${1:-https://github.com/mainarkler/Progect_analitycs.git}"
BRANCH_NAME="${2:-main}"
SPLIT_BRANCH="tmp/analytics_clean_arch_split"

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

if [[ ! -d "analytics_clean_arch" ]]; then
  echo "ERROR: folder analytics_clean_arch not found in $(pwd)"
  exit 1
fi

echo "[1/5] Creating split branch from subdirectory analytics_clean_arch..."
# Recreate split branch each run for idempotence
if git show-ref --verify --quiet "refs/heads/${SPLIT_BRANCH}"; then
  git branch -D "${SPLIT_BRANCH}" >/dev/null 2>&1 || true
fi

git subtree split --prefix=analytics_clean_arch -b "${SPLIT_BRANCH}" >/dev/null

TEMP_REMOTE_NAME="newrepo-temp"
if git remote | grep -q "^${TEMP_REMOTE_NAME}$"; then
  git remote remove "${TEMP_REMOTE_NAME}"
fi

echo "[2/5] Adding temporary remote: ${NEW_REMOTE_URL}"
git remote add "${TEMP_REMOTE_NAME}" "${NEW_REMOTE_URL}"

echo "[3/5] Pushing split branch to ${NEW_REMOTE_URL}:${BRANCH_NAME} ..."
set +e
git push "${TEMP_REMOTE_NAME}" "${SPLIT_BRANCH}:${BRANCH_NAME}"
PUSH_EXIT=$?
set -e

echo "[4/5] Cleaning up temporary remote and branch..."
git remote remove "${TEMP_REMOTE_NAME}"
git branch -D "${SPLIT_BRANCH}" >/dev/null 2>&1 || true

if [[ ${PUSH_EXIT} -ne 0 ]]; then
  echo "[5/5] Push failed (likely auth/permissions)."
  echo "Run manually after auth setup:"
  echo "  git push ${NEW_REMOTE_URL} ${SPLIT_BRANCH}:${BRANCH_NAME}"
  exit ${PUSH_EXIT}
fi

echo "[5/5] Done. analytics_clean_arch has been pushed as standalone project to ${NEW_REMOTE_URL} (${BRANCH_NAME})."
