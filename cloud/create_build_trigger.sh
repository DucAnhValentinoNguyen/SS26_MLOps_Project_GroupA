#!/usr/bin/env bash
# M21 — Continuous containers: auto-rebuild the training image on every push to
# `main`, via a Cloud Build 2nd-gen GitHub trigger in europe-west4 (matching the
# rest of the project). Reuses cloud/cloudbuild.train.yaml.
#
# ── PREREQUISITES ──
#  * cloud/cloudbuild.train.yaml on origin/main — the trigger reads the build
#    config from the repo at build time (already pushed).
#  * The Cloud Build P4SA needs Secret Manager perms (2nd-gen stores the GitHub
#    OAuth token there). One-time:
#      gcloud projects add-iam-policy-binding "$PROJECT" \
#        --member="serviceAccount:service-<PROJECT_NUMBER>@gcp-sa-cloudbuild.iam.gserviceaccount.com" \
#        --role=roles/secretmanager.admin --condition=None
#  * You need ADMIN on the GitHub repo to install the Cloud Build GitHub App.
#
# Idempotent. The GitHub OAuth can't be scripted: on the first run this creates
# the connection and prints the authorization link; after you authorize in the
# browser, re-run it to link the repo and create the trigger.
set -euo pipefail

PROJECT="${PROJECT:-paligemma-scienceqa}"
REGION="${REGION:-europe-west4}"            # 2nd-gen trigger lives in this region
CONNECTION="${CONNECTION:-paligemma-gh}"
REPO_RES="${REPO_RES:-paligemma-repo}"
REPO_OWNER="${REPO_OWNER:-DucAnhValentinoNguyen}"
REPO_NAME="${REPO_NAME:-SS26_MLOps_Project_GroupA}"
REMOTE_URI="https://github.com/${REPO_OWNER}/${REPO_NAME}.git"

# 1. Host connection (GitHub App OAuth).
if ! gcloud builds connections describe "$CONNECTION" \
       --region="$REGION" --project="$PROJECT" >/dev/null 2>&1; then
  echo ">>> creating Cloud Build GitHub connection '$CONNECTION' in $REGION ..."
  gcloud builds connections create github "$CONNECTION" \
    --region="$REGION" --project="$PROJECT"
fi

STAGE="$(gcloud builds connections describe "$CONNECTION" \
  --region="$REGION" --project="$PROJECT" \
  --format='value(installationState.stage)' 2>/dev/null || true)"
if [ "$STAGE" != "COMPLETE" ]; then
  echo
  echo "!!! ACTION REQUIRED — authorize Cloud Build to access GitHub (needs admin"
  echo "    on ${REPO_OWNER}/${REPO_NAME}). Open and complete this link, then re-run:"
  gcloud builds connections describe "$CONNECTION" \
    --region="$REGION" --project="$PROJECT" \
    --format='value(installationState.actionUri)'
  exit 0
fi

# 2. Link the specific repository to the connection.
if ! gcloud builds repositories describe "$REPO_RES" \
       --connection="$CONNECTION" --region="$REGION" --project="$PROJECT" >/dev/null 2>&1; then
  echo ">>> linking repository ${REPO_OWNER}/${REPO_NAME} ..."
  gcloud builds repositories create "$REPO_RES" \
    --remote-uri="$REMOTE_URI" --connection="$CONNECTION" \
    --region="$REGION" --project="$PROJECT"
fi

# 3. Create the push trigger (rebuilds only on image-relevant changes).
if ! gcloud builds triggers describe paligemma-train-build \
       --region="$REGION" --project="$PROJECT" >/dev/null 2>&1; then
  echo ">>> creating trigger 'paligemma-train-build' ..."
  gcloud builds triggers create github \
    --name=paligemma-train-build \
    --project="$PROJECT" --region="$REGION" \
    --repository="projects/${PROJECT}/locations/${REGION}/connections/${CONNECTION}/repositories/${REPO_RES}" \
    --branch-pattern='^main$' \
    --build-config=cloud/cloudbuild.train.yaml \
    --included-files='dockerfiles/train.dockerfile,src/**,pyproject.toml,uv.lock'
fi

echo
echo "Trigger ready. Fire it once now to verify (without waiting for a push):"
echo "  gcloud builds triggers run paligemma-train-build --region=$REGION --branch=main --project=$PROJECT"
