"""S3-compatible storage client.

Targets Supabase Storage (S3-compatible API) in production and LocalStack in local dev.
Credentials are read from environment variables — never baked in.

Required env vars:
    S3_ENDPOINT_URL      — e.g. http://localhost:4566 (LocalStack) or Supabase S3 URL
    S3_ACCESS_KEY_ID     — access key
    S3_SECRET_ACCESS_KEY — secret key

Optional:
    S3_REGION            — defaults to "us-east-1"
"""

import os
from pathlib import Path

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from loguru import logger


def get_s3_client() -> BaseClient:
    """Build a boto3 S3 client from environment variables.

    Uses path-style addressing, which is required for LocalStack and Supabase Storage.
    Timeouts are kept short and retries use botocore's "standard" mode (exponential
    backoff with jitter) so a hung storage endpoint fails in about a minute instead of ~5.

    Raises:
        KeyError: If S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, or S3_SECRET_ACCESS_KEY are unset.
    """
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(
            s3={"addressing_style": "path"},
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def upload_file(
    client: BaseClient, local_path: Path, bucket: str, key: str, content_type: str
) -> None:
    """Upload a local file to S3.

    Args:
        client: A boto3 S3 client (from get_s3_client()).
        local_path: Path to the local file to upload.
        bucket: Target S3 bucket name.
        key: Object key (path) within the bucket.
        content_type: MIME type for the object (e.g. "application/x-ndjson", "text/plain").

    Raises:
        botocore.exceptions.BotoCoreError: On upload failure.
    """
    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=f,
            ContentType=content_type,
        )
    logger.info(f"Uploaded {local_path} → s3://{bucket}/{key}")
