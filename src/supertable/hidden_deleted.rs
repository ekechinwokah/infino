// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Consolidated deleted-user-`_id` set for the hidden vector-index table.
//!
//! User deletes tombstone only the user table; hidden cell superfiles keep
//! deleted rows physically present until drain/compaction removes them. The
//! hidden manifest fast payload carries this encoded set inline, so vector
//! search consults resident manifest bytes and never performs a deleted-set
//! GET.

use std::sync::Arc;

use crate::supertable::manifest::ManifestSnapshot;

/// Magic prefix on a packed deleted-user-`_id` set.
const DELETED_IDS_MAGIC: &[u8; 4] = b"HDEL";

/// Wire-format version for [`DELETED_IDS_MAGIC`] blobs.
const DELETED_IDS_VERSION: u8 = 1;

/// Header: magic (4) + version (1) + count (4).
const DELETED_IDS_HEADER_LEN: usize = 4 + 1 + 4;

/// Bytes per serialized `_id` (a little-endian `i128`).
const DELETED_ID_LEN: usize = 16;

/// Serialize the consolidated deleted user-`_id` set. The caller passes a
/// sorted, deduplicated slice so the on-disk order is canonical.
pub(crate) fn encode_deleted_ids(sorted_ids: &[i128]) -> Vec<u8> {
    let mut out = Vec::with_capacity(DELETED_IDS_HEADER_LEN + sorted_ids.len() * DELETED_ID_LEN);
    out.extend_from_slice(DELETED_IDS_MAGIC);
    out.push(DELETED_IDS_VERSION);
    out.extend_from_slice(&(sorted_ids.len() as u32).to_le_bytes());
    for id in sorted_ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum HiddenDeletedError {
    #[error("deleted-id set truncated")]
    Truncated,
    #[error("deleted-id set bad magic")]
    BadMagic,
    #[error("deleted-id set unsupported version {0}")]
    UnsupportedVersion(u8),
}

/// Parse a deleted-`_id` set written by [`encode_deleted_ids`].
pub(crate) fn decode_deleted_ids(bytes: &[u8]) -> Result<Vec<i128>, HiddenDeletedError> {
    if bytes.len() < DELETED_IDS_HEADER_LEN {
        return Err(HiddenDeletedError::Truncated);
    }
    if &bytes[0..4] != DELETED_IDS_MAGIC {
        return Err(HiddenDeletedError::BadMagic);
    }
    let version = bytes[4];
    if version != DELETED_IDS_VERSION {
        return Err(HiddenDeletedError::UnsupportedVersion(version));
    }
    let count = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
    let body = &bytes[DELETED_IDS_HEADER_LEN..];
    if body.len() != count * DELETED_ID_LEN {
        return Err(HiddenDeletedError::Truncated);
    }
    let mut ids = Vec::with_capacity(count);
    for chunk in body.chunks_exact(DELETED_ID_LEN) {
        let mut buf = [0u8; DELETED_ID_LEN];
        buf.copy_from_slice(chunk);
        ids.push(i128::from_le_bytes(buf));
    }
    Ok(ids)
}

/// Decode the hidden index's resident deleted user-`_id` set from the
/// manifest. Returns an empty set when none is stamped. There is deliberately
/// no storage fallback here: the two-blob contract requires this state to ride
/// in the hidden manifest fast payload.
pub(crate) fn deleted_user_ids(manifest: &ManifestSnapshot) -> Result<Arc<Vec<i128>>, HiddenDeletedError> {
    let Some(bytes) = manifest.deleted_user_ids_inline() else {
        return Ok(Arc::new(Vec::new()));
    };
    Ok(Arc::new(decode_deleted_ids(bytes)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deleted_ids_encode_decode_roundtrip() {
        let ids: Vec<i128> = vec![i128::MIN, -1, 0, 1, 42, 1 << 100, i128::MAX];
        let bytes = encode_deleted_ids(&ids);
        assert_eq!(decode_deleted_ids(&bytes).expect("decode"), ids);
        assert!(decode_deleted_ids(&[]).is_err());
        assert!(
            decode_deleted_ids(&encode_deleted_ids(&[]))
                .expect("empty")
                .is_empty()
        );
    }
}
