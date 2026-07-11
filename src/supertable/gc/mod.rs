// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use std::{
    collections::HashSet,
    time::{Duration, SystemTime},
};

use tracing::{debug, warn};

use crate::{
    Supertable,
    runtime_bridge::bridge_on_runtime,
    supertable::{
        ManifestSnapshot,
        error::GcError,
        handle::SupertableInner,
        manifest::{
            SUPERFILE_DATA_DIR,
            commit::{MANIFEST_DIR, MANIFEST_PARTS_DIR, POINTER_PATH, manifest_uri},
        },
        slow_vector_state::STORAGE_PREFIX as SLOW_VECTOR_STATE_STORAGE_PREFIX,
    },
};

/// Minimum age of a storage object before [`gc_storage_sweep_for_inner`] may
/// delete it. Sized so snapshot-pinned readers can finish cold fetches against
/// superseded superfiles after a manifest swap.
#[cfg_attr(test, allow(dead_code))]
pub(crate) const DEFAULT_SUPERFILE_RECLAIM_GRACE: Duration = Duration::from_secs(5 * 60);

#[derive(Debug, Default, Clone)]
pub struct GcReport {
    pub objects_deleted: u64,
    pub bytes_freed: u64,
    pub objects_skipped_live: u64,
    pub objects_skipped_too_new: u64,
    pub delete_errors: u64,
}

fn build_live_set(manifest: &ManifestSnapshot) -> HashSet<String> {
    let mut live = HashSet::new();
    live.insert(POINTER_PATH.to_string());
    live.insert(manifest_uri(manifest.manifest_id));
    for entry in manifest.get_all_list_entries() {
        live.insert(entry.uri.clone());
    }
    for sf in manifest.get_all_superfiles() {
        live.insert(sf.uri.storage_path());
    }
    // Slow-CAS entry blob: the URI is read straight off the manifest-list
    // ref — sync, no fetch. Superseded blobs (older drains) are absent from
    // the current list and get swept once past the safety gap.
    if let Some((uri, _)) = manifest.slow_vector_state_blob() {
        live.insert(uri.to_owned());
    }
    live
}

impl Supertable {
    pub fn gc(&self, safety_gap: Duration) -> Result<GcReport, GcError> {
        bridge_on_runtime(self.gc_async(safety_gap), &self.inner().query_runtime())
    }

    pub(crate) async fn gc_async(&self, safety_gap: Duration) -> Result<GcReport, GcError> {
        gc_storage_sweep_for_inner(self.inner(), safety_gap).await
    }
}

/// Delete storage objects not referenced by the current manifest once they are
/// older than `safety_gap`. Supersedes inline post-commit deletes so readers
/// pinned to an older snapshot cannot lose bytes mid-fetch.
pub(super) async fn gc_storage_sweep_for_inner(
    inner: &SupertableInner,
    safety_gap: Duration,
) -> Result<GcReport, GcError> {
    let storage = inner.options.storage.clone().ok_or(GcError::NoStorage)?;
    let manifest = inner.manifest.load_full();
    let live = build_live_set(&manifest);
    let cutoff = SystemTime::now()
        .checked_sub(safety_gap)
        .unwrap_or(SystemTime::UNIX_EPOCH);

    let mut report = GcReport::default();

    for prefix in [
        MANIFEST_DIR,
        MANIFEST_PARTS_DIR,
        SUPERFILE_DATA_DIR,
        SLOW_VECTOR_STATE_STORAGE_PREFIX,
    ] {
        let entries = storage.list_with_prefix_metadata(prefix).await?;
        for (key, meta) in entries {
            if live.contains(&key) {
                report.objects_skipped_live += 1;
                continue;
            }
            if meta.last_modified >= cutoff {
                report.objects_skipped_too_new += 1;
                continue;
            }
            match storage.delete(&key).await {
                Ok(()) => {
                    report.objects_deleted += 1;
                    report.bytes_freed += meta.size;
                }
                Err(e) => {
                    warn!(object = %key, error = %e, "gc: failed to delete orphan object");
                    report.delete_errors += 1;
                }
            }
        }
    }

    debug!(
        deleted = report.objects_deleted,
        bytes_freed = report.bytes_freed,
        delete_errors = report.delete_errors,
        "gc sweep complete"
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use tempfile::tempdir;
    use uuid::Uuid;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{
            SupertableOptions,
            manifest::{
                ManifestSnapshot, SuperfileEntry, SuperfileUri,
                list::{FORMAT_VERSION, Manifest, PartitionStrategy},
                part::ContentHash,
            },
            slow_vector_state,
        },
        test_helpers::default_supertable_options,
    };

    /// Bucket count for a minimal hash-partitioned manifest list fixture.
    const TEST_HASH_BUCKETS: u32 = 1;

    /// ManifestSnapshot id for a single-list live-set fixture.
    const TEST_MANIFEST_ID: u64 = 0;

    fn opts() -> Arc<SupertableOptions> {
        Arc::new(default_supertable_options())
    }

    fn sf_entry(uri: SuperfileUri) -> Arc<SuperfileEntry> {
        Arc::new(SuperfileEntry {
            birth_version: 0,
            superfile_id: Uuid::new_v4(),
            uri,
            n_docs: 1,
            id_min: 0,
            id_max: 0,
            scalar_stats: HashMap::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: vec![],
            partition_hint: None,
            vector_layout: crate::superfile::vector::layout::VectorLayout::Ivf,
            subsection_offsets: None,
        })
    }

    #[test]
    fn build_live_set_contains_pointer_and_manifest_uri() {
        let manifest = ManifestSnapshot::empty(opts());
        let live = build_live_set(&manifest);
        assert!(live.contains(POINTER_PATH));
        assert!(live.contains(&manifest_uri(manifest.manifest_id)));
    }

    #[test]
    fn build_live_set_contains_superfile_uris() {
        let uri = SuperfileUri::new_v4();
        let manifest = ManifestSnapshot::empty(opts()).with_appended(vec![sf_entry(uri)]);
        let live = build_live_set(&manifest);
        assert!(live.contains(&uri.storage_path()));
    }

    #[test]
    fn build_live_set_does_not_contain_older_manifest_uris() {
        let uri = SuperfileUri::new_v4();
        let manifest = ManifestSnapshot::empty(opts()).with_appended(vec![sf_entry(uri)]);
        assert_eq!(manifest.manifest_id, 1);
        let live = build_live_set(&manifest);
        assert!(!live.contains(&manifest_uri(0)));
        assert!(!live.contains(&manifest_uri(2)));
    }

    /// The slow-CAS entry blob referenced from the list is live; anything
    /// else under its prefix (superseded drains, orphans from a crash
    /// between PUT and stamp) is sweepable, and a ref-less manifest keeps
    /// nothing there.
    #[test]
    fn build_live_set_contains_slow_vector_state_blob() {
        let dir = tempdir().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let hash = ContentHash::of(b"slow state");
        let uri = slow_vector_state::storage_path(&hash);
        let orphan = slow_vector_state::storage_path(&ContentHash::of(b"orphan"));
        let manifest = ManifestSnapshot::new(
            TEST_MANIFEST_ID,
            opts(),
            Vec::new(),
            Some(storage),
            Some(Manifest {
                format_version: FORMAT_VERSION.into(),
                manifest_id: TEST_MANIFEST_ID,
                options_hash: ContentHash::of(b"options"),
                schema: Vec::new(),
                id_column: "_id".into(),
                fts_columns: Vec::new(),
                vector_columns: Vec::new(),
                partition_strategy: PartitionStrategy::Hash {
                    column: "_id".into(),
                    n_buckets: TEST_HASH_BUCKETS,
                },
                vector_index_storage_prefix: None,
                global_vector_index: None,
                drained_ranges: Default::default(),
                deleted_user_ids_inline: None,
                slow_vector_state_uri: Some(uri.clone()),
                slow_vector_state_content_hash: Some(hash),
                parts: Vec::new(),
            }),
        );
        let live = build_live_set(&manifest);
        assert!(live.contains(&uri), "referenced blob must be live");
        assert!(
            !live.contains(&orphan),
            "unreferenced blob must be sweepable"
        );

        // A manifest without a ref keeps nothing under the prefix live.
        let bare = ManifestSnapshot::empty(opts());
        let live = build_live_set(&bare);
        assert!(!live.contains(&uri));
    }
}
