// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Counts object-store requests (HEAD / GET / PUT, including multipart
//! parts) and byte volumes during a bench window. The cost model prices
//! each lifecycle phase (ingest, drain, cold open, per-query fetch) from
//! these measured counts — never from estimates.
//!
//! One blind spot, by design: reads issued through
//! `StorageProvider::object_store_handle` hand the caller the raw inner
//! `ObjectStore`, bypassing this wrapper, so requests on that escape
//! hatch are not counted. Today that path serves row materialization
//! (`take_rows_object_store`); the `_id` + `score` search shapes the
//! metered iterations run do not hit it.

use std::{
    fmt,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use async_trait::async_trait;
use bytes::Bytes;
use infino::storage::{ObjectMeta, StorageError, StorageProvider};
use object_store::{
    MultipartUpload, PutPayload, PutResult, Result as ObjectStoreResult, UploadPart,
};

/// Request + byte counts observed in one metering window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ObjectStoreMeter {
    pub head_count: u64,
    pub get_count: u64,
    pub get_bytes: u64,
    pub put_count: u64,
    pub put_bytes: u64,
}

impl ObjectStoreMeter {
    /// Counts accumulated since an `earlier` snapshot of the same meter —
    /// the per-phase delta the cost model prices.
    pub fn since(&self, earlier: &ObjectStoreMeter) -> ObjectStoreMeter {
        ObjectStoreMeter {
            head_count: self.head_count.saturating_sub(earlier.head_count),
            get_count: self.get_count.saturating_sub(earlier.get_count),
            get_bytes: self.get_bytes.saturating_sub(earlier.get_bytes),
            put_count: self.put_count.saturating_sub(earlier.put_count),
            put_bytes: self.put_bytes.saturating_sub(earlier.put_bytes),
        }
    }

    /// Read-class requests (HEAD + GET) — billed at the GET rate.
    pub fn read_requests(&self) -> u64 {
        self.head_count + self.get_count
    }
}

/// One cold consumer's metered windows, split at the phase boundaries the
/// cost model prices separately: the one-time table open, the first query
/// on the cold cache (the per-query cold fetch), and the same query
/// repeated immediately on the same fresh consumer — a cache fill-lag
/// probe, *not* a steady-state warm number (steady-state warm is metered
/// separately on a cache-hot consumer).
#[derive(Debug, Clone, Copy)]
pub struct ColdStoreSplit {
    pub open: ObjectStoreMeter,
    pub first_query: ObjectStoreMeter,
    pub repeat_query: ObjectStoreMeter,
}

struct MeterCounters {
    head_count: AtomicU64,
    get_count: AtomicU64,
    get_bytes: AtomicU64,
    put_count: AtomicU64,
    put_bytes: AtomicU64,
}

impl MeterCounters {
    fn snapshot(&self) -> ObjectStoreMeter {
        ObjectStoreMeter {
            head_count: self.head_count.load(Ordering::Relaxed),
            get_count: self.get_count.load(Ordering::Relaxed),
            get_bytes: self.get_bytes.load(Ordering::Relaxed),
            put_count: self.put_count.load(Ordering::Relaxed),
            put_bytes: self.put_bytes.load(Ordering::Relaxed),
        }
    }

    fn record_get(&self, bytes: u64) {
        self.get_count.fetch_add(1, Ordering::Relaxed);
        self.get_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_put(&self, bytes: u64) {
        self.put_count.fetch_add(1, Ordering::Relaxed);
        self.put_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// Storage provider wrapper that meters request counts and byte volumes.
pub struct MeteredStorage {
    provider: Arc<dyn StorageProvider>,
    counters: Arc<MeterCounters>,
}

struct CountingStorage {
    inner: Arc<dyn StorageProvider>,
    counters: Arc<MeterCounters>,
}

impl CountingStorage {
    fn new(inner: Arc<dyn StorageProvider>, counters: Arc<MeterCounters>) -> Self {
        Self { inner, counters }
    }
}

impl fmt::Debug for CountingStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountingStorage").finish_non_exhaustive()
    }
}

pub fn wrap(storage: Arc<dyn StorageProvider>) -> MeteredStorage {
    let counters = Arc::new(MeterCounters {
        head_count: AtomicU64::new(0),
        get_count: AtomicU64::new(0),
        get_bytes: AtomicU64::new(0),
        put_count: AtomicU64::new(0),
        put_bytes: AtomicU64::new(0),
    });
    let provider: Arc<dyn StorageProvider> =
        Arc::new(CountingStorage::new(storage, Arc::clone(&counters)));
    MeteredStorage { provider, counters }
}

impl MeteredStorage {
    pub fn provider(&self) -> Arc<dyn StorageProvider> {
        Arc::clone(&self.provider)
    }

    pub fn snapshot(&self) -> ObjectStoreMeter {
        self.counters.snapshot()
    }
}

/// Multipart-upload wrapper: each part is one billable PUT, and the
/// completion call is one more (matches S3 `UploadPart` +
/// `CompleteMultipartUpload` billing; the create call is counted by
/// [`CountingStorage::put_multipart`]). Aborts are failure-path cleanup
/// and are left uncounted.
struct CountingUpload {
    inner: Box<dyn MultipartUpload>,
    counters: Arc<MeterCounters>,
}

impl fmt::Debug for CountingUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountingUpload").finish_non_exhaustive()
    }
}

#[async_trait]
impl MultipartUpload for CountingUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        self.counters.record_put(data.content_length() as u64);
        self.inner.put_part(data)
    }

    async fn complete(&mut self) -> ObjectStoreResult<PutResult> {
        self.counters.record_put(0);
        self.inner.complete().await
    }

    async fn abort(&mut self) -> ObjectStoreResult<()> {
        self.inner.abort().await
    }
}

#[async_trait]
impl StorageProvider for CountingStorage {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.counters.head_count.fetch_add(1, Ordering::Relaxed);
        self.inner.head(uri).await
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        let (bytes, meta) = self.inner.get(uri).await?;
        self.counters.record_get(bytes.len() as u64);
        Ok((bytes, meta))
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let bytes = self.inner.get_range(uri, range).await?;
        self.counters.record_get(bytes.len() as u64);
        Ok(bytes)
    }

    async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
        let (bytes, size) = self.inner.tail(uri, len).await?;
        self.counters.record_get(bytes.len() as u64);
        Ok((bytes, size))
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        self.counters.record_put(bytes.len() as u64);
        self.inner.put_atomic(uri, bytes).await
    }

    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        self.counters.record_put(bytes.len() as u64);
        self.inner.put_if_match(uri, bytes, expected_etag).await
    }

    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        // The create call itself is a billable request.
        self.counters.record_put(0);
        let inner = self.inner.put_multipart(uri).await?;
        Ok(Box::new(CountingUpload {
            inner,
            counters: Arc::clone(&self.counters),
        }))
    }

    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.inner.delete(uri).await
    }

    async fn list_with_prefix(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        self.inner.list_with_prefix(prefix).await
    }

    fn object_store_handle(
        &self,
        uri: &str,
    ) -> Option<(Arc<dyn object_store::ObjectStore>, object_store::path::Path)> {
        self.inner.object_store_handle(uri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn since_subtracts_fieldwise_and_saturates() {
        let earlier = ObjectStoreMeter {
            head_count: 1,
            get_count: 10,
            get_bytes: 100,
            put_count: 5,
            put_bytes: 50,
        };
        let later = ObjectStoreMeter {
            head_count: 1,
            get_count: 25,
            get_bytes: 400,
            put_count: 9,
            put_bytes: 90,
        };
        let delta = later.since(&earlier);
        assert_eq!(delta.head_count, 0);
        assert_eq!(delta.get_count, 15);
        assert_eq!(delta.get_bytes, 300);
        assert_eq!(delta.put_count, 4);
        assert_eq!(delta.put_bytes, 40);
        // Windows never run backwards; saturate instead of wrapping if a
        // caller ever crosses snapshots.
        assert_eq!(earlier.since(&later).get_count, 0);
    }

    #[test]
    fn read_requests_sums_head_and_get() {
        let m = ObjectStoreMeter {
            head_count: 2,
            get_count: 3,
            ..Default::default()
        };
        assert_eq!(m.read_requests(), 5);
    }
}
