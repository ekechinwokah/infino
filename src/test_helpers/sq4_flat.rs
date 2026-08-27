// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Test/bench access to the flat 4-bit vector index.
//!
//! The index itself is engine code — [`crate::superfile::vector::flat`] —
//! not a measurement fixture. This module exists only to expose it to
//! benches and integration tests under the `test-helpers` feature, so a
//! harness can construct one from an fp32 corpus without the crate widening
//! its public API.
//!
//! It deliberately holds no logic. An earlier revision carried a scan of its
//! own here, which meant the published 4-bit numbers came from a path no
//! table could be configured to use.

pub use crate::superfile::vector::flat::Sq4FlatIndex;
