// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Order-preserving scalar key encodings for the hidden **scalar-index
//! family** — value-organized shards in the ONE hidden supertable that
//! answer SQL predicates on indexed columns with a dictionary probe
//! plus one contiguous `_id` span read.
//!
//! The family is the text family with this encoder in front of it: the
//! drain merge-sorts a column's `(encoded_value, _id)` pairs and emits
//! value-range shards through the existing FTS merge builder, so the
//! shard dictionary, split logic, manifest bloom/range routing,
//! `drained_ranges` watermark, and HDEL filtering are all reused. What
//! the encoder must guarantee is exactly one property:
//!
//! > `encode(a) < encode(b)` (bytewise) **iff** `a < b` (typed),
//!
//! so numeric/temporal **range predicates become contiguous key
//! ranges** in the dictionary — the reason tokenized values can't
//! serve ranges (`"9" > "100"` lexicographically).
//!
//! Encodings (all big-endian so bytewise order is value order):
//! - signed integers: two's-complement with the sign bit flipped
//!   (`i64 → u64 ^ SIGN_FLIP_U64`), so negatives sort below positives;
//! - unsigned integers: raw big-endian;
//! - floats: the classic IEEE-754 total-order trick — positive values
//!   set the sign bit, negative values flip ALL bits — which orders
//!   `-∞ < … < -0.0 < +0.0 < … < +∞ < NaN` (`-0.0` is normalized to
//!   `+0.0` first so the two compare equal instead of adjacent);
//! - decimal128: the i128 sign-flip encoding (scale is a column
//!   property — one column has one scale, so raw integer order is
//!   value order);
//! - strings: raw UTF-8 bytes (UTF-8 bytewise order is code-point
//!   order);
//! - booleans / dates / timestamps: their integer representations.
//!
//! **NULLs are not encoded.** SQL predicates never match NULL
//! (`x = 5`, `x < 5`, `x IN (…)` are all false or unknown for NULL),
//! so NULL rows are simply absent from the index; the shard summary
//! carries a null count so covering `COUNT(*)` stays exact.

use arrow_array::{
    Array, BooleanArray, Date32Array, Date64Array, Decimal128Array, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray, StringArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, TimeUnit};
use datafusion::scalar::ScalarValue;

/// Reserved column-name prefix for scalar-index columns inside hidden
/// text shards: `inf.sidx.<user column>`. It builds on the options
/// layer's reserved `inf.` prefix — user column names can never start
/// with it — so index columns cannot collide with user FTS columns.
pub(crate) const SCALAR_INDEX_COLUMN_PREFIX: &str = "inf.sidx.";

/// The hidden-blob column name carrying `column`'s scalar index.
pub(crate) fn scalar_index_column_name(column: &str) -> String {
    format!("{SCALAR_INDEX_COLUMN_PREFIX}{column}")
}

/// Whether `name` is a scalar-index column (reserved-prefix test).
pub(crate) fn is_scalar_index_column(name: &str) -> bool {
    name.starts_with(SCALAR_INDEX_COLUMN_PREFIX)
}

/// XOR mask that flips a two's-complement sign bit so signed values
/// sort unsigned-bytewise: negatives land below positives.
const SIGN_FLIP_U64: u64 = 1 << 63;
/// [`SIGN_FLIP_U64`] for 128-bit values (decimal128 storage).
const SIGN_FLIP_U128: u128 = 1 << 127;

/// Encode an `i64` (also dates / timestamps stored as i64 units).
pub(crate) fn encode_i64(v: i64) -> [u8; 8] {
    ((v as u64) ^ SIGN_FLIP_U64).to_be_bytes()
}

/// Encode a `u64`.
pub(crate) fn encode_u64(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

/// Encode an `i128` (decimal128 storage integer; the column's scale is
/// fixed, so integer order is value order).
pub(crate) fn encode_i128(v: i128) -> [u8; 16] {
    ((v as u128) ^ SIGN_FLIP_U128).to_be_bytes()
}

/// Encode an `f64` in IEEE-754 total order:
/// `-∞ < … < -0.0=+0.0 < … < +∞ < NaN`.
///
/// `-0.0` normalizes to `+0.0` so the two encode identically (SQL
/// treats them as equal); every NaN bit pattern is canonicalized to
/// one key that sorts above `+∞`, so a NaN-carrying row is indexed
/// deterministically instead of scattering across payload bits.
pub(crate) fn encode_f64(v: f64) -> [u8; 8] {
    let v = if v.is_nan() {
        f64::NAN
    } else if v == 0.0 {
        0.0
    } else {
        v
    };
    let bits = v.to_bits();
    let ordered = if bits & SIGN_FLIP_U64 == 0 {
        bits | SIGN_FLIP_U64
    } else {
        !bits
    };
    ordered.to_be_bytes()
}

/// Encode an `f32` by widening: `f32 → f64` is exact, so one key
/// space serves both float widths.
pub(crate) fn encode_f32(v: f32) -> [u8; 8] {
    encode_f64(f64::from(v))
}

/// Encode a boolean (`false < true`).
pub(crate) fn encode_bool(v: bool) -> [u8; 1] {
    [u8::from(v)]
}

/// Encode a string: raw UTF-8 (bytewise order == code-point order).
pub(crate) fn encode_str(v: &str) -> &[u8] {
    v.as_bytes()
}

/// Uppercase-hex alphabet for FST-safe keys.
const HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";

/// Order-preserving, FST-safe key: uppercase hex of `bytes`.
///
/// The dictionary layer requires UTF-8 term keys and reserves
/// `FST_SEPARATOR` (`0x1F`); raw big-endian encodings satisfy
/// neither. Uppercase hex is UTF-8, contains no reserved bytes, and
/// preserves bytewise order (hex digits ascend `0-9 < A-F` in ASCII)
/// at 2x the key bytes — largely absorbed by the dictionary's prefix
/// compression. Range-predicate bounds must pass through the same
/// hex step so bound comparisons stay in one key space.
pub(crate) fn hex_key(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX_UPPER[usize::from(b >> 4)]);
        out.push(HEX_UPPER[usize::from(b & 0x0F)]);
    }
    out
}

/// Encode row `row` of `array` as a scalar-index dictionary key
/// (order-preserving bytes, hex-armored via [`hex_key`]), or `None`
/// for a NULL row (SQL predicates never match NULL, so NULL rows are
/// simply absent from the index; shard summaries carry the null
/// count).
///
/// Callers guarantee the array's type passed the options layer's
/// `scalar_index_type_supported` check; an unsupported type here is a
/// contract violation and returns `None` (the row is skipped, never
/// mis-encoded).
pub(crate) fn encode_array_value(array: &dyn Array, row: usize) -> Option<Vec<u8>> {
    if array.is_null(row) {
        return None;
    }
    let any = array.as_any();
    let key: Vec<u8> = match array.data_type() {
        DataType::Int8 => encode_i64(i64::from(any.downcast_ref::<Int8Array>()?.value(row))).into(),
        DataType::Int16 => {
            encode_i64(i64::from(any.downcast_ref::<Int16Array>()?.value(row))).into()
        }
        DataType::Int32 => {
            encode_i64(i64::from(any.downcast_ref::<Int32Array>()?.value(row))).into()
        }
        DataType::Int64 => encode_i64(any.downcast_ref::<Int64Array>()?.value(row)).into(),
        DataType::UInt8 => {
            encode_u64(u64::from(any.downcast_ref::<UInt8Array>()?.value(row))).into()
        }
        DataType::UInt16 => {
            encode_u64(u64::from(any.downcast_ref::<UInt16Array>()?.value(row))).into()
        }
        DataType::UInt32 => {
            encode_u64(u64::from(any.downcast_ref::<UInt32Array>()?.value(row))).into()
        }
        DataType::UInt64 => encode_u64(any.downcast_ref::<UInt64Array>()?.value(row)).into(),
        DataType::Float32 => encode_f32(any.downcast_ref::<Float32Array>()?.value(row)).into(),
        DataType::Float64 => encode_f64(any.downcast_ref::<Float64Array>()?.value(row)).into(),
        DataType::Boolean => encode_bool(any.downcast_ref::<BooleanArray>()?.value(row)).into(),
        DataType::Utf8 => encode_str(any.downcast_ref::<StringArray>()?.value(row)).to_vec(),
        DataType::LargeUtf8 => {
            encode_str(any.downcast_ref::<LargeStringArray>()?.value(row)).to_vec()
        }
        DataType::Date32 => {
            encode_i64(i64::from(any.downcast_ref::<Date32Array>()?.value(row))).into()
        }
        DataType::Date64 => encode_i64(any.downcast_ref::<Date64Array>()?.value(row)).into(),
        DataType::Timestamp(unit, _) => {
            // One column has one unit + zone, so raw i64 order is
            // instant order within the index.
            let v = match unit {
                TimeUnit::Second => any.downcast_ref::<TimestampSecondArray>()?.value(row),
                TimeUnit::Millisecond => {
                    any.downcast_ref::<TimestampMillisecondArray>()?.value(row)
                }
                TimeUnit::Microsecond => {
                    any.downcast_ref::<TimestampMicrosecondArray>()?.value(row)
                }
                TimeUnit::Nanosecond => any.downcast_ref::<TimestampNanosecondArray>()?.value(row),
            };
            encode_i64(v).into()
        }
        DataType::Decimal128(_, _) => {
            encode_i128(any.downcast_ref::<Decimal128Array>()?.value(row)).into()
        }
        _ => return None,
    };
    Some(hex_key(&key))
}

/// Encode a DataFusion literal as a scalar-index dictionary key —
/// the predicate-side twin of [`encode_array_value`], sharing the
/// same per-type encoders so a pushed-down `col = lit` probes exactly
/// the key the drain wrote for matching rows. `None` for NULL
/// literals (no row matches), unsupported types, or a literal whose
/// type doesn't losslessly match the column's encoding family
/// (callers fall back to the stats-pruned scan).
pub(crate) fn encode_literal(value: &ScalarValue) -> Option<Vec<u8>> {
    let key: Vec<u8> = match value {
        ScalarValue::Int8(Some(v)) => encode_i64(i64::from(*v)).into(),
        ScalarValue::Int16(Some(v)) => encode_i64(i64::from(*v)).into(),
        ScalarValue::Int32(Some(v)) => encode_i64(i64::from(*v)).into(),
        ScalarValue::Int64(Some(v)) => encode_i64(*v).into(),
        ScalarValue::UInt8(Some(v)) => encode_u64(u64::from(*v)).into(),
        ScalarValue::UInt16(Some(v)) => encode_u64(u64::from(*v)).into(),
        ScalarValue::UInt32(Some(v)) => encode_u64(u64::from(*v)).into(),
        ScalarValue::UInt64(Some(v)) => encode_u64(*v).into(),
        ScalarValue::Float32(Some(v)) => encode_f32(*v).into(),
        ScalarValue::Float64(Some(v)) => encode_f64(*v).into(),
        ScalarValue::Boolean(Some(v)) => encode_bool(*v).into(),
        ScalarValue::Utf8(Some(v))
        | ScalarValue::LargeUtf8(Some(v))
        | ScalarValue::Utf8View(Some(v)) => encode_str(v).to_vec(),
        ScalarValue::Date32(Some(v)) => encode_i64(i64::from(*v)).into(),
        ScalarValue::Date64(Some(v)) => encode_i64(*v).into(),
        ScalarValue::TimestampSecond(Some(v), _)
        | ScalarValue::TimestampMillisecond(Some(v), _)
        | ScalarValue::TimestampMicrosecond(Some(v), _)
        | ScalarValue::TimestampNanosecond(Some(v), _) => encode_i64(*v).into(),
        ScalarValue::Decimal128(Some(v), _, _) => encode_i128(*v).into(),
        _ => return None,
    };
    Some(hex_key(&key))
}

/// The UTF-8 form of an encoded key, as the dictionary APIs take
/// term strings. Hex keys are ASCII by construction.
pub(crate) fn key_to_term(key: Vec<u8>) -> String {
    String::from_utf8(key).expect("hex keys are ASCII")
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{encode_bool, encode_f64, encode_i64, encode_i128, encode_str, encode_u64};

    /// Bytewise order of encodings must equal typed order — the single
    /// property the whole family rests on.
    fn assert_order<T: Copy + PartialOrd + std::fmt::Debug, const N: usize>(
        a: T,
        b: T,
        enc: impl Fn(T) -> [u8; N],
    ) {
        let (ea, eb) = (enc(a), enc(b));
        match a.partial_cmp(&b) {
            Some(ord) => assert_eq!(ea.cmp(&eb), ord, "order mismatch for {a:?} vs {b:?}"),
            None => unreachable!("callers exclude unordered pairs"),
        }
    }

    proptest! {
        #[test]
        fn i64_order_preserved(a: i64, b: i64) {
            assert_order(a, b, encode_i64);
        }

        #[test]
        fn u64_order_preserved(a: u64, b: u64) {
            assert_order(a, b, encode_u64);
        }

        #[test]
        fn i128_order_preserved(a: i128, b: i128) {
            assert_order(a, b, encode_i128);
        }

        #[test]
        fn f64_order_preserved(a: f64, b: f64) {
            // NaN pairs are unordered under PartialOrd; the encoding's
            // NaN policy is pinned by the unit tests below instead.
            prop_assume!(!a.is_nan() && !b.is_nan());
            assert_order(a, b, encode_f64);
        }

        #[test]
        fn str_order_preserved(a: String, b: String) {
            let ord = a.cmp(&b);
            prop_assert_eq!(encode_str(&a).cmp(encode_str(&b)), ord);
        }
    }

    #[test]
    fn f64_total_order_pins() {
        // -0.0 and +0.0 must encode identically (SQL equality).
        assert_eq!(encode_f64(-0.0), encode_f64(0.0));
        // NaN canonicalizes to one key that sorts above +infinity.
        assert_eq!(encode_f64(f64::NAN), encode_f64(-f64::NAN));
        assert!(encode_f64(f64::NAN) > encode_f64(f64::INFINITY));
        // Spot-check the full ladder.
        let ladder = [
            f64::NEG_INFINITY,
            f64::MIN,
            -1.5,
            -f64::MIN_POSITIVE,
            0.0,
            f64::MIN_POSITIVE,
            1.5,
            f64::MAX,
            f64::INFINITY,
        ];
        for pair in ladder.windows(2) {
            assert!(
                encode_f64(pair[0]) < encode_f64(pair[1]),
                "{} must encode below {}",
                pair[0],
                pair[1]
            );
        }
    }

    proptest! {
        #[test]
        fn hex_key_order_preserved(a: Vec<u8>, b: Vec<u8>) {
            prop_assert_eq!(
                super::hex_key(&a).cmp(&super::hex_key(&b)),
                a.cmp(&b)
            );
        }
    }

    #[test]
    fn hex_key_is_fst_safe() {
        let all_bytes: Vec<u8> = (0..=u8::MAX).collect();
        let key = super::hex_key(&all_bytes);
        assert!(std::str::from_utf8(&key).is_ok(), "hex keys are UTF-8");
        // FST_SEPARATOR (0x1F) can never appear in a hex key.
        assert!(!key.contains(&0x1F));
    }

    #[test]
    fn bool_order_pins() {
        assert!(encode_bool(false) < encode_bool(true));
    }

    #[test]
    fn signed_boundaries_pin() {
        assert!(encode_i64(i64::MIN) < encode_i64(-1));
        assert!(encode_i64(-1) < encode_i64(0));
        assert!(encode_i64(0) < encode_i64(i64::MAX));
        assert!(encode_i128(i128::MIN) < encode_i128(0));
        assert!(encode_i128(0) < encode_i128(i128::MAX));
    }
}
