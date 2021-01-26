//! This crate implements a hash table that can be used as is in its binary, on-disk format.
//! The goal is to provide a high performance data structure that can be used without any significant up-front decoding.
//! The implementation makes no assumptions about alignment or endianess of the underlying data,
//! so a table encoded on one platform can be used on any other platform and
//! the binary data can be mapped into memory at arbitrary addresses.
//!
//!
//! ## Usage
//!
//! In order to use the hash table one needs to implement the `Config` trait.
//! This trait defines how the table is encoded and what hash function is used.
//! With a `Config` in place the `HashTableBuilder` type can be used to build and serialize a hash table.
//! The `HashTable` type can then be used to create an almost zero-cost view of the serialized hash table.
//!
//! ```rust
//!
//! use odht::{HashTable, HashTableBuilder, Config, FxHashFn};
//!
//! struct MyConfig;
//!
//! impl Config for MyConfig {
//!
//!     type Key = u64;
//!     type Value = u32;
//!
//!     type RawKey = [u8; 8];
//!     type RawValue = [u8; 4];
//!
//!     type H = FxHashFn;
//!
//!     #[inline] fn encode_key(k: &Self::Key) -> Self::RawKey { k.to_le_bytes() }
//!     #[inline] fn encode_value(v: &Self::Value) -> Self::RawValue { v.to_le_bytes() }
//!     #[inline] fn decode_key(k: &Self::RawKey) -> Self::Key { u64::from_le_bytes(*k) }
//!     #[inline] fn decode_value(v: &Self::RawValue) -> Self::Value { u32::from_le_bytes(*v)}
//! }
//!
//! fn main() {
//!     let mut builder = HashTableBuilder::<MyConfig>::with_capacity(3, 0.95);
//!
//!     builder.insert(&1, &2);
//!     builder.insert(&3, &4);
//!     builder.insert(&5, &6);
//!
//!     let serialized: Vec<u8> = {
//!         let mut data = std::io::Cursor::new(Vec::new());
//!         builder.serialize(&mut data).unwrap();
//!         data.into_inner()
//!     };
//!
//!     let table = HashTable::<MyConfig>::from_serialized(&serialized[..]).unwrap();
//!
//!     assert_eq!(table.get(&1), Some(2));
//!     assert_eq!(table.get(&3), Some(4));
//!     assert_eq!(table.get(&5), Some(6));
//! }
//! ```

mod error;
mod fxhash;
mod raw_table;
mod unhash;

use error::Error;
pub use fxhash::FxHashFn;
pub use unhash::UnHashFn;

use raw_table::{ByteArray, Entry, EntryMetadata, RawIter, RawTable, RawTableMut};
use std::{convert::TryInto, marker::PhantomData, mem::size_of};

/// This trait provides a complete "configuration" for a hash table, i.e. it
/// defines the key and value types, how these are encoded and what hash
/// function is being used.
///
/// Implementations of the `encode_key` and `encode_value` methods must encode
/// the given key/value into a fixed size array. The encoding must be
/// deterministic (i.e. no random padding bytes) and must be independent of
/// platform endianess. It is always highly recommended to mark these methods
/// as #[inline].
pub trait Config {
    type Key;
    type Value;

    // The RawKey and RawValue types must always be a fixed size array of bytes,
    // e.g. [u8; 4].
    type RawKey: ByteArray;
    type RawValue: ByteArray;

    type H: HashFn;

    /// Implementations of the `encode_key` and `encode_value` methods must encode
    /// the given key/value into a fixed size array. See above for requirements.
    fn encode_key(k: &Self::Key) -> Self::RawKey;

    /// Implementations of the `encode_key` and `encode_value` methods must encode
    /// the given key/value into a fixed size array. See above for requirements.
    fn encode_value(v: &Self::Value) -> Self::RawValue;

    fn decode_key(k: &Self::RawKey) -> Self::Key;
    fn decode_value(v: &Self::RawValue) -> Self::Value;
}

/// This trait represents hash functions as used by HashTable and
/// HashTableBuilder.
pub trait HashFn: Eq {
    fn hash(bytes: &[u8]) -> u32;
}

/// `HashTableBuilder` is used building and then persisting hash tables. It
/// does provide methods for looking up values but
pub struct HashTableBuilder<C: Config> {
    _config: PhantomData<C>,
    raw_metadata: Vec<EntryMetadata>,
    raw_data: Vec<Entry<C::RawKey, C::RawValue>>,
    mod_mask: usize,
    max_load_factor: f32,
    max_item_count: usize,
    item_count: usize,
}

impl<C: Config> HashTableBuilder<C> {
    pub fn with_capacity(item_count: usize, max_load_factor: f32) -> HashTableBuilder<C> {
        assert!(max_load_factor < 1.0);
        assert!(max_load_factor > 0.1);

        let slots_needed = (item_count as f32 / max_load_factor).ceil() as usize;
        let capacity = slots_needed.checked_next_power_of_two().unwrap();
        assert!(capacity > 0);

        let mod_mask = capacity.checked_sub(1).unwrap();
        let max_item_count = (capacity as f32 * max_load_factor).ceil() as usize;

        HashTableBuilder {
            _config: PhantomData::default(),
            raw_metadata: vec![EntryMetadata::default(); capacity],
            raw_data: vec![Entry::default(); capacity],
            mod_mask,
            max_load_factor,
            max_item_count,
            item_count: 0,
        }
    }

    /// Retrieves the value for the given key. Returns `None` if no entry is found.
    #[inline]
    pub fn get(&self, key: &C::Key) -> Option<C::Value> {
        let raw_key = C::encode_key(key);
        self.as_raw().find(&raw_key).map(C::decode_value)
    }

    /// Inserts the given key-value pair into the table.
    /// Grows the table if necessary.
    #[inline]
    pub fn insert(&mut self, key: &C::Key, value: &C::Value) {
        if self.item_count == self.max_item_count {
            self.grow();
        }

        assert!(self.item_count < self.max_item_count);

        let raw_key = C::encode_key(key);
        let raw_value = C::encode_value(value);

        if self.as_raw_mut().insert(raw_key, raw_value) {
            self.item_count += 1;
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, C> {
        Iter(RawIter::new(&self.raw_metadata[..], &self.raw_data[..]))
    }

    pub fn from_iterator<I: IntoIterator<Item = (C::Key, C::Value)>>(
        it: I,
        max_load_factor: f32,
    ) -> Self {
        let it = it.into_iter();

        let known_size = match it.size_hint() {
            (min, Some(max)) => {
                if min == max {
                    Some(max)
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(known_size) = known_size {
            let mut table = HashTableBuilder::with_capacity(known_size, max_load_factor);

            let initial_capacity = table.raw_data.len();

            for (k, v) in it {
                table.insert(&k, &v);
            }

            assert_eq!(table.len(), known_size);
            assert_eq!(table.raw_data.len(), initial_capacity);

            table
        } else {
            let items: Vec<_> = it.collect();
            Self::from_iterator(items, max_load_factor)
        }
    }

    #[inline]
    fn as_raw(&self) -> RawTable<'_, C::RawKey, C::RawValue, C::H> {
        RawTable::new(&self.raw_metadata[..], &self.raw_data[..], self.mod_mask)
    }

    #[inline]
    fn as_raw_mut(&mut self) -> RawTableMut<'_, C::RawKey, C::RawValue, C::H> {
        RawTableMut::new(
            &mut self.raw_metadata[..],
            &mut self.raw_data[..],
            self.mod_mask,
        )
    }

    #[inline(never)]
    #[cold]
    fn grow(&mut self) {
        let mut new_table = Self::with_capacity(self.item_count * 2, self.max_load_factor);

        {
            let mut new_table = new_table.as_raw_mut();

            for (raw_metadata, raw_data) in self.as_raw().iter() {
                new_table.insert_entry(raw_metadata, *raw_data);
            }
        }

        *self = new_table;
    }

    pub fn serialize(&self, w: &mut dyn std::io::Write) -> Result<(), Box<dyn std::error::Error>> {
        assert!(self.raw_data.len().is_power_of_two());

        let header = Header::new::<C>(self.item_count, self.raw_data.len());

        header.serialize(w)?;

        w.write_all(self.as_raw().metadata_bytes())?;
        w.write_all(self.as_raw().entry_data_bytes())?;

        Ok(())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.item_count
    }
}

impl<C: Config> std::fmt::Debug for HashTableBuilder<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "(item_count={}, mod_mask={:x}, max_item_count={}, max_load_factor={})",
            self.item_count, self.mod_mask, self.max_item_count, self.max_load_factor
        )?;

        writeln!(f, "{:?}", self.as_raw())
    }
}

/// This type provides a cheap to construct readonly view on a persisted
/// hash table.
pub struct HashTable<'a, C: Config> {
    _config: PhantomData<C>,
    raw_metadata: &'a [EntryMetadata],
    raw_data: &'a [Entry<C::RawKey, C::RawValue>],
    mod_mask: usize,
    item_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Header {
    tag: [u8; 4],
    size_of_metadata: u8,
    size_of_key: u8,
    size_of_value: u8,
    size_of_header: u8,
    item_count: [u8; 8],
    capacity: [u8; 8],
}

const HEADER_TAG: [u8; 4] = *b"ODHT";
const HEADER_SIZE: usize = size_of::<Header>();

impl Header {
    fn new<C: Config>(item_count: usize, capacity: usize) -> Header {
        Header {
            tag: HEADER_TAG,
            size_of_metadata: size_of::<EntryMetadata>().try_into().unwrap(),
            size_of_key: size_of::<C::RawKey>().try_into().unwrap(),
            size_of_value: size_of::<C::RawValue>().try_into().unwrap(),
            size_of_header: size_of::<Header>().try_into().unwrap(),
            item_count: (item_count as u64).to_le_bytes(),
            capacity: (capacity as u64).to_le_bytes(),
        }
    }

    fn item_count(&self) -> usize {
        u64::from_le_bytes(self.item_count) as usize
    }

    fn capacity(&self) -> usize {
        u64::from_le_bytes(self.capacity) as usize
    }

    fn metadata_offset(&self) -> isize {
        HEADER_SIZE as isize
    }

    fn data_offset(&self) -> isize {
        (HEADER_SIZE + self.capacity() * size_of::<EntryMetadata>()) as isize
    }

    fn serialize(&self, w: &mut dyn std::io::Write) -> Result<(), Box<dyn std::error::Error>> {
        let bytes =
            unsafe { std::slice::from_raw_parts(self as *const _ as *const u8, HEADER_SIZE) };

        Ok(w.write_all(bytes)?)
    }

    fn from_serialized<C: Config>(data: &[u8]) -> Result<Header, Error> {
        if data.len() < HEADER_SIZE {
            return Err(Error(format!("Provided data not big enough for header.")));
        }

        let header = unsafe { *(data.as_ptr() as *const Header) };

        if header.tag != HEADER_TAG {
            return Err(Error(format!(
                "Expected header tag {:?} but found {:?}",
                HEADER_TAG, header.tag
            )));
        }

        check_expected_size::<EntryMetadata>(header.size_of_metadata)?;
        check_expected_size::<C::RawKey>(header.size_of_key)?;
        check_expected_size::<C::RawValue>(header.size_of_value)?;
        check_expected_size::<Header>(header.size_of_header)?;

        let bytes_per_entry =
            size_of::<Entry<C::RawKey, C::RawValue>>() + size_of::<EntryMetadata>();

        if data.len() < HEADER_SIZE + header.capacity() * bytes_per_entry {
            return Err(Error(format!(
                "Provided data not big enough for capacity {}",
                header.capacity()
            )));
        }

        if !header.capacity().is_power_of_two() {
            return Err(Error(format!(
                "Capacity of hashtable should be a power of two but is {}",
                header.capacity()
            )));
        }

        return Ok(header);

        fn check_expected_size<T>(expected_size: u8) -> Result<(), Error> {
            if expected_size as usize != size_of::<T>() {
                Err(Error(format!(
                    "Expected size of Config::RawValue to be {} but the encoded \
                     table specifies {}. This indicates an encoding mismatch.",
                    size_of::<T>(),
                    expected_size
                )))
            } else {
                Ok(())
            }
        }
    }
}

impl<'a, C: Config> HashTable<'a, C> {
    pub fn from_serialized(data: &[u8]) -> Result<HashTable<'_, C>, Box<dyn std::error::Error>> {
        assert!(std::mem::align_of::<Entry<C::RawKey, C::RawValue>>() == 1);

        let header = Header::from_serialized::<C>(data)?;

        let raw_metadata: &[EntryMetadata] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr().offset(header.metadata_offset()) as *const EntryMetadata,
                header.capacity(),
            )
        };

        let raw_data: &[Entry<C::RawKey, C::RawValue>] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr().offset(header.data_offset()) as *const Entry<C::RawKey, C::RawValue>,
                header.capacity(),
            )
        };

        let table = HashTable {
            _config: PhantomData::default(),
            raw_metadata,
            raw_data,
            mod_mask: header.capacity() - 1,
            item_count: header.item_count(),
        };

        table.as_raw().sanity_check_hashes(3)?;

        Ok(table)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.item_count
    }

    #[inline]
    pub fn get(&self, key: &C::Key) -> Option<C::Value> {
        let raw_key = C::encode_key(key);
        self.as_raw().find(&raw_key).map(C::decode_value)
    }

    #[inline]
    fn as_raw(&self) -> RawTable<'_, C::RawKey, C::RawValue, C::H> {
        RawTable::new(self.raw_metadata, self.raw_data, self.mod_mask)
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, C> {
        Iter(RawIter::new(self.raw_metadata, self.raw_data))
    }
}

pub struct Iter<'a, C: Config>(RawIter<'a, C::RawKey, C::RawValue>);

impl<'a, C: Config> Iterator for Iter<'a, C> {
    type Item = (C::Key, C::Value);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, entry)| {
            let key = C::decode_key(&entry.key);
            let value = C::decode_value(&entry.value);

            (key, value)
        })
    }
}

#[cfg(test)]
mod tests {

    use std::{convert::TryInto, io::Cursor};

    use super::*;

    enum TestConfig {}

    impl Config for TestConfig {
        type RawKey = [u8; 4];
        type RawValue = [u8; 4];

        type Key = u32;
        type Value = u32;

        type H = FxHashFn;

        fn encode_key(k: &Self::Key) -> Self::RawKey {
            k.to_le_bytes()
        }

        fn encode_value(v: &Self::Value) -> Self::RawValue {
            v.to_le_bytes()
        }

        fn decode_key(k: &Self::RawKey) -> Self::Key {
            u32::from_le_bytes(k[..].try_into().unwrap())
        }

        fn decode_value(v: &Self::RawValue) -> Self::Value {
            u32::from_le_bytes(v[..].try_into().unwrap())
        }
    }

    fn make_test_items(count: usize) -> Vec<(u32, u32)> {
        if count == 0 {
            return vec![];
        }

        let mut items = vec![];

        if count > 1 {
            let steps = (count - 1) as u32;
            let step = u32::MAX / steps;

            for i in 0..steps {
                let x = i * step;
                items.push((x, u32::MAX - x));
            }
        }

        items.push((u32::MAX, 0));

        items.sort();
        items.dedup();
        assert_eq!(items.len(), count);

        items
    }

    #[test]
    fn from_iterator() {
        for count in 0..33 {
            let items = make_test_items(count);
            let table = HashTableBuilder::<TestConfig>::from_iterator(items.clone(), 0.95);
            assert_eq!(table.len(), items.len());

            let mut actual_items: Vec<_> = table.iter().collect();
            actual_items.sort();

            assert_eq!(items, actual_items);
        }
    }

    #[test]
    fn hash_table_at_different_alignments() {
        let items = make_test_items(33);

        let mut serialized = {
            let table: HashTableBuilder<TestConfig> =
                HashTableBuilder::from_iterator(items.clone(), 0.95);

            assert_eq!(table.len(), items.len());

            let mut stream = Cursor::new(Vec::new());

            table.serialize(&mut stream).unwrap();

            stream.into_inner()
        };

        for alignment_shift in 0..4 {
            let data = &serialized[alignment_shift..];

            let table = HashTable::<TestConfig>::from_serialized(data).unwrap();

            assert_eq!(table.len(), items.len());

            for (key, value) in items.iter() {
                assert_eq!(table.get(key), Some(*value));
            }

            serialized.insert(0, 0xFFu8);
        }
    }
}
