// This test makes sure that a hash table generated with SIMD support
// can be loaded on a platform without SIMD support.

struct FxConfig;

impl odht::Config for FxConfig {
    type Key = u64;
    type Value = u32;

    type EncodedKey = [u8; 8];
    type EncodedValue = [u8; 4];

    type H = odht::FxHashFn;

    #[inline]
    fn encode_key(k: &Self::Key) -> Self::EncodedKey {
        k.to_le_bytes()
    }

    #[inline]
    fn encode_value(v: &Self::Value) -> Self::EncodedValue {
        v.to_le_bytes()
    }

    #[inline]
    fn decode_key(k: &Self::EncodedKey) -> Self::Key {
        u64::from_le_bytes(*k)
    }

    #[inline]
    fn decode_value(v: &Self::EncodedValue) -> Self::Value {
        u32::from_le_bytes(*v)
    }
}

const FILE_NAME_NO_SIMD: &str = "odht_hash_table_no_simd";
const FILE_NAME_WITH_SIMD: &str = "odht_hash_table_with_simd";

#[cfg(feature = "no_simd")]
const WRITE_FILE_NAME: &str = FILE_NAME_NO_SIMD;
#[cfg(not(feature = "no_simd"))]
const WRITE_FILE_NAME: &str = FILE_NAME_WITH_SIMD;

#[cfg(feature = "no_simd")]
const READ_FILE_NAME: &'static str = FILE_NAME_WITH_SIMD;

#[cfg(not(feature = "no_simd"))]
const READ_FILE_NAME: &'static str = FILE_NAME_NO_SIMD;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let make_entries = || (0 .. 70_000_u64).map(|x| (x * x, x as u32)).collect::<Vec<_>>();

    if std::env::args_os().find(|arg| arg == "write").is_some() {
        let hash_table = odht::HashTableOwned::<FxConfig>::from_iterator(make_entries(), 85);
        let mut path = std::env::temp_dir();
        path.push(WRITE_FILE_NAME);
        std::fs::write(&path, hash_table.raw_bytes())?;
        eprintln!("Wrote hash table with {} bytes to {}", hash_table.raw_bytes().len(), path.display());
    }

    if std::env::args_os().find(|arg| arg == "read").is_some() {
        let mut path = std::env::temp_dir();
        path.push(READ_FILE_NAME);
        eprintln!("Trying to load hash table from {}", path.display());
        let data = std::fs::read(&path)?;
        let hash_table = odht::HashTable::<FxConfig, _>::from_raw_bytes(data)?;
        eprintln!("Loaded hash table with {} bytes from {}", hash_table.raw_bytes().len(), path.display());
        let expected_entries = make_entries();

        eprintln!("Comparing hash table to expected values.");
         // Check that we can read the data
        assert_eq!(hash_table.len(), expected_entries.len());
        for (key, value) in expected_entries {
            assert_eq!(hash_table.get(&key), Some(value));
        }

        eprintln!("Success");
    }

    Ok(())
}
