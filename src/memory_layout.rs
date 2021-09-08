use std::{
    borrow::{Borrow, BorrowMut},
    convert::TryInto,
    marker::PhantomData,
    mem::{align_of, size_of},
};

use crate::Config;
use crate::{
    error::Error,
    raw_table::{Entry, EntryMetadata, RawTable},
    swisstable_group_query::GROUP_SIZE,
    Factor,
};

const CURRENT_FILE_FORMAT_VERSION: [u8; 4] = [0, 0, 0, 1];

#[repr(C)]
#[derive(Clone)]
pub(crate) struct Header {
    tag: [u8; 4],
    size_of_metadata: u8,
    size_of_key: u8,
    size_of_value: u8,
    size_of_header: u8,

    item_count: [u8; 8],
    slot_count: [u8; 8],

    file_format_version: [u8; 4],
    max_load_factor: [u8; 2],

    // Let's keep things at least 8 byte aligned
    padding: [u8; 2],
}

const HEADER_TAG: [u8; 4] = *b"ODHT";
const HEADER_SIZE: usize = size_of::<Header>();

impl Header {
    pub fn sanity_check<'a, C: Config>(&self, raw_bytes: &[u8]) -> Result<(), Error> {
        assert!(align_of::<Header>() == 1);
        assert!(HEADER_SIZE % 8 == 0);

        if self.tag != HEADER_TAG {
            return Err(Error(format!(
                "Expected header tag {:?} but found {:?}",
                HEADER_TAG, self.tag
            )));
        }

        if self.file_format_version != CURRENT_FILE_FORMAT_VERSION {
            return Err(Error(format!(
                "Expected file format version {:?} but found {:?}",
                CURRENT_FILE_FORMAT_VERSION, self.file_format_version
            )));
        }

        check_expected_size::<EntryMetadata>("EntryMetadata", self.size_of_metadata)?;
        check_expected_size::<C::EncodedKey>("Config::EncodedKey", self.size_of_key)?;
        check_expected_size::<C::EncodedValue>("Config::EncodedValue", self.size_of_value)?;
        check_expected_size::<Header>("Header", self.size_of_header)?;

        if raw_bytes.len() != bytes_needed::<C>(self.slot_count()) {
            return Err(Error(format!(
                "Provided allocation has wrong size for slot count {}",
                self.slot_count()
            )));
        }

        if !self.slot_count().is_power_of_two() {
            return Err(Error(format!(
                "Slot count of hashtable should be a power of two but is {}",
                self.slot_count()
            )));
        }

        return Ok(());

        fn check_expected_size<T>(name: &str, expected_size: u8) -> Result<(), Error> {
            if expected_size as usize != size_of::<T>() {
                Err(Error(format!(
                    "Expected size of {} to be {} but the encoded \
                     table specifies {}. This indicates an encoding mismatch.",
                    name,
                    size_of::<T>(),
                    expected_size
                )))
            } else {
                Ok(())
            }
        }
    }

    #[inline]
    pub fn item_count(&self) -> usize {
        u64::from_le_bytes(self.item_count) as usize
    }

    #[inline]
    pub fn set_item_count(&mut self, item_count: usize) {
        self.item_count = (item_count as u64).to_le_bytes();
    }

    #[inline]
    pub fn slot_count(&self) -> usize {
        u64::from_le_bytes(self.slot_count) as usize
    }

    #[inline]
    pub fn max_load_factor(&self) -> Factor {
        Factor(u16::from_le_bytes(self.max_load_factor))
    }

    #[inline]
    fn metadata_offset<C: Config>(&self) -> isize {
        let slot_count = self.slot_count();
        let size_of_entry = size_of::<Entry<C::EncodedKey, C::EncodedValue>>();
        (HEADER_SIZE + slot_count * size_of_entry) as isize
    }

    #[inline]
    fn entry_data_offset(&self) -> isize {
        HEADER_SIZE as isize
    }

    fn initialize<C: Config>(
        raw_bytes: &mut [u8],
        slot_count: usize,
        item_count: usize,
        max_load_factor: Factor,
    ) {
        assert_eq!(raw_bytes.len(), bytes_needed::<C>(slot_count));

        let header = Header {
            tag: HEADER_TAG,
            size_of_metadata: size_of::<EntryMetadata>().try_into().unwrap(),
            size_of_key: size_of::<C::EncodedKey>().try_into().unwrap(),
            size_of_value: size_of::<C::EncodedValue>().try_into().unwrap(),
            size_of_header: size_of::<Header>().try_into().unwrap(),
            item_count: (item_count as u64).to_le_bytes(),
            slot_count: (slot_count as u64).to_le_bytes(),
            file_format_version: CURRENT_FILE_FORMAT_VERSION,
            max_load_factor: max_load_factor.0.to_le_bytes(),
            padding: [0u8; 2],
        };

        assert_eq!(header.sanity_check::<C>(raw_bytes), Ok(()));

        unsafe {
            *(raw_bytes.as_mut_ptr() as *mut Header) = header;
        }
    }
}

/// An allocation holds a byte array that is guaranteed to conform to the
/// hash table's binary layout.
#[derive(Clone, Copy)]
pub(crate) struct Allocation<C, M>
where
    C: Config,
{
    bytes: M,
    _config: PhantomData<C>,
}

impl<C, M> Allocation<C, M>
where
    C: Config,
    M: Borrow<[u8]>,
{
    pub fn from_raw_bytes(raw_bytes: M) -> Result<Allocation<C, M>, Error> {
        let allocation = Allocation {
            bytes: raw_bytes,
            _config: PhantomData::default(),
        };

        allocation
            .header()
            .sanity_check::<C>(allocation.bytes.borrow())?;

        // Check that the hash function provides the expected hash values.
        {
            let (entry_metadata, entry_data) = allocation.data_slices();
            RawTable::<C::EncodedKey, C::EncodedValue, C::H>::new(entry_metadata, entry_data)
                .sanity_check_hashes(10)?;
        }

        Ok(allocation)
    }

    #[inline]
    pub unsafe fn from_raw_bytes_unchecked(raw_bytes: M) -> Allocation<C, M> {
        Allocation {
            bytes: raw_bytes,
            _config: PhantomData::default(),
        }
    }

    #[inline]
    pub fn header(&self) -> &Header {
        let raw_bytes = self.bytes.borrow();
        debug_assert!(raw_bytes.len() >= HEADER_SIZE);

        let header: &Header = unsafe { &*(raw_bytes.as_ptr() as *const Header) };

        debug_assert_eq!(header.sanity_check::<C>(raw_bytes), Ok(()));

        header
    }

    #[inline]
    pub fn data_slices(&self) -> (&[EntryMetadata], &[Entry<C::EncodedKey, C::EncodedValue>]) {
        let raw_bytes = self.bytes.borrow();
        let slot_count = self.header().slot_count();
        let entry_data_offset = self.header().entry_data_offset();
        let metadata_offset = self.header().metadata_offset::<C>();

        let entry_metadata = unsafe {
            std::slice::from_raw_parts(
                raw_bytes.as_ptr().offset(metadata_offset) as *const EntryMetadata,
                slot_count + GROUP_SIZE,
            )
        };

        let entry_data = unsafe {
            std::slice::from_raw_parts(
                raw_bytes.as_ptr().offset(entry_data_offset)
                    as *const Entry<C::EncodedKey, C::EncodedValue>,
                slot_count,
            )
        };

        unsafe {
            debug_assert_eq!(
                raw_bytes.as_ptr().offset(raw_bytes.len() as isize),
                entry_metadata
                    .as_ptr()
                    .offset(entry_metadata.len() as isize) as *const u8
            );

            debug_assert_eq!(
                entry_data.as_ptr().offset(entry_data.len() as isize) as *const u8,
                entry_metadata.as_ptr() as *const u8
            );
        }

        (entry_metadata, entry_data)
    }

    #[inline]
    pub fn raw_bytes(&self) -> &[u8] {
        self.bytes.borrow()
    }
}

impl<C, M> Allocation<C, M>
where
    C: Config,
    M: BorrowMut<[u8]>,
{
    #[inline]
    pub fn header_mut(&mut self) -> &mut Header {
        let raw_bytes = self.bytes.borrow_mut();
        debug_assert!(raw_bytes.len() >= HEADER_SIZE);

        let header: &mut Header = unsafe { &mut *(raw_bytes.as_mut_ptr() as *mut Header) };

        debug_assert_eq!(header.sanity_check::<C>(raw_bytes), Ok(()));

        header
    }

    #[inline]
    pub fn data_slices_mut<'a>(
        &'a mut self,
    ) -> (
        &'a mut [EntryMetadata],
        &'a mut [Entry<C::EncodedKey, C::EncodedValue>],
    ) {
        let slot_count = self.header().slot_count();
        let metadata_offset = self.header().metadata_offset::<C>();
        let entry_data_offset = self.header().entry_data_offset();

        let raw_bytes = self.bytes.borrow_mut();

        let entry_metadata = unsafe {
            std::slice::from_raw_parts_mut(
                raw_bytes.as_mut_ptr().offset(metadata_offset) as *mut EntryMetadata,
                slot_count + GROUP_SIZE,
            )
        };

        let entry_data = unsafe {
            std::slice::from_raw_parts_mut(
                raw_bytes.as_mut_ptr().offset(entry_data_offset)
                    as *mut Entry<C::EncodedKey, C::EncodedValue>,
                slot_count,
            )
        };

        unsafe {
            debug_assert_eq!(
                raw_bytes.as_ptr().offset(raw_bytes.len() as isize),
                entry_metadata
                    .as_ptr()
                    .offset(entry_metadata.len() as isize) as *const u8
            );

            debug_assert_eq!(
                entry_data.as_ptr().offset(entry_data.len() as isize) as *const u8,
                entry_metadata.as_ptr() as *const u8
            );
        }

        (entry_metadata, entry_data)
    }
}

#[inline]
pub(crate) fn bytes_needed<C: Config>(slot_count: usize) -> usize {
    assert!(slot_count.is_power_of_two());
    let size_of_entry = size_of::<Entry<C::EncodedKey, C::EncodedValue>>();
    let size_of_metadata = size_of::<EntryMetadata>();

    HEADER_SIZE + slot_count * size_of_entry + (slot_count + GROUP_SIZE) * size_of_metadata
}

pub(crate) fn allocate<C: Config>(
    slot_count: usize,
    item_count: usize,
    max_load_factor: Factor,
) -> Allocation<C, Box<[u8]>> {
    let bytes = vec![0u8; bytes_needed::<C>(slot_count)].into_boxed_slice();
    init_in_place::<C, _>(bytes, slot_count, item_count, max_load_factor)
}

pub(crate) fn init_in_place<C: Config, M: BorrowMut<[u8]>>(
    mut bytes: M,
    slot_count: usize,
    item_count: usize,
    max_load_factor: Factor,
) -> Allocation<C, M> {
    Header::initialize::<C>(bytes.borrow_mut(), slot_count, item_count, max_load_factor);

    let mut allocation = Allocation {
        bytes,
        _config: PhantomData::default(),
    };

    {
        let (metadata, data) = allocation.data_slices_mut();
        metadata.fill(0xFF);
        data.fill(Default::default());
    }

    allocation
}
