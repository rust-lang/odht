#[cfg(target_arch = "x86")]
use core::arch::x86;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as x86;

pub const GROUP_SIZE: usize = 16;

pub struct GroupQuery {
    matches: u16,
    empty: u16,
}

impl GroupQuery {
    #[inline]
    pub fn from(group: &[u8; GROUP_SIZE], h2: u8) -> GroupQuery {
        assert!(std::mem::size_of::<x86::__m128i>() == GROUP_SIZE);

        unsafe {
            let group = x86::_mm_loadu_si128(group as *const _ as *const x86::__m128i);
            let cmp_byte = x86::_mm_cmpeq_epi8(group, x86::_mm_set1_epi8(h2 as i8));
            let matches = x86::_mm_movemask_epi8(cmp_byte) as u16;
            let empty = x86::_mm_movemask_epi8(group) as u16;

            GroupQuery { matches, empty }
        }
    }

    #[inline]
    pub fn any_empty(&self) -> bool {
        self.empty != 0
    }

    #[inline]
    pub fn first_empty(&self) -> Option<usize> {
        if self.empty == 0 {
            None
        } else {
            Some(lowest_bit_set_non_zero(self.empty))
        }
    }
}

impl Iterator for GroupQuery {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.matches == 0 {
            None
        } else {
            let index = lowest_bit_set_non_zero(self.matches);

            // Clear the lowest bit
            // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
            self.matches &= self.matches - 1;

            Some(index)
        }
    }
}

#[inline]
fn lowest_bit_set_non_zero(x: u16) -> usize {
    debug_assert!(x != 0);

    cfg_if::cfg_if! {
        if #[cfg(feature = "nightly")] {
            unsafe {
                std::intrinsics::cttz_nonzero(x) as usize
            }
        } else {
            x.trailing_zeros() as usize
        }
    }
}
