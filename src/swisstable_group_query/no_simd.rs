use super::GROUP_SIZE;

pub struct GroupQuery<'a> {
    group: &'a [u8; GROUP_SIZE],
    seen: usize,
    first_empty_index: usize,
    h2: u8,
}

impl<'a> GroupQuery<'a> {
    #[inline]
    pub fn from(group: &[u8; GROUP_SIZE], h2: u8) -> GroupQuery {
        GroupQuery {
            group,
            seen: 0,
            first_empty_index: usize::MAX,
            h2,
        }
    }

    #[inline]
    pub fn any_empty(&self) -> bool {
        debug_assert!(self.seen == GROUP_SIZE);
        self.first_empty_index < GROUP_SIZE
    }

    #[inline]
    pub fn first_empty(&self) -> Option<usize> {
        debug_assert!(self.seen == GROUP_SIZE);
        if self.first_empty_index < GROUP_SIZE {
            Some(self.first_empty_index)
        } else {
            None
        }
    }
}

impl<'a> Iterator for GroupQuery<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.seen >= GROUP_SIZE {
                return None;
            }

            let i = self.seen;
            self.seen += 1;

            let byte = self.group[i];

            if byte == self.h2 {
                return Some(i);
            } else if self.first_empty_index >= GROUP_SIZE && is_empty(byte) {
                self.first_empty_index = i;
            }
        }
    }
}

#[inline]
fn is_empty(byte: u8) -> bool {
    (byte & 0b1000_0000) != 0
}
