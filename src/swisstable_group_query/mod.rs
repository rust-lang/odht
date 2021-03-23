cfg_if::cfg_if! {
    if #[cfg(all(
        target_feature = "sse2",
        any(target_arch = "x86", target_arch = "x86_64"),
        not(miri),
        not(feature = "no_simd"),
    ))] {
        mod sse2;
        use sse2 as imp;
    } else {
        mod no_simd;
        use no_simd as imp;
    }
}

pub(crate) use imp::GroupQuery;

pub(crate) const GROUP_SIZE: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;

    const EMPTY_GROUP: [u8; 16] = [
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ];

    #[test]
    fn full() {
        let mut q = GroupQuery::from(&[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], 42);

        assert_eq!(Iterator::count(&mut q), 0);
        assert!(!q.any_empty());
        assert_eq!(q.first_empty(), None);
    }

    #[test]
    fn all_empty() {
        let mut q = GroupQuery::from(&EMPTY_GROUP, 31);

        assert_eq!(Iterator::count(&mut q), 0);
        assert!(q.any_empty());
        assert_eq!(q.first_empty(), Some(0));
    }

    #[test]
    fn partially_filled() {
        for filled_up_to_index in 0..=14 {
            let mut group = EMPTY_GROUP;

            for i in 0..=filled_up_to_index {
                group[i] = 42;
            }

            let mut q = GroupQuery::from(&group, 77);

            assert_eq!(Iterator::count(&mut q), 0);
            assert!(q.any_empty());
            assert_eq!(q.first_empty(), Some(filled_up_to_index + 1));
        }
    }

    #[test]
    fn match_iter() {
        let group = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0];

        let mut q = GroupQuery::from(&group, 1);

        let matches: Vec<usize> = (&mut q).collect();

        assert_eq!(matches, vec![1, 4, 7, 12, 13]);
        assert!(!q.any_empty());
        assert_eq!(q.first_empty(), None);
    }

    #[test]
    fn match_iter_with_emtpy() {
        let group = [0, 1, 0, 255, 1, 0, 0, 1, 0, 255, 0, 0, 1, 1, 0, 0];

        let mut q = GroupQuery::from(&group, 1);

        let matches: Vec<usize> = (&mut q).collect();

        assert_eq!(matches, vec![1, 4, 7, 12, 13]);
        assert!(q.any_empty());
        assert_eq!(q.first_empty(), Some(3));
    }
}
