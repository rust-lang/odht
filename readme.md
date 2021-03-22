![CI Status](https://github.com/michaelwoerister/odht/actions/workflows/ci.yml/badge.svg)

# odht

A Rust crate for hash tables that can be mapped from disk into memory without the need for up-front decoding.
The goal of the implementation is to provide a data structure that

- can be used exactly in the format it is stored on disk,
- provides roughly the same performance as a `HashMap` from Rust's standard library,
- has a completely deterministic binary representation,
- is platform and endianess independent, so that data serialized on one system can be used on any other system, and
- is independent of alignment requirements so that
  - its use is not restricted to certain classes of CPUs, and
  - the data structure can be mapped to arbitrary memory addresses.
