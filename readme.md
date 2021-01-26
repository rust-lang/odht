# odht - an on-disk hash table

This crate implements a hash table that can be used as is in its binary, on-disk format.
The goal is to provide a high performance data structure that can be used without any significant up-front decoding.
The implementation makes no assumptions about alignment or endianess of the underlying data,
so a table encoded on one platform can be used on any other platform and
the binary data can be mapped into memory at arbitrary addresses.
