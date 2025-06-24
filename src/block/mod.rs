use std::{
    fmt::{self, Debug, Display},
    io::Write,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, Sub},
};

use num_traits::{NumCast, Signed, ToPrimitive};

use crate::{
    bitstream::{BitStreamReader, BitStreamWriter},
    error::Result,
    Decodable, Encodable, Error, FromBytes, ToBytes,
};

pub(crate) mod quantization;
pub(crate) mod zigzag;

const BLOCK_COLS: usize = 8;
const BLOCK_ROWS: usize = 8;
const BLOCK_SIZE: usize = BLOCK_COLS * BLOCK_ROWS;
/// If everything is the same in the block then the diff will be 0
/// If everything is off by 1 then it will 64
/// So 128, or off by 2, seems like a reasonable value for now.
pub(crate) const REASONABLE_SUM_OF_ABS_DIFF_I16: i16 = (BLOCK_COLS * BLOCK_ROWS * 2) as i16;

pub struct Block<T>(pub [T; BLOCK_SIZE]);

unsafe impl<T> Send for Block<T> where T: Send {}
unsafe impl<T> Sync for Block<T> where T: Sync {}

impl<const N: usize, T> Decodable for Block<T>
where
    T: Debug + num_traits::FromBytes<Bytes = [u8; N]>,
{
    type Output = Self;

    fn decode<R>(stream: &mut BitStreamReader<R>) -> Result<Self::Output>
    where
        R: std::io::Read,
    {
        let (block, size) =
            Block::from_bytes(&stream.read_slice(Block::<T>::size() * std::mem::size_of::<T>())?);
        assert_eq!(size, Block::<T>::size() * std::mem::size_of::<T>());
        Ok(block)
    }
}

impl<const N: usize, T> Encodable for Block<T>
where
    T: num_traits::ToBytes<Bytes = [u8; N]>,
{
    fn encode<W>(&self, stream: &mut BitStreamWriter<W>) -> Result<()>
    where
        W: Write,
    {
        let bytes = self.to_bytes();
        stream.write_slice(&bytes)?;

        Ok(())
    }
}

impl<T> Clone for Block<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Copy for Block<T> where T: Copy {}

impl<T> Debug for Block<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Block").field("inner", &self.0).finish()
    }
}

impl<T> Default for Block<T>
where
    T: Copy + Default,
{
    fn default() -> Self {
        Self([T::default(); BLOCK_SIZE])
    }
}

impl<T> Display for Block<T>
where
    T: Copy + Clone + Debug + Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Block").field("inner", &self.0).finish()
    }
}

impl<T> Eq for Block<T> where T: Eq {}

impl<T> PartialEq for Block<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T> Block<T> {
    /// Get a value at row, col
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> T
    where
        T: Copy,
    {
        self.0[row * BLOCK_COLS + col]
    }

    /// Set a value at row, col
    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.0[row * BLOCK_COLS + col] = value;
    }

    /// Get a reference to value at row, col
    #[inline(always)]
    pub fn get_ref(&self, row: usize, col: usize) -> &T {
        &self.0[row * BLOCK_COLS + col]
    }

    /// Get a mutable reference to value at row, col
    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.0[row * BLOCK_COLS + col]
    }
}

// Direct indexing into flat array for better performance
impl<T> Index<usize> for Block<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> IndexMut<usize> for Block<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T> Add for Block<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut next = self;
        for r in 0..BLOCK_ROWS {
            for c in 0..BLOCK_COLS {
                let idx = r * BLOCK_COLS + c;
                next[idx] = self[idx] + other[idx];
            }
        }
        next
    }
}

impl<T> Sub for Block<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut next = self;
        for r in 0..BLOCK_ROWS {
            for c in 0..BLOCK_COLS {
                let idx = r * BLOCK_COLS + c;
                next[idx] = self[idx] - other[idx];
            }
        }
        next
    }
}

impl<T> Div<T> for Block<T>
where
    T: Div<Output = T> + Copy + DivAssign,
{
    type Output = Self;

    fn div(self, other: T) -> Self {
        let mut next = self;
        for r in 0..BLOCK_ROWS {
            for c in 0..BLOCK_COLS {
                let idx = r * BLOCK_COLS + c;
                next[idx] /= other;
            }
        }
        next
    }
}

impl<T> Div<Block<T>> for Block<T>
where
    T: Div<Output = T> + Copy,
{
    type Output = Self;

    fn div(self, other: Block<T>) -> Self {
        let mut next = self;
        for r in 0..BLOCK_ROWS {
            for c in 0..BLOCK_COLS {
                let idx = r * BLOCK_COLS + c;
                next[idx] = self[idx] / other[idx];
            }
        }
        next
    }
}

impl<T> Mul<Block<T>> for Block<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Self;

    fn mul(self, other: Block<T>) -> Self {
        let mut next = self;
        for r in 0..BLOCK_ROWS {
            for c in 0..BLOCK_COLS {
                let idx = r * BLOCK_COLS + c;
                next[idx] = self[idx] * other[idx];
            }
        }
        next
    }
}

pub(crate) struct BlockIter<'a, T> {
    block: &'a Block<T>,
    index: usize,
}

impl<T> Iterator for BlockIter<'_, T>
where
    T: Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= BLOCK_SIZE {
            None
        } else {
            let value = self.block.0[self.index];
            self.index += 1;
            Some(value)
        }
    }
}

impl<T> From<&[T]> for Block<T>
where
    T: Copy + Default,
{
    fn from(slice: &[T]) -> Self {
        let data: [T; BLOCK_SIZE] = slice.try_into().expect("block worth of data");
        Self(data)
    }
}

impl<T> From<[T; BLOCK_SIZE]> for Block<T>
where
    T: Copy + Default,
{
    fn from(data: [T; BLOCK_SIZE]) -> Self {
        Self(data)
    }
}

impl<T> From<Block<T>> for [T; BLOCK_SIZE]
where
    T: Copy + Default,
{
    fn from(block: Block<T>) -> Self {
        block.0
    }
}

impl<T> Block<T> {
    pub(crate) const fn cols() -> usize {
        BLOCK_COLS
    }

    pub(crate) const fn rows() -> usize {
        BLOCK_ROWS
    }

    pub(crate) const fn size() -> usize {
        BLOCK_COLS * BLOCK_ROWS
    }

    pub(crate) fn iter(&self) -> BlockIter<'_, T> {
        BlockIter {
            block: self,
            index: 0,
        }
    }
}

impl<T> Block<T>
where
    T: Copy + Default + NumCast,
{
    pub(crate) fn try_convert_from<U>(value: Block<U>) -> Result<Self>
    where
        U: Copy + ToPrimitive + 'static,
    {
        let mut next = Block::default();
        for r in 0..BLOCK_ROWS {
            for c in 0..BLOCK_COLS {
                let idx = r * BLOCK_COLS + c;
                let converted_value = match std::any::TypeId::of::<U>() {
                    id if id == std::any::TypeId::of::<f32>() => {
                        let f_val = value[idx].to_f32().ok_or(Error::BlockConversion)?;
                        <T as NumCast>::from(f_val.round()).ok_or(Error::BlockConversion)?
                    }
                    id if id == std::any::TypeId::of::<f64>() => {
                        let f_val = value[idx].to_f64().ok_or(Error::BlockConversion)?;
                        <T as NumCast>::from(f_val.round()).ok_or(Error::BlockConversion)?
                    }
                    _ => <T as NumCast>::from(value[idx]).ok_or(Error::BlockConversion)?,
                };
                next[idx] = converted_value;
            }
        }
        Ok(next)
    }
}

impl<T> Block<T>
where
    T: Copy + ToPrimitive + 'static,
{
    pub(crate) fn try_convert_to<U>(self) -> Result<Block<U>>
    where
        U: Copy + Default + NumCast + 'static,
    {
        Block::<U>::try_convert_from(self)
    }

    pub fn convert_to<U>(self) -> Block<U>
    where
        U: Copy + Default + NumCast + 'static,
    {
        self.try_convert_to().unwrap()
    }
}

impl<T> Block<T>
where
    T: Signed + Default + AddAssign + Copy,
{
    pub fn sum_of_abs_difference(&self, other: &Block<T>) -> T {
        let mut sum = T::default();
        for r in 0..BLOCK_ROWS {
            for c in 0..BLOCK_COLS {
                let idx = r * BLOCK_COLS + c;
                sum += (self[idx] - other[idx]).abs();
            }
        }
        sum
    }
}

impl<T> Block<T>
where
    T: PartialOrd + Copy,
{
    pub fn clamp(self, min: T, max: T) -> Self {
        let mut block = self;

        for r in 0..BLOCK_ROWS {
            for c in 0..BLOCK_COLS {
                let idx = r * BLOCK_COLS + c;
                if block[idx] < min {
                    block[idx] = min;
                } else if block[idx] > max {
                    block[idx] = max;
                }
            }
        }

        block
    }
}

impl<const N: usize, T> ToBytes for Block<T>
where
    T: num_traits::ToBytes<Bytes = [u8; N]>,
{
    fn to_bytes(&self) -> Vec<u8> {
        self.0.iter().flat_map(|x| x.to_be_bytes()).collect()
    }
}

impl<const N: usize, T> FromBytes for Block<T>
where
    T: Debug + num_traits::FromBytes<Bytes = [u8; N]>,
{
    fn from_bytes(bytes: &[u8]) -> (Self, usize) {
        let block_bytes = &bytes[..N * BLOCK_SIZE];
        let vec: Vec<T> = block_bytes
            .chunks_exact(N)
            .map(|x| T::from_be_bytes(x.try_into().expect("bytes")))
            .collect();
        let array: [T; BLOCK_SIZE] = vec.try_into().expect("block size array");
        (Self(array), N * BLOCK_SIZE)
    }
}

pub(crate) struct Blocks<T>(Vec<Block<T>>);

impl<T> Blocks<T> {
    pub(crate) fn new(blocks: Vec<Block<T>>) -> Self {
        Self(blocks)
    }

    pub(crate) fn iter(&self) -> BlocksIter<'_, T> {
        BlocksIter {
            idx: 0,
            blocks: &self.0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> std::ops::Index<usize> for Blocks<T> {
    type Output = Block<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

pub(crate) struct BlocksIter<'a, T> {
    idx: usize,
    blocks: &'a [Block<T>],
}

impl<'a, T> Iterator for BlocksIter<'a, T> {
    type Item = &'a Block<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.blocks.len() {
            None
        } else {
            let block = &self.blocks[self.idx];
            self.idx += 1;
            Some(block)
        }
    }
}

impl<const N: usize, T> ToBytes for Blocks<T>
where
    T: num_traits::ToBytes<Bytes = [u8; N]>,
{
    fn to_bytes(&self) -> Vec<u8> {
        let mut res = Vec::with_capacity(size_of::<usize>() + (self.0.len() * Block::<T>::size()));

        res.extend_from_slice(&self.0.len().to_be_bytes());
        for block in &self.0 {
            res.extend(block.to_bytes());
        }
        res
    }
}

impl<const N: usize, T> FromBytes for Blocks<T>
where
    T: Debug + num_traits::FromBytes<Bytes = [u8; N]>,
{
    fn from_bytes(bytes: &[u8]) -> (Self, usize)
    where
        Self: Sized,
    {
        let mut offset = 0;

        let len = usize::from_be_bytes(
            bytes[offset..offset + size_of::<usize>()]
                .try_into()
                .expect("usize"),
        );

        let mut blocks = Vec::with_capacity(len);

        offset += size_of::<usize>();

        for _ in 0..len {
            let (block, size) = Block::<T>::from_bytes(&bytes[offset..]);
            blocks.push(block);
            offset += size;
        }
        (Self(blocks), offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_default() {
        let block: Block<i32> = Block::default();
        for r in 0..Block::<i32>::rows() {
            for c in 0..Block::<i32>::cols() {
                assert_eq!(block.get(r, c), 0);
            }
        }
    }

    #[test]
    fn test_block_dimensions() {
        assert_eq!(Block::<i32>::cols(), 8);
        assert_eq!(Block::<i32>::rows(), 8);
    }

    #[test]
    fn test_block_indexing() {
        let mut block: Block<i32> = Block::default();

        // Test index access
        assert_eq!(block.get(0, 0), 0);

        // Test index mutation
        block.set(0, 0, 42);
        block.set(7, 7, 99);
        assert_eq!(block.get(0, 0), 42);
        assert_eq!(block.get(7, 7), 99);
    }

    #[test]
    fn test_block_from_array() {
        let arr = [1; 64];
        let block: Block<i32> = Block::from(arr);

        for r in 0..Block::<i32>::rows() {
            for c in 0..Block::<i32>::cols() {
                assert_eq!(block.get(r, c), 1);
            }
        }
    }

    #[test]
    fn test_block_from_array_with_different_values() {
        let mut arr = [0; 64];
        for i in 0..64 {
            arr[i] = i as i32;
        }
        let block: Block<i32> = Block::from(arr);

        for r in 0..Block::<i32>::rows() {
            for c in 0..Block::<i32>::cols() {
                assert_eq!(block.get(r, c), (r * BLOCK_COLS + c) as i32);
            }
        }
    }

    #[test]
    fn test_block_clone() {
        let mut block: Block<i32> = Block::default();
        block.set(0, 0, 42);
        block.set(7, 7, 99);

        let cloned = block;
        assert_eq!(cloned.get(0, 0), 42);
        assert_eq!(cloned.get(7, 7), 99);
    }

    #[test]
    fn test_block_copy() {
        let mut block: Block<i32> = Block::default();
        block.set(0, 0, 42);
        block.set(7, 7, 99);

        // Uses Copy trait
        let copied = block;
        assert_eq!(copied.get(0, 0), 42);
        assert_eq!(copied.get(7, 7), 99);
    }

    #[test]
    fn test_block_equality() {
        let mut block1: Block<i32> = Block::default();
        let mut block2: Block<i32> = Block::default();

        assert_eq!(block1, block2);

        block1.set(0, 0, 42);
        assert_ne!(block1, block2);

        block2.set(0, 0, 42);
        assert_eq!(block1, block2);
    }

    #[test]
    fn test_block_add() {
        let mut block1: Block<i32> = Block::default();
        let mut block2: Block<i32> = Block::default();

        block1.set(0, 0, 10);
        block1.set(7, 7, 20);
        block2.set(0, 0, 5);
        block2.set(7, 7, 15);

        let result = block1 + block2;
        assert_eq!(result.get(0, 0), 15);
        assert_eq!(result.get(7, 7), 35);
        assert_eq!(result.get(0, 1), 0); // Other elements should be 0
    }

    #[test]
    fn test_block_div() {
        let mut block: Block<i32> = Block::default();
        block.set(0, 0, 20);
        block.set(7, 7, 100);

        let result = block / 2;
        assert_eq!(result.get(0, 0), 10);
        assert_eq!(result.get(7, 7), 50);
        assert_eq!(result.get(0, 1), 0); // Other elements should be 0
    }

    #[test]
    fn test_block_iter() {
        let mut block: Block<i32> = Block::default();
        block.set(0, 0, 1);
        block.set(0, 1, 2);
        block.set(1, 0, 9); // row 1, col 0 = index 8

        let mut iter = block.iter();
        assert_eq!(iter.next(), Some(1)); // [0][0]
        assert_eq!(iter.next(), Some(2)); // [0][1]
        assert_eq!(iter.next(), Some(0)); // [0][2]

        // Skip to the 8th element (row 1, col 0)
        for _ in 0..5 {
            iter.next();
        }
        assert_eq!(iter.next(), Some(9)); // [1][0]
    }

    #[test]
    fn test_block_iter_full() {
        let arr = [1; 64];
        let block: Block<i32> = Block::from(arr);

        let collected: Vec<i32> = block.iter().collect();
        assert_eq!(collected.len(), 64);
        assert!(collected.iter().all(|&x| x == 1));
    }

    #[test]
    fn test_block_iter_empty_at_end() {
        let block: Block<i32> = Block::default();
        let mut iter = block.iter();

        // Consume all 64 elements
        for _ in 0..64 {
            assert!(iter.next().is_some());
        }

        // Should be None after consuming all
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_block_i16_to_f64() {
        let mut block: Block<i16> = Block::default();
        block.set(0, 0, 100);
        block.set(7, 7, -50);

        let f64_block = block.convert_to::<f64>();
        assert_eq!(f64_block.get(0, 0), 100.0);
        assert_eq!(f64_block.get(7, 7), -50.0);
        assert_eq!(f64_block.get(0, 1), 0.0);
    }

    #[test]
    fn test_block_i32_to_f64() {
        let mut block: Block<i32> = Block::default();
        block.set(0, 0, 1000);
        block.set(7, 7, -500);

        let f64_block = block.convert_to::<f64>();
        assert_eq!(f64_block.get(0, 0), 1000.0);
        assert_eq!(f64_block.get(7, 7), -500.0);
        assert_eq!(f64_block.get(0, 1), 0.0);
    }

    #[test]
    fn test_block_f64_to_i32() {
        let mut block: Block<f64> = Block::default();
        block.set(0, 0, 100.7);
        block.set(7, 7, -50.3);
        block.set(0, 1, 25.5);

        let i32_block = block.convert_to::<i32>();
        assert_eq!(i32_block.get(0, 0), 101); // 100.7 rounded
        assert_eq!(i32_block.get(7, 7), -50); // -50.3 rounded
        assert_eq!(i32_block.get(0, 1), 26); // 25.5 rounded
    }

    #[test]
    fn test_block_f64_to_i32_rounding() {
        let mut block: Block<f64> = Block::default();
        block.set(0, 0, 2.4); // Should round to 2
        block.set(0, 1, 2.5); // Should round to 3
        block.set(0, 2, 2.6); // Should round to 3
        block.set(1, 0, -2.4); // Should round to -2
        block.set(1, 1, -2.5); // Should round to -3
        block.set(1, 2, -2.6); // Should round to -3

        let i32_block = block.convert_to::<i32>();
        assert_eq!(i32_block.get(0, 0), 2);
        assert_eq!(i32_block.get(0, 1), 3);
        assert_eq!(i32_block.get(0, 2), 3);
        assert_eq!(i32_block.get(1, 0), -2);
        assert_eq!(i32_block.get(1, 1), -3);
        assert_eq!(i32_block.get(1, 2), -3);
    }

    #[test]
    fn test_block_debug_format() {
        let block: Block<i32> = Block::default();
        let debug_str = format!("{block:?}");
        assert!(debug_str.contains("Block"));
        assert!(debug_str.contains("inner"));
    }

    #[test]
    fn test_block_display_format() {
        let block: Block<i32> = Block::default();
        let display_str = format!("{block}");
        assert!(display_str.contains("Block"));
        assert!(display_str.contains("inner"));
    }

    #[test]
    fn test_mixed_type_operations() {
        // Test conversion chain: i16 -> f64 -> i32
        let mut block_i16: Block<i16> = Block::default();
        block_i16.set(0, 0, 100);
        block_i16.set(7, 7, -50);

        let block_f64 = block_i16.convert_to::<f64>();
        let block_i32 = block_f64.convert_to::<i32>();

        assert_eq!(block_i32.get(0, 0), 100);
        assert_eq!(block_i32.get(7, 7), -50);
    }

    #[test]
    fn test_block_operations_with_different_values() {
        let mut block1: Block<i32> = Block::default();
        let mut block2: Block<i32> = Block::default();

        // Fill with different patterns
        for r in 0..Block::<i32>::rows() {
            for c in 0..Block::<i32>::cols() {
                block1.set(r, c, (r * BLOCK_COLS + c) as i32);
                block2.set(r, c, ((r * BLOCK_COLS + c) * 2) as i32);
            }
        }

        let sum = block1 + block2;
        let quotient = block2 / 2;

        // Check a few values
        assert_eq!(sum.get(0, 0), 0); // 0 + 0
        assert_eq!(sum.get(0, 1), 3); // 1 + 2
        assert_eq!(sum.get(1, 0), 24); // 8 + 16

        assert_eq!(quotient.get(0, 0), 0); // 0 / 2
        assert_eq!(quotient.get(0, 1), 1); // 2 / 2
        assert_eq!(quotient.get(1, 0), 8); // 16 / 2
    }
}
