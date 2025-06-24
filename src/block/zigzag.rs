use super::Block;

#[rustfmt::skip]
const ZIGZAG: [[usize; 8]; 8] = [
     [0,  1,  5,  6, 14, 15, 27, 28],
     [2,  4,  7, 13, 16, 26, 29, 42],
     [3,  8, 12, 17, 25, 30, 41, 43],
     [9, 11, 18, 24, 31, 40, 44, 53],
     [10, 19, 23, 32, 39, 45, 52, 54],
     [20, 22, 33, 38, 46, 51, 55, 60],
     [21, 34, 37, 47, 50, 56, 59, 61],
     [35, 36, 48, 49, 57, 58, 62, 63],
];

impl<T> Block<T>
where
    T: Copy,
{
    pub fn zigzag(self) -> Self {
        let mut next = self.0;
        let temp = next;

        for r in 0..8 {
            for c in 0..8 {
                let zz = ZIGZAG[r][c];
                let zz_r = zz / 8;
                let zz_c = zz % 8;
                next[r * 8 + c] = temp[zz_r * 8 + zz_c];
            }
        }

        Self(next)
    }

    pub fn zagzig(self) -> Self {
        let mut next = self.0;
        let temp = next;

        for r in 0..8 {
            for c in 0..8 {
                let zz = ZIGZAG[r][c];
                let zz_r = zz / 8;
                let zz_c = zz % 8;
                next[zz_r * 8 + zz_c] = temp[r * 8 + c];
            }
        }

        Self(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_zagzig() {
        let block: Block<i32> = Block([
            52, 55, 61, 66, 70, 61, 64, 73, 63, 59, 55, 90, 109, 85, 69, 72, 62, 59, 68, 113, 144,
            104, 66, 73, 63, 58, 71, 122, 154, 106, 70, 69, 67, 61, 68, 104, 126, 88, 68, 70, 79,
            65, 60, 70, 77, 68, 58, 75, 85, 71, 64, 59, 55, 61, 65, 83, 87, 79, 69, 68, 65, 76, 78,
            94,
        ]);

        assert_eq!(block.zigzag().zagzig(), block);
    }
}
