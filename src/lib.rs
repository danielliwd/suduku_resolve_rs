
use anyhow::{Result, anyhow};

use std::io::prelude::*;
use std::io::{BufWriter, Write, BufReader, Read};
use std::cmp::Reverse;

#[allow(unused_imports)]
use log::{info, trace};

type Board = Vec<Vec<u8>>;
type Cell = (usize, usize);
pub struct Suduku{
    data: Board,
}

pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    // sort_unstable_by_key is faster than sort_by_key
    indices.sort_unstable_by_key(|&i| Reverse(&data[i]));
    indices
}

impl Suduku{
    pub fn new() -> Self{
        Suduku{data: vec![vec![0; 9]; 9]}
    }

    /// create a Suduku object by given `Board` data
    ///
    /// # Arguments
    ///
    /// * `data` - suduku Board data
    ///
    /// # Examples
    ///
    /// ```
    /// let mut suduku = Suduku::create(vec![
    ///     vec![9,6,0, 0,3,0,  0,0,0],
    ///     vec![0,0,0, 8,0,0,  0,5,0],
    ///     vec![0,7,0, 0,0,0,  0,0,0],
    ///
    ///     vec![4,0,5, 0,0,0,  9,0,0],
    ///     vec![2,0,0, 0,0,0,  3,0,0],
    ///     vec![0,0,0, 7,0,0,  0,0,0],
    ///
    ///     vec![0,0,0, 5,0,0,  0,8,7],
    ///     vec![0,0,0, 0,9,6,  0,0,0],
    ///     vec![0,0,0, 0,0,0,  0,0,0],
    /// ]);
    /// ```
    ///
    pub fn create(data: Board) -> Self{
        Suduku{data}
    }

    pub fn load_from_file(file: &str) -> Result<Suduku>{
        Self::load_from(std::fs::File::open(file)?)
    }

    pub fn load_from_buffer(buf: &[u8]) -> Result<Suduku>{
        let mut data: Vec<u8> = Vec::with_capacity(9*9);
        for c in buf{
            if *c >= b'0' && *c <= b'9' {
                data.push(*c - b'0')
            }
        }
        if data.len() < 9*9{
            return Err(anyhow!("data length is not enough"));
        }

        data.truncate(9*9);
        let chunks: Vec<Vec<u8>> = data.chunks(9).map(|chunk| chunk.to_vec()).collect();
        Ok(Suduku::create(chunks))
    }
    pub fn load_from(mut reader: impl std::io::Read) -> Result<Suduku>{
        let mut buf = Vec::with_capacity(1024);
        reader.read_to_end(&mut buf).unwrap();
        return Self::load_from_buffer(&buf[..])
    }


    pub fn to_string(&self) -> Result<String>{
        let mut bufwriter = BufWriter::new(Vec::with_capacity((9*4 as usize).pow(2)));
        for i in 0..9{
            for j in 0..9{
                let n = self.data[i][j];
                bufwriter.write(&[n + b'0', b' ']).unwrap();
                if j%3==2{
                    bufwriter.write(&[b' ']).unwrap();
                }
            }
            bufwriter.write(&[b'\n']).unwrap();
            if i%3 ==2 {
                bufwriter.write(&[b'\n']).unwrap();
            }
        }

        Ok(String::from_utf8(bufwriter.into_inner()?)?)
    }

    pub fn resolv(&mut self)->Result<()>{
        let mut num_counter_vec = Self::static_data(&self.data);
        let n_order = argsort(&num_counter_vec[..])
            .into_iter()
            .filter(|i|*i!=0)
            .map(|i|i as u8)
            .collect();
        println!("n_order: {:?}", n_order);
        Self::resolv_inner(&mut self.data, (0,0), 0, &n_order)
    }


    fn static_data(data: &Board)-> Vec<u8>{
        let mut num_counter_vec = vec![0;10];
        for i in 0..9{
        for j in 0..9{
            let n: usize = data[i][j].into();
            if n!=0 {
                num_counter_vec[n] += 1;
            }

        }
        }
    
        num_counter_vec

    }

    fn resolv_inner(board: &mut Board, pos: Cell, level: usize, n_order: &Vec<u8>) -> Result<()> {
        fn conflict_with(board: &Board, pos: Cell, val: u8) -> bool{

            // 冲突检查
            for i in 0..9 {

                // 九宫格pos
                let posx: Cell= (pos.0 - pos.0 % 3 + i / 3, pos.1 - pos.1 % 3 + i % 3);

                if 
                // 行
                board[pos.0][i] == val || 
                // 列
                board[i][pos.1] == val ||
                // 九宫格
                board[posx.0][posx.1] == val
                {
                    return true
                }

            }

            false
        }

        fn find_next_empty_pos(board: &Board, pos: Cell) -> Option<Cell> {

            let (mut i,mut j) = pos;
            loop {
                if i >= 9 {
                    break;
                }
                if j >= 9 {
                    i+=1;
                    j=0;
                    continue;
                }

                if board[i][j] == 0 {
                    return Some((i,j))
                }

                j+=1;


            }

            None
        }

        loop{

            if let Some(next_pos) = find_next_empty_pos(board, pos){
                //for n in 1..=9{
                for n in &vec![5u8, 7, 9, 3, 6, 8, 2, 4, 1]{
                //for n in n_order{
                    if conflict_with(board, next_pos, *n){
                        continue
                    }
                    // if not conflict, fill n to board
                    trace!("{: <3$}{:?} try: {}","", next_pos, *n, level*2 );
                    board[next_pos.0][next_pos.1] = *n ;
                    if let Ok(result) = Self::resolv_inner(board, next_pos, level+1, n_order){
                        return Ok(result)
                    }else{
                        // recover and continue the next try
                        trace!("{: <2$}{:?} backward", "", next_pos, level*2);
                        board[next_pos.0][next_pos.1] = 0;
                    }
                }
                return Err(anyhow!("can't resolv"))
            } else{
                break;
            }
        }
        
        Ok(())
    }

}

#[cfg(test)]
mod test{
    use super::*;

    #[test]
    fn test_load(){
        let data: Vec<u8> = vec![
        9,6,0, 0,3,0,  0,0,0,
        0,0,0, 8,0,0,  0,5,0,
        0,7,0, 0,0,0,  0,0,0,

        4,0,5, 0,0,0,  9,0,0,
        2,0,0, 0,0,0,  3,0,0,
        0,0,0, 7,0,0,  0,0,0,

        0,0,0, 5,0,0,  0,8,7,
        0,0,0, 0,9,6,  0,0,0,
        0,0,0, 0,0,0,  0,0,0,
        ].into_iter().map(|i|i+b'0').collect();

        println!("{:?}", data);

        let reader = BufReader::new(data.as_slice());
        let mut suduku = Suduku::load_from(reader).unwrap();
        println!("suduku: {:?}", suduku.to_string().unwrap());

        let result = suduku.resolv();
        println!("resolved? {:?}\n", result);
        println!("suduku: {:?}", suduku.to_string().unwrap());

    }

}
