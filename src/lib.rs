
use anyhow::{Result, anyhow};

#[allow(unused_imports)]
use std::io::prelude::*;
use std::io::{BufWriter, Write, BufReader, Read};
use std::cmp::Reverse;

#[allow(unused_imports)]
use log::{info, trace};

use bitmaps::Bitmap;

fn get_candidates(b: Bitmap<10>) -> Vec<u8>{
    let mut candidates = Vec::with_capacity(9);
    let mut from_idx = 0;
    loop{
        if let Some(index) = b.next_index(from_idx){
            from_idx = index;
            candidates.push(index as u8);
        } else{
            break;
        }

    }
    candidates
}

pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    // sort_unstable_by_key is faster than sort_by_key
    indices.sort_unstable_by_key(|&i| Reverse(&data[i]));
    indices
}

type Board = Vec<Vec<u8>>;
type Cell = (usize, usize);
type FilledState = Vec<Bitmap<10>>;
pub enum CombType{
    Row,
    Col,
}
pub struct SudukuState{
    /// 行填充状态 rows_filled_state[0].get(9) == true 表示 第零行9已被填充
    rows_filled_state: FilledState,
    /// 列填充状态 cols_filled_state[1].get(8) == true 表示 第一列8已被填充
    cols_filled_state: FilledState,
    /// 9 x 九宫格填充状态 grids_filled_state[3].get(7) == true 表示 第三宫格7已被填充
    grids_filled_state: FilledState,
   
    // 侯选数字 candidates[1][2] 表示第1行第2列的侯选数字列表
    // 数字0~9 共10个
    candidates: Vec<Vec<Bitmap<10>>>,

}

pub struct Suduku{
    data: Board,
    state: SudukuState,
}

impl Default for SudukuState{
    fn default() -> Self{
        fn new_filled_state()-> FilledState{
            let mut tpl = Bitmap::<10>::mask(1);
            let tpl = tpl.as_value();

            vec![
                Bitmap::<10>::from_value(*tpl),
                Bitmap::<10>::from_value(*tpl),
                Bitmap::<10>::from_value(*tpl),

                Bitmap::<10>::from_value(*tpl),
                Bitmap::<10>::from_value(*tpl),
                Bitmap::<10>::from_value(*tpl),

                Bitmap::<10>::from_value(*tpl),
                Bitmap::<10>::from_value(*tpl),
                Bitmap::<10>::from_value(*tpl),
            ]
        }

        let mut candidates_tpl = Bitmap::<10>::mask(10);
        candidates_tpl.set(0, false);
        let candidates_tpl = candidates_tpl.as_value();

        let mut candidates = Vec::with_capacity(9);
        for i in 0..9{
            let mut row = Vec::with_capacity(9);
            for j in 0..9{
                    row.push(Bitmap::<10>::from_value(*candidates_tpl));
            }
            candidates.push(row);
        }
        SudukuState{
            rows_filled_state: new_filled_state(),
            cols_filled_state: new_filled_state(),
            grids_filled_state: new_filled_state(),
            candidates,
        }
    }
}

impl SudukuState{
    fn fill(&mut self, pos: Cell, n: u8)-> Result<()>{
        assert!(n > 0 && n < 10);

        let (r,c) = pos;
        assert!(r < 9 && c < 9);

        let n = n as usize;

        if self.rows_filled_state[r].set(n, true) {
            return Err(anyhow!("conflict"))
        }

        if self.cols_filled_state[c].set(n, true){
            return Err(anyhow!("conflict"))
        }

        let grid = (r / 3) * 3 +  (c / 3);
        if self.grids_filled_state[grid].set(n, true){
            return Err(anyhow!("conflict"))
        }

        // remove candidates
        self.candidates[r][c] &= Bitmap::new();
        for i in 0..9{
            // 行
            self.candidates[r][i].set(n, false);
            // 列
            self.candidates[i][c].set(n, false);
            // 宫格
            let gr = r - r%3 + i / 3 ;
            let gc = c - c % 3 + i % 3;
            self.candidates[gr][gc].set(n, false);
        }
        Ok(())
    }

    fn candidates_eliminate(&mut self, cell: Cell, n: u8) -> bool{
        self.candidates[cell.0][cell.1].set(n as usize, false)
    }

    fn get_candidates(&self, pos:Cell) -> Vec<u8>{
       get_candidates(self.candidates[pos.0][pos.1])
    }
    /// 消除术，宫格条式消除
    /// 当宫格内行或列的candidates， 不在其他行或列里, 则对应的整行或整列非本宫格,应消除这些candidates
    /// 比如 (0,1) (0,2) 共同candidates 是 [2,3], 且在 (1,0) (1,1) (1,2) (2,0) (2,1) (2, 2)中都没有2,3. 则应该消除 (0,p) p=3,4,5,6,7,9的candidates中的[2,3]
    fn grid_line_candidates_eliminate(&mut self) -> usize{
        let mut counter = 0;
        for i in 0..9{
            for (cell1, cell2, n) in self.find_grid_line_combose(i){
                //println!("grid_line_combose {:?}{:?} with {}", cell1, cell2, n);
                if cell1.0 == cell2.0{
                    // remove row
                    for col in 0..9{
                        if col / 3 != cell1.1/3{
                            //println!("grid_line_combose eliminate ({},{})", cell1.0, col);
                            let old = self.candidates_eliminate((cell1.0, col), n);
                            if old == true{
                                counter+=1;
                            }
                        }
                    }

                }else if cell1.1==cell2.1{
                    // remove col
                    for row in 0..9{
                        if row / 3 != cell1.0/3{
                            //println!("grid_line_combose eliminate ({},{})", row, cell1.1);
                            let old=self.candidates_eliminate((row, cell1.1), n);
                            if old == true{
                                counter+=1;
                            }
                        }
                    }

                }
            }
        }
        counter
    }

    /// 找到某一行、列、宫格中，有唯一可填数值的单元
    fn find_only_one_candidates(&mut self) -> Vec<(Cell, u8)> {
        let mut result :Vec<(Cell, u8)> = Vec::new();
        for num in 0..9{
        //for num in 5..6
            // 一行， 一列， 一个宫格为一组
            for grp in 0..9{
                //println!("num {}. group id {}", num, grp);
            //for grp in 0..1
                let mut row_counter = 0;
                let mut col_counter = 0;
                let mut grid_counter = 0;

                let mut row_unit_last: Cell = (0,0);
                let mut col_unit_last: Cell = (0,0);
                let mut grid_unit_last: Cell = (0,0);
                // 每组9个单元格
                for unit in 0..9{
                    // 行 
                    let row_unit = self.candidates[grp][unit];
                    let col_unit = self.candidates[unit][grp];
                    // lefttop_cell
                    let lt :Cell = (grp / 3 * 3, grp % 3 * 3);
                    let gr = lt.0 + unit / 3 ;
                    let gc = lt.1 + unit % 3;
                    let grid_unit = self.candidates[gr][gc];
                    //println!("({},{}): {:?}", grp, unit, get_candidates(row_unit));
                    //println!("({},{}): {:?}", unit, grp, get_candidates(col_unit));
                    //println!("({},{}): {:?}", gr, gc, get_candidates(grid_unit));

                    if row_unit.get(num){
                        row_counter+=1;
                        row_unit_last = (grp, unit);
                    }

                    if col_unit.get(num){
                        col_counter+=1;
                        col_unit_last = (unit, grp);
                    }

                    if grid_unit.get(num){
                        grid_counter+=1;
                        grid_unit_last = (gr, gc);
                    }

                }
                fn push_to_result(result: &mut Vec<(Cell, u8)>, cell: Cell, n:u8){
                    if let None = result.iter().position(|i|i.0.0 == cell.0 && i.0.1 == cell.1 && i.1 == n) {
                        result.push((cell, n));
                    }
                }
                if row_counter==1 {
                    push_to_result(&mut result, row_unit_last, num as u8);
                    //println!("group id {}. result {:?}", grp, result);
                    continue; // next grp
                }
                if col_counter==1 {
                    push_to_result(&mut result, col_unit_last, num as u8);
                    //println!("group id {}. result {:?}", grp, result);
                    continue; // next grp
                }
                if grid_counter==1 {
                    push_to_result(&mut result, grid_unit_last, num as u8);
                    //println!("group id {}. result {:?}", grp, result);
                    continue; // next grp
                }
            }

        }

        result
    }
    /// 某个宫格内一行中某两个单元, 如果其共同侯选N 在宫格内其他行都不存在，则可以断定该行非宫格内的单元，不可能填入N
    /// 列也相同
    /// 找到宫格内这种直线组合
    fn find_grid_line_combose(&mut self, grid_id: usize) -> Vec<(Cell, Cell, u8)>{
        let mut result : Vec<(Cell,Cell, u8)> = Vec::new();
        // lefttop_cell
        let lt :Cell = (grid_id / 3 * 3, grid_id % 3 * 3);
        // all combose
        const ALL_COMB: &[(CombType,Cell,Cell)] = &[
            // 行
            (CombType::Row, (0,0), (0,1)),
            (CombType::Row, (0,0), (0,2)),
            (CombType::Row, (0,1), (0,2)),

            (CombType::Row, (1,0), (1,1)),
            (CombType::Row, (1,0), (1,2)),
            (CombType::Row, (1,1), (1,2)),

            (CombType::Row, (2,0), (2,1)),
            (CombType::Row, (2,0), (2,2)),
            (CombType::Row, (2,1), (2,2)),

            // 列
            (CombType::Col, (0,0), (1,0)),
            (CombType::Col, (0,0), (2,0)),
            (CombType::Col, (1,0), (2,0)),
 
            (CombType::Col, (0,1), (1,1)),
            (CombType::Col, (0,1), (2,1)),
            (CombType::Col, (1,1), (2,1)),
 
            (CombType::Col, (0,2), (1,2)),
            (CombType::Col, (0,2), (2,2)),
            (CombType::Col, (1,2), (2,2)),
        ];

        for (comb_type, p1, p2) in ALL_COMB{
            // cell1
            let c1 :Cell = (p1.0 + lt.0, p1.1 + lt.1);  
            // cell2
            let c2 :Cell = (p2.0 + lt.0, p2.1 + lt.1);  
            let c1_can = self.candidates[c1.0][c1.1] ; 
            let c2_can = self.candidates[c2.0][c2.1];
            if c1_can.is_empty() || c2_can.is_empty(){
                continue
            }
            //println!("combose: {:?} {:?}", c1, c2);

            // combose common candidates bitmap
            let common = c1_can & c2_can; 
            if common.is_empty(){
                continue;
            }

            // 对比相邻的行列
            let mut compare_cells: Vec<Cell> = Vec::with_capacity(6);
            match comb_type{
                CombType::Row => {
                    for row in 0..3{
                        if row == c1.0 {
                            // same row
                            continue;
                        }
                        for col in 0..3{
                            compare_cells.push((row+lt.0, col+lt.1))
                        }
                    }
                },
                CombType::Col => {
                    for col in 0..3{
                        if col == c1.1 {
                            // same col
                            continue;
                        }
                        for row in 0..3{
                            compare_cells.push((row+lt.0, col+lt.1))
                        }
                    }

                },
            }

            let mut candidates_for_elimate: Vec<u8> = Vec::with_capacity(4);
            for element in get_candidates(common) {
                let mut founded = true;
                for cell in &compare_cells{
                    let compare_cell_candidate = self.candidates[cell.0][cell.1];
                    if compare_cell_candidate.get(element as usize){
                        founded = false;
                        break;
                    }
                }
                if founded{
                    result.push((c1, c2, element))
                }
            }

        }
        result
    }

}

impl Suduku{
    pub fn new() -> Self{
        Suduku{
            data: vec![vec![0; 9]; 9], 
            state: SudukuState::default(),
        }
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
    /// use suduku_resolve::Suduku;
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
        Suduku{
            data,
            state: SudukuState::default(),
        }
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
    pub fn resolv(&mut self)-> Result<()>{
        self.init_state();

        loop{
            loop {
                // 不断填充只有唯一侯选的单元
                let result = self.state.find_only_one_candidates();
                // println!("result: {:?}", result);
                for (cell, num) in &result{
                    let result = self.fill(*cell, *num);
                    // println!("fill result: {:?}", result);
                }
                // println!("suduku:\n{}", suduku.to_string().unwrap());
                if result.len() == 0{
                    break;
                }
            }

            // 当唯一侯选都填充完毕，则进行一次宫格直线的侯选消除, 消除后再检查填充唯一侯选
            let n = self.state.grid_line_candidates_eliminate();
            // println!("suduku:\n{}", suduku.to_string().unwrap());
            if n == 0{
                break;
            }
        }

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

    fn fill(&mut self, pos: Cell, n: u8)-> Result<()>{
        self.data[pos.0][pos.1] = n;
        self.state.fill(pos, n)
    }
    fn get(&mut self, pos: Cell) -> u8{
        self.data[pos.0][pos.1]
    }
    fn erase(&mut self, _pos: Cell)-> Result<()>{
        Ok(())
    }
    fn init_state(&mut self) -> Result<()>{
        for i in 0..9{
            for j in 0..9{
                let n = self.data[i][j];
                if n != 0{
                    if let Err(err) = self.state.fill((i,j), n) {
                        return Err(err);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test{
    use super::*;

    #[ignore]
    #[test]
    fn test_bitmap(){
        //let mut b = Bitmap::<10>::mask(10);
        let mut b = Bitmap::<10>::new();
        b.set(2, true);
        b.set(7, true);
        println!("1: {}", b.get(1));
        println!("2: {}", b.get(2));
        println!("7: {}", b.get(7));

        println!("{}", b.next_index(0).unwrap());
    }

    #[ignore]
    #[test]
    fn test_suduku_state() {
        let mut state = SudukuState::default();
        println!("state row0 with digit 9: {}", state.rows_filled_state[0].get(9));
        println!("state col8 with digit 0: {}", state.cols_filled_state[8].get(0));
        state.fill((1,1), 3); // same row
        state.fill((1,2), 4);
        state.fill((1,3), 5);
        state.fill((3,5), 7); // same column
        state.fill((4,5), 8);
        state.fill((2,4), 9); // same grid
        assert_eq!(vec![1,2, 6],state.get_candidates((1,5)));
    }

    #[ignore]
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

        // let result = suduku.resolv();
        // println!("resolved? {:?}\n", result);
        // println!("suduku: {:?}", suduku.to_string().unwrap());
    }

    #[test]
    fn test_resolv(){
        let data = r#"
        960 030 000
        000 800 050
        070 000 000

        405 000 900
        200 000 300
        000 700 000

        000 500 087
        000 096 000
        000 000 000
        "#; 
        println!("{:?}", data);

        let reader = BufReader::new(data.as_bytes());
        let mut suduku = Suduku::load_from(reader).unwrap();
        println!("suduku:\n{}", suduku.to_string().unwrap());

        // let result = suduku.init_state();
        // println!("result: {:?}", result);
        // // for i in 0..9{
        // //     for j in 0..9{
        // //         println!("candidates for {:?}: {:?}", (i,j) as Cell ,suduku.state.get_candidates((i, j)));
        // //     }
        // // }
        // let result = suduku.state.find_only_one_candidates();
        // println!("result: {:?}", result);
        // for (cell, num) in &result{
        //     let result = suduku.fill(*cell, *num);
        //     println!("fill result: {:?}", result);
        // }

        // println!("suduku:\n{}", suduku.to_string().unwrap());
        // // for i in 0..9{
        // //     for j in 0..9{
        // //         println!("candidates for {:?}: {:?}", (i,j) as Cell ,suduku.state.get_candidates((i, j)));
        // //     }
        // // }

        // // let result = suduku.state.find_only_one_candidates();
        // // println!("result: {:?}", result);
        // // for (cell, num) in &result{
        // //     let result = suduku.fill(*cell, *num);
        // //     println!("fill result: {:?}", result);
        // // }
        // // println!("suduku:\n{}", suduku.to_string().unwrap());

        // // let result = suduku.state.find_only_one_candidates();
        // // println!("result: {:?}", result);
        // // for (cell, num) in &result{
        // //     let result = suduku.fill(*cell, *num);
        // //     println!("fill result: {:?}", result);
        // // }
        // // println!("suduku:\n{}", suduku.to_string().unwrap());

        // // let result = suduku.state.find_only_one_candidates();
        // // println!("result: {:?}", result);
        // // for (cell, num) in &result{
        // //     let result = suduku.fill(*cell, *num);
        // //     println!("fill result: {:?}", result);
        // // }
        // // println!("suduku:\n{}", suduku.to_string().unwrap());

        // // suduku.state.grid_line_candidates_eliminate();
        // // println!("suduku:\n{}", suduku.to_string().unwrap());

        // // let result = suduku.state.find_only_one_candidates();
        // // println!("result: {:?}", result);
        // // for (cell, num) in &result{
        // //     let result = suduku.fill(*cell, *num);
        // //     println!("fill result: {:?}", result);
        // // }
        // // println!("suduku:\n{}", suduku.to_string().unwrap());

        // // let result = suduku.state.find_only_one_candidates();
        // // println!("result: {:?}", result);
        // // for (cell, num) in &result{
        // //     let result = suduku.fill(*cell, *num);
        // //     println!("fill result: {:?}", result);
        // // }
        // // println!("suduku:\n{}", suduku.to_string().unwrap());

        // // let result = suduku.state.find_only_one_candidates();
        // // println!("result: {:?}", result);
        // // for (cell, num) in &result{
        // //     let result = suduku.fill(*cell, *num);
        // //     println!("fill result: {:?}", result);
        // // }
        // // println!("suduku:\n{}", suduku.to_string().unwrap());

        // // let result = suduku.state.find_only_one_candidates();
        // // println!("result: {:?}", result);
        // // for (cell, num) in &result{
        // //     let result = suduku.fill(*cell, *num);
        // //     println!("fill result: {:?}", result);
        // // }
        // // println!("suduku:\n{}", suduku.to_string().unwrap());

        // // let result = suduku.state.find_only_one_candidates();
        // // println!("result: {:?}", result);
        // // for (cell, num) in &result{
        // //     let result = suduku.fill(*cell, *num);
        // //     println!("fill result: {:?}", result);
        // // }
        // // println!("suduku:\n{}", suduku.to_string().unwrap());

        // // suduku.state.grid_line_candidates_eliminate();
        // // println!("suduku:\n{}", suduku.to_string().unwrap());

        // loop{
        // loop {
        //     let result = suduku.state.find_only_one_candidates();
        //     println!("result: {:?}", result);
        //     for (cell, num) in &result{
        //         let result = suduku.fill(*cell, *num);
        //         println!("fill result: {:?}", result);
        //     }
        //     println!("suduku:\n{}", suduku.to_string().unwrap());
        //     if result.len() == 0{
        //         break;
        //     }
        // }

        // let n = suduku.state.grid_line_candidates_eliminate();
        // println!("suduku:\n{}", suduku.to_string().unwrap());
        //     if n == 0{
        //         break;
        //     }
        // }
        // for i in 0..9{
        //     for j in 0..9{
        //         println!("candidates for {:?}: {:?}", (i,j) as Cell ,suduku.state.get_candidates((i, j)));
        //     }
        // }

        let result = suduku.resolv();
        println!("resolved? {:?}\n", result);
        println!("suduku:\n{}", suduku.to_string().unwrap());
    }
}
