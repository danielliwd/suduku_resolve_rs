
use anyhow::{Result, anyhow, Context};
use std::mem;

#[allow(unused_imports)]
use std::io::prelude::*;
use std::io::{BufWriter, Write, BufReader, Read};
use std::cmp::Reverse;
use bitmaps::Bitmap;

#[allow(unused_imports)]
use log::{info, trace};

const N: usize=9;
type Board = [[u8;N];N];
type Cell = (usize, usize);

// 每个单元格关联的单元格数
const RN: usize = N + N / 3 * 4 - 1;
/// 关联矩阵。 某个单元格所在行，列，宫格的相关联单元格。
type RelatedMatrix = [[[Cell;RN];N];N];

const fn create_related_matrix()-> RelatedMatrix{
    let mut matrix = [[[(0,0);RN];N];N];
    const fn matrix_cell(pos: Cell)-> [Cell;RN]{
        let mut cells = [(0,0);RN];
        let mut id = 0;
        let mut c = 0;
        while c < N {
            // 行
            let (row,col) = (pos.0, c);
            if c!=pos.1{
                cells[id] = (row,col);
                id +=1;
            }
            // 列
            let (row, col) = (c, pos.1);
            if c!=pos.0{
                cells[id] = (row,col);
                id +=1;
            }
            let (row,col) = (pos.0 - pos.0 % 3 + c / 3, pos.1 - pos.1 % 3 + c % 3);
            if row!=pos.0 && col != pos.1{
                cells[id] = (row,col);
                id +=1;
            }
            // 宫格
            c+=1;
        }
        cells
    }
    let mut row = 0;
    while row<N{
        let mut col = 0;
        while col< N {
            matrix[row][col] = matrix_cell((row,col));
            col+=1;
        }
        row += 1;
    }
    matrix
}

// #[allow(long_running_const_eval)]
const RELATED_MATRIX : RelatedMatrix = create_related_matrix();


type Combose = (Cell, Cell);
type Compare = [Cell;6];
type GridLineComboseAndCompare = [(Combose, Compare);18];

#[allow(long_running_const_eval)]
const GRID_LINE_COMBOSE_AND_COMPARE: GridLineComboseAndCompare = create_grid_line_combose_and_compare();
const fn create_grid_line_combose_and_compare()-> GridLineComboseAndCompare{
    //let mut comb = [([(0,0);2],[(0,0);6]);18];
    let mut comb: GridLineComboseAndCompare = [
        // 行
        (((0,0), (0,1)),[(0,0);6]),
        (((0,0), (0,2)),[(0,0);6]), 
        (((0,1), (0,2)),[(0,0);6]),

        (((1,0), (1,1)),[(0,0);6]),
        (((1,0), (1,2)),[(0,0);6]),
        (((1,1), (1,2)),[(0,0);6]),

        (((2,0), (2,1)),[(0,0);6]),
        (((2,0), (2,2)),[(0,0);6]),
        (((2,1), (2,2)),[(0,0);6]),

        // 列
        (((0,0), (1,0)),[(0,0);6]),
        (((0,0), (2,0)),[(0,0);6]),
        (((1,0), (2,0)),[(0,0);6]),

        (((0,1), (1,1)),[(0,0);6]),
        (((0,1), (2,1)),[(0,0);6]),
        (((1,1), (2,1)),[(0,0);6]),

        (((0,2), (1,2)),[(0,0);6]),
        (((0,2), (2,2)),[(0,0);6]),
        (((1,2), (2,2)),[(0,0);6]),
    ];
    const fn comb_compare_with(towcell: Combose)-> Compare{
        let mut comp : Compare = [(0,0);6];
        let mut id = 0;
        let (cell1, cell2) = towcell;
        if cell1.0 == cell2.0 {
            // 行组合
            let mut row = 0;
            while row<3{
                if row == cell1.0 {
                    row+=1;
                    continue;
                }
                comp[id] = (row, 0);
                comp[id+1] = (row, 1);
                comp[id+2] = (row, 2);
                id += 3;
                row+=1;
            }
        } else if cell1.1 == cell2.1 {
            // 列组合
            let mut col = 0;
            while col <3{
                if col == cell1.1 {
                    col+=1;
                    continue;
                }
                comp[id] = (0, col);
                comp[id+1] = (1, col);
                comp[id+2] = (2, col);
                id += 3;
                col+=1;
            }
        }
        comp
    }
    let mut c = 0;
    while c < 18{
        comb[c].1 = comb_compare_with(comb[c].0);
        c+=1;
    }
    comb
}


// get bits in bitmap
#[inline(always)]
fn get_bits(b: Bitmap<10>) -> Vec<u8>{
    let mut bits = Vec::with_capacity(N);
    let mut from_idx = 0;
    loop{
        if let Some(index) = b.next_index(from_idx){
            from_idx = index;
            bits.push(index as u8);
        } else{
            break;
        }

    }
    bits
}

/// 行、列的填充状态, 某数字被填充，则 get(n) == true
/// bitmap可以进行位运算, 比较方便
type FilledState = [Bitmap<10>;N];
type Candidates = [[Bitmap<10>;N];N];

#[derive(Clone, Copy)]
pub struct SudokuState{
    /// 空格数量
    blank_counts: u8,
    /// 行填充状态 rows_filled_state[0].get(9) == true 表示 第零行9已被填充
    rows_filled_state: FilledState,
    /// 列填充状态 cols_filled_state[1].get(8) == true 表示 第一列8已被填充
    cols_filled_state: FilledState,
    /// 9 x 九宫格填充状态 grids_filled_state[3].get(7) == true 表示 第三宫格7已被填充
    grids_filled_state: FilledState,
   
    /// 侯选
    /// 侯选数字 candidates[1][2] 表示第1行第2列的侯选数字列表
    /// 数字0~9 共10个
    /// if candidates[1][2].get(3) == true, 表示3是(1,2)这个cell的侯选
    candidates: Candidates,
}

#[derive(Clone, Copy)]
pub struct Sudoku{
    board: Board,
    state: SudokuState,
}


impl Sudoku {
    fn create()-> Sudoku{
        Sudoku{
            board: [[0;N];N],
            state: SudokuState::default(),
        }
    }

    fn from_bytes(data: &[u8])-> Sudoku{
        assert!(data.len() >= N*N);
        let mut sudoku = Self::create();
        // row
        let mut row=0;
        // col
        let mut col=0;
        for num in data.iter().take(N*N){
            if *num != 0 {
                sudoku.board[row][col] = *num;
            }
            col+=1;
            if col>=N{
                row+=1;
                col=0;
            }
            if row >= N {
                break;
            }
        }
        sudoku
    }
    pub fn load_from_buffer(buf: &[u8]) -> Result<Sudoku>{
        let mut data = [0u8;N*N];
        let mut idx = 0;

        for c in buf{
            if idx > N*N {
                break;
            }
            if *c >= b'0' && *c <= b'9' {
                data[idx] = *c - b'0';
                idx+=1;
            }
        }
        Ok(Self::from_bytes(&data[..]))
    }
    pub fn load(mut reader: impl std::io::Read) -> Result<Sudoku>{
        let mut buf = Vec::with_capacity(1024);
        reader.read_to_end(&mut buf).map(|i|i as u8 -b'0').unwrap();
        Self::load_from_buffer(&buf[..])
    }

    pub fn load_from_file(file: &str) -> Result<Sudoku>{
        Self::load(std::fs::File::open(file)?)
    }

    fn init_state(&mut self) -> Result<()>{
        for i in 0..N{
            for j in 0..N{
                let n = self.board[i][j];
                if n != 0{
                    if let Err(err) = self.state.fill((i,j), n) {
                        return Err(err);
                    }
                }
            }
        }
        Ok(())
    }

    fn fill(&mut self, pos: Cell, n: u8)-> Result<()>{
        self.board[pos.0][pos.1] = n;
        self.state.fill(pos, n)
    }

    /// 一轮次的拟人聪明解法
    pub fn resolv_inner_smart_once(&mut self)-> Result<bool>{
        // 只要还有单一侯选，则一直填充
        while self.resolv_inner_fill_only_one()? > 0{
        };

        //println!("has no only one:\n{}", self.to_string());
        // 当唯一侯选都填充完毕，则进行一次宫格直线的侯选消除, 消除后再检查填充唯一侯选
        let counter = self.state.line_candidates_eliminate()?;
        //println!("line candidates_eliminate counts: {}", counter);

        Ok(counter > 0)
    }
    /// 填充只有唯一侯选的单元, 返回填充单元格数量
    pub fn resolv_inner_fill_only_one(&mut self)->Result<usize>{
        let result = self.state.find_only_one_candidates();
        //println!("only candidate fill: {:?}", result);
        for (cell, num) in &result{
            self.fill(*cell, *num)?;
        }
        Ok(result.len())
    }
    fn resolv_inner(&mut self)-> Result<()>{
        if self.state.blank_counts == 0 {
            return Ok(())
        }

        while self.resolv_inner_smart_once().with_context(||format!("err?:\n{}", self.to_string()))? {};

        if self.state.blank_counts == 0 {
            return Ok(())
        }

        if let Some(cell) = self.state.find_empty_pos_with_less_candidates() {
            //println!("candidate for try: {:?}", self.state.get_candidates(cell));
            for num in self.state.get_candidates(cell){
                let mut backup = self.clone();
                //println!("try {:?} {}", cell, num);
                backup.fill(cell, num);
                if let Ok(_) = backup.resolv_inner(){
                    mem::replace(self, backup);
                    return Ok(())
                } else{
                    //println!("rollback {:?} {}", cell, num);
                }
            }
        } else{
            return Err(anyhow!("can't find empty pos."))
        }

        Err(anyhow!("can't resolv."))
    }
    pub fn resolv(&mut self)-> Result<()>{
        self.init_state();
        self.resolv_inner()?;

        Ok(())
    }
}

impl ToString for Sudoku{
    fn to_string(&self) -> String{
        let mut bufwriter = BufWriter::new(Vec::with_capacity((N*4 as usize).pow(2)));
        for i in 0..N{
            for j in 0..N{
                let n = self.board[i][j];
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

        String::from_utf8(bufwriter.into_inner().unwrap()).unwrap()
    }
}

impl Default for SudokuState{
    fn default() -> Self{
        fn new_filled_state()-> FilledState{
            // 0 总是默认填充, 值为1
            let mut tpl = Bitmap::<10>::mask(1);
            // let tpl = tpl.as_value();
            // let bits = Bitmap::<10>::from_value(*tpl),
            // bitmap has copy trait
            [tpl;N]
        }

        // 1~9默认都是侯选, 0 永远不是侯选
        let mut candidates_tpl = Bitmap::<10>::mask(10);
        candidates_tpl.set(0, false);
        SudokuState{
            blank_counts: (N*N) as u8,
            rows_filled_state: new_filled_state(),
            cols_filled_state: new_filled_state(),
            grids_filled_state: new_filled_state(),
            candidates: [[candidates_tpl;N];N],
        }
    }
}
impl ToString for SudokuState{
    fn to_string(&self) -> String{
        let mut bufwriter = BufWriter::new(Vec::with_capacity((N*4 as usize).pow(2)));
        bufwriter.write("candidates:\n".as_bytes()).unwrap();
        for i in 0..N{
            for j in 0..N{
                let line = format!("({},{}): {:?}\n", i, j, self.get_candidates((i, j)));
                bufwriter.write(line.as_bytes()).unwrap();
            }
        }
        String::from_utf8(bufwriter.into_inner().unwrap()).unwrap()
    }
}

impl SudokuState{
    pub fn get_candidates(&self, cell: Cell)-> Vec<u8>{
        get_bits(self.candidates[cell.0][cell.1])
    }

    fn fill(&mut self, pos: Cell, num: u8)-> Result<()>{
        assert!(num > 0 && num < 10);

        let (row,col) = pos;
        assert!(row < N && col < N);

        let num = num as usize;

        if self.rows_filled_state[row].set(num, true) {
            return Err(anyhow!("conflict at row {}", row))
        }

        if self.cols_filled_state[col].set(num, true){
            return Err(anyhow!("conflict at col {}", col))
        }

        let grid = (row / 3) * 3 +  (col / 3);
        if self.grids_filled_state[grid].set(num, true){
            return Err(anyhow!("conflict at grid {}", grid))
        }

        // remove candidates
        self.candidates[row][col] = Bitmap::new();
        // remove candidates for this cell
        for cell in RELATED_MATRIX[row][col]{
            let mut cans = &mut self.candidates[cell.0][cell.1];
            // true 改成 false 时触发冲突检查
            if cans.set(num, false){
                // 如果清空侯选，则表示冲突出现
                if cans.is_empty(){
                    return Err(anyhow!("conflict cause by filling {:?} with {}", pos, num))
                }
            }
        }
        self.blank_counts -= 1;
        //println!("fill {:?}, {} Ok. left: {}", pos, num, self.blank_counts);
        Ok(())
    }

    fn find_empty_pos_with_less_candidates(&self)-> Option<Cell>{
        let mut cell = (0,0);
        let mut min_cans_len = 10;

        for row in 0..N {
            for col in 0..N {
                let can = self.candidates[row][col];
                let can_len = can.len();
                if can_len>0 && can_len < min_cans_len {
                    min_cans_len = can_len;
                    cell = (row, col);
                }
                if min_cans_len <= 2{
                    return Some(cell);
                }
            }
        }
        if min_cans_len < 10{
            return Some(cell);
        }
        None
    }
    
    /// 找到某一行、列、宫格中，有唯一可填数值的单元, 输出需要去重
    fn find_only_one_candidates(&self) -> Vec<(Cell, u8)> {
        let mut result :Vec<(Cell, u8)> = Vec::new();
        let result_ref = &mut result;
        let mut push_to_result = move |cell: Cell, n:u8| {
            if let None = result_ref.into_iter().position(|i|i.0.0 == cell.0 && i.0.1 == cell.1 && i.1 == n) {
                result_ref.push((cell, n));
            }
        };

        #[inline(always)]
        fn to_row_cell(grp: usize, unit:usize)->Cell{(grp,unit)}

        #[inline(always)]
        fn to_col_cell(grp: usize, unit: usize)-> Cell{(unit,grp)}

        #[inline(always)]
        fn to_grid_cell(grp: usize, unit: usize)-> Cell{
            let lt :Cell = (grp / 3 * 3, grp % 3 * 3);
            let gr = lt.0 + unit / 3 ;
            let gc = lt.1 + unit % 3;
            (gr,gc)}

        let mut only_one_candidate_count = 0;
        for row in 0..N{
            for col in 0..N{
                let can = self.candidates[row][col];
                if can.len() == 1{
                    only_one_candidate_count += 1;
                    push_to_result((row, col), can.first_index().unwrap() as u8);
                }
            }
        }

        if only_one_candidate_count > 0 {
            return result;
        }

        // 检查每一组(行、列、宫格) 的所有单元格(cell) 是否只有一个单元格具有唯一侯选
        // 共9组
        for grp in 0..N{
            // 计数器，对应0~9, 0永远计数为0
            let mut row_counter = [0;10];
            let mut row_last_cell = [(0,0);10];
            let mut col_counter = [0;10];
            let mut col_last_cell = [(0,0);10];
            let mut grid_counter = [0;10];
            let mut grid_last_cell = [(0,0);10];
            // 检查组中每个单元格, 共9格
            for unit in 0..N{
                // 行统计
                let (row, col) = to_row_cell(grp, unit);
                let cans = self.candidates[row][col];
                // 每个单元格计数
                for num in 1..=9{
                    if cans.get(num){
                        row_counter[num] += 1;
                        row_last_cell[num] = (row, col);
                    }
                }

                // 列统计
                let (row, col) = to_col_cell(grp, unit);
                let cans = self.candidates[row][col];
                // 每个单元格计数
                for num in 1..=9{
                    if cans.get(num){
                        col_counter[num] += 1;
                        col_last_cell[num] = (row, col);
                    }
                }

                // 宫格统计
                let (row, col) = to_grid_cell(grp, unit);
                let cans = self.candidates[row][col];
                // 每个单元格计数
                for num in 1..=9{
                    if cans.get(num){
                        grid_counter[num] += 1;
                        grid_last_cell[num] = (row, col);
                    }
                }

            }
            for num in 1..=9{
                if row_counter[num] == 1{
                    push_to_result(row_last_cell[num], num as u8);
                }
                if col_counter[num] == 1{
                    push_to_result(col_last_cell[num], num as u8);
                }
                if grid_counter[num] == 1{
                    push_to_result(grid_last_cell[num], num as u8);
                }
            }
        }

        result
    }

    /// 某个宫格内一行中某两个单元, 如果其共同侯选N 在宫格内其他行都不存在，则可以断定该行其他单元(非宫格内的)，不可能填入N
    /// 列也相同
    /// 找到宫格内这种直线组合
    fn find_line_combose_in_grid(&self, grid_id: usize) -> Vec<(Cell, Cell, u8)>{
        let mut result : Vec<(Cell,Cell,u8)> = Vec::new();

        // grid lefttop cell
        let lt :Cell = (grid_id / 3 * 3, grid_id % 3 * 3);

        for ((p1, p2),comp) in GRID_LINE_COMBOSE_AND_COMPARE{
            // cell1
            let cell1 :Cell = (p1.0 + lt.0, p1.1 + lt.1);  
            // cell2
            let cell2 :Cell = (p2.0 + lt.0, p2.1 + lt.1);  
            let cell1_can = self.candidates[cell1.0][cell1.1] ; 
            let cell2_can = self.candidates[cell2.0][cell2.1];
            if cell1_can.is_empty() || cell2_can.is_empty(){
                continue;
            }
            //println!("combose: {:?} {:?}", c1, c2);

            // combose common candidates bitmap
            // 共同的侯选
            let common = cell1_can & cell2_can; 
            if common.is_empty(){
                continue;
            }

            // 每个共同候选，对比相邻行、列
            for num in get_bits(common){
                let mut founded = true;
                // 对比相邻的行列
                for cell in comp{
                    let this_cans = self.candidates[cell.0+lt.0][cell.1+lt.1];
                    if this_cans.get(num as usize){
                        founded = false;
                        break;
                    }
                }

                if founded{
                    result.push((cell1, cell2, num));
                }
            }
        }
        result
    }

    /// 消除一个侯选，如果该侯选被成功消除，返回Ok(true), 否则返回Ok(false)
    /// 当出现冲突时抛出错误
    fn candidates_eliminate(&mut self, cell: Cell, n: u8) -> Result<bool>{
        //println!("eliminate {} in {:?}. {:?}", n, cell, self.get_candidates(cell));
        let old = self.candidates[cell.0][cell.1].set(n as usize, false);
        if old == true && self.candidates[cell.0][cell.1].is_empty(){
            //println!("old {} {:?}", old, self.get_candidates(cell));
            // 如果由 true 改为 false, 并且侯选被清空, 则该单元格无法填充, 前置步骤有错误导致的冲突
            return Err(anyhow!("found conflict when eliminate"));
        }
        Ok(old)
    }

    /// 直线消除
    /// 当宫格内某行或列的共同candidates， 不在同宫格其他行或列里, 则对应的整行或整列非本宫格,应消除这些candidates
    /// 比如 (0,1) (0,2) 共同candidates 是 [2,3], 且在 (1,0) (1,1) (1,2) (2,0) (2,1) (2, 2)中都没有2,3. 则应该消除 (0,p) p=3,4,5,6,7,9的candidates中的[2,3]
    /// 返回消除的数量
    fn line_candidates_eliminate(&mut self) -> Result<usize>{
        fn eliminate_by_grid(state: &mut SudokuState, id: usize)->Result<usize>{
            let mut counter = 0;
            for (cell1, cell2, n) in state.find_line_combose_in_grid(id){
                //println!("line combose in grid-{}: {:?}-{:?} {}", id, cell1, cell2, n);
                if cell1.0 == cell2.0{
                    // 行消除, 除了本宫格
                    for icol in 0..N{
                        if icol / 3 == cell1.1/3{
                            continue;
                        }
                        if state.candidates_eliminate((cell1.0, icol), n)?{
                            counter += 1;
                        }
                        
                    }

                }else if cell1.1==cell2.1{
                    // 列消除, 除了本宫格
                    for row in 0..N{
                        if row / 3 == cell1.0/3{
                            continue;
                        }
                            //println!("grid_line_combose eliminate ({},{})", row, cell1.1);
                        if state.candidates_eliminate((row, cell1.1), n)? {
                            counter += 1;
                        }
                    }
                }
            }
            Ok(counter)
        }

        let mut counter = 0;
        for id in 0..N{
            counter += eliminate_by_grid(self, id)?;
        }
        Ok(counter)
    }

}


#[cfg(test)]
mod test{
    use super::*;

    #[ignore]
    #[test]
    fn test_load(){
        let mut data: [u8;N*N] = [
        9,6,0, 0,3,0,  0,0,0,
        0,0,0, 8,0,0,  0,5,0,
        0,7,0, 0,0,0,  0,0,0,

        4,0,5, 0,0,0,  9,0,0,
        2,0,0, 0,0,0,  3,0,0,
        0,0,0, 7,0,0,  0,0,0,

        0,0,0, 5,0,0,  0,8,7,
        0,0,0, 0,9,6,  0,0,0,
        0,0,0, 0,0,0,  0,0,0,
        ];

        for cell in data.iter_mut(){
            *cell = *cell + b'0';
        }
        println!("{:?}", data);

        let reader = BufReader::new(data.as_slice());
        let mut sudoku = Sudoku::load(reader).unwrap();
        println!("sudoku:\n{}", sudoku.to_string());
    }

    #[ignore]
    #[test]
    fn test_load_str(){
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
        let mut sudoku = Sudoku::load(reader).unwrap();
        println!("sudoku:\n{}", sudoku.to_string());
    }

    #[ignore]
    #[test]
    fn test_sudoku_state() {
        let mut state = SudokuState::default();
        println!("state row0 with digit 9: {}", state.rows_filled_state[0].get(9));
        println!("state col8 with digit 0: {}", state.cols_filled_state[8].get(0));
        state.fill((1,1), 3); // same row
        state.fill((1,2), 4);
        state.fill((1,3), 5);
        state.fill((3,5), 7); // same column
        state.fill((4,5), 8);
        state.fill((2,4), 9); // same grid
        assert_eq!(vec![1,2,6],state.get_candidates((1,5)));

        for row in 0..9{
            for col in 0..9{
                case_sudoku_state_fill((row, col), 1);
            }
        }
    }

    fn case_sudoku_state_fill(pos: Cell, num: u8) {
        let mut state = SudokuState::default();
        state.fill(pos, num);
        for col in 0..9{
            assert_eq!(state.candidates[pos.0][col].get(num as usize), false);
        }
        for row in 0..9{
            assert_eq!(state.candidates[row][pos.1].get(num as usize), false);
        }
        let lt = (pos.0 - pos.0 % 3, pos.1 - pos.1 %3);
        for r_off in 0..3{
            for c_off in 0..3{
                let row = lt.0 + r_off;
                let col = lt.1 + c_off;
                assert_eq!(state.candidates[row][pos.1].get(num as usize), false);
            }
        }
    }

    #[ignore]
    #[test]
    fn test_resolv(){
        let data: [u8;N*N] = [
        9,6,0, 0,3,0, 0,0,0,
        0,0,0, 8,0,0, 0,5,0,
        0,7,0, 0,0,0, 0,0,0,

        4,0,5, 0,0,0, 9,0,0,
        2,0,0, 0,0,0, 3,0,0,
        0,0,0, 7,0,0, 0,0,0,

        0,0,0, 5,0,0, 0,8,7,
        0,0,0, 0,9,6, 0,0,0,
        0,0,0, 0,0,0, 0,0,0,
        ];
        let mut sudoku = Sudoku::from_bytes(&data[..]);
        println!("sudoku:\n{}", sudoku.to_string());
        println!("left:{}", sudoku.state.blank_counts);
        let result = sudoku.resolv();
        println!("result: {:?}", result);

        println!("sudoku:\n{}", sudoku.to_string());
        println!("sudoku state:\n{}", sudoku.state.to_string());

    }

    #[ignore]
    #[test]
    fn test_const(){
        println!("{:?}", RELATED_MATRIX);
        println!("{:?}", GRID_LINE_COMBOSE_AND_COMPARE);
    }
}

