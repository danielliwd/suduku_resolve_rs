#[allow(unused_imports)]
use log::{info, trace};
use anyhow::{Result};

use sudoku_resolve::Sudoku;

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    env_logger::init();

    let mut sudoku = Sudoku::load_from_file("suduku.txt")?;

    println!("{}",sudoku.to_string());

    let result = sudoku.resolv();
    println!("resolved? {:?}\n", result);

    println!("{}",sudoku.to_string());

    Ok(())
}
