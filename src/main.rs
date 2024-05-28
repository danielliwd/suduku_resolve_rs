#[allow(unused_imports)]
use log::{info, trace};
use anyhow::{Result};

use suduku_resolve::Suduku;

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    env_logger::init();

    //let mut suduku = Suduku::load_from_file("suduku.txt")?;

    // println!("{}",suduku.to_string()?);

    // let result = suduku.resolv();
    // println!("resolved? {:?}\n", result);

    // println!("{}",suduku.to_string()?);

    Ok(())
}
