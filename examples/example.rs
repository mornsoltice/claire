use claire_vector::operations::vector::Function;
use claire_vector::operations::vector::Vector;
use claire_vector::vector;
use claire_vector::{claire_pdx, claire_pdy, claire_pdz, f};

fn main() {
    // Vector operations
    let claire_u: Vector = vector!(2, 3);
    let claire_v: Vector = vector!(3.5, std::f64::consts::PI, 2.0);
    println!("claire_u: {}", claire_u);
    println!("claire_v: {}", claire_v);

    // Derivatives
    let f: Function = f!(x, y, z, x * y.powi(2) + z);
    let a: f64 = claire_pdx!(f, 1.0, 2.0, 9.0);
    let b: f64 = claire_pdy!(f, 1.0, 2.0, 9.0);
    let c: f64 = claire_pdz!(f, 1.0, 2.0, 9.0);

    println!("∂f/∂x at (1.0, 2.0, 9.0) = {}", a);
    println!("∂f/∂y at (1.0, 2.0, 9.0) = {}", b);
    println!("∂f/∂z at (1.0, 2.0, 9.0) = {}", c);
}
