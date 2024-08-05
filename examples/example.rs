use claire::operations::vector::Vector;
use claire::vector;

fn main() {
    // Vector operations
    let claire_u: Vector = vector!(2, 3);
    let claire_v: Vector = vector!(3.5, std::f64::consts::PI, 2.0);
    println!("claire_u: {}", claire_u);
    println!("claire_v: {}", claire_v);
}
