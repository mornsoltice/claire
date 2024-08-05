//! claire: a Rust library for vector calculus.
//! this crate provides easy-to-use tools for multivariable and vector calculus in Rust, enabling
//! numerical computation of various operations efficiently.

use dyn_clone::DynClone;
use rand::Rng;
use std::fmt::{self, Display, Formatter};

// Define constants
pub const Δ: f64 = 5e-6;
const G: f64 = 6.6743e-11;
const C: f64 = 299_792_458.0;

// Define 2D and 3D vector structures
#[derive(Copy, Clone, Debug)]
pub struct _Vector2 {
    pub x: f64,
    pub y: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct _Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

// Implement methods for 2D vectors
impl _Vector2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    pub fn modulus(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
    pub fn get_tuple(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

// Implement methods for 3D vectors
impl _Vector3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn modulus(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
    pub fn get_tuple(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
}

// Implement dot and cross product operations for 2D and 3D vectors
impl std::ops::Rem for _Vector2 {
    type Output = f64;
    fn rem(self, rhs: Self) -> Self::Output {
        self.x * rhs.y - self.y * rhs.x
    }
}

impl std::ops::Rem for _Vector3 {
    type Output = _Vector3;
    fn rem(self, rhs: Self) -> Self::Output {
        _Vector3 {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

// Implement display for vectors
impl Display for _Vector2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "⟨{:.5}, {:.5}⟩", self.x, self.y)
    }
}

impl Display for _Vector3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "⟨{:.5}, {:.5}, {:.5}⟩", self.x, self.y, self.z)
    }
}

// Define the Vector enum to handle both 2D and 3D vectors
#[derive(Copy, Clone, Debug)]
pub enum Vector {
    TwoD(_Vector2),
    ThreeD(_Vector3),
}

// Implement display for the Vector enum
impl Display for Vector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Vector::TwoD(v) => write!(f, "{}", v),
            Vector::ThreeD(v) => write!(f, "{}", v),
        }
    }
}

// Implement methods for the Vector enum
impl Vector {
    pub fn new_2d(x: f64, y: f64) -> Self {
        Vector::TwoD(_Vector2 { x, y })
    }

    pub fn new_3d(x: f64, y: f64, z: f64) -> Self {
        Vector::ThreeD(_Vector3 { x, y, z })
    }

    pub fn x(&self) -> f64 {
        match self {
            Vector::TwoD(v) => v.x,
            Vector::ThreeD(v) => v.x,
        }
    }

    pub fn y(&self) -> f64 {
        match self {
            Vector::TwoD(v) => v.y,
            Vector::ThreeD(v) => v.y,
        }
    }

    pub fn z(&self) -> f64 {
        match self {
            Vector::TwoD(_) => 0.,
            Vector::ThreeD(v) => v.z,
        }
    }
}

// Implement additional operations for the Vector enum
impl PartialEq for Vector {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Vector::TwoD(u), Vector::TwoD(v)) => u.x == v.x && u.y == v.y,
            (Vector::ThreeD(u), Vector::ThreeD(v)) => u.x == v.x && u.y == v.y && u.z == v.z,
            _ => false,
        }
    }
}

// Implement dot product for vectors
impl std::ops::Mul for Vector {
    type Output = f64;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Vector::TwoD(v1), Vector::TwoD(v2)) => v1.x * v2.x + v1.y * v2.y,
            (Vector::ThreeD(v1), Vector::ThreeD(v2)) => v1.x * v2.x + v1.y * v2.y + v1.z * v2.z,
            _ => panic!("there is no dot product for 2x3"),
        }
    }
}

// Implement scalar multiplication for vectors
impl std::ops::Mul<f64> for Vector {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        match self {
            Vector::TwoD(v) => Vector::TwoD(_Vector2 {
                x: v.x * scalar,
                y: v.y * scalar,
            }),
            Vector::ThreeD(v) => Vector::ThreeD(_Vector3 {
                x: v.x * scalar,
                y: v.y * scalar,
                z: v.z * scalar,
            }),
        }
    }
}

// Scalar Function
#[doc(hidden)]
pub trait F1DClone: DynClone + Fn(f64) -> f64 {
    fn call(&self, x: f64) -> f64;
}
#[doc(hidden)]
pub trait F2DClone: DynClone + Fn(f64, f64) -> f64 {
    fn call(&self, x: f64, y: f64) -> f64;
}
#[doc(hidden)]
pub trait F3DClone: DynClone + Fn(f64, f64, f64) -> f64 {
    fn call(&self, x: f64, y: f64, z: f64) -> f64;
}

impl<F> F1DClone for F
where
    F: 'static + Fn(f64) -> f64 + Clone,
{
    fn call(&self, x: f64) -> f64 {
        self(x)
    }
}
impl<F> F2DClone for F
where
    F: 'static + Fn(f64, f64) -> f64 + Clone,
{
    fn call(&self, x: f64, y: f64) -> f64 {
        self(x, y)
    }
}
impl<F> F3DClone for F
where
    F: 'static + Fn(f64, f64, f64) -> f64 + Clone,
{
    fn call(&self, x: f64, y: f64, z: f64) -> f64 {
        self(x, y, z)
    }
}

dyn_clone::clone_trait_object!(F1DClone);
dyn_clone::clone_trait_object!(F2DClone);
dyn_clone::clone_trait_object!(F3DClone);

#[doc(hidden)]
pub struct _Function1D {
    pub f: Box<dyn F1DClone>,
    pub expression: String,
}
impl _Function1D {
    fn call(&self, x: f64) -> f64 {
        (self.f)(x)
    }
}
#[doc(hidden)]
pub struct _Function2D {
    pub f: Box<dyn F2DClone>,
    pub expression: String,
}
impl _Function2D {
    fn call(&self, x: f64, y: f64) -> f64 {
        (self.f)(x, y)
    }
}
#[doc(hidden)]
pub struct _Function3D {
    pub f: Box<dyn F3DClone>,
    pub expression: String,
}
impl _Function3D {
    fn call(&self, x: f64, y: f64, z: f64) -> f64 {
        (self.f)(x, y, z)
    }
}

impl Clone for _Function1D {
    fn clone(&self) -> _Function1D {
        _Function1D {
            f: dyn_clone::clone_box(&*self.f),
            expression: self.expression.clone(),
        }
    }
}
impl Clone for _Function2D {
    fn clone(&self) -> _Function2D {
        _Function2D {
            f: dyn_clone::clone_box(&*self.f),
            expression: self.expression.clone(),
        }
    }
}
impl Clone for _Function3D {
    fn clone(&self) -> _Function3D {
        _Function3D {
            f: dyn_clone::clone_box(&*self.f),
            expression: self.expression.clone(),
        }
    }
}

// Define the Function enum to encapsulate different function types
#[derive(Clone)]
pub enum Function {
    F1D(_Function1D),
    F2D(_Function2D),
    F3D(_Function3D),
}

impl Function {
    pub fn call(&self, x: f64, y: Option<f64>, z: Option<f64>) -> f64 {
        match self {
            Function::F1D(f) => f.call(x),
            Function::F2D(f) => f.call(x, y.unwrap()),
            Function::F3D(f) => f.call(x, y.unwrap(), z.unwrap()),
        }
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Function::F1D(func) => write!(f, "Function1D: {}", func.expression),
            Function::F2D(func) => write!(f, "Function2D: {}", func.expression),
            Function::F3D(func) => write!(f, "Function3D: {}", func.expression),
        }
    }
}

// Define macros for function creation
#[macro_export]
macro_rules! f {
    ($x:ident, $body:expr) => {
        Function::F1D(_Function1D {
            f: Box::new(move |$x: f64| $body),
            expression: stringify!($body).to_string(),
        })
    };
    ($x:ident, $y:ident, $body:expr) => {
        Function::F2D(_Function2D {
            f: Box::new(move |$x: f64, $y: f64| $body),
            expression: stringify!($body).to_string(),
        })
    };
    ($x:ident, $y:ident, $z:ident, $body:expr) => {
        Function::F3D(_Function3D {
            f: Box::new(move |$x: f64, $y: f64, $z: f64| $body),
            expression: stringify!($body).to_string(),
        })
    };
}

// partial derivatives
#[doc(hidden)]
pub fn ddx_s(f: &Function, args: Vec<f64>) -> f64 {
    match f {
        Function::F1D(f) => (f.call(args[0] + Δ) - f.call(args[0])) / Δ,
        Function::F2D(f) => (f.call(args[0] + Δ, args[1]) - f.call(args[0], args[1])) / Δ,
        Function::F3D(f) => {
            (f.call(args[0] + Δ, args[1], args[2]) - f.call(args[0], args[1], args[2])) / Δ
        }
    }
}

#[doc(hidden)]
pub fn ddy_s(f: &Function, args: Vec<f64>) -> f64 {
    match f {
        Function::F1D(_) => {
            panic!("Can't take partial derivative with respect to y of a 1D function")
        }
        Function::F2D(f) => (f.call(args[0], args[1] + Δ) - f.call(args[0], args[1])) / Δ,
        Function::F3D(f) => {
            (f.call(args[0], args[1] + Δ, args[2]) - f.call(args[0], args[1], args[2])) / Δ
        }
    }
}

#[doc(hidden)]
pub fn ddz_s(f: &Function, args: Vec<f64>) -> f64 {
    match f {
        Function::F1D(_) => {
            panic!("Can't take partial derivative with respect to z of a 1D function")
        }
        Function::F2D(_) => {
            panic!("Can't take partial derivative with respect to z of a 2D function")
        }
        Function::F3D(f) => {
            (f.call(args[0], args[1], args[2] + Δ) - f.call(args[0], args[1], args[2])) / Δ
        }
    }
}

// Macros for partial derivatives
#[macro_export]
macro_rules! claire_pdx {
    ($f:expr, $x:expr) => {
        ddx_s(&$f, vec![$x as f64])
    };
    ($f:expr, $x:expr, $y:expr) => {
        ddx_s(&$f, vec![$x as f64, $y as f64])
    };
    ($f:expr, $x:expr, $y:expr, $z:expr) => {
        ddx_s(&$f, vec![$x as f64, $y as f64, $z as f64])
    };
}

#[macro_export]
macro_rules! claire_pdy {
    ($f:expr, $x:expr, $y:expr) => {
        ddy_s(&$f, vec![$x as f64, $y as f64])
    };
    ($f:expr, $x:expr, $y:expr, $z:expr) => {
        ddy_s(&$f, vec![$x as f64, $y as f64, $z as f64])
    };
}

#[macro_export]
macro_rules! claire_pdz {
    ($f:expr, $x:expr, $y:expr, $z:expr) => {
        ddz_s(&$f, vec![$x as f64, $y as f64, $z as f64])
    };
}

impl Function {
    fn clone(&self) -> Function {
        match self {
            Function::F1D(f) => Function::F1D(f.clone()),
            Function::F2D(f) => Function::F2D(f.clone()),
            Function::F3D(f) => Function::F3D(f.clone()),
        }
    }
}

// Implement more operations as needed...

// Macros for vector creation and modulus calculation
#[macro_export]
macro_rules! vector {
    ($x:expr, $y:expr) => {
        Vector::new_2d($x as f64, $y as f64)
    };
    ($x:expr, $y:expr, $z:expr) => {
        Vector::new_3d($x as f64, $y as f64, $z as f64)
    };
}

#[macro_export]
macro_rules! md {
    ($v:expr) => {
        modulus(&$v)
    };
}

