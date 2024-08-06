//! claire: a Rust library for vector calculus.
//! this crate provides easy-to-use tools for multivariable and vector calculus in Rust, enabling
//! numerical computation of various operations efficiently.
#![feature(unboxed_closures, fn_traits)]

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
    pub fn expression(&self) -> &str {
        match self {
            Function::F1D(func) => &func.expression,
            Function::F2D(func) => &func.expression,
            Function::F3D(func) => &func.expression,
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

// creates a 1-3 variable function
// Function trait implementations
impl FnOnce<(f64,)> for Function {
    type Output = f64;

    extern "rust-call" fn call_once(self, args: (f64,)) -> Self::Output {
        match self {
            Function::F1D(f) => f.call(args.0),
            Function::F2D(_) => panic!("1D function can't take 2 arguments"),
            Function::F3D(_) => panic!("1D function can't take 3 arguments"),
        }
    }
}

impl Fn<(f64,)> for Function {
    extern "rust-call" fn call(&self, args: (f64,)) -> Self::Output {
        match self {
            Function::F1D(f) => f.call(args.0),
            Function::F2D(_) => panic!("1D function can't take 2 arguments"),
            Function::F3D(_) => panic!("1D function can't take 3 arguments"),
        }
    }
}

impl FnMut<(f64,)> for Function {
    extern "rust-call" fn call_mut(&mut self, args: (f64,)) -> Self::Output {
        match self {
            Function::F1D(f) => f.call(args.0),
            Function::F2D(_) => panic!("1D function can't take 2 arguments"),
            Function::F3D(_) => panic!("1D function can't take 3 arguments"),
        }
    }
}

impl FnOnce<(f64, f64)> for Function {
    type Output = f64;

    extern "rust-call" fn call_once(self, args: (f64, f64)) -> Self::Output {
        match self {
            Function::F1D(_) => panic!("1D function can't take 2 arguments"),
            Function::F2D(f) => f.call(args.0, args.1),
            Function::F3D(_) => panic!("3D function can't take 2 arguments"),
        }
    }
}

impl Fn<(f64, f64)> for Function {
    extern "rust-call" fn call(&self, args: (f64, f64)) -> Self::Output {
        match self {
            Function::F1D(_) => panic!("1D function can't take 2 arguments"),
            Function::F2D(f) => f.call(args.0, args.1),
            Function::F3D(_) => panic!("3D function can't take 2 arguments"),
        }
    }
}

impl FnMut<(f64, f64)> for Function {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64)) -> Self::Output {
        match self {
            Function::F1D(_) => panic!("1D function can't take 2 arguments"),
            Function::F2D(f) => f.call(args.0, args.1),
            Function::F3D(_) => panic!("3D function can't take 2 arguments"),
        }
    }
}

impl FnOnce<(f64, f64, f64)> for Function {
    type Output = f64;

    extern "rust-call" fn call_once(self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            Function::F1D(_) => panic!("1D function can't take 2 arguments"),
            Function::F2D(_) => panic!("2D function can't take 3 arguments"),
            Function::F3D(f) => f.call(args.0, args.1, args.2),
        }
    }
}

impl Fn<(f64, f64, f64)> for Function {
    extern "rust-call" fn call(&self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            Function::F1D(_) => panic!("1D function can't take 2 arguments"),
            Function::F2D(_) => panic!("2D function can't take 3 arguments"),
            Function::F3D(f) => f.call(args.0, args.1, args.2),
        }
    }
}

impl FnMut<(f64, f64, f64)> for Function {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            Function::F1D(_) => panic!("1D function can't take 2 arguments"),
            Function::F2D(_) => panic!("2D function can't take 3 arguments"),
            Function::F3D(f) => f.call(args.0, args.1, args.2),
        }
    }
}

impl FnOnce<(Vector,)> for Function {
    type Output = f64;

    extern "rust-call" fn call_once(self, args: (Vector,)) -> Self::Output {
        match self {
            Function::F1D(_) => panic!("1D function can't take a vector as an argument"),
            Function::F2D(f) => match args.0 {
                Vector::TwoD(v) => f.call(v.x, v.y),
                Vector::ThreeD(_) => panic!("3D vector passed to 2D function"),
            },
            Function::F3D(f) => {
                match args.0 {
                    Vector::TwoD(v) => f.call(v.x, v.y, 0.), // 2D vector passed to 3D function with z = 0
                    Vector::ThreeD(v) => f.call(v.x, v.y, v.z),
                }
            }
        }
    }
}

impl Fn<(Vector,)> for Function {
    extern "rust-call" fn call(&self, args: (Vector,)) -> Self::Output {
        match self {
            Function::F1D(_) => panic!("1D function can't take a vector as an argument"),
            Function::F2D(f) => match args.0 {
                Vector::TwoD(v) => f.call(v.x, v.y),
                Vector::ThreeD(_) => panic!("3D vector passed to 2D function"),
            },
            Function::F3D(f) => match args.0 {
                Vector::TwoD(_) => panic!("2D vector passed to 3D function"),
                Vector::ThreeD(v) => f.call(v.x, v.y, v.z),
            },
        }
    }
}

impl FnMut<(Vector,)> for Function {
    extern "rust-call" fn call_mut(&mut self, args: (Vector,)) -> Self::Output {
        match self {
            Function::F1D(_) => panic!("1D function can't take a vector as an argument"),
            Function::F2D(f) => match args.0 {
                Vector::TwoD(v) => f.call(v.x, v.y),
                Vector::ThreeD(_) => panic!("3D vector passed to 2D function"),
            },
            Function::F3D(f) => match args.0 {
                Vector::TwoD(_) => panic!("2D vector passed to 3D function"),
                Vector::ThreeD(v) => f.call(v.x, v.y, v.z),
            },
        }
    }
}

/// Calculus limits for functions
/// # Limit
/// The limit macro takes in a [Function] of n variables and n f64's, representing the point to take
/// the limit at. It also uses the => syntax, i actually want to add -> but it is not allowed so i use =>
/// ## Examples
///
/// use claire_vector::*;
/// let f:Function = f!(x, (x.powi(2)-25.)/(x-5.)); // (x^2-25)/(x-5)
/// let a:f64 = limit!(f => 5);
/// assert_eq!(a, 10.0); // Analytically it's 10

#[macro_export]
macro_rules! limit {
    ($f:expr => $x:expr) => {
        limit_s(&$f, vec![$x as f64])
    };
    ($f:expr => $x:expr, $y:expr) => {
        limit_s(&$f, vec![$x as f64, $y as f64])
    };
    ($f:expr => $x:expr, $y:expr, $z:expr) => {
        limit_s(&$f, vec![$x as f64, $y as f64, $z as f64])
    };
}

#[doc(hidden)]
pub fn limit_s(f: &Function, args: Vec<f64>) -> f64 {
    let delta = 1e-10;
    match f {
        Function::F1D(func) => {
            let x = args[0];
            (func.call(x + delta) + func.call(x - delta)) / 2.0
        }
        Function::F2D(func) => {
            let (x, y) = (args[0], args[1]);
            (func.call(x + delta, y) + func.call(x - delta, y)) / 2.0
        }
        Function::F3D(func) => {
            let (x, y, z) = (args[0], args[1], args[2]);
            (func.call(x + delta, y, z) + func.call(x - delta, y, z)) / 2.0
        }
    }
}

// Vector Functions
#[derive(Clone)]
#[doc(hidden)]
pub struct _VectorFunction2D {
    pub f1: Function,
    pub f2: Function,
    pub potential: Option<Function>,
}

#[derive(Clone)]
#[doc(hidden)]
pub struct _VectorFunction3D {
    pub f1: Function,
    pub f2: Function,
    pub f3: Function,
    pub potential: Option<Function>,
}

/// Vector functions for 2D and 3D spaces
/// # Vector Function
/// Vector functions are functions of two or three variables that return two or three-dimensional [Vector]s. \
/// You can create one using the [vector_function!] macro. \
/// Vector functions evaluate just like [Function]s and rust functions, and you can even evaluate them on [Vector]s. \
/// You can take its partial derivatives with the [claire_dxv!], [claire_dyv!] and [claire_dzv!] macros, adding the 'v' at the end
/// to signal that it is the partial of a vector function, and that it returns a [Vector] as such. \
/// Furthermore, you can use the [curl!] and [div!] macro to evaluate its rotational and divergence at a point. \
/// ## Examples
///
/// use claire_vector::*;
/// let v:VectorFunction = vector_function!(x, y, -y, x);
/// assert_eq!(v(0., 1.), vector!(-1, 0)); // v(0, 1) = (-1, 0)
///
#[derive(Clone)]
pub enum VectorFunction {
    TwoD(_VectorFunction2D),
    ThreeD(_VectorFunction3D),
}

impl VectorFunction {
    fn potential(&self, args: Vec<f64>) -> f64 {
        match self {
            VectorFunction::TwoD(v) => {
                if let Some(f) = &v.potential {
                    return match f {
                        Function::F1D(func) => func.call(args[0]),
                        Function::F2D(func) => func.call(args[0], args[1]),
                        Function::F3D(func) => func.call(args[0], args[1], args[2]),
                    };
                } else {
                    f64::NAN
                }
            }
            VectorFunction::ThreeD(v) => {
                if let Some(f) = &v.potential {
                    return match f {
                        Function::F1D(func) => func.call(args[0]),
                        Function::F2D(func) => func.call(args[0], args[1]),
                        Function::F3D(func) => func.call(args[0], args[1], args[2]),
                    };
                } else {
                    f64::NAN
                }
            }
        }
    }
}

impl Display for VectorFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorFunction::TwoD(v) => {
                write!(f, "⟨{:.5}, {:.5}⟩", v.f1.expression(), v.f2.expression())
            }
            VectorFunction::ThreeD(v) => {
                write!(
                    f,
                    "⟨{:.5}, {:.5}, {:.5}⟩",
                    v.f1.expression(),
                    v.f2.expression(),
                    v.f3.expression()
                )
            }
        }
    }
}

/// Creates vector functions
/// # Vector Function macro
/// This macro takes in two or three identifiers (the variable names) and two or three expressions that may use
/// those variables, and outputs a [VectorFunction].
/// ## Examples
///
/// use claire_vector::*;
/// let F:VectorFunction = vector_function!(x, y, z, y.exp(), x + z, x*y*z);
///
#[macro_export]
macro_rules! vector_function {
    ($x:ident, $y:ident, $f1:expr, $f2:expr) => {
        VectorFunction::TwoD(_VectorFunction2D {
            f1: Function::F2D(_Function2D {
                f: Box::new(|$x: f64, $y: f64| $f1),
                expression: String::from(stringify!($f1)),
            }),
            f2: Function::F2D(_Function2D {
                f: Box::new(|$x: f64, $y: f64| $f2),
                expression: String::from(stringify!($f2)),
            }),
            potential: Option::None,
        })
    };
    ($x:ident, $y:ident, $z:ident, $f1:expr, $f2:expr, $f3:expr) => {
        VectorFunction::ThreeD(_VectorFunction3D {
            f1: Function::F3D(_Function3D {
                f: Box::new(|$x: f64, $y: f64, $z: f64| $f1),
                expression: String::from(stringify!($f1)),
            }),
            f2: Function::F3D(_Function3D {
                f: Box::new(|$x: f64, $y: f64, $z: f64| $f2),
                expression: String::from(stringify!($f2)),
            }),
            f3: Function::F3D(_Function3D {
                f: Box::new(|$x: f64, $y: f64, $z: f64| $f3),
                expression: String::from(stringify!($f3)),
            }),
            potential: Option::None,
        })
    };
}

impl FnOnce<(f64, f64)> for VectorFunction {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(v) => Vector::TwoD(_Vector2::new(
                (v.f1)(args.0, args.1),
                (v.f2)(args.0, args.1),
            )),
            VectorFunction::ThreeD(_) => panic!("3D vector function can't take 2 arguments"),
        }
    }
}

impl Fn<(f64, f64)> for VectorFunction {
    extern "rust-call" fn call(&self, args: (f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(v) => Vector::TwoD(_Vector2::new(
                (v.f1)(args.0, args.1),
                (v.f2)(args.0, args.1),
            )),
            VectorFunction::ThreeD(_) => panic!("3D vector function can't take 2 arguments"),
        }
    }
}

impl FnMut<(f64, f64)> for VectorFunction {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(v) => Vector::TwoD(_Vector2::new(
                (v.f1)(args.0, args.1),
                (v.f2)(args.0, args.1),
            )),
            VectorFunction::ThreeD(_) => panic!("3D vector function can't take 2 arguments"),
        }
    }
}

impl FnOnce<(f64, f64, f64)> for VectorFunction {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(_) => panic!("2D vector function can't take 3 arguments"),
            VectorFunction::ThreeD(v) => Vector::ThreeD(_Vector3::new(
                (v.f1)(args.0, args.1, args.2),
                (v.f2)(args.0, args.1, args.2),
                (v.f3)(args.0, args.1, args.2),
            )),
        }
    }
}

impl Fn<(f64, f64, f64)> for VectorFunction {
    extern "rust-call" fn call(&self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(_) => panic!("2D vector function can't take 3 arguments"),
            VectorFunction::ThreeD(v) => Vector::ThreeD(_Vector3::new(
                (v.f1)(args.0, args.1, args.2),
                (v.f2)(args.0, args.1, args.2),
                (v.f3)(args.0, args.1, args.2),
            )),
        }
    }
}

impl FnMut<(f64, f64, f64)> for VectorFunction {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(_) => panic!("2D vector function can't take 3 arguments"),
            VectorFunction::ThreeD(v) => Vector::ThreeD(_Vector3::new(
                (v.f1)(args.0, args.1, args.2),
                (v.f2)(args.0, args.1, args.2),
                (v.f3)(args.0, args.1, args.2),
            )),
        }
    }
}

impl FnOnce<(Vector,)> for VectorFunction {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (Vector,)) -> Self::Output {
        match self {
            VectorFunction::TwoD(f) => match args.0 {
                Vector::TwoD(v) => Vector::TwoD(_Vector2::new((f.f1)(v.x, v.y), (f.f2)(v.x, v.y))),
                Vector::ThreeD(_) => panic!("3D vector passed to 2D vector function"),
            },
            VectorFunction::ThreeD(f) => match args.0 {
                Vector::TwoD(v) => Vector::ThreeD(_Vector3::new(
                    (f.f1)(v.x, v.y, 0.),
                    (f.f2)(v.x, v.y, 0.),
                    (f.f3)(v.x, v.y, 0.),
                )),
                Vector::ThreeD(v) => Vector::ThreeD(_Vector3::new(
                    (f.f1)(v.x, v.y, v.z),
                    (f.f2)(v.x, v.y, v.z),
                    (f.f3)(v.x, v.y, v.z),
                )),
            },
        }
    }
}

impl Fn<(Vector,)> for VectorFunction {
    extern "rust-call" fn call(&self, args: (Vector,)) -> Self::Output {
        match self {
            VectorFunction::TwoD(f) => match args.0 {
                Vector::TwoD(v) => Vector::TwoD(_Vector2::new((f.f1)(v.x, v.y), (f.f2)(v.x, v.y))),
                Vector::ThreeD(_) => panic!("3D vector passed to 2D vector function"),
            },
            VectorFunction::ThreeD(f) => match args.0 {
                Vector::TwoD(v) => Vector::ThreeD(_Vector3::new(
                    (f.f1)(v.x, v.y, 0.),
                    (f.f2)(v.x, v.y, 0.),
                    (f.f3)(v.x, v.y, 0.),
                )),
                Vector::ThreeD(v) => Vector::ThreeD(_Vector3::new(
                    (f.f1)(v.x, v.y, v.z),
                    (f.f2)(v.x, v.y, v.z),
                    (f.f3)(v.x, v.y, v.z),
                )),
            },
        }
    }
}

impl FnMut<(Vector,)> for VectorFunction {
    extern "rust-call" fn call_mut(&mut self, args: (Vector,)) -> Self::Output {
        match self {
            VectorFunction::TwoD(f) => match args.0 {
                Vector::TwoD(v) => Vector::TwoD(_Vector2::new((f.f1)(v.x, v.y), (f.f2)(v.x, v.y))),
                Vector::ThreeD(_) => panic!("3D vector passed to 2D vector function"),
            },
            VectorFunction::ThreeD(f) => match args.0 {
                Vector::TwoD(v) => Vector::ThreeD(_Vector3::new(
                    (f.f1)(v.x, v.y, 0.),
                    (f.f2)(v.x, v.y, 0.),
                    (f.f3)(v.x, v.y, 0.),
                )),
                Vector::ThreeD(v) => Vector::ThreeD(_Vector3::new(
                    (f.f1)(v.x, v.y, v.z),
                    (f.f2)(v.x, v.y, v.z),
                    (f.f3)(v.x, v.y, v.z),
                )),
            },
        }
    }
}

// Partial Derivatives
#[doc(hidden)]
pub fn ddx_v(v: &VectorFunction, args: Vec<f64>) -> Vector {
    match v {
        VectorFunction::TwoD(v) => Vector::TwoD(_Vector2::new(
            ((v.f1)(args[0] + Δ, args[1]) - (v.f1)(args[0], args[1])) / Δ,
            ((v.f2)(args[0] + Δ, args[1]) - (v.f2)(args[0], args[1])) / Δ,
        )),
        VectorFunction::ThreeD(v) => Vector::ThreeD(_Vector3::new(
            ((v.f1)(args[0] + Δ, args[1], args[2]) - (v.f1)(args[0], args[1], args[2])) / Δ,
            ((v.f2)(args[0] + Δ, args[1], args[2]) - (v.f2)(args[0], args[1], args[2])) / Δ,
            ((v.f3)(args[0] + Δ, args[1], args[2]) - (v.f3)(args[0], args[1], args[2])) / Δ,
        )),
    }
}

#[doc(hidden)]
pub fn ddy_v(v: &VectorFunction, args: Vec<f64>) -> Vector {
    match v {
        VectorFunction::TwoD(v) => Vector::TwoD(_Vector2::new(
            ((v.f1)(args[0], args[1] + Δ) - (v.f1)(args[0], args[1])) / Δ,
            ((v.f2)(args[0], args[1] + Δ) - (v.f2)(args[0], args[1])) / Δ,
        )),
        VectorFunction::ThreeD(v) => Vector::ThreeD(_Vector3::new(
            ((v.f1)(args[0], args[1] + Δ, args[2]) - (v.f1)(args[0], args[1], args[2])) / Δ,
            ((v.f2)(args[0], args[1] + Δ, args[2]) - (v.f2)(args[0], args[1], args[2])) / Δ,
            ((v.f3)(args[0], args[1] + Δ, args[2]) - (v.f3)(args[0], args[1], args[2])) / Δ,
        )),
    }
}

#[doc(hidden)]
pub fn ddz_v(v: &VectorFunction, args: Vec<f64>) -> Vector {
    match v {
        VectorFunction::TwoD(_) => {
            panic!("Can't take partial with respect to z of a 2D vector function")
        }
        VectorFunction::ThreeD(v) => Vector::ThreeD(_Vector3::new(
            ((v.f1)(args[0], args[1], args[2] + Δ) - (v.f1)(args[0], args[1], args[2])) / Δ,
            ((v.f2)(args[0], args[1], args[2] + Δ) - (v.f2)(args[0], args[1], args[2])) / Δ,
            ((v.f3)(args[0], args[1], args[2] + Δ) - (v.f3)(args[0], args[1], args[2])) / Δ,
        )),
    }
}

/// Partial x for a vector function
/// # Vector Partial x
/// This macro takes a [VectorFunction] and n f64's representing a point
/// and returns the derivative with respect to the first variable at that point.
/// ## Examples
///
/// use claire_vector::*;
/// let F:VectorFunction = vector_function!(x, y, -2.*y, x);
/// let v:Vector = ddxv!(F, 2, 1.);
/// assert_eq!(v, vector!(0, 0.9999999999621422)); // Analytically, it's (0, 1)
///
#[macro_export]
macro_rules! ddxv {
    ($f:expr, $x:expr, $y:expr) => {
        ddx_v(&$f, vec![$x as f64, $y as f64])
    };
    ($f:expr, $x:expr, $y:expr, $z:expr) => {
        ddx_v(&$f, vec![$x as f64, $y as f64, $z as f64])
    };
}

/// Partial y for a vector function
/// # Vector Partial y
/// This macro takes a [VectorFunction] and n f64's representing a point
/// and returns the derivative with respect to the second variable at that point.
/// ## Examples
///
/// use claire_vector::*;
/// let F:VectorFunction = vector_function!(x, y, -2.*y, x);
/// let v:Vector = ddyv!(F, 2, 1.);
/// assert_eq!(v, vector!(-2.0000000000131024, 0)); // Analytically, it's (-2, 0)
///
#[macro_export]
macro_rules! ddyv {
    ($f:expr, $x:expr, $y:expr) => {
        ddy_v(&$f, vec![$x as f64, $y as f64])
    };
    ($f:expr, $x:expr, $y:expr, $z:expr) => {
        ddy_v(&$f, vec![$x as f64, $y as f64, $z as f64])
    };
}

/// Partial z for a vector function
/// # Vector Partial z
/// This macro takes a [VectorFunction] and n f64's representing a point
/// and returns the derivative with respect to the third variable at that point.
/// ## Examples
///
/// use claire_vector::*;
/// let F:VectorFunction = vector_function!(x, y, z, x, y, z.powi(2));
/// let v:Vector = ddzv!(F, 2, 1, 2);
/// assert_eq!(v, vector!(0, 0, 4.000004999760165)); // Analytically, it's (0, 0, 4)
///
#[macro_export]
macro_rules! ddzv {
    ($f:expr, $x:expr, $y:expr, $z:expr) => {
        ddz_v(&$f, vec![$x as f64, $y as f64, $z as f64])
    };
}

// Curl
#[doc(hidden)]
pub fn curl(v: &VectorFunction, args: Vec<f64>) -> Vector {
    match v {
        VectorFunction::TwoD(_) => {
            let (x, y) = (args[0], args[1]);
            vector!(0, 0, ddx_v(v, vec![x, y]).y() - ddy_v(v, vec![x, y]).x())
        }
        VectorFunction::ThreeD(_) => {
            let (x, y, z) = (args[0], args[1], args[2]);
            vector!(
                ddy_v(v, vec![x, y, z]).z() - ddz_v(v, vec![x, y, z]).y(),
                ddz_v(v, vec![x, y, z]).x() - ddx_v(v, vec![x, y, z]).z(),
                ddx_v(v, vec![x, y, z]).y() - ddy_v(v, vec![x, y, z]).x()
            )
        }
    }
}

/// Curl of a vector function at a point
/// # Curl
/// The curl macro takes in a [VectorFunction] and n f64's representing the point to evaluate, and
/// returns a vector with the result of applying the rotational operator to it.
/// ## Examples
///
/// use claire_vector::*;
/// let f = vector_function!(x, y, -y, x);
/// let c:Vector = curl!(f, 2, 3);
/// assert_eq!(c, vector!(0, 0, 1.9999999999242843)) // Analytically its (0, 0, 2)
///
#[macro_export]
macro_rules! curl {
    ($v:expr, $x:expr, $y:expr) => {
        curl(&$v, vec![$x as f64, $y as f64])
    };
    ($v:expr, $x:expr, $y:expr, $z:expr) => {
        curl(&$v, vec![$x as f64, $y as f64, $z as f64])
    };
}

// Div
#[doc(hidden)]
pub fn div(v: &VectorFunction, args: Vec<f64>) -> f64 {
    match v {
        VectorFunction::TwoD(_) => {
            let (x, y) = (args[0], args[1]);
            ddx_v(v, vec![x, y]).x() + ddy_v(v, vec![x, y]).y()
        }
        VectorFunction::ThreeD(_) => {
            let (x, y, z) = (args[0], args[1], args[2]);
            ddx_v(v, vec![x, y, z]).x() + ddy_v(v, vec![x, y, z]).y() + ddz_v(v, vec![x, y, z]).z()
        }
    }
}

/// Divergence of a vector function at a point
/// # Divergence
/// The divergence macro takes a [VectorFunction] and n f64's representing the point to evaluate,
/// and returns a f64 as the result of applying the divergence operator to the function at that point.
/// ## Examples
///
/// use claire_vector::*;
/// let f = vector_function!(x, y, 2.*x, 2.*y);
/// let a:f64 = div!(f, 0, 0);
/// assert_eq!(a, 4.);
///
#[macro_export]
macro_rules! div {
    ($v:expr, $x:expr, $y:expr) => {
        div(&$v, vec![$x as f64, $y as f64])
    };
    ($v:expr, $x:expr, $y:expr, $z:expr) => {
        div(&$v, vec![$x as f64, $y as f64, $z as f64])
    };
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
