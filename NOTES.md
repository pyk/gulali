# Data Type?
Do we need introduce new data type for multidimensional array?
I think we don't need that.

Let's just use:

    Vec<Vec<T>> as 2D Array
    Vec<Vec<Vec<T>>> as 3D Array

Our focus will be on 2D array and 3D array creation like:

    zeros()

that can return 2D array filled with zeros. Is it possible?
we will se.

If we only use `Vec<Vec<T>>` or `Vec<Vec<Vec<T>>>`, we will
not have an information about the array it self. Like, it is
two dimensional or tree dimensional?

Yeah, but we don't need to keep that information right?
let's see.

btw, we can't use one function `zeros()` to build zero-valued
two dimensional and three dimensional array because rust
is [doesn't support function overloading](https://blog.rust-lang.org/2015/05/11/traits.html).

> *Overloading*. Rust does not support traditional overloading
> where the same method is defined with multiple signatures.
> But traits provide much of the benefit of overloading: if
> a method is defined generically over a trait, it can be
> called with any type implementing that trait. Compared to
> traditional overloading, this has two advantages. First,
> it means the overloading is less ad hoc: once you understand
> a trait, you immediately understand the overloading pattern
> of any APIs using it. Second, it is extensible: you can
> effectively provide new overloads downstream from a method
> by providing new trait implementations.

So, we can use trait to solve this right? but how?

# with_size, with_shape, one_dim, two_dim ...

We will replace `Vec::with_size(x)` and `Vec::with_shape(&[x])`
with `Vec::one_dim(x)`, `Vec::two_dim(a,b)`, `Vec::three_dim(a,b,c)`
and `Vec::four_dim(a,b,c,d)`. I think it's more intuitive.

For example:

```
let a: Vec<Vec<i64>> = Vec::two_dim(x, y)
```

It clearly says, initialize 2D vector. Oke, Nice!

# Developer Experience first
I think we already got the basics, we need to follow numpy docs
first before continue adding new feature.

Read the Numpy docs -> Modify the docs for `np` -> Write
and example.

# Store N-Dimensional Array as One array
Btw, we can store n-dimensional array as one array.

> More generally, in a k-dimensional array, the address of an
> element with indices i1, i2, ..., ik is
>
> B + c1 · i1 + c2 · i2 + ... + ck · ik.
>

From [wikipedia](https://en.wikipedia.org/wiki/Array_data_structure#Multidimensional_arrays)

If we build something like that, we can access the data using `arr[i][j][k]`
syntax right? I don't know let's try.

We stuck on this

    impl<T> Index<usize> for Array<T> {
        type Output = T;

        fn index(&self, index: usize) -> &T {
            &self.data[index]
            // Index::index(&*self, index)
        }
    }

How to implement the `[i][j]` part? if we returns the element
then we can't do reindex. What we need is:

    dimension -> syntax -> expected returns type
    if ndim == 1 -> arr[i] -> T
    if ndim == 2 -> arr[i] -> Arrray<T>
    if ndim == 2 -> arr[i][j] -> T
    if ndim == 3 -> arr[i] -> Arrray<T>
    if ndim == 3 -> arr[i][j] -> Arrray<T>
    if ndim == 3 -> arr[i][j][k] -> T

Btw berarti kalo mau `arr[i][j]` berarti `Array<Array<T>>` dong ya?
bener juga hmmm ...

nah yg tak pengen dia tetep `Array<T>` si.


# Rust Module
Btw, we can move the vector builder to it's own module: `builder`.

Example of code the structure: [rand]

[rand]: https://github.com/rust-random/rand/tree/master/src

and we can split the builder into their own file like: `zeros.rs`,
`ones.rs`, `full.rs` and so on. We will add the test on each file
like the following.

# Rename the project to Gibbs

Gibbs is named after [Josiah Willard Gibbs](https://www.britannica.com/science/vector-mathematics).

Hmmmmmmm

The name is very confusing:
[https://www.google.com/search?q=Gibbs](https://www.google.com/search?q=Gibbs).

I think it's not appropriate to use that name. Let's brainstorming first:

```
vector
multidimensional
vector extension
VectorX
```

Anw we can use [name generator](https://namelix.com/app/?keywords=science+vector+operations)
wkwk.

```
vectwise
hivector
solvect
savec
wavec
covect
solvecto
covecto
eigenic
```

fuk ternyata susah banget cari nama ASW.
udah ah np aja. tai.

I think I'm gonna use Gulali as a name,
Gulali is an Indonesian-flavored candy.
It's sweet and everyone favorite. I think it
describe how this library so swell. I want to
make library that easy-to-use and learn.


# Vector creation

We will use [Numpy Array Creation Routines] as the reference.

We already implement:

```
ones()
zeros()
full()
```

NumPy array creation routines:

```
empty(shape[, dtype, order])	Return a new array of given shape and type, without initializing entries.
empty_like(prototype[, dtype, order, subok])	Return a new array with the same shape and type as a given array.
eye(N[, M, k, dtype, order])	Return a 2-D array with ones on the diagonal and zeros elsewhere.
identity(n[, dtype])	Return the identity array.
ones_like(a[, dtype, order, subok])	Return an array of ones with the same shape and type as a given array.
zeros_like(a[, dtype, order, subok])	Return an array of zeros with the same shape and type as a given array.
full_like(a, fill_value[, dtype, order, subok])	Return a full array with the same shape and type as a given array.
```

anw kita fokus ke [Numpy Array Creation] dulu untuk edit `src/builders/mod.rs`.
keyh.


[Numpy Array Creation]: https://docs.scipy.org/doc/numpy-1.16.1/user/basics.creation.html
[Numpy Array Creation Routines]: https://docs.scipy.org/doc/numpy-1.16.1/reference/routines.array-creation.html

Let's create `np.arange` API:

```
numpy.arange([start, ]stop, [step, ]dtype=None) -> returns 1D array
```

```
>>> np.arange(3)
array([0, 1, 2])
>>> np.arange(3.0)
array([ 0.,  1.,  2.])
>>> np.arange(3,7)
array([3, 4, 5, 6])
>>> np.arange(3,7,2)
array([3, 5])
```

So maybe we ...


Oke `range()` udah. Sip.

# Init vs Generate
Currently we use the following API to create range vector:

```rust
let range2: Vec<f64> = Vec::range()
    .start_at(1.0)
    .stop_at(3.0)
    .init();
assert_eq!(range2, [1.0, 2.0]);
```

Inspired by [MATLAB linspace()], we can use `generate` instead
of `init`. It is far more intuitive. Ok.

[MATLAB linspace()]: https://www.mathworks.com/help/matlab/ref/linspace.html


# Rust error handling
btw how to handle the error?

I have a function like the following:

```
pub fn generate(self) -> Vec<T>
```

it generates `Vec<T>` based on `self`.

There are two cases:
1. `stop` value should be specified. otherwise it returns None.
2. `start >= stop` it returns None.

# API Dilemma
So, currently we have the following API to generate multidimensional
vector:

```rust
let arr: Vec<u8> = Vec::one_dim(2).ones();
let arr: Vec<Vec<u8>> = Vec::two_dim(2, 2).ones();
let arr: Vec<Vec<Vec<u8>>> = Vec::three_dim(2, 2, 2).ones();
let arr: Vec<Vec<Vec<Vec<u8>>>> = Vec::four_dim(2, 2, 2, 2).ones();
```

and the following api to generate numerical range vector:

```rust
let range: Vec<f64> = Vec::range()
    .start_at(1.0)
    .stop_at(3.0)
    .init();

let lin: Vec<f64> = Vec::linspace()
    .start_at(1.0)
    .stop_at(3.0)
    .with_size(4)
    .generate()
    .unwrap();
```

first of all it's not consistent.

Currently:

```rust
// Create two-dimensional vector with shape [3, 3]
// filled with zeros
let matrix: Vec<Vec<i32>> = Vec::two_dim(3, 3).zeros();
```

We should say **Generate** for consistency:

```rust
// Generate n-dimensional vector with shape [3, 3]
// filled with zeros
let matrix: Vec<Vec<i32>> = Vec::zeros()
    .with_shape([3, 3])
    .generate();
```

anw how about `full`?

```rust
let matrix: Vec<Vec<i32>> = Vec::full()
    .of_value(2.5)
    .with_shape([2, 2])
    .generate();
```

hmmm i think it's not intuitive. Let's get inspiration.

Numpy

```python
# Return a new array of given shape and type, filled with fill_value.
np.full((2, 2), np.inf)
np.full((2, 2), 10)
```

hmm I can't find any
[related function](https://www.mathworks.com/help/matlab/matrices-and-arrays.html)
in matlab.

Wording:

- Return a new array of given shape and type, filled with `fill_value`.
- Generate a new vector of given shape, filled with `fill_value`.

```rust
let matrix: Vec<Vec<i32>> = Vec::new()
    .with_shape([2, 2])
    .full_of(2)
    .generate();

// Generate a new m-dimensional vector of given shape,
// filled with zeros
let matrix: Vec<Vec<i32>> = Vec::new()
    .with_shape([2, 2])
    .zeros()
    .generate();
```

Mantul! Jadi gini ya

```rust
// Generate a new n-dimensional vector of given shape,
// filled with zeros
let matrix: Vec<Vec<i32>> = Vec::new()
    .with_shape([2, 2])
    .zeros()
    .generate();

// Generate a new n-dimensional vector of given shape,
// filled with ones
let matrix: Vec<Vec<i32>> = Vec::new()
    .with_shape([2, 2])
    .ones()
    .generate();

// Generate a new n-dimensional vector of given shape,
// filled with 2
let matrix: Vec<Vec<i32>> = Vec::new()
    .with_shape([2, 2])
    .full_of(2)
    .generate();

// Generate a new n-dimensional vector of given shape,
// filled with 2
let matrix: Vec<Vec<i32>> = Vec::new()
    .with_shape([2, 2])
    .full_of(2)
    .generate();

// Generate a new range vector
let range: Vec<f64> = Vec::new()
    .range()
    .start_at(1.0)
    .stop_at(3.0)
    .generate();

// Generate a new linearly spaced vector
let lin: Vec<f64> = Vec::new()
    .linspace()
    .start_at(1.0)
    .stop_at(3.0)
    .with_size(4)
    .generate();
```

Btw, we can't use `Vec::new()` because it returns
`Vec<T>`. Not works for n-dimensional vector.

How about:

```rust
Vec::ndim(); // n-dimensional vector
Vec::new_ndim() // new n-dimensional vector;
```

So, we will revisit the API like the following:


```rust
// Generate a new n-dimensional vector of given shape,
// filled with zeros
let matrix: Vec<Vec<i32>> = Vec::ndim()
    .with_shape([2, 2])
    .zeros()
    .generate();

// Generate a new n-dimensional vector of given shape,
// filled with ones
let matrix: Vec<Vec<i32>> = Vec::ndim()
    .with_shape([2, 2])
    .ones()
    .generate();

// Generate a new n-dimensional vector of given shape,
// filled with 2
let matrix: Vec<Vec<i32>> = Vec::ndim()
    .with_shape([2, 2])
    .full_of(2)
    .generate();

// Generate a new n-dimensional vector of given shape,
// filled with 2
let matrix: Vec<Vec<i32>> = Vec::ndim()
    .with_shape([2, 2])
    .full_of(2)
    .generate();

// Generate a new range vector
let range: Vec<f64> = Vec::ndim()
    .range()
    .start_at(1.0)
    .stop_at(3.0)
    .generate();

// Generate a new linearly spaced vector
let lin: Vec<f64> = Vec::ndim()
    .linspace()
    .start_at(1.0)
    .stop_at(3.0)
    .with_size(4)
    .generate();
```

Oh forgot, we can't use the same method trait like this:

```
let arr: Vec<u8> = Vec::ndim();
let arr: Vec<Vec<u8>> = Vec::ndim();
let arr: Vec<Vec<Vec<u8>>> = Vec::ndim();
let arr: Vec<Vec<Vec<Vec<u8>>>> = Vec::ndim();
```

if we implement `ndim` for other `Vec<T>`, it will raise
a conflict:

```
error[E0119]: conflicting implementations of trait `builders::NDimensional` for type `std::vec::Vec<std::vec::Vec<_>>`:
   --> src/builders/mod.rs:167:1
    |
158 | / impl<T> NDimensional for Vec<T>
159 | | where
160 | |     T: Num + FromPrimitive + Copy,
161 | | {
...   |
164 | |     }
165 | | }
    | |_- first implementation here
166 |
167 | / impl<T> NDimensional for Vec<Vec<T>>
168 | | where
169 | |     T: Num + FromPrimitive + Copy,
170 | | {
...   |
173 | |     }
174 | | }
    | |_^ conflicting implementation for `std::vec::Vec<std::vec::Vec<_>>`
    |
    = note: upstream crates may add new impl of trait `num::Num` for type `std::vec::Vec<_>` in future versions
    = note: upstream crates may add new impl of trait `num::FromPrimitive` for type `std::vec::Vec<_>` in future versions
    = note: upstream crates may add new impl of trait `std::marker::Copy` for type `std::vec::Vec<_>` in future versions
```

so our previous solution works perfectly:

```
let arr: Vec<u8> = Vec::one_dim();
let arr: Vec<Vec<u8>> = Vec::two_dim();
let arr: Vec<Vec<Vec<u8>>> = Vec::three_dim();
let arr: Vec<Vec<Vec<Vec<u8>>>> = Vec::four_dim();
```

OK.

Hmmm let's modify it a litle bit.

Instead of returns `Vec<T>` in `one_dim()`,
we may return a new struct.